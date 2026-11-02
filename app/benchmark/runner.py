import csv
import json
import logging
import yaml
from sqlalchemy import select
from pathlib import Path
from statistics import mean
from app.evaluation.dataset import load_jsonl, validate_dataset
from app.evaluation.metrics import answer_relevancy, faithfulness_heuristic, recall_at_k, mean_reciprocal_rank
from app.providers.factory import get_provider
from app.rag.engine import RAGEngine
from app.database.models import Experiment, EvaluationResult, DatasetMetadata, get_session_factory

logger = logging.getLogger(__name__)


def estimate_cost(model: str, rows: list[dict], pricing_path: str = "config/model_pricing.yaml") -> float:
    """Estimate USD from provider-reported tokens; unknown prices/usages remain unpriced."""
    path = Path(pricing_path)
    if not path.exists():
        return 0.0
    prices = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model_price = prices.get("models", {}).get(model)
    if not model_price:
        return 0.0
    total = 0.0
    for row in rows:
        input_tokens, output_tokens = row.get("input_tokens"), row.get("output_tokens")
        if input_tokens is not None:
            total += input_tokens * float(model_price.get("input_per_1m", 0)) / 1_000_000
        if output_tokens is not None:
            total += output_tokens * float(model_price.get("output_per_1m", 0)) / 1_000_000
    return round(total, 8)


class BenchmarkRunner:
    def __init__(self, database_url: str | None = None):
        self.Session = get_session_factory() if database_url is None else None
        if database_url:
            from app.database.models import configure_database
            self.Session = configure_database(database_url)

    def run(self, dataset_path="datasets/golden_dataset.jsonl", provider_name="mock", model=None,
            embedding_model="hashing", chunk_size=500, chunk_overlap=80, top_k=3):
        records = load_jsonl(dataset_path)
        errors = validate_dataset(records)
        if errors:
            raise ValueError("Dataset validation failed: " + "; ".join(errors))
        provider = get_provider(provider_name, model)
        rag = RAGEngine("datasets/sample_documents", chunk_size, chunk_overlap, embedding_model)
        rows = []
        for item in records:
            retrieved = rag.retrieve(item["question"], top_k)
            context = "\n".join(chunk.text for chunk, _ in retrieved)
            generated = provider.generate(item["question"], context)
            rel = answer_relevancy(item["question"], generated["answer"])
            faith = faithfulness_heuristic(generated["answer"], context)
            relevant_source = item.get("relevant_chunk_id", "").split(":", 1)[0]
            relevant = {chunk.id for chunk in rag.retriever.chunks if chunk.source == relevant_source}
            ids = [chunk.id for chunk, _ in retrieved]
            recall = recall_at_k(ids, relevant, top_k) if relevant else 0.0
            mrr = mean_reciprocal_rank(ids, relevant) if relevant else 0.0
            rows.append({"question_id": item["id"], "question": item["question"],
                         "reference_answer": item["reference_answer"], "generated_answer": generated["answer"],
                         "retrieved_context": context, "relevancy": rel, "faithfulness": faith,
                         "hallucination": 1.0 if faith < 0.5 else 0.0, "recall_at_k": recall, "mrr": mrr, "latency_ms": generated["latency_ms"],
                         "input_tokens": generated["input_tokens"], "output_tokens": generated["output_tokens"]})
        summary = {"provider": provider.name, "model": provider.model, "embedding_model": embedding_model,
                   "total_questions": len(rows), "average_latency_ms": mean(r["latency_ms"] for r in rows) if rows else 0,
                   "answer_relevancy": mean(r["relevancy"] for r in rows) if rows else 0,
                   "faithfulness": mean(r["faithfulness"] for r in rows) if rows else 0,
                   "hallucination_rate": mean(r["hallucination"] for r in rows) if rows else 0,
                   "recall_at_k": mean(r["recall_at_k"] for r in rows) if rows else 0,
                   "mrr": mean(r["mrr"] for r in rows) if rows else 0,
                   "estimated_cost_usd": estimate_cost(provider.model, rows)}
        with self.Session() as session:
            dataset_name = Path(dataset_path).stem
            metadata = session.scalar(select(DatasetMetadata).where(DatasetMetadata.name == dataset_name))
            if metadata is None:
                session.add(DatasetMetadata(name=dataset_name, version="v1", record_count=len(records)))
            experiment = Experiment(provider=provider.name, model=provider.model, embedding_model=embedding_model,
                                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k, total_questions=len(rows),
                                    average_latency_ms=summary["average_latency_ms"],
                                    answer_relevancy=summary["answer_relevancy"], faithfulness=summary["faithfulness"],
                                    hallucination_rate=summary["hallucination_rate"], recall_at_k=summary["recall_at_k"], mrr=summary["mrr"],
                                    estimated_cost_usd=summary["estimated_cost_usd"])
            session.add(experiment)
            session.flush()
            for row in rows:
                session.add(EvaluationResult(experiment_id=experiment.id, **row))
            session.commit()
            experiment_id = experiment.id
        output_dir = Path("artifacts")
        output_dir.mkdir(exist_ok=True)
        with (output_dir / f"experiment_{experiment_id}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader(); writer.writerows(rows)
        (output_dir / f"experiment_{experiment_id}.json").write_text(
            json.dumps({"experiment_id": experiment_id, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log_mlflow(experiment_id, summary, chunk_size, top_k, embedding_model)
        return {"experiment_id": experiment_id, **summary, "results": rows}

    @staticmethod
    def _log_mlflow(experiment_id, summary, chunk_size, top_k, embedding_model):
        try:
            from app.core.config import get_settings
            settings = get_settings()
            if not settings.enable_mlflow:
                return
            import mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("persian-ai-evaluation")
            with mlflow.start_run(run_name=f"benchmark-{experiment_id}"):
                mlflow.log_params({"model": summary["model"], "embedding_model": embedding_model,
                                   "chunk_size": chunk_size, "top_k": top_k, "dataset_version": "v1"})
                mlflow.log_metrics({k: summary[k] for k in ("recall_at_k", "mrr", "answer_relevancy", "faithfulness", "average_latency_ms", "estimated_cost_usd")})
                for artifact in Path("artifacts").glob(f"experiment_{experiment_id}.*"):
                    mlflow.log_artifact(str(artifact))
        except Exception as exc:
            logger.warning("MLflow logging skipped; SQLite results are saved: %s", exc)
