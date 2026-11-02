from fastapi import FastAPI, HTTPException
from sqlalchemy import select, func
from app.core.schemas import BenchmarkRequest
from app.database.models import Experiment, EvaluationResult, get_session_factory
from app.benchmark.runner import BenchmarkRunner
from app.evaluation.dataset import load_jsonl

app = FastAPI(title="Persian AI Evaluation API", version="0.1.0", description="Persian LLM/RAG benchmark MVP")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/datasets")
def datasets():
    return [{"name": "golden_dataset", "path": "datasets/golden_dataset.jsonl",
             "records": len(load_jsonl("datasets/golden_dataset.jsonl"))}]


@app.get("/models")
def models():
    return [{"provider": "mock", "model": "mock-context-extractive-v1"},
            {"provider": "openai-compatible", "model": "configured by OPENAI_MODEL"},
            {"provider": "gemini", "model": "configured by GEMINI_MODEL"}]


@app.post("/experiments/run")
def run_experiment(request: BenchmarkRequest):
    try:
        return BenchmarkRunner().run(request.dataset_path, request.provider, request.model,
                                     request.embedding_model, request.chunk_size,
                                     request.chunk_overlap, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/experiments")
def experiments():
    with get_session_factory()() as session:
        items = session.scalars(select(Experiment).order_by(Experiment.id.desc())).all()
        return [{"id": x.id, "created_at": x.created_at.isoformat(), "provider": x.provider,
                 "model": x.model, "total_questions": x.total_questions,
                 "average_latency_ms": x.average_latency_ms, "answer_relevancy": x.answer_relevancy,
                 "faithfulness": x.faithfulness, "hallucination_rate": x.hallucination_rate,
                 "recall_at_k": x.recall_at_k, "mrr": x.mrr,
                 "estimated_cost_usd": x.estimated_cost_usd} for x in items]


@app.get("/experiments/{experiment_id}")
def experiment_detail(experiment_id: int):
    with get_session_factory()() as session:
        item = session.get(Experiment, experiment_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return {"id": item.id, "provider": item.provider, "model": item.model,
                "total_questions": item.total_questions, "answer_relevancy": item.answer_relevancy,
                "faithfulness": item.faithfulness, "hallucination_rate": item.hallucination_rate,
                "recall_at_k": item.recall_at_k, "mrr": item.mrr}


@app.get("/results/{experiment_id}")
def results(experiment_id: int):
    with get_session_factory()() as session:
        rows = session.scalars(select(EvaluationResult).where(EvaluationResult.experiment_id == experiment_id)).all()
        return [{"question_id": r.question_id, "question": r.question, "reference_answer": r.reference_answer,
                 "generated_answer": r.generated_answer, "retrieved_context": r.retrieved_context,
                 "relevancy": r.relevancy, "faithfulness": r.faithfulness, "hallucination": r.hallucination,
                 "recall_at_k": r.recall_at_k,
                 "mrr": r.mrr, "latency_ms": r.latency_ms} for r in rows]


@app.get("/metrics/summary")
def metrics_summary():
    with get_session_factory()() as session:
        count, avg_rel, avg_faith, avg_latency = session.query(
            func.count(Experiment.id), func.avg(Experiment.answer_relevancy),
            func.avg(Experiment.faithfulness), func.avg(Experiment.average_latency_ms)).one()
        return {"experiments": count, "average_answer_relevancy": avg_rel or 0,
                "average_faithfulness": avg_faith or 0, "average_latency_ms": avg_latency or 0}
