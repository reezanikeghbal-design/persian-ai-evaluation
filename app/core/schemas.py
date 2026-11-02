from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    dataset_path: str = "datasets/golden_dataset.jsonl"
    provider: str = "mock"
    model: str | None = None
    embedding_model: str = "hashing"
    chunk_size: int = Field(default=500, ge=100, le=3000)
    chunk_overlap: int = Field(default=80, ge=0, le=500)
    top_k: int = Field(default=3, ge=1, le=20)


class ExperimentSummary(BaseModel):
    id: int
    created_at: str
    provider: str
    model: str
    total_questions: int
    average_latency_ms: float
    answer_relevancy: float
    faithfulness: float
    recall_at_k: float
    mrr: float
