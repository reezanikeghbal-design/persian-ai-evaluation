from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine, String, Float, Integer, DateTime, ForeignKey, Text, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    provider: Mapped[str] = mapped_column(String(80))
    model: Mapped[str] = mapped_column(String(160))
    embedding_model: Mapped[str] = mapped_column(String(200))
    dataset_version: Mapped[str] = mapped_column(String(80), default="v1")
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=80)
    top_k: Mapped[int] = mapped_column(Integer)
    total_questions: Mapped[int] = mapped_column(Integer)
    average_latency_ms: Mapped[float] = mapped_column(Float, default=0)
    answer_relevancy: Mapped[float] = mapped_column(Float, default=0)
    faithfulness: Mapped[float] = mapped_column(Float, default=0)
    hallucination_rate: Mapped[float] = mapped_column(Float, default=0)
    recall_at_k: Mapped[float] = mapped_column(Float, default=0)
    mrr: Mapped[float] = mapped_column(Float, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0)
    results: Mapped[list["EvaluationResult"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    question_id: Mapped[str] = mapped_column(String(80))
    question: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str] = mapped_column(Text)
    generated_answer: Mapped[str] = mapped_column(Text)
    retrieved_context: Mapped[str] = mapped_column(Text)
    relevancy: Mapped[float] = mapped_column(Float)
    faithfulness: Mapped[float] = mapped_column(Float)
    hallucination: Mapped[float] = mapped_column(Float, default=0)
    recall_at_k: Mapped[float] = mapped_column(Float)
    mrr: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    experiment: Mapped[Experiment] = relationship(back_populates="results")


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    version: Mapped[str] = mapped_column(String(80))
    record_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


_engine = None
_Session = None


def configure_database(url: str | None = None):
    global _engine, _Session
    db_url = url or get_settings().database_url
    kwargs = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    if db_url.startswith("sqlite:///") and ":memory:" not in db_url:
        database_path = db_url.removeprefix("sqlite:///")
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(db_url, connect_args=kwargs)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    return _Session


def get_session_factory():
    global _Session
    if _Session is None:
        configure_database()
    return _Session
