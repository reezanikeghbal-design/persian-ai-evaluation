from app.rag.engine import RAGEngine
from app.providers.mock import MockProvider


def test_rag_retrieves_persian_document():
    engine = RAGEngine("datasets/sample_documents", chunk_size=500, chunk_overlap=80, embedding_model="hashing")
    results = engine.retrieve("Docker کانتینر برنامه", top_k=3)
    assert results
    assert any("technology.txt" in chunk.source for chunk, _ in results)


def test_mock_provider_is_offline_and_returns_contract():
    result = MockProvider().generate("پرسش", "متن زمینه")
    assert result["answer"] == "متن زمینه"
    assert result["model"]
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["latency_ms"] >= 0
