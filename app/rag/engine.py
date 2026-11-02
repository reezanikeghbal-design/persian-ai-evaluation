from pathlib import Path
from .chunker import split_text
from .retriever import VectorRetriever, HashingEmbedder, SentenceTransformerEmbedder


class RAGEngine:
    def __init__(self, documents_dir: str = "datasets/sample_documents", chunk_size: int = 500,
                 chunk_overlap: int = 80, embedding_model: str = "hashing"):
        chunks = []
        for path in sorted(Path(documents_dir).glob("*.txt")):
            chunks.extend(split_text(path.read_text(encoding="utf-8"), chunk_size, chunk_overlap, path.name))
        embedder = HashingEmbedder() if embedding_model == "hashing" else SentenceTransformerEmbedder(embedding_model)
        self.retriever = VectorRetriever(chunks, embedder)

    def retrieve(self, question: str, top_k: int = 3):
        return self.retriever.retrieve(question, top_k)
