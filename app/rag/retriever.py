import hashlib
import re
import numpy as np
from .chunker import Chunk


class HashingEmbedder:
    """Deterministic zero-dependency fallback embedder (lexical, not semantic)."""
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[\w\u0600-\u06ff]+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dimensions
                matrix[row, index] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class VectorRetriever:
    def __init__(self, chunks: list[Chunk], embedder=None):
        self.chunks = chunks
        self.embedder = embedder or HashingEmbedder()
        self.vectors = self.embedder.encode([chunk.text for chunk in chunks]) if chunks else np.zeros((0, 384), dtype=np.float32)
        self.index = None
        try:
            import faiss
            if len(self.vectors):
                self.index = faiss.IndexFlatIP(self.vectors.shape[1])
                self.index.add(self.vectors)
        except ImportError:
            pass

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        vector = self.embedder.encode([query])
        if self.index is not None:
            scores, indices = self.index.search(vector, min(top_k, len(self.chunks)))
            pairs = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            scores = self.vectors @ vector[0]
            pairs = sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.chunks[i], score) for i, score in pairs]
