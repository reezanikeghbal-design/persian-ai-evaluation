from dataclasses import dataclass
from ..core.text import normalize_persian


@dataclass
class Chunk:
    id: str
    text: str
    source: str


def split_text(text: str, chunk_size: int = 500, overlap: int = 80, source: str = "unknown") -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    clean = normalize_persian(text)
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        piece = clean[start:end].strip()
        if piece:
            chunks.append(Chunk(id=f"{source}:{len(chunks)}", text=piece, source=source))
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks
