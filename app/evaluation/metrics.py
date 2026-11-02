from collections.abc import Sequence
from ..core.text import normalize_persian


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant IDs retrieved in top-k; exact given correct relevance labels."""
    if not relevant_ids:
        return 0.0
    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of first relevant result; exact given relevance labels."""
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _tokens(text: str) -> set[str]:
    return {t for t in normalize_persian(text).lower().split() if len(t) > 1}


def answer_relevancy(question: str, answer: str) -> float:
    """Token-overlap heuristic, not a semantic or human-validated relevance score."""
    q, a = _tokens(question), _tokens(answer)
    if not q or not a:
        return 0.0
    return len(q & a) / len(q)


def faithfulness_heuristic(answer: str, context: str) -> float:
    """Share of answer tokens found in context; weak grounding proxy, not factual verification."""
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(context)
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def hallucination_rate(grounding_scores: Sequence[float], threshold: float = 0.5) -> float:
    if not grounding_scores:
        return 0.0
    return sum(score < threshold for score in grounding_scores) / len(grounding_scores)
