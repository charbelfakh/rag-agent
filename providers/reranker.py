"""Cross-encoder reranking and no-op passthrough when reranking is disabled."""
import os

from providers.base import Reranker

RERANKER_MAX_CHARS = int(os.getenv("RERANKER_MAX_CHARS", "512"))


def truncate_for_rerank(text: str, max_chars: int | None = None) -> str:
    """Truncate chunk text for cross-encoder input; full text stays in the chunk dict."""
    limit = RERANKER_MAX_CHARS if max_chars is None else max_chars
    if len(text) <= limit:
        return text
    return text[:limit]


def _attach_rerank_scores(chunks: list[dict], scores: list[float], top_n: int) -> list[dict]:
    """Preserve vector scores and attach rerank scores; ``score`` stays the ranking value."""
    ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    result = []
    for chunk, rerank_score in ranked[:top_n]:
        vector_score = float(chunk.get("vector_score", chunk.get("score", 0.0)))
        score = float(rerank_score)
        result.append(
            {
                **chunk,
                "vector_score": vector_score,
                "rerank_score": score,
                "score": score,
            }
        )
    return result


class NoOpReranker(Reranker):
    """Passthrough reranker that preserves vector scores when reranking is off."""

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        selected = chunks[:top_n]
        return [
            {
                **chunk,
                "vector_score": float(chunk.get("score", 0.0)),
                "rerank_score": float(chunk.get("score", 0.0)),
            }
            for chunk in selected
        ]


class CrossEncoderReranker(Reranker):
    """Re-score retrieved chunks with a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        # CPU keeps GPU VRAM available for embed/LLM workloads.
        self._model = CrossEncoder(self.model_name, device="cpu")

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        if not chunks:
            return []

        pairs = [
            (query, truncate_for_rerank(chunk.get("text", "")))
            for chunk in chunks
        ]
        scores = self._model.predict(pairs)

        return _attach_rerank_scores(chunks, list(scores), top_n)
