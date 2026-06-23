"""Public sync RAG entry point for programmatic callers (e.g. a future GraphQL layer)."""
from __future__ import annotations

from providers.rag_pipeline import query


def answer(question: str) -> dict:
    """Public RAG entry point. Returns ``{"answer": str, "sources": list[dict]}``."""
    if not question or not question.strip():
        raise ValueError("question must be non-empty")

    result = query(question)
    return {
        "answer": result["answer"],
        "sources": result.get("citations") or [],
    }
