"""Public sync RAG entry point for programmatic callers (e.g. a future GraphQL layer)."""
from __future__ import annotations

import threading
from typing import Iterator

from providers.rag_pipeline import query, stream_query


def answer(
    question: str,
    *,
    top_k: int | None = None,
    vendor_filter: str | None = None,
    document_type_filter: str | None = None,
) -> dict:
    """Public RAG entry point. Returns ``{"answer": str, "sources": list[dict]}``.

    Optional ``top_k`` and ``vendor_filter`` / ``document_type_filter`` mirror the
    REST ``/query`` knobs so non-SSE callers (e.g. the GraphQL layer) can scope
    retrieval. They are only forwarded when set, preserving the bare
    ``query(question)`` call for the default case.
    """
    if not question or not question.strip():
        raise ValueError("question must be non-empty")

    kwargs: dict = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if vendor_filter is not None:
        kwargs["vendor_filter"] = vendor_filter
    if document_type_filter is not None:
        kwargs["document_type_filter"] = document_type_filter

    result = query(question, **kwargs)
    return {
        "answer": result["answer"],
        "sources": result.get("citations") or [],
    }


def answer_stream(
    question: str,
    *,
    top_k: int | None = None,
    vendor_filter: str | None = None,
    document_type_filter: str | None = None,
    cancel_event: threading.Event | None = None,
) -> Iterator[dict]:
    """Streaming RAG entry point. Returns an iterator of normalized event dicts.

    Each event is one of:
      ``{"token": str}``        — an answer token (emit in order to the client)
      ``{"done": True, "sources": list[dict]}`` — terminal event with citations
      ``{"done": True, "error": str}``          — terminal error event

    Stage/meta events from the underlying pipeline are dropped; only tokens and
    the terminal done/error are surfaced, keeping the public contract minimal.
    Filters mirror :func:`answer` and are only forwarded when set.

    Raises ``ValueError`` at call time (not first iteration) for an empty
    question. ``cancel_event`` lets the caller stop LLM generation when the
    client goes away, mirroring the REST ``/query`` disconnect handling.
    """
    if not question or not question.strip():
        raise ValueError("question must be non-empty")

    kwargs: dict = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if vendor_filter is not None:
        kwargs["vendor_filter"] = vendor_filter
    if document_type_filter is not None:
        kwargs["document_type_filter"] = document_type_filter
    if cancel_event is not None:
        kwargs["cancel_event"] = cancel_event

    return _iter_stream_events(question, kwargs)


def _iter_stream_events(question: str, kwargs: dict) -> Iterator[dict]:
    for event in stream_query(question, **kwargs):
        if event.get("token") is not None:
            yield {"token": event["token"]}
        elif event.get("done"):
            if event.get("error"):
                yield {"done": True, "error": event["error"]}
            else:
                yield {"done": True, "sources": event.get("citations") or []}
