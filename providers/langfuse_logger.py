"""Fire-and-forget Langfuse observability for cache hits and full query analytics."""
import logging
import os
import threading

logger = logging.getLogger(__name__)

_langfuse = None


def _is_analytics_enabled() -> bool:
    return os.getenv("ANALYTICS_ENABLED", "false").lower() in ("true", "1", "yes")


def _get_langfuse():
    global _langfuse
    if _langfuse is None:
        if not os.getenv("LANGFUSE_SECRET_KEY") or not os.getenv("LANGFUSE_PUBLIC_KEY"):
            return None
        from langfuse import Langfuse

        _langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or "http://localhost:3000",
        )
    return _langfuse


def log_cache_hit(question: str, answer: str, similarity: float) -> None:
    """Record a semantic-cache hit in Langfuse when configured."""
    try:
        lf = _get_langfuse()
        if lf is None:
            return
        span = lf.start_observation(
            name="rag_query",
            input={"question": question},
            metadata={
                "cache_hit": True,
                "similarity": similarity,
                "tags": ["cache_hit"],
            },
        )
        span.update(
            output={"answer": answer},
            metadata={"cache_hit": True, "similarity": similarity},
        )
        span.end()
        lf.flush()
    except Exception:
        pass


_STAGE_SPANS: tuple[tuple[str, str], ...] = (
    ("embed", "embed_ms"),
    ("hyde", "hyde_ms"),
    ("qdrant_search", "qdrant_search_ms"),
    ("rerank", "rerank_ms"),
    ("pre_stream", "pre_stream_ms"),
    ("llm_ttft", "llm_ttft_ms"),
    ("llm_generation", "llm_generation_ms"),
)


def _start_stage_span(parent, lf, stage_name: str, duration_ms: int):
    kwargs: dict = {
        "name": stage_name,
        "metadata": {"duration_ms": duration_ms},
    }
    start_child = getattr(parent, "start_observation", None) if parent is not None else None
    if callable(start_child):
        span = start_child(**kwargs)
    else:
        span = lf.start_observation(**kwargs)
    span.end()
    return span


def _log_query_sync(data: dict) -> None:
    try:
        lf = _get_langfuse()
        if lf is None:
            return
        metadata = {k: v for k, v in data.items() if k != "question"}
        parent = lf.start_observation(
            name="rag_query",
            input={"question": data["question"]},
            metadata=metadata,
        )
        for stage_name, ms_key in _STAGE_SPANS:
            duration_ms = data.get(ms_key)
            if duration_ms is None:
                continue
            _start_stage_span(parent, lf, stage_name, int(duration_ms))
        parent.update(
            output={"answer_length": data.get("answer_length")},
            metadata=metadata,
        )
        parent.end()
        lf.flush()
    except Exception as exc:
        logger.warning("Langfuse query log failed: %s", exc)


def log_feedback(
    *,
    question: str,
    answer: str,
    rating: int,
    trace_id: str | None = None,
    comment: str | None = None,
) -> None:
    """Post user thumbs-up/down to Langfuse on a background thread."""
    if not _is_analytics_enabled():
        return

    def _sync() -> None:
        try:
            lf = _get_langfuse()
            if lf is None:
                return
            value = 1.0 if rating > 0 else 0.0
            lf.create_score(
                name="user_feedback",
                value=value,
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=comment or f"rating={rating}",
                metadata={
                    "question": question[:500],
                    "answer_length": len(answer),
                    "rating": rating,
                },
            )
            lf.flush()
        except Exception as exc:
            logger.warning("Langfuse feedback log failed: %s", exc)

    threading.Thread(target=_sync, daemon=True).start()


def log_query(
    *,
    question: str,
    trace_id: str | None = None,
    hyde_used: bool,
    cache_hit: bool,
    web_fallback_used: bool,
    web_fallback_reason: str | None,
    sufficiency_check_result: str,
    chunks_retrieved: int | None,
    chunks_after_rerank: int | None,
    chunks_in_prompt: int = 0,
    top_chunk_score: float | None,
    top_chunk_source: str | None,
    top_chunk_page: int | None,
    top_chunk_vendor: str | None = None,
    top_chunk_document_type: str | None = None,
    top_vector_score: float | None = None,
    mean_vector_score: float | None = None,
    top_rerank_score: float | None = None,
    mean_rerank_score: float | None = None,
    embed_ms: int | None = None,
    hyde_ms: int | None = None,
    qdrant_search_ms: int | None = None,
    rerank_ms: int | None = None,
    prompt_token_count: int | None = None,
    pre_stream_ms: int | None = None,
    llm_ttft_ms: int | None = None,
    llm_generation_ms: int | None = None,
    prompt_eval_count: int | None = None,
    answer_token_count: int | None = None,
    cache_skipped_reason: str | None = None,
    citations_count: int = 0,
    insufficient_answer: bool = False,
    answer_outcome: str = "ok",
    generation_error: str | None = None,
    empty_retrieval: bool = False,
    answer_length: int,
    latency_ms: int,
    history_turns: int,
) -> None:
    """Emit full query analytics to Langfuse on a background thread."""
    if not _is_analytics_enabled():
        return
    payload = {
        "question": question,
        "trace_id": trace_id,
        "hyde_used": hyde_used,
        "cache_hit": cache_hit,
        "web_fallback_used": web_fallback_used,
        "web_fallback_reason": web_fallback_reason,
        "sufficiency_check_result": sufficiency_check_result,
        "chunks_retrieved": chunks_retrieved,
        "chunks_after_rerank": chunks_after_rerank,
        "chunks_in_prompt": chunks_in_prompt,
        "top_chunk_score": top_chunk_score,
        "top_chunk_source": top_chunk_source,
        "top_chunk_page": top_chunk_page,
        "top_chunk_vendor": top_chunk_vendor,
        "top_chunk_document_type": top_chunk_document_type,
        "top_vector_score": top_vector_score,
        "mean_vector_score": mean_vector_score,
        "top_rerank_score": top_rerank_score,
        "mean_rerank_score": mean_rerank_score,
        "embed_ms": embed_ms,
        "hyde_ms": hyde_ms,
        "qdrant_search_ms": qdrant_search_ms,
        "rerank_ms": rerank_ms,
        "prompt_token_count": prompt_token_count,
        "pre_stream_ms": pre_stream_ms,
        "llm_ttft_ms": llm_ttft_ms,
        "llm_generation_ms": llm_generation_ms,
        "prompt_eval_count": prompt_eval_count,
        "answer_token_count": answer_token_count,
        "cache_skipped_reason": cache_skipped_reason,
        "citations_count": citations_count,
        "insufficient_answer": insufficient_answer,
        "answer_outcome": answer_outcome,
        "empty_retrieval": empty_retrieval,
        "answer_length": answer_length,
        "latency_ms": latency_ms,
        "history_turns": history_turns,
    }
    if generation_error:
        payload["generation_error"] = generation_error
    threading.Thread(target=_log_query_sync, args=(payload,), daemon=True).start()
