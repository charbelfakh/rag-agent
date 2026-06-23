"""Retrieval orchestration between ``rag_pipeline`` and provider backends.

Runs embed, optional HyDE, semantic-cache lookup, Qdrant search, hybrid image merge,
and reranking. ``rag_pipeline`` builds the prompt; this module only assembles chunks.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import re

from providers.factory import get_embedder, get_llm, get_reranker, get_vector_store
from providers.hybrid_retrieval import (
    is_hybrid_retrieval_enabled,
    merge_text_and_image_hits,
    reciprocal_rank_fusion,
)
from providers.langfuse_logger import log_cache_hit
from providers.query_enhancements import (
    dynamic_rerank_top_n,
    is_dynamic_rerank_top_n_enabled,
    is_map_reduce_enabled,
    is_speculative_generation_enabled,
    map_reduce_search,
    speculative_prompt_chunks,
    start_rerank_future,
)
from providers.semantic_cache import get_semantic_cache

DEFINITIONAL_PATTERN = re.compile(
    r"\b(what is|what are|what's|define|explain|describe|tell me about)\b",
    re.IGNORECASE,
)

_VIDEO_DEMO_INTENT = re.compile(
    r"\b(demo|demonstration|tutorial|video|walkthrough|showcase|watch)\b",
    re.IGNORECASE,
)
_VIDEO_DEMO_TOPIC = re.compile(
    r"\b(bin[\s-]*pick(?:ing)?|3d\s*camera|mech[\s-]?eye|vision[\s-]?guided)\b",
    re.IGNORECASE,
)


def should_supplement_video_transcript_retrieval(question: str) -> bool:
    """Boost video transcript hits when the user asks for a demo/tutorial on a vision topic."""
    if not question.strip():
        return False
    return bool(_VIDEO_DEMO_INTENT.search(question) and _VIDEO_DEMO_TOPIC.search(question))


def is_hyde_enabled() -> bool:
    """Return whether HyDE hypothetical-document expansion is enabled."""
    return os.getenv("HYDE_ENABLED", "false").lower() in ("true", "1", "yes")


def _is_reranker_enabled() -> bool:
    return os.getenv("RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")


def is_two_stage_retrieval_enabled() -> bool:
    """Return whether bi-encoder fetch runs before cross-encoder rerank."""
    return os.getenv("TWO_STAGE_RETRIEVAL_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _retrieval_top_k(request_top_k: int) -> int:
    if is_two_stage_retrieval_enabled():
        return int(os.getenv("BI_ENCODER_TOP_K", "50"))
    if _is_reranker_enabled():
        return int(os.getenv("RERANKER_FETCH_K", "50"))
    return request_top_k


def _search_filter(
    vendor_filter: str | None,
    document_type_filter: str | None,
) -> dict | None:
    payload: dict[str, str] = {}
    if vendor_filter:
        payload["vendor"] = vendor_filter.strip().lower()
    if document_type_filter:
        payload["document_type"] = document_type_filter.strip().lower()
    return payload or None


def _rerank_score_floor() -> float:
    return float(os.getenv("RERANK_SCORE_FLOOR", "0"))


def filter_chunks_by_score_floor(chunks: list[dict]) -> list[dict]:
    """Drop chunks below ``RERANK_SCORE_FLOOR``; keep the top hit if all would be removed."""
    floor = _rerank_score_floor()
    if floor <= 0 or not chunks:
        return chunks
    kept = [c for c in chunks if float(c.get("score", 0.0)) >= floor]
    return kept if kept else chunks[:1]


def score_stats(chunks: list[dict], field: str = "score") -> tuple[float | None, float | None]:
    """Return top score and mean score for ``field`` across ``chunks`` (or ``None``, ``None``)."""
    if not chunks:
        return None, None
    scores = [float(chunk.get(field, 0.0)) for chunk in chunks]
    return scores[0], sum(scores) / len(scores)


HYDE_PROMPT = """You are an industrial automation expert. Write a short hypothetical passage (2-3 sentences maximum) that would answer the question below, as if it appeared in a technical manual for the relevant product or vendor. Reply with only the passage — no preamble, labels, or explanation.

Question: {question}"""


def get_search_text(question: str, llm) -> tuple[str, bool]:
    """Return retrieval text and whether HyDE replaced the raw question."""
    if is_hyde_enabled() and not DEFINITIONAL_PATTERN.search(question):
        passage = llm.generate(HYDE_PROMPT.format(question=question)).strip()
        return passage, True
    return question, False


def is_parallel_orchestrator_enabled() -> bool:
    """Return whether embed and HyDE may run concurrently (``QUERY_ORCHESTRATOR_ENABLED``)."""
    return os.getenv("QUERY_ORCHESTRATOR_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


@dataclass
class RetrievalContext:
    question: str
    top_k: int
    history_turns: int
    vendor_filter: str | None
    document_type_filter: str | None
    analytics: object


@dataclass
class RetrievalResult:
    chunks: list[dict]
    question_vector: list[float] | None
    search_text: str
    hyde_used: bool
    use_cache: bool
    cached_answer: str | None = None
    cache_similarity: float | None = None


class QueryOrchestrator:
    """Coordinate retrieval stages for one question (embed through rerank)."""

    def run_retrieval(self, ctx: RetrievalContext) -> RetrievalResult:
        """Embed, optionally hit the semantic cache, search Qdrant, and rerank.

        Returns cached answer metadata when the cache hits; otherwise ranked chunks
        and the vectors/text used for search.
        """
        embedder = get_embedder()
        # Eager singleton init: store may probe embed dimensions even on a cache hit.
        store = get_vector_store()
        llm = get_llm()
        cache = get_semantic_cache()
        # History invalidates cache keys; vendor/product filters disable cache in rag_pipeline.
        use_cache = cache.enabled and ctx.history_turns == 0
        if cache.enabled and ctx.history_turns > 0:
            ctx.analytics.cache_skipped_reason = "history"
            cache.record_skip("history")

        question_vector: list[float] | None = None
        search_text = ctx.question
        hyde_used = False
        embed_ms = 0

        hyde_eligible = is_hyde_enabled() and not DEFINITIONAL_PATTERN.search(ctx.question)
        parallel_hyde = is_parallel_orchestrator_enabled() and hyde_eligible

        if use_cache or parallel_hyde:
            # Overlap question embed with HyDE LLM call to hide HyDE latency before search.
            executor = ThreadPoolExecutor(max_workers=2)
            t_embed = time.perf_counter()
            embed_future = executor.submit(embedder.embed, [ctx.question])
            hyde_future = (
                executor.submit(get_search_text, ctx.question, llm)
                if parallel_hyde
                else None
            )
            question_vector = embed_future.result()[0]
            embed_ms = int((time.perf_counter() - t_embed) * 1000)

            if use_cache:
                hit = cache.lookup(question_vector)
                if hit is not None:
                    answer, similarity = hit
                    log_cache_hit(ctx.question, answer, similarity)
                    ctx.analytics.cache_hit = True
                    executor.shutdown(wait=False, cancel_futures=True)
                    return RetrievalResult(
                        chunks=[],
                        question_vector=question_vector,
                        search_text=search_text,
                        hyde_used=False,
                        use_cache=True,
                        cached_answer=answer,
                        cache_similarity=similarity,
                    )

            if hyde_future:
                t_hyde = time.perf_counter()
                search_text, hyde_used = hyde_future.result()
                if hyde_used:
                    ctx.analytics.hyde_ms = int((time.perf_counter() - t_hyde) * 1000)
                ctx.analytics.hyde_used = hyde_used
            executor.shutdown(wait=False)
        elif hyde_eligible:
            t_hyde = time.perf_counter()
            search_text, hyde_used = get_search_text(ctx.question, llm)
            if hyde_used:
                ctx.analytics.hyde_ms = int((time.perf_counter() - t_hyde) * 1000)
            ctx.analytics.hyde_used = hyde_used

        t_embed = time.perf_counter()
        if hyde_used:
            search_vector = embedder.embed([search_text])[0]
        elif question_vector is not None:
            search_vector = question_vector
        else:
            question_vector = embedder.embed([ctx.question])[0]
            search_vector = question_vector
        embed_ms += int((time.perf_counter() - t_embed) * 1000)
        ctx.analytics.embed_ms = embed_ms

        fetch_k = _retrieval_top_k(ctx.top_k)
        filter_payload = _search_filter(ctx.vendor_filter, ctx.document_type_filter)
        t_search = time.perf_counter()
        if is_map_reduce_enabled():
            chunks = map_reduce_search(
                ctx.question,
                embedder=embedder,
                store=store,
                top_k=fetch_k,
                filter_payload=filter_payload,
            )
        else:
            chunks = store.search(
                search_vector,
                top_k=fetch_k,
                filter_payload=filter_payload,
            )
        for hit in chunks:
            hit.setdefault("vector_score", float(hit.get("score", 0.0)))
        if should_supplement_video_transcript_retrieval(ctx.question):
            video_filter = dict(filter_payload or {})
            video_filter["content_type"] = "video_transcript"
            video_hits = store.search(
                search_vector,
                top_k=min(fetch_k, 30),
                filter_payload=video_filter,
            )
            for hit in video_hits:
                hit.setdefault("vector_score", float(hit.get("score", 0.0)))
            if video_hits:
                chunks = reciprocal_rank_fusion(chunks, video_hits)[:fetch_k]
                ctx.analytics.video_transcript_supplement = True
        if is_hybrid_retrieval_enabled():
            image_filter = dict(filter_payload or {})
            image_filter["content_type"] = "image"
            image_hits = store.search(
                search_vector,
                top_k=fetch_k,
                filter_payload=image_filter,
            )
            chunks = merge_text_and_image_hits(chunks, image_hits, top_k=fetch_k)
        ctx.analytics.qdrant_search_ms = int((time.perf_counter() - t_search) * 1000)
        ctx.analytics.chunks_retrieved = len(chunks)
        ctx.analytics.top_vector_score, ctx.analytics.mean_vector_score = score_stats(
            chunks, "vector_score"
        )

        rerank_ms = None
        if _is_reranker_enabled() and chunks:
            top_n = int(os.getenv("RERANKER_TOP_N", "5"))
            if is_dynamic_rerank_top_n_enabled():
                # Lazy import: rag_pipeline imports this module at load time.
                from providers.rag_pipeline import estimate_prompt_tokens, format_chunk_for_prompt

                top_n = dynamic_rerank_top_n(
                    chunks,
                    prompt_token_budget=int(os.getenv("MAX_PROMPT_TOKENS", "12000")),
                    estimate_tokens=lambda chunk: estimate_prompt_tokens(
                        format_chunk_for_prompt(chunk)
                    ),
                )
            t_rerank = time.perf_counter()
            reranker = get_reranker()
            rerank_future = None
            if is_speculative_generation_enabled():
                rerank_future = start_rerank_future(
                    ThreadPoolExecutor(max_workers=1),
                    reranker,
                    ctx.question,
                    chunks,
                    top_n,
                )
                chunks = speculative_prompt_chunks(
                    chunks, rerank_future, default_top_n=top_n
                )
            else:
                chunks = reranker.rerank(ctx.question, chunks, top_n)
            rerank_ms = int((time.perf_counter() - t_rerank) * 1000)
        ctx.analytics.rerank_ms = rerank_ms
        ctx.analytics.chunks_after_rerank = len(chunks)
        chunks = filter_chunks_by_score_floor(chunks)

        return RetrievalResult(
            chunks=chunks,
            question_vector=question_vector,
            search_text=search_text,
            hyde_used=hyde_used,
            use_cache=use_cache,
        )
