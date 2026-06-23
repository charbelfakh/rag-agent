"""Advanced retrieval/generation helpers: map-reduce, speculative, dynamic top-N (Sprints O, P)."""
from __future__ import annotations

import os
import re
from concurrent.futures import Future, ThreadPoolExecutor

MAP_REDUCE_SPLIT = re.compile(r"\s+(?:and|;)\s+", re.IGNORECASE)


def is_map_reduce_enabled() -> bool:
    return os.getenv("MAP_REDUCE_RETRIEVAL_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def is_speculative_generation_enabled() -> bool:
    return os.getenv("SPECULATIVE_GENERATION_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def is_dynamic_rerank_top_n_enabled() -> bool:
    return os.getenv("DYNAMIC_RERANK_TOP_N_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def split_sub_questions(question: str) -> list[str]:
    """Split compound questions on ``and`` / ``;`` for map-reduce retrieval."""
    parts = [part.strip() for part in MAP_REDUCE_SPLIT.split(question) if part.strip()]
    return parts if len(parts) > 1 else [question]


def map_reduce_search(
    question: str,
    *,
    embedder,
    store,
    top_k: int,
    filter_payload: dict | None,
) -> list[dict]:
    """Run retrieval per sub-question and merge unique chunks by chunk_id."""
    merged: dict[str, dict] = {}
    for sub_q in split_sub_questions(question):
        vector = embedder.embed([sub_q])[0]
        hits = store.search(vector, top_k=top_k, filter_payload=filter_payload)
        for hit in hits:
            key = hit.get("chunk_id") or f"{hit.get('source')}:{hit.get('page')}"
            prev = merged.get(key)
            if prev is None or float(hit.get("score", 0)) > float(prev.get("score", 0)):
                merged[key] = hit
    return sorted(merged.values(), key=lambda row: float(row.get("score", 0)), reverse=True)


def start_rerank_future(executor: ThreadPoolExecutor, reranker, question: str, chunks: list[dict], top_n: int) -> Future:
    """Submit rerank work so generation can start on top-1 while rerank finishes."""
    return executor.submit(reranker.rerank, question, chunks, top_n)


def speculative_prompt_chunks(
    chunks: list[dict],
    rerank_future: Future | None,
    *,
    default_top_n: int,
) -> list[dict]:
    """Use reranked chunks when ready; otherwise top-1 for faster TTFT."""
    if not is_speculative_generation_enabled() or not chunks:
        return chunks[:default_top_n]
    if rerank_future is not None and rerank_future.done():
        try:
            reranked = rerank_future.result()
            if reranked:
                return reranked[:default_top_n]
        except Exception:
            pass
    return chunks[:1]


def dynamic_rerank_top_n(
    chunks: list[dict],
    *,
    prompt_token_budget: int,
    estimate_tokens,
) -> int:
    """Adapt reranker top-N to fit measured prompt token budget."""
    if not is_dynamic_rerank_top_n_enabled() or not chunks:
        return int(os.getenv("RERANKER_TOP_N", "5"))
    max_n = int(os.getenv("RERANKER_TOP_N", "5"))
    used = 0
    selected = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk)
        if selected > 0 and used + cost > prompt_token_budget:
            break
        used += cost
        selected += 1
        if selected >= max_n:
            break
    return max(1, selected or 1)
