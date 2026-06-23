"""Reciprocal-rank fusion for text + image retrieval (Sprint M M2)."""
from __future__ import annotations

import os


def is_hybrid_retrieval_enabled() -> bool:
    return os.getenv("HYBRID_RETRIEVAL_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def hit_fusion_key(hit: dict) -> str:
    """Stable dedupe key for RRF across text, image, and timestamped video hits."""
    chunk_id = hit.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    start_seconds = hit.get("start_seconds")
    if start_seconds is not None:
        return f"{hit.get('source', '')}:{float(start_seconds):.3f}"
    return f"{hit.get('source', '')}:{hit.get('page', 0)}"


def reciprocal_rank_fusion(
    *hit_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse ranked lists with RRF; each hit must have ``chunk_id`` or ``source`` key."""
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}

    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit_fusion_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            rows[key] = hit

    fused = []
    for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        row = dict(rows[key])
        row["score"] = score
        row["fusion_score"] = score
        fused.append(row)
    return fused


def merge_text_and_image_hits(
    text_hits: list[dict],
    image_hits: list[dict],
    *,
    top_k: int,
) -> list[dict]:
    """Fuse text and image hit lists with RRF, capped at ``top_k``."""
    if not image_hits:
        return text_hits[:top_k]
    if not text_hits:
        return image_hits[:top_k]
    return reciprocal_rank_fusion(text_hits, image_hits)[:top_k]
