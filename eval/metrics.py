"""Retrieval evaluation metrics for offline golden-set runs."""


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    if not expected_sources:
        return 0.0
    top = retrieved_sources[:k]
    return 1.0 if any(src in top for src in expected_sources) else 0.0


def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    for index, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1.0 / index
    return 0.0


def caption_recall_at_k(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
    k: int,
) -> float:
    if not expected_sources:
        return 0.0
    for chunk in retrieved_chunks[:k]:
        if chunk.get("content_type") != "image_caption":
            continue
        if chunk.get("source") in expected_sources:
            return 1.0
    return 0.0


def strict_caption_recall_at_k(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
    k: int,
    expected_caption_substring: str | None = None,
) -> float:
    if not expected_sources:
        return 0.0
    needle = (expected_caption_substring or "").strip().lower()
    for chunk in retrieved_chunks[:k]:
        if chunk.get("content_type") != "image_caption":
            continue
        if chunk.get("source") not in expected_sources:
            continue
        if needle and needle not in (chunk.get("text") or "").lower():
            continue
        return 1.0
    return 0.0


def aggregate_metrics(per_item: list[dict]) -> dict:
    if not per_item:
        return {
            "count": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "mean_top_vector_score": None,
            "mean_embed_ms": None,
            "mean_search_ms": None,
            "mean_rerank_ms": None,
        }

    count = len(per_item)
    summary = {
        "count": count,
        "recall_at_5": sum(item["recall_at_5"] for item in per_item) / count,
        "recall_at_10": sum(item["recall_at_10"] for item in per_item) / count,
        "mrr": sum(item["mrr"] for item in per_item) / count,
        "mean_top_vector_score": _mean_optional(
            item.get("top_vector_score") for item in per_item
        ),
        "mean_embed_ms": _mean_optional(item.get("embed_ms") for item in per_item),
        "mean_search_ms": _mean_optional(item.get("search_ms") for item in per_item),
        "mean_rerank_ms": _mean_optional(item.get("rerank_ms") for item in per_item),
    }
    if any("caption_recall_at_5" in item for item in per_item):
        summary["caption_recall_at_5"] = sum(
            item.get("caption_recall_at_5", 0.0) for item in per_item
        ) / count
        summary["caption_recall_at_10"] = sum(
            item.get("caption_recall_at_10", 0.0) for item in per_item
        ) / count
        summary["strict_caption_recall_at_5"] = sum(
            item.get("strict_caption_recall_at_5", 0.0) for item in per_item
        ) / count
        summary["strict_caption_recall_at_10"] = sum(
            item.get("strict_caption_recall_at_10", 0.0) for item in per_item
        ) / count
    return summary


def _mean_optional(values) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def compare_to_baseline(
    current: dict,
    baseline: dict,
    *,
    recall_tolerance: float = 0.05,
) -> list[str]:
    """Return regression messages; empty list means within tolerance."""
    regressions: list[str] = []
    for key in ("recall_at_5", "recall_at_10", "mrr"):
        base = baseline.get(key)
        cur = current.get(key)
        if base is None or cur is None:
            continue
        if cur < base - recall_tolerance:
            regressions.append(
                f"{key} dropped from {base:.3f} to {cur:.3f} "
                f"(tolerance {recall_tolerance:.2f})"
            )
    return regressions
