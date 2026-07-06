#!/usr/bin/env python3
"""Offline retrieval evaluation against a golden JSONL dataset.

Usage:
    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --dataset eval/dataset.jsonl --k 10
    python eval/run_retrieval_eval.py --check-baseline
    python eval/run_retrieval_eval.py --write-baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import (
    aggregate_metrics,
    caption_recall_at_k,
    compare_to_baseline,
    recall_at_k,
    reciprocal_rank,
    strict_caption_recall_at_k,
)
from providers.factory import get_embedder, get_reranker, get_vector_store
from providers.rag_pipeline import _is_reranker_enabled, _retrieval_top_k, get_search_text, score_stats


def load_dataset(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return items


def retrieve_for_item(
    item: dict,
    fetch_k: int,
    rerank_top_n: int,
    content_type_filter: str | None = None,
) -> dict:
    from providers.factory import get_llm

    question = item["question"]
    llm = get_llm()
    embedder = get_embedder()
    store = get_vector_store()

    t_hyde = time.perf_counter()
    search_text, hyde_used = get_search_text(question, llm)
    hyde_ms = int((time.perf_counter() - t_hyde) * 1000) if hyde_used else None

    t_embed = time.perf_counter()
    vector = embedder.embed([search_text])[0]
    embed_ms = int((time.perf_counter() - t_embed) * 1000)

    filter_payload = None
    if content_type_filter:
        filter_payload = {"content_type": content_type_filter}

    t_search = time.perf_counter()
    chunks = store.search(
        vector, top_k=fetch_k, filter_payload=filter_payload, query_text=question
    )
    search_ms = int((time.perf_counter() - t_search) * 1000)

    top_vector_score, _ = score_stats(chunks)
    rerank_ms = None
    if _is_reranker_enabled() and chunks:
        t_rerank = time.perf_counter()
        chunks = get_reranker().rerank(question, chunks, rerank_top_n)
        rerank_ms = int((time.perf_counter() - t_rerank) * 1000)

    sources = [chunk.get("source", "") for chunk in chunks]
    expected = item.get("expected_sources", [])
    caption_substring = item.get("expected_caption_substring")

    result = {
        "id": item.get("id", question[:40]),
        "recall_at_5": recall_at_k(sources, expected, 5),
        "recall_at_10": recall_at_k(sources, expected, 10),
        "mrr": reciprocal_rank(sources, expected),
        "caption_recall_at_5": caption_recall_at_k(chunks, expected, 5),
        "caption_recall_at_10": caption_recall_at_k(chunks, expected, 10),
        "strict_caption_recall_at_5": strict_caption_recall_at_k(
            chunks, expected, 5, caption_substring
        ),
        "strict_caption_recall_at_10": strict_caption_recall_at_k(
            chunks, expected, 10, caption_substring
        ),
        "top_vector_score": top_vector_score,
        "embed_ms": embed_ms,
        "search_ms": search_ms,
        "rerank_ms": rerank_ms,
        "hyde_ms": hyde_ms,
        "retrieved_sources": sources[:10],
    }
    if item.get("vendor"):
        result["vendor"] = item["vendor"]
    if item.get("tags"):
        result["tags"] = item["tags"]
    return result


def run_eval(
    dataset_path: Path,
    fetch_k: int,
    rerank_top_n: int,
    content_type_filter: str | None = None,
) -> dict:
    items = load_dataset(dataset_path)
    per_item = [
        retrieve_for_item(
            item,
            fetch_k=fetch_k,
            rerank_top_n=rerank_top_n,
            content_type_filter=content_type_filter,
        )
        for item in items
    ]
    summary = aggregate_metrics(per_item)
    return {
        "dataset": str(dataset_path),
        "fetch_k": fetch_k,
        "rerank_top_n": rerank_top_n,
        "content_type_filter": content_type_filter,
        "summary": summary,
        "items": per_item,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline retrieval evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval" / "dataset.jsonl",
        help="Golden set JSONL path",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "eval" / "baseline.json",
        help="Baseline metrics JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write full report JSON to this path",
    )
    parser.add_argument("--fetch-k", type=int, default=None, help="Vector fetch K")
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Exit 1 if metrics regress vs baseline",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite baseline.json with current summary",
    )
    parser.add_argument(
        "--langfuse-experiment",
        action="store_true",
        help="Sync dataset and publish scores to Langfuse Experiments",
    )
    parser.add_argument(
        "--langfuse-dataset",
        default=None,
        help="Langfuse dataset name (default: LANGFUSE_EVAL_DATASET env)",
    )
    parser.add_argument(
        "--langfuse-run-name",
        default=None,
        help="Langfuse experiment run name",
    )
    parser.add_argument(
        "--content-type-filter",
        choices=("none", "text", "image_caption"),
        default="none",
        help="Restrict Qdrant search to content_type (default: no filter)",
    )
    args = parser.parse_args()

    fetch_k = args.fetch_k if args.fetch_k is not None else _retrieval_top_k(5)
    content_type_filter = (
        None if args.content_type_filter == "none" else args.content_type_filter
    )
    report = run_eval(
        args.dataset,
        fetch_k=fetch_k,
        rerank_top_n=args.rerank_top_n,
        content_type_filter=content_type_filter,
    )
    summary = report["summary"]

    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.output}")

    if args.write_baseline:
        baseline_payload = {
            "dataset": args.dataset.as_posix(),
            **summary,
        }
        args.baseline.write_text(
            json.dumps(baseline_payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote baseline to {args.baseline}")

    if args.check_baseline:
        if not args.baseline.exists():
            print(
                f"ERROR: baseline file not found: {args.baseline} — "
                "run --write-baseline against live services first",
                file=sys.stderr,
            )
            return 1
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline.get("count", 0) == 0:
            print(
                "ERROR: baseline count is 0 — run --write-baseline against live services first",
                file=sys.stderr,
            )
            return 1
        regressions = compare_to_baseline(summary, baseline)
        if regressions:
            for message in regressions:
                print(f"REGRESSION: {message}", file=sys.stderr)
            return 1

    if args.langfuse_experiment:
        from eval.langfuse_experiment import run_retrieval_experiment

        experiment = run_retrieval_experiment(
            args.dataset,
            dataset_name=args.langfuse_dataset,
            run_name=args.langfuse_run_name,
            fetch_k=fetch_k,
            rerank_top_n=args.rerank_top_n,
        )
        if experiment is None:
            print("Langfuse experiment skipped (credentials missing)", file=sys.stderr)
            return 1
        print(json.dumps(
            {
                "dataset_name": experiment["dataset_name"],
                "run_name": experiment["run_name"],
                "item_count": experiment["item_count"],
                "aggregate": experiment.get("aggregate", {}),
            },
            indent=2,
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
