#!/usr/bin/env python3
"""Compare two cross-encoder reranker models on the golden retrieval set (Sprint L rank 52)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import aggregate_metrics, recall_at_k, reciprocal_rank
from eval.run_retrieval_eval import load_dataset
from providers.factory import get_embedder, get_vector_store
from providers.reranker import CrossEncoderReranker
from providers.rag_pipeline import get_search_text, score_stats


def _retrieve_with_reranker(
    item: dict,
    *,
    fetch_k: int,
    rerank_top_n: int,
    reranker: CrossEncoderReranker,
) -> dict:
    from providers.factory import get_llm

    question = item["question"]
    llm = get_llm()
    embedder = get_embedder()
    store = get_vector_store()

    search_text, _ = get_search_text(question, llm)
    vector = embedder.embed([search_text])[0]
    chunks = store.search(vector, top_k=fetch_k)
    top_vector_score, _ = score_stats(chunks)

    if chunks:
        chunks = reranker.rerank(question, chunks, rerank_top_n)

    sources = [chunk.get("source", "") for chunk in chunks]
    expected = item.get("expected_sources", [])
    return {
        "id": item.get("id", question[:40]),
        "recall_at_5": recall_at_k(sources, expected, 5),
        "recall_at_10": recall_at_k(sources, expected, 10),
        "mrr": reciprocal_rank(sources, expected),
        "top_vector_score": top_vector_score,
        "retrieved_sources": sources[:10],
    }


def run_reranker_ab(
    dataset_path: Path,
    *,
    model_a: str,
    model_b: str,
    fetch_k: int = 20,
    rerank_top_n: int = 5,
    limit: int | None = None,
) -> dict:
    items = load_dataset(dataset_path)
    if limit is not None:
        items = items[:limit]

    reranker_a = CrossEncoderReranker(model_name=model_a)
    reranker_b = CrossEncoderReranker(model_name=model_b)

    results_a = [
        _retrieve_with_reranker(
            item,
            fetch_k=fetch_k,
            rerank_top_n=rerank_top_n,
            reranker=reranker_a,
        )
        for item in items
    ]
    results_b = [
        _retrieve_with_reranker(
            item,
            fetch_k=fetch_k,
            rerank_top_n=rerank_top_n,
            reranker=reranker_b,
        )
        for item in items
    ]

    summary_a = aggregate_metrics(results_a)
    summary_b = aggregate_metrics(results_b)
    winner = "a"
    if summary_b.get("recall_at_5", 0) > summary_a.get("recall_at_5", 0):
        winner = "b"
    elif summary_b.get("recall_at_5", 0) == summary_a.get("recall_at_5", 0):
        if summary_b.get("mrr", 0) > summary_a.get("mrr", 0):
            winner = "b"

    return {
        "dataset": str(dataset_path),
        "model_a": model_a,
        "model_b": model_b,
        "winner": winner,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "items_a": results_a,
        "items_b": results_b,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    parser.add_argument(
        "--model-a",
        default=None,
        help="Reranker A (default: RERANKER_MODEL env)",
    )
    parser.add_argument(
        "--model-b",
        default=None,
        help="Reranker B (default: RERANKER_AB_MODEL env)",
    )
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import os

    model_a = args.model_a or os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    model_b = args.model_b or os.getenv(
        "RERANKER_AB_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"
    )

    report = run_reranker_ab(
        args.dataset,
        model_a=model_a,
        model_b=model_b,
        fetch_k=args.fetch_k,
        rerank_top_n=args.rerank_top_n,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "winner": report["winner"],
                "model_a": report["model_a"],
                "model_b": report["model_b"],
                "summary_a": report["summary_a"],
                "summary_b": report["summary_b"],
            },
            indent=2,
        )
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
