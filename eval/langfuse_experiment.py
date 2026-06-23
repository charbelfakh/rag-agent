"""Sync retrieval golden set to Langfuse Datasets and run experiment scoring."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def get_langfuse_client():
    if not _langfuse_enabled():
        return None
    from langfuse import Langfuse

    return Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or "http://localhost:3000",
    )


def ensure_dataset(dataset_name: str, description: str = "") -> None:
    lf = get_langfuse_client()
    if lf is None:
        return
    try:
        lf.create_dataset(name=dataset_name, description=description or None)
    except Exception as exc:
        logger.debug("Dataset create skipped (%s)", exc)


def sync_dataset(dataset_path: Path, dataset_name: str) -> int:
    """Create or update Langfuse dataset items from JSONL golden set."""
    from eval.run_retrieval_eval import load_dataset

    lf = get_langfuse_client()
    if lf is None:
        logger.warning("Langfuse credentials missing; skipping dataset sync")
        return 0

    ensure_dataset(dataset_name, description=f"Synced from {dataset_path}")
    items = load_dataset(dataset_path)
    synced = 0
    for item in items:
        item_id = item.get("id") or item["question"][:80]
        lf.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": item["question"]},
            expected_output={
                "expected_sources": item.get("expected_sources", []),
            },
            id=item_id,
        )
        synced += 1
    lf.flush()
    return synced


def _build_experiment_items(dataset_path: Path) -> list:
    from eval.run_retrieval_eval import load_dataset
    from langfuse.experiment import LocalExperimentItem

    rows = load_dataset(dataset_path)
    return [
        LocalExperimentItem(
            id=row.get("id") or row["question"][:80],
            input={"question": row["question"]},
            expected_output={"expected_sources": row.get("expected_sources", [])},
        )
        for row in rows
    ]


def run_retrieval_experiment(
    dataset_path: Path,
    *,
    dataset_name: str | None = None,
    run_name: str | None = None,
    fetch_k: int,
    rerank_top_n: int,
    sync_items: bool = True,
) -> dict | None:
    """Run retrieval eval as a Langfuse experiment with per-item scores."""
    from eval.metrics import recall_at_k, reciprocal_rank
    from eval.run_retrieval_eval import retrieve_for_item
    from langfuse.experiment import Evaluation

    lf = get_langfuse_client()
    if lf is None:
        return None

    name = dataset_name or os.getenv("LANGFUSE_EVAL_DATASET", "rag-retrieval-golden")
    run = run_name or os.getenv("LANGFUSE_EVAL_RUN_NAME", "retrieval-eval")

    if sync_items:
        sync_dataset(dataset_path, name)

    def task(*, item: Any, **kwargs) -> dict:
        row = {
            "id": getattr(item, "id", None) or item.get("id"),
            "question": item.input["question"],
            "expected_sources": (item.expected_output or {}).get(
                "expected_sources", []
            ),
        }
        return retrieve_for_item(row, fetch_k=fetch_k, rerank_top_n=rerank_top_n)

    def recall_at_5_evaluator(*, output: dict, expected_output: dict, **kwargs):
        expected = (expected_output or {}).get("expected_sources", [])
        value = recall_at_k(output.get("retrieved_sources", []), expected, 5)
        return Evaluation(name="recall_at_5", value=value)

    def recall_at_10_evaluator(*, output: dict, expected_output: dict, **kwargs):
        expected = (expected_output or {}).get("expected_sources", [])
        value = recall_at_k(output.get("retrieved_sources", []), expected, 10)
        return Evaluation(name="recall_at_10", value=value)

    def mrr_evaluator(*, output: dict, expected_output: dict, **kwargs):
        expected = (expected_output or {}).get("expected_sources", [])
        value = reciprocal_rank(output.get("retrieved_sources", []), expected)
        return Evaluation(name="mrr", value=value)

    data = _build_experiment_items(dataset_path)
    result = lf.run_experiment(
        name=name,
        run_name=run,
        data=data,
        task=task,
        evaluators=[recall_at_5_evaluator, recall_at_10_evaluator, mrr_evaluator],
        metadata={
            "fetch_k": str(fetch_k),
            "rerank_top_n": str(rerank_top_n),
            "dataset_path": str(dataset_path),
        },
    )
    lf.flush()

    aggregate = {}
    for key in ("recall_at_5", "recall_at_10", "mrr"):
        if hasattr(result, "format") and key in str(result.format()):
            pass
    if hasattr(result, "run_evaluations"):
        for evaluation in result.run_evaluations or []:
            aggregate[evaluation.name] = evaluation.value

    return {
        "dataset_name": name,
        "run_name": run,
        "item_count": len(data),
        "aggregate": aggregate,
        "result": result,
    }
