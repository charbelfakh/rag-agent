#!/usr/bin/env python3
"""End-to-end answer grading on the golden set via LLM-as-judge.

Usage:
    python eval/run_answer_eval.py
    python eval/run_answer_eval.py --dataset eval/dataset.jsonl --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.llm_judge import grade_answer
from eval.run_retrieval_eval import load_dataset
from providers.factory import get_llm
from providers.rag_pipeline import query


def run_answer_eval(dataset_path: Path, *, limit: int | None = None) -> dict:
    llm = get_llm()
    items = load_dataset(dataset_path)
    if limit is not None:
        items = items[:limit]

    results = []
    for item in items:
        question = item["question"]
        rag = query(question, top_k=5)
        answer = rag.get("answer", "")
        judgment = grade_answer(
            llm,
            question=question,
            answer=answer,
            expected_sources=item.get("expected_sources"),
        )
        results.append(
            {
                "id": item.get("id", question[:40]),
                "question": question,
                "answer_preview": answer[:240],
                "pass": judgment["pass"],
                "score": judgment["score"],
                "reason": judgment["reason"],
            }
        )

    count = len(results)
    pass_rate = (
        sum(1 for row in results if row["pass"]) / count if count else 0.0
    )
    mean_score = sum(row["score"] for row in results) / count if count else 0.0
    return {
        "dataset": str(dataset_path),
        "count": count,
        "pass_rate": pass_rate,
        "mean_judge_score": mean_score,
        "items": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_answer_eval(args.dataset, limit=args.limit)
    print(json.dumps(
        {
            "count": report["count"],
            "pass_rate": report["pass_rate"],
            "mean_judge_score": report["mean_judge_score"],
        },
        indent=2,
    ))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
