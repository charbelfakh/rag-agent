"""Eval harness unit tests — exercises gate logic with no live services required.

These tests are the CI stand-in for the full live recall gate. The real gate
(run_retrieval_eval.py --check-baseline) must be run locally against live
Qdrant + Ollama before merging any retrieval-path change.
"""
import json
import sys
from pathlib import Path

import pytest

import eval.run_retrieval_eval as run_retrieval_eval
from eval.metrics import aggregate_metrics, compare_to_baseline


# ---------------------------------------------------------------------------
# compare_to_baseline — caption gating
# ---------------------------------------------------------------------------


class TestCompareCaptionGating:
    def test_caption_recall_regression_is_caught(self):
        regressions = compare_to_baseline(
            {"recall_at_5": 0.93, "recall_at_10": 0.93, "mrr": 0.75,
             "caption_recall_at_5": 0.20, "strict_caption_recall_at_5": 0.18},
            {"recall_at_5": 0.93, "recall_at_10": 0.93, "mrr": 0.75,
             "caption_recall_at_5": 0.37, "strict_caption_recall_at_5": 0.35},
            recall_tolerance=0.05,
        )
        assert any("caption_recall_at_5" in r for r in regressions)
        assert any("strict_caption_recall_at_5" in r for r in regressions)

    def test_caption_regression_within_tolerance_passes(self):
        regressions = compare_to_baseline(
            {"recall_at_5": 0.93, "recall_at_10": 0.93, "mrr": 0.75,
             "caption_recall_at_5": 0.34, "strict_caption_recall_at_5": 0.32},
            {"recall_at_5": 0.93, "recall_at_10": 0.93, "mrr": 0.75,
             "caption_recall_at_5": 0.37, "strict_caption_recall_at_5": 0.35},
            recall_tolerance=0.05,
        )
        assert regressions == []

    def test_caption_metrics_absent_in_baseline_skipped(self):
        """Baseline without caption metrics does not gate caption recall."""
        regressions = compare_to_baseline(
            {"recall_at_5": 0.93, "recall_at_10": 0.93, "mrr": 0.75,
             "caption_recall_at_5": 0.0, "strict_caption_recall_at_5": 0.0},
            {"recall_at_5": 0.93, "recall_at_10": 0.93, "mrr": 0.75},
        )
        assert regressions == []

    def test_text_recall_regression_still_caught(self):
        regressions = compare_to_baseline(
            {"recall_at_5": 0.80, "recall_at_10": 0.80, "mrr": 0.60},
            {"recall_at_5": 0.93, "recall_at_10": 0.93, "mrr": 0.75},
            recall_tolerance=0.05,
        )
        assert any("recall_at_5" in r for r in regressions)


# ---------------------------------------------------------------------------
# baseline.json guard — count == 0 must be detected at the metric layer
# ---------------------------------------------------------------------------


class TestZeroCountBaseline:
    def test_zero_count_baseline_has_no_gated_keys(self):
        """A zeroed baseline has no useful metrics to gate against."""
        zeroed = {
            "dataset": "eval/dataset.jsonl",
            "count": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
        }
        current = {
            "recall_at_5": 0.93,
            "recall_at_10": 0.93,
            "mrr": 0.75,
        }
        # With a zeroed baseline, current metrics are always >= 0 − tolerance,
        # so compare_to_baseline must return no regressions (it'll compare
        # 0.93 vs 0.00 which is fine). The count=0 guard in run_retrieval_eval
        # prevents this comparison from happening in the CLI; this test
        # documents the expected fallthrough behavior of compare_to_baseline
        # itself.
        regressions = compare_to_baseline(current, zeroed)
        assert regressions == []

    def test_populated_baseline_json_count_is_nonzero(self):
        """The committed baseline.json must have a real item count."""
        baseline_path = Path(__file__).resolve().parent.parent / "eval" / "baseline.json"
        if not baseline_path.exists():
            pytest.skip("baseline.json not present")
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert data.get("count", 0) > 0, (
            "baseline.json count is 0 — run: "
            "python eval/run_retrieval_eval.py --write-baseline"
        )

    def test_populated_baseline_has_caption_metrics(self):
        """Baseline must include caption recall after dataset merge."""
        baseline_path = Path(__file__).resolve().parent.parent / "eval" / "baseline.json"
        if not baseline_path.exists():
            pytest.skip("baseline.json not present")
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        if data.get("count", 0) == 0:
            pytest.skip("baseline not populated")
        assert "caption_recall_at_5" in data, (
            "baseline.json missing caption_recall_at_5 — regenerate with merged dataset"
        )
        assert "strict_caption_recall_at_5" in data, (
            "baseline.json missing strict_caption_recall_at_5 — regenerate with merged dataset"
        )


# ---------------------------------------------------------------------------
# --check-baseline CLI guards — misconfiguration must fail loudly, not pass
# ---------------------------------------------------------------------------


class TestCheckBaselineCliGuards:
    def _run_main(self, monkeypatch, tmp_path, baseline_content):
        dataset = tmp_path / "dataset.jsonl"
        dataset.write_text(
            '{"id": "q1", "question": "x", "expected_sources": ["a.pdf"]}\n',
            encoding="utf-8",
        )
        baseline = tmp_path / "baseline.json"
        if baseline_content is not None:
            baseline.write_text(json.dumps(baseline_content), encoding="utf-8")
        summary = {"count": 1, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0}
        monkeypatch.setattr(
            run_retrieval_eval,
            "run_eval",
            lambda *args, **kwargs: {"summary": summary, "items": []},
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_retrieval_eval.py",
                "--check-baseline",
                "--dataset", str(dataset),
                "--baseline", str(baseline),
                "--fetch-k", "5",
            ],
        )
        return run_retrieval_eval.main()

    def test_missing_baseline_file_exits_nonzero(self, monkeypatch, tmp_path):
        assert self._run_main(monkeypatch, tmp_path, baseline_content=None) == 1

    def test_zero_count_baseline_exits_nonzero(self, monkeypatch, tmp_path):
        zeroed = {"count": 0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
        assert self._run_main(monkeypatch, tmp_path, zeroed) == 1

    def test_populated_baseline_within_tolerance_passes(self, monkeypatch, tmp_path):
        ok = {"count": 1, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0}
        assert self._run_main(monkeypatch, tmp_path, ok) == 0


# ---------------------------------------------------------------------------
# aggregate_metrics — caption keys included when any item has them
# ---------------------------------------------------------------------------


class TestAggregateWithCaptionItems:
    def test_caption_keys_in_summary_when_present(self):
        per_item = [
            {
                "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0,
                "caption_recall_at_5": 1.0, "caption_recall_at_10": 1.0,
                "strict_caption_recall_at_5": 1.0, "strict_caption_recall_at_10": 1.0,
            },
            {
                "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0,
                "caption_recall_at_5": 0.0, "caption_recall_at_10": 0.0,
                "strict_caption_recall_at_5": 0.0, "strict_caption_recall_at_10": 0.0,
            },
        ]
        summary = aggregate_metrics(per_item)
        assert summary["caption_recall_at_5"] == pytest.approx(0.5)
        assert summary["strict_caption_recall_at_5"] == pytest.approx(0.5)

    def test_caption_keys_absent_when_not_in_items(self):
        per_item = [
            {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0},
            {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0},
        ]
        summary = aggregate_metrics(per_item)
        assert "caption_recall_at_5" not in summary
