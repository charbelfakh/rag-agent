"""Sprint B unit tests: offline retrieval eval metrics."""
import json
from pathlib import Path

import pytest

from eval.metrics import (
    aggregate_metrics,
    caption_recall_at_k,
    compare_to_baseline,
    recall_at_k,
    reciprocal_rank,
    strict_caption_recall_at_k,
)
from eval.run_retrieval_eval import load_dataset


class TestRetrievalMetrics:
    def test_recall_at_k_hit(self):
        assert recall_at_k(["a.pdf", "b.pdf"], ["b.pdf"], k=2) == 1.0

    def test_recall_at_k_miss(self):
        assert recall_at_k(["a.pdf"], ["b.pdf"], k=5) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["x.pdf", "y.pdf", "target.pdf"], ["target.pdf"]) == pytest.approx(
            1 / 3
        )
        assert reciprocal_rank(["x.pdf"], ["target.pdf"]) == 0.0

    def test_aggregate_metrics(self):
        per_item = [
            {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "top_vector_score": 0.9},
            {"recall_at_5": 0.0, "recall_at_10": 1.0, "mrr": 0.5, "top_vector_score": 0.7},
        ]
        summary = aggregate_metrics(per_item)
        assert summary["count"] == 2
        assert summary["recall_at_5"] == 0.5
        assert summary["mrr"] == 0.75
        assert summary["mean_top_vector_score"] == pytest.approx(0.8)

    def test_compare_to_baseline_regression(self):
        regressions = compare_to_baseline(
            {"recall_at_5": 0.4, "recall_at_10": 0.5, "mrr": 0.3},
            {"recall_at_5": 0.6, "recall_at_10": 0.6, "mrr": 0.5},
            recall_tolerance=0.05,
        )
        assert len(regressions) >= 1
        assert "recall_at_5" in regressions[0]

    def test_compare_to_baseline_within_tolerance(self):
        regressions = compare_to_baseline(
            {"recall_at_5": 0.58, "recall_at_10": 0.6, "mrr": 0.5},
            {"recall_at_5": 0.6, "recall_at_10": 0.6, "mrr": 0.5},
            recall_tolerance=0.05,
        )
        assert regressions == []


class TestCaptionMetrics:
    def test_caption_recall_requires_image_caption_type(self):
        chunks = [
            {"source": "a.pdf", "content_type": "text"},
            {"source": "a.pdf", "content_type": "image_caption", "text": "figure"},
        ]
        assert caption_recall_at_k(chunks, ["a.pdf"], k=2) == 1.0
        assert caption_recall_at_k(chunks[:1], ["a.pdf"], k=1) == 0.0

    def test_strict_caption_recall_requires_substring(self):
        chunks = [
            {
                "source": "a.pdf",
                "content_type": "image_caption",
                "text": "PORT IS ALLOCATED",
            }
        ]
        assert (
            strict_caption_recall_at_k(chunks, ["a.pdf"], 1, "port is allocated")
            == 1.0
        )
        assert strict_caption_recall_at_k(chunks, ["a.pdf"], 1, "STOPPED") == 0.0

    def test_strict_without_substring_matches_caption_recall(self):
        chunks = [
            {"source": "a.pdf", "content_type": "image_caption", "text": "ui menu"}
        ]
        assert caption_recall_at_k(chunks, ["a.pdf"], k=1) == 1.0
        assert strict_caption_recall_at_k(chunks, ["a.pdf"], 1, None) == 1.0


class TestDatasetLoader:
    def test_load_dataset_skips_comments_and_blanks(self, tmp_path: Path):
        path = tmp_path / "set.jsonl"
        path.write_text(
            '# comment\n\n{"id": "1", "question": "q?", "expected_sources": ["a.pdf"]}\n',
            encoding="utf-8",
        )
        items = load_dataset(path)
        assert len(items) == 1
        assert items[0]["id"] == "1"

    def test_repo_dataset_jsonl_is_valid(self):
        path = Path(__file__).resolve().parent.parent / "eval" / "dataset.jsonl"
        items = load_dataset(path)
        assert len(items) >= 3
        for item in items:
            assert "question" in item
            assert "expected_sources" in item

    def test_baseline_json_is_valid(self):
        path = Path(__file__).resolve().parent.parent / "eval" / "baseline.json"
        baseline = json.loads(path.read_text(encoding="utf-8"))
        assert "recall_at_5" in baseline
