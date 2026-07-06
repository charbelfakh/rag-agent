"""Tests for the embedder bake-off harness (pure logic, no live services)."""
from eval.run_embedder_bakeoff import select_sample, slugify_model, summarize_run


class TestSlugifyModel:
    def test_colon_and_dash_normalized(self):
        assert slugify_model("mxbai-embed-large:latest") == "mxbai_embed_large_latest"

    def test_plain_name(self):
        assert slugify_model("bge-m3") == "bge_m3"


class TestSelectSample:
    def _chunks(self):
        return [
            {"text": "a", "source": "golden-1.pdf"},
            {"text": "b", "source": "distractor-1.pdf"},
            {"text": "c", "source": "golden-2.pdf"},
            {"text": "d", "source": "distractor-2.pdf"},
            {"text": "e", "source": "distractor-3.pdf"},
        ]

    def test_expected_source_chunks_always_included(self):
        sample = select_sample(
            self._chunks(), {"golden-1.pdf", "golden-2.pdf"}, sample_size=2
        )
        sources = [c["source"] for c in sample]
        assert "golden-1.pdf" in sources
        assert "golden-2.pdf" in sources

    def test_distractors_fill_to_sample_size(self):
        sample = select_sample(
            self._chunks(), {"golden-1.pdf", "golden-2.pdf"}, sample_size=4
        )
        assert len(sample) == 4
        distractors = [c for c in sample if c["source"].startswith("distractor")]
        assert len(distractors) == 2

    def test_deterministic_for_same_seed(self):
        first = select_sample(self._chunks(), {"golden-1.pdf"}, 3, seed=7)
        second = select_sample(self._chunks(), {"golden-1.pdf"}, 3, seed=7)
        assert first == second


class TestSummarizeRun:
    def test_aggregates_metrics_and_latency(self):
        per_item = [
            {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0},
            {"recall_at_5": 0.0, "recall_at_10": 1.0, "mrr": 0.5},
        ]
        summary = summarize_run(per_item, embed_ms=[100, 300])
        assert summary["recall_at_5"] == 0.5
        assert summary["recall_at_10"] == 1.0
        assert summary["mrr"] == 0.75
        assert summary["mean_query_embed_ms"] == 200

    def test_empty_run(self):
        assert summarize_run([], []) == {"count": 0}
