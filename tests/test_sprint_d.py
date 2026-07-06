"""Sprint D tests: LLM timing, cache counters, score floor."""
from unittest.mock import MagicMock


import providers.rag_pipeline as rag_pipeline
from providers.rag_pipeline import (
    QueryAnalytics,
    filter_chunks_by_score_floor,
    stream_query,
)


class TestScoreFloor:
    def test_drops_low_scoring_chunks(self, monkeypatch):
        monkeypatch.setenv("RERANK_SCORE_FLOOR", "0.5")
        chunks = [
            {"score": 0.9, "text": "a"},
            {"score": 0.2, "text": "b"},
        ]
        result = filter_chunks_by_score_floor(chunks)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_keeps_one_chunk_if_all_below_floor(self, monkeypatch):
        monkeypatch.setenv("RERANK_SCORE_FLOOR", "0.9")
        chunks = [{"score": 0.1, "text": "a"}, {"score": 0.2, "text": "b"}]
        result = filter_chunks_by_score_floor(chunks)
        assert len(result) == 1


class TestLlmTiming:
    def test_stream_emits_retrieval_stage_and_llm_ttft(self, monkeypatch):
        plan = rag_pipeline.GenerationPlan(
            prompt="Answer:",
            citations=[],
            analytics=QueryAnalytics(
                question="q",
                chunks_after_rerank=1,
                qdrant_search_ms=5,
            ),
        )
        monkeypatch.setattr(rag_pipeline, "_build_generation_plan", lambda *a, **k: plan)

        class FakeLLM:
            last_stream_stats = {"prompt_eval_count": 100, "eval_count": 50}

            def generate_stream(self, prompt, cancel_event=None):
                yield "Hi"
                self.last_stream_stats = {
                    "prompt_eval_count": 100,
                    "eval_count": 50,
                }

        monkeypatch.setattr(rag_pipeline, "get_llm", lambda: FakeLLM())
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        events = list(stream_query("q"))
        assert events[0]["stage"] == "retrieval_complete"
        meta = events[-1]["meta"]
        assert meta.get("llm_ttft_ms") is not None
        assert meta.get("pre_stream_ms") is not None
        assert meta.get("prompt_eval_count") == 100
        assert meta.get("answer_token_count") == 50


class TestCacheCounters:
    def test_record_skip_increments_history(self):
        from providers.semantic_cache import SemanticCache

        sc = SemanticCache()
        sc.enabled = True
        sc._redis = MagicMock()
        sc.record_skip("history")
        sc._redis.hincrby.assert_called_with(
            "semantic_cache:stats", "skipped_history", 1
        )
