"""Sprint E tests: staged SSE and parallel HyDE path."""
from unittest.mock import MagicMock

import pytest

import providers.rag_pipeline as rag_pipeline
from conftest import patch_retrieval_pipeline
from providers.rag_pipeline import GenerationPlan, QueryAnalytics, _build_generation_plan


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("HYDE_ENABLED", "true")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("SUFFICIENCY_CHECK_ENABLED", "false")
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")
    monkeypatch.delenv("ANALYTICS_ENABLED", raising=False)


def _chunk():
    return {
        "text": "body",
        "source": "data/a.pdf",
        "page": 0,
        "score": 0.8,
    }


class TestStagedSse:
    def test_retrieval_stage_event(self, monkeypatch):
        plan = GenerationPlan(
            prompt="p",
            analytics=QueryAnalytics(question="q", embed_ms=10),
        )
        import time

        event = rag_pipeline._retrieval_stage_event(plan, time.time())
        assert event["stage"] == "retrieval_complete"
        assert event["meta"]["embed_ms"] == 10


class TestParallelHyde:
    def test_cache_miss_runs_hyde_parallel_with_cache(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.side_effect = [
            [[0.1]],
            [[0.2]],
        ]
        store = MagicMock()
        store.search.return_value = [_chunk()]
        cache = MagicMock()
        cache.enabled = True
        cache.lookup.return_value = None
        llm = MagicMock()
        llm.generate.return_value = "HyDE passage about triggers."

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=llm,
            cache=cache,
            reranker=MagicMock(),
        )
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        plan = _build_generation_plan("How do I configure triggers?", top_k=5)

        assert plan.analytics.hyde_used is True
        assert plan.prompt is not None
        assert embedder.embed.call_count == 2
