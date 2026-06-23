"""Sprint A unit tests: embed reuse, sufficiency dedup, cache SSE batching."""
import os
from unittest.mock import MagicMock, patch

import pytest

import providers.rag_pipeline as rag_pipeline
from conftest import patch_retrieval_pipeline
from providers.rag_pipeline import (
    GenerationPlan,
    _build_generation_plan,
    _yield_text_as_tokens,
    check_context_sufficient,
    stream_query,
)


@pytest.fixture(autouse=True)
def disable_side_effects(monkeypatch):
    monkeypatch.setenv("HYDE_ENABLED", "false")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("SUFFICIENCY_CHECK_ENABLED", "false")
    monkeypatch.setenv("CACHE_SSE_WORD_CHUNK", "0")
    monkeypatch.delenv("ANALYTICS_ENABLED", raising=False)


def _chunk(text: str = "doc text", score: float = 0.9) -> dict:
    return {
        "text": text,
        "source": "data/manual.pdf",
        "page": 1,
        "section": "Intro",
        "score": score,
    }


class TestEmbedReuse:
    def test_cache_miss_reuses_single_question_embed(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        store = MagicMock()
        store.search.return_value = [_chunk()]
        cache = MagicMock()
        cache.enabled = True
        cache.lookup.return_value = None
        llm = MagicMock()

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=llm,
            cache=cache,
            reranker=MagicMock(),
        )

        plan = _build_generation_plan("What is Pekat?", top_k=5)

        assert plan.prompt is not None
        embedder.embed.assert_called_once_with(["What is Pekat?"])
        store.search.assert_called_once_with(
            [0.1, 0.2, 0.3],
            top_k=5,
            filter_payload={"vendor": "pekat"},
        )
        assert plan.question_vector == [0.1, 0.2, 0.3]

    def test_hyde_embeds_search_text_separately(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.side_effect = [
            [[0.1, 0.2]],  # question for cache lookup
            [[0.9, 0.8]],  # HyDE hypothetical passage
        ]
        store = MagicMock()
        store.search.return_value = [_chunk()]
        cache = MagicMock()
        cache.enabled = True
        cache.lookup.return_value = None
        llm = MagicMock()
        llm.generate.return_value = "HyDE passage about vision systems."

        monkeypatch.setenv("HYDE_ENABLED", "true")
        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=llm,
            cache=cache,
            reranker=MagicMock(),
        )

        _build_generation_plan("How do I configure triggers?", top_k=5)

        assert embedder.embed.call_count == 2
        embedder.embed.assert_any_call(["How do I configure triggers?"])
        store.search.assert_called_once_with(
            [0.9, 0.8], top_k=5, filter_payload=None
        )

    def test_empty_chunks_does_not_reembed_question(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.5]]
        store = MagicMock()
        store.search.return_value = []
        cache = MagicMock()
        cache.enabled = True
        cache.lookup.return_value = None

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=MagicMock(),
            cache=cache,
            reranker=MagicMock(),
        )

        plan = _build_generation_plan("missing topic", top_k=5)

        embedder.embed.assert_called_once()
        assert plan.immediate_answer is not None
        assert plan.question_vector == [0.5]


class TestSufficiencyDedup:
    def test_debug_log_does_not_call_sufficiency_llm(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = [_chunk()]
        llm = MagicMock()
        cache = MagicMock()
        cache.enabled = False

        monkeypatch.setenv("SUFFICIENCY_CHECK_ENABLED", "true")
        llm.generate.return_value = "YES"
        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=llm,
            cache=cache,
            reranker=MagicMock(),
        )
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        _build_generation_plan("What is Pekat?", top_k=5)

        # One sufficiency call in the real check path — not a duplicate in debug logging.
        assert llm.generate.call_count == 1

    def test_sufficiency_still_runs_once_when_enabled(self, monkeypatch):
        chunks = [_chunk("relevant excerpt")]
        llm = MagicMock()
        llm.generate.return_value = "YES"

        result = check_context_sufficient("What is Pekat?", chunks, llm)

        assert result is True
        llm.generate.assert_called_once()


class TestCacheSseBatching:
    def test_default_yields_single_token_event(self, monkeypatch):
        monkeypatch.setenv("CACHE_SSE_WORD_CHUNK", "0")
        events = list(_yield_text_as_tokens("Hello cached answer."))
        assert events == [{"token": "Hello cached answer."}]

    def test_word_chunk_mode_yields_fewer_events_than_chars(self, monkeypatch):
        monkeypatch.setenv("CACHE_SSE_WORD_CHUNK", "2")
        text = "one two three four five"
        events = list(_yield_text_as_tokens(text))
        assert len(events) == 3
        assert "".join(e["token"] for e in events) == text

    def test_cache_hit_stream_emits_one_token_before_done(self, monkeypatch):
        plan = GenerationPlan(
            cached_answer="Full cached response text.",
            citations=[],
            analytics=rag_pipeline.QueryAnalytics(question="q"),
        )
        monkeypatch.setattr(
            rag_pipeline,
            "_build_generation_plan",
            lambda *args, **kwargs: plan,
        )
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        events = list(stream_query("q"))

        token_events = [e for e in events if "token" in e]
        done_events = [e for e in events if e.get("done")]
        assert len(token_events) == 1
        assert token_events[0]["token"] == "Full cached response text."
        assert len(done_events) == 1
