"""Sprint B unit tests: analytics, stage timers, history cap, prompt tokens."""
import time
from unittest.mock import MagicMock

import pytest

import providers.rag_pipeline as rag_pipeline
from conftest import patch_retrieval_pipeline
from providers.rag_pipeline import (
    GENERATION_FAILURE_USER_MESSAGE,
    QueryAnalytics,
    _build_generation_plan,
    _cap_history_chars,
    _classify_answer_outcome,
    _resolve_generation_answer,
    estimate_prompt_tokens,
    format_history_block,
    is_insufficient_answer,
    query as run_query,
    score_stats,
    stream_query,
)


@pytest.fixture(autouse=True)
def disable_side_effects(monkeypatch):
    monkeypatch.setenv("HYDE_ENABLED", "false")
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("SUFFICIENCY_CHECK_ENABLED", "false")
    monkeypatch.setenv("HISTORY_MAX_CHARS", "4000")
    monkeypatch.delenv("ANALYTICS_ENABLED", raising=False)


def _chunk(text: str = "doc text", score: float = 0.9, source: str = "data/manual.pdf") -> dict:
    return {
        "text": text,
        "source": source,
        "page": 1,
        "section": "Intro",
        "score": score,
    }


def _reranked_chunk(vector_score: float, rerank_score: float) -> dict:
    return {
        **_chunk(score=rerank_score),
        "vector_score": vector_score,
        "rerank_score": rerank_score,
        "score": rerank_score,
    }


class TestScoreStats:
    def test_vector_scores_before_rerank(self):
        chunks = [_chunk(score=0.9), _chunk(score=0.7)]
        top, mean = score_stats(chunks)
        assert top == 0.9
        assert mean == pytest.approx(0.8)

    def test_rerank_field_scores(self):
        chunks = [_reranked_chunk(0.8, 0.95), _reranked_chunk(0.7, 0.6)]
        top, mean = score_stats(chunks, "rerank_score")
        assert top == 0.95
        assert mean == pytest.approx(0.775)


class TestHistoryCap:
    def test_cap_truncates_oldest_content(self, monkeypatch):
        monkeypatch.setenv("HISTORY_MAX_CHARS", "20")
        turns = [
            {"role": "user", "content": "1234567890"},
            {"role": "assistant", "content": "abcdefghijklmnop"},
        ]
        capped = _cap_history_chars(turns)
        assert len(capped) == 2
        assert capped[0]["content"] == "1234…"
        assert capped[1]["content"] == "abcdefghijklmnop"

    def test_format_history_respects_cap(self, monkeypatch):
        monkeypatch.setenv("HISTORY_MAX_CHARS", "15")
        history = [
            {"role": "user", "content": "x" * 20},
            {"role": "assistant", "content": "short"},
        ]
        block = format_history_block(history)
        assert "Conversation history:" in block
        assert len(block) < 80


class TestPromptTokens:
    def test_estimate_prompt_tokens(self):
        assert estimate_prompt_tokens("abcd") == 1
        assert estimate_prompt_tokens("a" * 400) == 100


class TestQueryAnalytics:
    def test_finalize_sets_insufficient_and_citations(self, monkeypatch):
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)
        analytics = QueryAnalytics(question="q")
        meta = analytics.finalize(
            "I don't have enough information to answer that.",
            time.time(),
            citations=[{"source": "a.pdf", "page": 1, "section": ""}],
            prompt="Context:\nhello\n\nQuestion: q\n\nAnswer:",
        )
        assert meta["insufficient_answer"] is True
        assert meta["citations_count"] == 1
        assert meta["prompt_token_count"] == estimate_prompt_tokens(
            "Context:\nhello\n\nQuestion: q\n\nAnswer:"
        )

    def test_empty_retrieval_flag(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = []
        cache = MagicMock()
        cache.enabled = False
        reranker = MagicMock()
        reranker.rerank.return_value = []

        monkeypatch.setenv("RERANKER_ENABLED", "true")
        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=MagicMock(),
            cache=cache,
            reranker=reranker,
        )

        plan = _build_generation_plan("missing", top_k=5)
        assert plan.immediate_answer is not None
        assert plan.analytics.empty_retrieval is True


class TestStageTimersAndScores:
    def test_analytics_include_timers_and_vector_scores(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = [_chunk(score=0.88)]
        cache = MagicMock()
        cache.enabled = False
        reranker = MagicMock()
        reranker.rerank.return_value = [_reranked_chunk(0.88, 0.92)]

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=MagicMock(),
            cache=cache,
            reranker=reranker,
        )
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        plan = _build_generation_plan("What is Pekat?", top_k=5)

        assert plan.analytics.embed_ms is not None
        assert plan.analytics.qdrant_search_ms is not None
        assert plan.analytics.rerank_ms is not None
        assert plan.analytics.top_vector_score == 0.88
        assert plan.analytics.top_rerank_score == 0.92
        assert plan.analytics.chunks_in_prompt == 1

    def test_stream_meta_exposes_sprint_b_fields(self, monkeypatch):
        plan = rag_pipeline.GenerationPlan(
            prompt="Context:\n" + ("word " * 50) + "\nQuestion: q\nAnswer:",
            citations=[{"source": "a.pdf", "page": 1, "section": ""}],
            analytics=QueryAnalytics(
                question="q",
                chunks_after_rerank=1,
                chunks_in_prompt=1,
                top_vector_score=0.81,
                top_rerank_score=0.93,
                embed_ms=120,
                qdrant_search_ms=15,
                rerank_ms=340,
            ),
        )
        monkeypatch.setattr(rag_pipeline, "_build_generation_plan", lambda *a, **k: plan)
        monkeypatch.setattr(rag_pipeline, "_stream_prompt", lambda *a, **k: (iter([]), []))
        monkeypatch.setattr(rag_pipeline, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        events = list(stream_query("q"))
        meta = events[-1]["meta"]

        assert meta["top_vector_score"] == 0.81
        assert meta["top_rerank_score"] == 0.93
        assert meta["embed_ms"] == 120
        assert meta["prompt_token_count"] is not None
        assert meta["citations_count"] == 1
        assert meta["insufficient_answer"] is True
        assert meta["answer_outcome"] == "empty_generation"

    def test_classify_answer_outcome_distinguishes_generation_failure(self):
        assert _classify_answer_outcome(
            "",
            generation_error=None,
            generation_timed_out=True,
            chunks_in_prompt=3,
        ) == "generation_timeout"
        assert _classify_answer_outcome(
            "",
            generation_error="connection refused",
            generation_timed_out=False,
            chunks_in_prompt=3,
        ) == "generation_error"
        assert _classify_answer_outcome(
            "",
            generation_error=None,
            generation_timed_out=False,
            chunks_in_prompt=3,
        ) == "empty_generation"
        assert _classify_answer_outcome(
            "I don't have enough information to answer that.",
            generation_error=None,
            generation_timed_out=False,
            chunks_in_prompt=3,
        ) == "insufficient_content"

    def test_resolve_generation_answer_substitutes_user_message(self):
        plan = rag_pipeline.GenerationPlan(
            citations=[{"source": "a.pdf", "page": 1, "section": ""}],
            analytics=QueryAnalytics(question="q", chunks_in_prompt=2),
        )
        display = _resolve_generation_answer(plan, "", plan.citations)
        assert display == GENERATION_FAILURE_USER_MESSAGE
        assert plan.analytics.answer_outcome == "empty_generation"

    def test_stream_empty_generation_yields_user_message(self, monkeypatch):
        plan = rag_pipeline.GenerationPlan(
            prompt="Context:\n" + ("word " * 50) + "\nQuestion: q\nAnswer:",
            citations=[{"source": "a.pdf", "page": 1, "section": ""}],
            analytics=QueryAnalytics(
                question="q",
                chunks_after_rerank=1,
                chunks_in_prompt=1,
            ),
        )
        monkeypatch.setattr(rag_pipeline, "_build_generation_plan", lambda *a, **k: plan)
        monkeypatch.setattr(rag_pipeline, "_stream_prompt", lambda *a, **k: (iter([]), []))
        monkeypatch.setattr(rag_pipeline, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        events = list(stream_query("q"))
        token_text = "".join(
            e.get("token", "") for e in events if "token" in e
        )
        meta = events[-1]["meta"]

        assert GENERATION_FAILURE_USER_MESSAGE in token_text
        assert meta["answer_outcome"] == "empty_generation"
        assert meta["insufficient_answer"] is True

    def test_generation_timeout_returns_user_message(self, monkeypatch):
        analytics = QueryAnalytics(
            question="q",
            chunks_after_rerank=1,
            chunks_in_prompt=2,
        )
        plan = rag_pipeline.GenerationPlan(
            prompt="Context:\n" + ("word " * 50) + "\nQuestion: q\nAnswer:",
            citations=[{"source": "a.pdf", "page": 1, "section": ""}],
            analytics=analytics,
        )
        monkeypatch.setattr(rag_pipeline, "_build_generation_plan", lambda *a, **k: plan)
        monkeypatch.setattr(
            rag_pipeline, "_answer_generation_timeout_seconds", lambda: 0.01
        )

        def slow_generate(prompt):
            time.sleep(0.2)
            return "late answer"

        llm = MagicMock()
        llm.generate = slow_generate
        monkeypatch.setattr(rag_pipeline, "get_llm", lambda: llm)
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        result = run_query("q")

        assert analytics.generation_timed_out is True
        assert result["meta"]["answer_outcome"] == "generation_timeout"
        assert result["answer"] == GENERATION_FAILURE_USER_MESSAGE
