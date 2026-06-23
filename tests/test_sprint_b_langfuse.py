"""Sprint B unit tests: Langfuse log_query extended payload."""
from unittest.mock import MagicMock, patch

import pytest

from providers import langfuse_logger


@pytest.fixture(autouse=True)
def enable_analytics(monkeypatch):
    monkeypatch.setenv("ANALYTICS_ENABLED", "true")


class TestLogQueryPayload:
    def test_log_query_includes_sprint_b_fields(self, monkeypatch):
        captured: list[dict] = []

        def fake_sync(data: dict) -> None:
            captured.append(data)

        monkeypatch.setattr(langfuse_logger, "_log_query_sync", fake_sync)

        def immediate_thread(target, args=(), daemon=True):
            target(*args)
            return MagicMock()

        with patch.object(langfuse_logger.threading, "Thread", side_effect=immediate_thread):
            langfuse_logger.log_query(
                question="How to calibrate?",
                hyde_used=False,
                cache_hit=False,
                web_fallback_used=False,
                web_fallback_reason=None,
                sufficiency_check_result="skipped",
                chunks_retrieved=20,
                chunks_after_rerank=5,
                chunks_in_prompt=5,
                top_chunk_score=0.91,
                top_chunk_source="data/manual.pdf",
                top_chunk_page=2,
                top_vector_score=0.75,
                mean_vector_score=0.62,
                top_rerank_score=0.91,
                mean_rerank_score=0.7,
                embed_ms=200,
                qdrant_search_ms=12,
                rerank_ms=450,
                prompt_token_count=3200,
                citations_count=3,
                insufficient_answer=False,
                empty_retrieval=False,
                answer_length=512,
                latency_ms=8000,
                history_turns=2,
            )

        assert len(captured) == 1
        payload = captured[0]
        assert payload["top_vector_score"] == 0.75
        assert payload["mean_vector_score"] == 0.62
        assert payload["top_rerank_score"] == 0.91
        assert payload["embed_ms"] == 200
        assert payload["prompt_token_count"] == 3200
        assert payload["citations_count"] == 3
        assert payload["insufficient_answer"] is False
        assert payload["empty_retrieval"] is False


class TestNestedObservations:
    def test_stage_spans_created_via_parent_start_observation(self, monkeypatch):
        parent = MagicMock()
        stage_names: list[str] = []
        parent.start_observation.side_effect = lambda **kwargs: (
            stage_names.append(kwargs["name"]) or MagicMock()
        )

        lf = MagicMock()
        lf.start_observation.return_value = parent

        monkeypatch.setattr(langfuse_logger, "_get_langfuse", lambda: lf)

        langfuse_logger._log_query_sync(
            {
                "question": "How to calibrate?",
                "hyde_used": False,
                "cache_hit": False,
                "web_fallback_used": False,
                "web_fallback_reason": None,
                "sufficiency_check_result": "skipped",
                "chunks_retrieved": 10,
                "embed_ms": 120,
                "qdrant_search_ms": 15,
                "answer_length": 100,
                "latency_ms": 5000,
                "history_turns": 0,
            }
        )

        lf.start_observation.assert_called_once()
        assert "parent_observation_id" not in lf.start_observation.call_args.kwargs
        assert parent.start_observation.call_count == 2
        assert stage_names == ["embed", "qdrant_search"]
        parent.update.assert_called_once()
        parent.end.assert_called_once()
        lf.flush.assert_called_once()

    def test_stage_spans_degrade_to_root_when_parent_has_no_start_observation(
        self, monkeypatch
    ):
        parent = object()
        root_calls: list[dict] = []

        def fake_start_observation(**kwargs):
            root_calls.append(kwargs)
            if len(root_calls) == 1:
                return parent
            return MagicMock()

        lf = MagicMock()
        lf.start_observation.side_effect = fake_start_observation

        monkeypatch.setattr(langfuse_logger, "_get_langfuse", lambda: lf)

        langfuse_logger._log_query_sync(
            {
                "question": "How to calibrate?",
                "hyde_used": False,
                "cache_hit": False,
                "web_fallback_used": False,
                "web_fallback_reason": None,
                "sufficiency_check_result": "skipped",
                "chunks_retrieved": 10,
                "embed_ms": 120,
                "answer_length": 100,
                "latency_ms": 5000,
                "history_turns": 0,
            }
        )

        assert len(root_calls) == 2
        assert root_calls[0]["name"] == "rag_query"
        assert root_calls[1]["name"] == "embed"
        assert "parent_observation_id" not in root_calls[1]
