"""Sprint G tests: early web fallback, incremental ingest, nested Langfuse spans."""
import hashlib
from unittest.mock import MagicMock

import pytest

import providers.rag_pipeline as rag_pipeline
from conftest import patch_retrieval_pipeline
from providers import langfuse_logger
from providers.metadata import (
    compute_file_content_hash,
    incremental_ingest_precheck,
    is_incremental_ingest_enabled,
    normalize_source,
    resolve_metadata,
)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("SUFFICIENCY_CHECK_ENABLED", "false")
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    monkeypatch.setenv("EARLY_WEB_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("INCREMENTAL_INGEST_ENABLED", "true")
    monkeypatch.delenv("ANALYTICS_ENABLED", raising=False)


def _chunk(score: float = 0.8):
    return {
        "text": "Calibration steps for the sensor.",
        "source": "data/manual.pdf",
        "page": 0,
        "score": score,
        "vector_score": score,
    }


class TestEarlyWebFallback:
    def test_score_indicates_early_web_on_low_top_score(self):
        analytics = rag_pipeline.QueryAnalytics(question="q", top_vector_score=0.2)
        assert rag_pipeline.score_indicates_early_web([_chunk(0.2)], 0.2, analytics)

    def test_score_indicates_early_web_disabled(self, monkeypatch):
        monkeypatch.setenv("EARLY_WEB_FALLBACK_ENABLED", "false")
        analytics = rag_pipeline.QueryAnalytics(question="q")
        assert not rag_pipeline.score_indicates_early_web([_chunk(0.1)], 0.1, analytics)

    def test_score_indicates_early_web_skips_when_rerank_is_strong(self):
        analytics = rag_pipeline.QueryAnalytics(question="q", top_vector_score=0.03)
        chunk = {
            **_chunk(2.4),
            "score": 2.4,
            "vector_score": 0.03,
            "rerank_score": 2.4,
        }
        assert not rag_pipeline.score_indicates_early_web([chunk], 2.4, analytics)

    def test_needs_web_fallback_uses_rerank_not_fused_score(self):
        chunk = {
            **_chunk(2.1),
            "score": 2.1,
            "vector_score": 0.03,
            "rerank_score": 2.1,
        }
        assert not rag_pipeline.needs_web_fallback("bin picking demo", [chunk], 0.03)

    def test_plan_uses_web_upfront_on_low_scores(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = [_chunk(0.15)]
        llm = MagicMock()
        web_hits = [
            {"title": "Web", "content": "Answer from web", "url": "http://x"}
        ]

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=llm,
            cache=MagicMock(enabled=False),
            reranker=MagicMock(),
        )
        monkeypatch.setattr(rag_pipeline, "web_search", lambda q: web_hits)
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        plan = rag_pipeline._build_generation_plan(
            "XYZ-9000 calibration procedure details", top_k=5
        )

        assert plan.analytics.web_fallback_used is True
        assert plan.analytics.web_fallback_reason in ("score_early", "upfront")
        assert plan.allow_insufficient_retry is False
        assert "Answer from web" in plan.prompt

    def test_stream_query_skips_retry_when_early_web_enabled(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = [_chunk(0.9)]
        llm = MagicMock()
        llm.generate_stream.return_value = iter(
            ["I don't have enough information in the documents."]
        )

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=llm,
            cache=MagicMock(enabled=False),
            reranker=MagicMock(),
        )
        monkeypatch.setattr(rag_pipeline, "web_search", MagicMock())
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        events = list(rag_pipeline.stream_query("Obscure topic?", top_k=5))

        rag_pipeline.web_search.assert_not_called()
        assert llm.generate_stream.call_count == 1
        assert events[-1]["done"] is True


class TestIncrementalIngest:
    def test_compute_file_content_hash(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("hello ingest", encoding="utf-8")
        expected = hashlib.sha256(b"hello ingest").hexdigest()
        assert compute_file_content_hash(str(path)) == expected

    def test_precheck_skips_unchanged_source(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("stable", encoding="utf-8")
        file_hash = compute_file_content_hash(str(path))
        normalized = normalize_source(str(path))

        store = MagicMock()
        store.get_source_content_hash.return_value = file_hash

        action, content_hash = incremental_ingest_precheck(str(path), normalized, store)

        assert action == "skip"
        assert content_hash == file_hash

    def test_precheck_replace_when_hash_differs(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("new content", encoding="utf-8")
        normalized = normalize_source(str(path))

        store = MagicMock()
        store.get_source_content_hash.return_value = "oldhash"

        action, content_hash = incremental_ingest_precheck(str(path), normalized, store)

        assert action == "replace"
        assert content_hash == compute_file_content_hash(str(path))

    def test_precheck_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INCREMENTAL_INGEST_ENABLED", "false")
        path = tmp_path / "doc.txt"
        path.write_text("x", encoding="utf-8")
        store = MagicMock()

        action, content_hash = incremental_ingest_precheck(
            str(path), normalize_source(str(path)), store
        )

        assert action == "ingest"
        assert content_hash == ""
        assert is_incremental_ingest_enabled() is False

    def test_content_hash_in_payload(self, tmp_path):
        path = tmp_path / "a.pdf"
        path.write_bytes(b"%PDF-1.4")
        meta = resolve_metadata(str(path))
        meta.content_hash = compute_file_content_hash(str(path))
        fields = meta.to_payload_fields()
        assert fields["content_hash"] == meta.content_hash


class TestNestedLangfuseSpans:
    def test_log_query_creates_stage_child_spans(self, monkeypatch):
        monkeypatch.setenv("ANALYTICS_ENABLED", "true")

        children: list[str] = []

        parent = MagicMock()
        parent.id = "parent-obs-1"

        def parent_start_observation(**kwargs):
            children.append(kwargs.get("name", ""))
            obs = MagicMock()
            obs.id = f"child-{kwargs.get('name', '')}"
            return obs

        parent.start_observation.side_effect = parent_start_observation

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
                "chunks_after_rerank": 5,
                "top_chunk_score": 0.5,
                "top_chunk_source": "data/a.pdf",
                "top_chunk_page": 1,
                "embed_ms": 120,
                "qdrant_search_ms": 15,
                "rerank_ms": 200,
                "pre_stream_ms": 500,
                "llm_ttft_ms": 800,
                "llm_generation_ms": 3000,
                "answer_length": 100,
                "latency_ms": 5000,
                "history_turns": 0,
            }
        )

        assert "embed" in children
        assert "qdrant_search" in children
        assert "llm_generation" in children
        lf.flush.assert_called_once()
