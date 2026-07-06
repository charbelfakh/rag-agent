"""Sprint J tests: feedback, orchestrator, two-stage, slim payload, judge, OTel."""
from unittest.mock import MagicMock, patch


from eval.llm_judge import grade_answer, parse_judge_response
from providers.metadata import slim_payload_text
from providers.query_orchestrator import (
    QueryOrchestrator,
    RetrievalContext,
    _retrieval_top_k,
    is_two_stage_retrieval_enabled,
    should_supplement_video_transcript_retrieval,
)


class TestTwoStageRetrieval:
    def test_bi_encoder_top_k_when_enabled(self, monkeypatch):
        monkeypatch.setenv("TWO_STAGE_RETRIEVAL_ENABLED", "true")
        monkeypatch.setenv("BI_ENCODER_TOP_K", "50")
        monkeypatch.setenv("RERANKER_ENABLED", "true")
        assert is_two_stage_retrieval_enabled() is True
        assert _retrieval_top_k(5) == 50


class TestPayloadSlimming:
    def test_slim_payload_text_truncates(self):
        text = "word " * 300
        stored, full = slim_payload_text(text, max_chars=100)
        assert len(stored) <= 105
        assert full == text

    def test_hydrate_search_result_pattern(self):
        payload = {"text": "preview…", "text_full": "full body", "score": 0.9}
        if payload.get("text_full"):
            payload["text"] = payload["text_full"]
        assert payload["text"] == "full body"


class TestQueryOrchestrator:
    def test_cache_hit_short_circuit(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")
        monkeypatch.setenv("HYDE_ENABLED", "false")
        monkeypatch.setenv("RERANKER_ENABLED", "false")

        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2]]
        store = MagicMock()
        cache = MagicMock()
        cache.enabled = True
        cache.lookup.return_value = ("cached", 0.99)

        analytics = MagicMock()
        analytics.history_turns = 0

        with patch("providers.query_orchestrator.get_embedder", lambda: embedder):
            with patch("providers.query_orchestrator.get_vector_store", lambda: store):
                with patch("providers.query_orchestrator.get_semantic_cache", lambda: cache):
                    with patch("providers.query_orchestrator.log_cache_hit", lambda *a, **k: None):
                        result = QueryOrchestrator().run_retrieval(
                        RetrievalContext(
                            question="q",
                            top_k=5,
                            history_turns=0,
                            vendor_filter=None,
                            document_type_filter=None,
                            analytics=analytics,
                        )
                        )

        assert result.cached_answer == "cached"
        assert result.chunks == []
        store.search.assert_not_called()

    def test_should_supplement_video_transcript_retrieval(self):
        assert should_supplement_video_transcript_retrieval(
            "Mech-Mind bin picking demo with 3D camera"
        )
        assert not should_supplement_video_transcript_retrieval(
            "How does surface flatness work in pekat?"
        )

    def test_video_transcript_supplement_search(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
        monkeypatch.setenv("HYDE_ENABLED", "false")
        monkeypatch.setenv("RERANKER_ENABLED", "false")

        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2]]
        store = MagicMock()
        store.search.side_effect = [
            [{"source": "manual.pdf", "content_type": "text", "score": 0.9}],
            [
                {
                    "source": "Random Bin Picking Tutorial Introduction",
                    "content_type": "video_transcript",
                    "start_seconds": 0.12,
                    "score": 0.8,
                }
            ],
        ]
        analytics = MagicMock()

        with patch("providers.query_orchestrator.get_embedder", lambda: embedder):
            with patch("providers.query_orchestrator.get_vector_store", lambda: store):
                result = QueryOrchestrator().run_retrieval(
                    RetrievalContext(
                        question="Mech-Mind bin picking demo with 3D camera",
                        top_k=5,
                        history_turns=0,
                        vendor_filter=None,
                        document_type_filter=None,
                        analytics=analytics,
                    )
                )

        assert store.search.call_count == 2
        assert analytics.video_transcript_supplement is True
        assert any(
            c.get("content_type") == "video_transcript" for c in result.chunks
        )


class TestLlmJudge:
    def test_parse_judge_response(self):
        parsed = parse_judge_response(
            'Sure. {"pass": true, "score": 0.9, "reason": "Grounded answer"}'
        )
        assert parsed["pass"] is True
        assert parsed["score"] == 0.9

    def test_grade_answer_calls_llm(self):
        llm = MagicMock()
        llm.generate.return_value = '{"pass": true, "score": 1.0, "reason": "ok"}'
        result = grade_answer(
            llm,
            question="How to calibrate?",
            answer="Follow the setup wizard.",
            expected_sources=["data/a.pdf"],
        )
        assert result["pass"] is True
        llm.generate.assert_called_once()


class TestFeedback:
    def test_log_feedback_posts_score(self, monkeypatch):
        monkeypatch.setenv("ANALYTICS_ENABLED", "true")
        lf = MagicMock()

        def immediate_thread(target, args=(), daemon=True):
            target(*args)
            return MagicMock()

        with patch("providers.langfuse_logger._get_langfuse", return_value=lf):
            with patch("providers.langfuse_logger.threading.Thread", side_effect=immediate_thread):
                from providers.langfuse_logger import log_feedback

                log_feedback(
                    question="q",
                    answer="a",
                    rating=1,
                    trace_id="trace-123",
                )

        lf.create_score.assert_called_once()
        assert lf.create_score.call_args.kwargs["trace_id"] == "trace-123"


class TestOtel:
    def test_record_query_span_noop_without_setup(self):
        from providers.otel_tracing import record_query_span

        record_query_span(
            question="q",
            latency_ms=10,
            cache_hit=False,
            chunks_retrieved=3,
            trace_id="abc",
        )
