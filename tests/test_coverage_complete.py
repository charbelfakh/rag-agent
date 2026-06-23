"""Gap-filling tests for modules added in Sprints G–K (full coverage audit)."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from conftest import patch_retrieval_pipeline
from eval.llm_judge import grade_answer, parse_judge_response
from providers import factory
from providers.doc_registry import DocumentRegistry
from providers.ingest_queue import is_ingest_queue_enabled, reset_ingest_queue
from providers.metadata import (
    build_chunk_payload,
    incremental_ingest_precheck,
    resolve_metadata,
    slim_payload_text,
)
from providers.otel_tracing import is_otel_enabled, record_query_span, setup_otel, shutdown_otel
from providers.query_orchestrator import (
    _retrieval_top_k,
    is_parallel_orchestrator_enabled,
)
from providers.semantic_cache import LegacySemanticCacheBackend, reset_semantic_cache
from providers.semantic_cache_redisvl import RedisVLSemanticCacheBackend

import providers.rag_pipeline as rag_pipeline
from providers.rag_pipeline import _build_generation_plan


class _FakeChunk:
    def __init__(self, text: str = "chunk body"):
        self.text = text
        self.page = 0
        self.section = ""


@pytest.fixture(autouse=True)
def reset_singletons():
    factory.reset_providers()
    reset_semantic_cache()
    reset_ingest_queue()
    shutdown_otel()
    yield
    factory.reset_providers()
    reset_semantic_cache()
    reset_ingest_queue()
    shutdown_otel()


class TestMetadataHelpers:
    def test_build_chunk_payload_with_slim_text(self):
        meta = resolve_metadata("data/manual.pdf")
        stored, full = slim_payload_text("x" * 50, max_chars=10)
        item = _FakeChunk(stored)
        payload = build_chunk_payload(item, meta, 0, total_chunks=1)
        assert payload["chunk_index"] == 0
        assert len(payload["text"]) <= 12

    def test_incremental_precheck_replace(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("updated", encoding="utf-8")
        store = MagicMock()
        store.get_source_content_hash.return_value = "stale"
        action, digest = incremental_ingest_precheck(str(path), "data/doc.txt", store)
        assert action == "replace"
        assert len(digest) == 64


class TestQueryOrchestratorCoverage:
    def test_parallel_orchestrator_flag_default(self, monkeypatch):
        monkeypatch.delenv("QUERY_ORCHESTRATOR_ENABLED", raising=False)
        assert is_parallel_orchestrator_enabled() is True

    def test_two_stage_fetch_k(self, monkeypatch):
        monkeypatch.setenv("TWO_STAGE_RETRIEVAL_ENABLED", "true")
        monkeypatch.setenv("BI_ENCODER_TOP_K", "50")
        assert _retrieval_top_k(5) == 50

    def test_trace_id_in_plan_meta(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
        monkeypatch.setenv("HYDE_ENABLED", "false")
        monkeypatch.setenv("RERANKER_ENABLED", "false")
        monkeypatch.setenv("EARLY_WEB_FALLBACK_ENABLED", "false")
        embedder = MagicMock()
        embedder.embed.return_value = [[0.2]]
        store = MagicMock()
        store.search.return_value = [
            {
                "text": "body",
                "source": "data/a.pdf",
                "page": 0,
                "score": 0.9,
            }
        ]
        cache = MagicMock(enabled=False)
        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=MagicMock(),
            cache=cache,
        )
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        plan = _build_generation_plan("What is Pekat?", top_k=5)
        assert plan.analytics.trace_id
        meta = plan.analytics.to_meta()
        assert meta["trace_id"] == plan.analytics.trace_id


class TestSemanticCacheRedisVL:
    def test_redisvl_lookup_hit(self):
        redis = MagicMock()
        index = MagicMock()
        index.query.return_value = [
            {
                "answer": "cached",
                "embedding": [1.0, 0.0],
                "vector_distance": 0.02,
            }
        ]

        backend = RedisVLSemanticCacheBackend.__new__(RedisVLSemanticCacheBackend)
        backend._redis = redis
        backend.threshold = 0.9
        backend.max_size = 10
        backend._index = index
        backend._VectorQuery = MagicMock(return_value=MagicMock())

        hit = backend.lookup([1.0, 0.0])
        assert hit == ("cached", pytest.approx(1.0, abs=0.01))

    def test_legacy_backend_store_respects_max_size(self):
        redis = MagicMock()
        redis.llen.side_effect = [10, 9]
        redis.pipeline.return_value = redis
        backend = LegacySemanticCacheBackend(redis, threshold=0.9, max_size=10)
        backend.store("q", [1.0], "a")
        redis.lpop.assert_called()


class TestOtelTracing:
    def test_setup_and_record_span(self, monkeypatch):
        monkeypatch.setenv("OTEL_ENABLED", "true")
        monkeypatch.setenv("OTEL_EXPORTER", "console")
        setup_otel("rag-agent-test")
        record_query_span(
            question="hello",
            latency_ms=42,
            cache_hit=False,
            chunks_retrieved=3,
            trace_id="trace-1",
        )
        assert is_otel_enabled() is True


class TestFactorySharding:
    def test_factory_uses_sharded_store(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE", "qdrant_local")
        monkeypatch.setenv("QDRANT_VENDOR_SHARDING", "true")

        fake_models = types.ModuleType("qdrant_client.models")
        for name in (
            "Distance",
            "FieldCondition",
            "Filter",
            "MatchValue",
            "PointStruct",
            "PayloadSchemaType",
            "VectorParams",
        ):
            setattr(fake_models, name, MagicMock())
        fake_client = types.ModuleType("qdrant_client")
        fake_client.QdrantClient = MagicMock(return_value=MagicMock())
        fake_client.models = fake_models

        with patch.dict(
            sys.modules,
            {"qdrant_client": fake_client, "qdrant_client.models": fake_models},
        ):
            from providers.qdrant_sharded_store import VendorShardedQdrantStore

            with patch.object(
                VendorShardedQdrantStore,
                "_ensure_collection",
                lambda self: None,
            ):
                with patch.object(
                    VendorShardedQdrantStore,
                    "_ensure_payload_indexes",
                    lambda self: None,
                ):
                    store = factory.get_vector_store()

        assert store.__class__.__name__ == "VendorShardedQdrantStore"


class TestOpenAIChatStream:
    def test_generate_stream_yields_tokens(self, monkeypatch):
        from providers.openai_chat_llm import OpenAIChatLLM

        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://vllm.test/v1")
        llm = OpenAIChatLLM()

        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            "data: [DONE]",
        ]

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_lines(self):
                yield from lines

            def raise_for_status(self):
                return None

        llm._client = MagicMock()
        llm._client.stream.return_value = FakeStream()

        tokens = list(llm.generate_stream("Question?"))
        assert tokens == ["Hi"]


class TestVendorShardedSearch:
    def test_merge_shard_results(self, monkeypatch):
        monkeypatch.setenv("QDRANT_VENDOR_SHARDING", "true")
        from providers.qdrant_sharded_store import VendorShardedQdrantStore
        from providers.qdrant_store import QdrantLocalStore

        store = VendorShardedQdrantStore.__new__(VendorShardedQdrantStore)
        store.shard_prefix = "rag_docs"
        store.default_collection = "rag_docs"
        store.collection = "rag_docs"
        store.client = MagicMock()
        store._collection_names = MagicMock(
            return_value=["rag_docs_mechmind", "rag_docs_pekat"]
        )
        store._ensure_collection = MagicMock()
        store._ensure_payload_indexes = MagicMock()

        def fake_search(self, vector, top_k=5, filter_payload=None):
            if self.collection.endswith("mechmind"):
                return [{"text": "a", "score": 0.9, "source": "a.pdf"}]
            return [{"text": "b", "score": 0.7, "source": "b.pdf"}]

        with patch.object(QdrantLocalStore, "search", fake_search):
            merged = VendorShardedQdrantStore.search(store, [0.1], top_k=2)
        assert len(merged) == 2
        assert merged[0]["score"] == 0.9


class TestScripts:
    def test_migrate_to_v2_dry_run(self, monkeypatch):
        from scripts.ops.migrate_to_v2 import migrate_collection

        client = MagicMock()
        client.count.return_value = MagicMock(count=12)
        info = MagicMock()
        info.config.params.vectors.size = 768
        client.get_collection.return_value = info
        monkeypatch.setenv("QDRANT_LOCAL_URL", "http://qdrant.test")

        with patch("scripts.ops.migrate_to_v2.QdrantClient", return_value=client):
            result = migrate_collection(source="rag_docs", target="rag_docs_v2", dry_run=True)

        assert result["points"] == 12
        assert result["dry_run"] is True

    def test_backfill_doc_registry_dry_run(self, monkeypatch, capsys):
        from scripts.ops import backfill_doc_registry

        fake_store = MagicMock()
        fake_store.list_sources.return_value = [
            {"source": "data/a.pdf", "vendor": "mechmind", "chunks": 2}
        ]
        monkeypatch.setattr(backfill_doc_registry, "get_vector_store", lambda: fake_store)
        monkeypatch.setattr(backfill_doc_registry, "reset_providers", lambda: None)
        monkeypatch.setattr(backfill_doc_registry, "reset_doc_registry", lambda: None)
        monkeypatch.setattr(sys, "argv", ["backfill_doc_registry", "--dry-run"])

        exit_code = backfill_doc_registry.main()
        assert exit_code == 0
        assert "Found 1 sources" in capsys.readouterr().out

    def test_ingest_worker_run_job_success(self, monkeypatch):
        from scripts.ingest import ingest_worker

        store = MagicMock()
        store.get.side_effect = lambda _job_id: {
            "job_id": "j1",
            "status": "queued",
            "started_at": 1.0,
        }
        monkeypatch.setattr(ingest_worker, "get_ingest_job_store", lambda: store)
        monkeypatch.setattr(ingest_worker, "ingest", lambda *a, **k: 4)

        ingest_worker._run_job(
            {
                "job_id": "j1",
                "ingest_path": "data/a.pdf",
                "vendor": None,
                "document_type": None,
            }
        )
        assert store.update.call_count >= 2
        final_kwargs = store.update.call_args_list[-1].kwargs
        assert final_kwargs["status"] == "done"


class TestEvalAnswer:
    def test_parse_judge_response_invalid(self):
        parsed = parse_judge_response("not json at all")
        assert parsed["pass"] is False

    def test_run_answer_eval_mocked(self, monkeypatch, tmp_path):
        import eval.run_answer_eval as answer_eval

        dataset = tmp_path / "set.jsonl"
        dataset.write_text(
            '{"id":"q1","question":"What is X?","expected_sources":["data/a.pdf"]}\n',
            encoding="utf-8",
        )
        llm = MagicMock()
        llm.generate.return_value = '{"pass": true, "score": 1.0, "reason": "ok"}'
        monkeypatch.setattr(answer_eval, "get_llm", lambda: llm)
        monkeypatch.setattr(
            answer_eval,
            "query",
            lambda question, top_k=5: {"answer": "X is a sensor."},
        )

        report = answer_eval.run_answer_eval(dataset, limit=1)
        assert report["count"] == 1
        assert report["pass_rate"] == 1.0

    def test_grade_answer_negative(self):
        llm = MagicMock()
        llm.generate.return_value = '{"pass": false, "score": 0.0, "reason": "bad"}'
        result = grade_answer(llm, question="q", answer="nope", expected_sources=[])
        assert result["pass"] is False


class TestIngestQueueFlag:
    def test_is_ingest_queue_enabled(self, monkeypatch):
        monkeypatch.setenv("INGEST_QUEUE_ENABLED", "true")
        assert is_ingest_queue_enabled() is True


class TestLangfuseTraceId:
    def test_log_query_carries_trace_id(self, monkeypatch):
        monkeypatch.setenv("ANALYTICS_ENABLED", "true")
        captured: list[dict] = []

        def fake_sync(data: dict) -> None:
            captured.append(data)

        class ImmediateThread:
            def __init__(self, target, args=(), daemon=True):
                self._target = target
                self._args = args

            def start(self):
                self._target(*self._args)

        monkeypatch.setattr("providers.langfuse_logger._log_query_sync", fake_sync)
        monkeypatch.setattr("providers.langfuse_logger.threading.Thread", ImmediateThread)
        from providers.langfuse_logger import log_query

        log_query(
            question="q",
            trace_id="trace-xyz",
            hyde_used=False,
            cache_hit=False,
            web_fallback_used=False,
            web_fallback_reason=None,
            sufficiency_check_result="skipped",
            chunks_retrieved=1,
            chunks_after_rerank=1,
            top_chunk_score=0.9,
            top_chunk_source="data/a.pdf",
            top_chunk_page=0,
            answer_length=10,
            latency_ms=100,
            history_turns=0,
        )
        assert captured[0]["trace_id"] == "trace-xyz"


class TestQdrantBlobIntegration:
    def test_search_hydrates_blob_text(self, monkeypatch):
        fake_models = types.ModuleType("qdrant_client.models")
        for name in (
            "Distance",
            "FieldCondition",
            "Filter",
            "MatchValue",
            "PointStruct",
            "PayloadSchemaType",
            "VectorParams",
        ):
            setattr(fake_models, name, MagicMock())
        fake_client_mod = types.ModuleType("qdrant_client")
        client = MagicMock()
        hit = MagicMock()
        hit.payload = {
            "text": "preview…",
            "text_uri": "blob://abc.txt",
            "source": "data/a.pdf",
        }
        hit.score = 0.8
        client.query_points.return_value = MagicMock(points=[hit])
        fake_client_mod.QdrantClient = MagicMock(return_value=client)
        fake_client_mod.models = fake_models

        blob = MagicMock()
        blob.hydrate_payload.side_effect = lambda p: {**p, "text": "full chunk text"}

        with patch("providers.blob_store.get_blob_store", lambda: blob):
            from providers.qdrant_store import QdrantLocalStore

            store = QdrantLocalStore.__new__(QdrantLocalStore)
            store.collection = "rag_docs"
            store.client = client
            results = QdrantLocalStore.search(store, [0.1], top_k=1)

        assert results[0]["text"] == "full chunk text"
        blob.hydrate_payload.assert_called_once()
