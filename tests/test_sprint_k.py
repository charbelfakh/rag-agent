"""Sprint K tests: ingest queue, blob store, vendor sharding, vLLM LLM."""
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from providers.blob_store import BlobStore, reset_blob_store
from providers.ingest_queue import IngestQueue, reset_ingest_queue
from providers.openai_chat_llm import OpenAIChatLLM
from providers.qdrant_sharded_store import (
    VendorShardedQdrantStore,
    normalize_vendor_shard,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    reset_ingest_queue()
    reset_blob_store()
    yield
    reset_ingest_queue()
    reset_blob_store()


class TestIngestQueue:
    def test_enqueue_and_dequeue(self):
        redis = MagicMock()
        redis.ping.return_value = True
        redis.lpush.return_value = 1
        redis.brpop.return_value = (
            "ingest:queue",
            json.dumps({"job_id": "j1", "ingest_path": "data/a.pdf"}),
        )
        redis.setex.return_value = True

        queue = IngestQueue()
        queue._redis = redis

        assert queue.enqueue({"job_id": "j1", "ingest_path": "data/a.pdf"}) is True
        job = queue.dequeue(block_timeout=1)
        assert job["job_id"] == "j1"
        redis.lpush.assert_called_once()


class TestBlobStore:
    def test_externalize_and_hydrate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BLOB_STORAGE_ENABLED", "true")
        monkeypatch.setenv("BLOB_PAYLOAD_PREVIEW_CHARS", "20")
        store = BlobStore(root=str(tmp_path / "blobs"))

        payload = {
            "chunk_id": "abc",
            "text": "x" * 100,
            "source": "data/a.pdf",
        }
        slim = store.externalize_payload_text(payload)
        assert slim["text_uri"].startswith("blob://")
        assert len(slim["text"]) <= 25

        hydrated = store.hydrate_payload(slim)
        assert hydrated["text"] == payload["text"]


class TestVendorSharding:
    def test_normalize_vendor_shard(self):
        assert normalize_vendor_shard("Mech Mind") == "mech_mind"

    def test_routes_upsert_to_vendor_collection(self, monkeypatch):
        monkeypatch.setenv("QDRANT_VENDOR_SHARDING", "true")
        monkeypatch.setenv("QDRANT_COLLECTION_PREFIX", "rag_docs")

        store = VendorShardedQdrantStore.__new__(VendorShardedQdrantStore)
        store.shard_prefix = "rag_docs"
        store.default_collection = "rag_docs"
        store.collection = "rag_docs"
        store.client = MagicMock()
        store._collection_names = MagicMock(return_value=[])
        store._ensure_collection = MagicMock()
        store._ensure_payload_indexes = MagicMock()

        with patch("providers.blob_store.get_blob_store") as blob_factory:
            blob_factory.return_value.externalize_payload_text.side_effect = (
                lambda payload: payload
            )
            VendorShardedQdrantStore.upsert(
                store,
                ids=["1"],
                vectors=[[0.1]],
                payloads=[{"vendor": "mechmind", "text": "hi", "source": "data/a.pdf"}],
            )

        assert store.client.upsert.called
        collection_name = store.client.upsert.call_args.kwargs.get(
            "collection_name"
        ) or store.client.upsert.call_args.args[0]
        assert collection_name == "rag_docs_mechmind"


class TestVllmLlm:
    def test_generate_uses_chat_completions(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://vllm.test/v1")
        llm = OpenAIChatLLM()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Answer text"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            )

        transport = httpx.MockTransport(handler)
        llm._client = httpx.Client(transport=transport)

        answer = llm.generate("Question?")
        assert answer == "Answer text"
        assert llm.last_stream_stats["prompt_eval_count"] == 10

    def test_factory_selects_vllm_provider(self, monkeypatch):
        from providers import factory

        factory.reset_providers()
        monkeypatch.setenv("LLM_PROVIDER", "vllm")
        with patch("providers.openai_chat_llm.OpenAIChatLLM") as constructor:
            constructor.return_value = MagicMock(name="vllm")
            first = factory.get_llm()
            second = factory.get_llm()
        assert first is second
        constructor.assert_called_once()
        factory.reset_providers()
