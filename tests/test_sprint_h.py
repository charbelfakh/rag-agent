"""Sprint H tests: TEI embed, doc registry, web compression, Redis ingest jobs."""
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from providers import factory
from providers.doc_registry import DocumentRegistry, reset_doc_registry
from providers.ingest_jobs import IngestJobStore, reset_ingest_job_store
from providers.metadata import resolve_metadata
from providers.tei_embed import TEIEmbedder
from providers.rag_pipeline import (
    compress_web_results,
    compress_web_snippet,
    fetch_web_results,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    factory.reset_providers()
    reset_doc_registry()
    reset_ingest_job_store()
    yield
    factory.reset_providers()
    reset_doc_registry()
    reset_ingest_job_store()


class TestTEIEmbedder:
    def test_embed_batches_via_http(self, monkeypatch):
        monkeypatch.setenv("TEI_BASE_URL", "http://tei.test")
        embedder = TEIEmbedder()
        embedder.batch_size = 2

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            inputs = body["inputs"]
            return httpx.Response(
                200,
                json={"embeddings": [[float(i)] * 3 for i, _ in enumerate(inputs)]},
            )

        transport = httpx.MockTransport(handler)
        embedder._client = httpx.Client(transport=transport)

        vectors = embedder.embed(["a", "b", "c"])
        assert len(vectors) == 3
        assert len(vectors[0]) == 3

    def test_factory_selects_tei_provider(self, monkeypatch):
        monkeypatch.setenv("EMBED_PROVIDER", "tei")
        with patch("providers.tei_embed.TEIEmbedder") as constructor:
            constructor.return_value = MagicMock(name="tei")
            first = factory.get_embedder()
            second = factory.get_embedder()
        assert first is second
        constructor.assert_called_once()


class TestDocumentRegistry:
    def test_upsert_and_list(self, tmp_path):
        db = tmp_path / "registry.db"
        registry = DocumentRegistry(db_path=str(db))
        meta = resolve_metadata("docs/pekat/guide.pdf")
        registry.upsert(meta, chunk_count=42, content_hash="abc123")

        docs = registry.list_documents()
        assert len(docs) == 1
        assert docs[0]["source"] == "docs/pekat/guide.pdf"
        assert docs[0]["vendor"] == "pekat"
        assert docs[0]["chunks"] == 42
        assert docs[0]["content_hash"] == "abc123"

    def test_delete_removes_row(self, tmp_path):
        db = tmp_path / "registry.db"
        registry = DocumentRegistry(db_path=str(db))
        meta = resolve_metadata("data/manual.pdf")
        registry.upsert(meta, chunk_count=5)
        assert registry.delete("data/manual.pdf") is True
        assert registry.list_documents() == []


class TestWebCompression:
    def test_compress_snippet_at_sentence_boundary(self):
        text = "First sentence is here. " + ("word " * 400)
        compressed = compress_web_snippet(text, max_chars=80)
        assert len(compressed) <= 85
        assert compressed.endswith("…")

    def test_compress_web_results_applies_to_all(self, monkeypatch):
        monkeypatch.setenv("WEB_SNIPPET_MAX_CHARS", "100")
        results = [
            {"title": "A", "url": "http://a", "content": "x" * 2000},
            {"title": "B", "url": "http://b", "content": "short"},
        ]
        compressed = compress_web_results(results)
        assert len(compressed[0]["content"]) < len(results[0]["content"])
        assert compressed[1]["content"] == "short"

    def test_fetch_web_results_compresses(self, monkeypatch):
        monkeypatch.setenv("WEB_SNIPPET_MAX_CHARS", "50")
        long_content = "Intro. " + ("detail " * 100)

        with patch(
            "providers.rag_pipeline.web_search",
            return_value=[{"title": "T", "url": "http://t", "content": long_content}],
        ):
            results = fetch_web_results("question")
        assert len(results[0]["content"]) <= 55


class TestIngestJobStore:
    def test_memory_fallback_create_and_get(self, monkeypatch):
        monkeypatch.setenv("INGEST_JOBS_REDIS_ENABLED", "false")
        store = IngestJobStore()
        job = {
            "job_id": "job-1",
            "status": "ingesting",
            "stage": "reading",
            "started_at": 1.0,
            "chunks_done": 0,
            "chunks_total": 0,
        }
        store.create(job)
        loaded = store.get("job-1")
        assert loaded["status"] == "ingesting"

    def test_mark_stale_ingesting_as_interrupted(self, monkeypatch):
        monkeypatch.setenv("INGEST_JOBS_REDIS_ENABLED", "false")
        monkeypatch.setenv("INGEST_JOB_STALE_SECONDS", "10")
        store = IngestJobStore()
        store.create(
            {
                "job_id": "old-job",
                "status": "ingesting",
                "stage": "embedding",
                "started_at": 0.0,
                "chunks_done": 1,
                "chunks_total": 10,
            }
        )
        count = store.mark_stale_ingesting_as_interrupted()
        assert count == 1
        assert store.get("old-job")["status"] == "interrupted"
