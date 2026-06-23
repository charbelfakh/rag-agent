"""Sprint C gap closure tests."""
import sys
import types
from unittest.mock import MagicMock

import pytest

from providers.metadata import build_chunk_payload, resolve_metadata


class _FakeChunk:
    def __init__(self, text="x"):
        self.text = text
        self.page = 0
        self.section = ""


class TestTotalChunksPatch:
    def test_list_sources_includes_total_chunks(self, monkeypatch):
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
        fake_qdrant = types.ModuleType("qdrant_client")
        fake_qdrant.models = fake_models
        monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant)
        monkeypatch.setitem(sys.modules, "qdrant_client.models", fake_models)

        from providers.qdrant_store import QdrantLocalStore

        record = MagicMock()
        record.payload = {
            "source": "docs/pekat/a.pdf",
            "vendor": "pekat",
            "total_chunks": 42,
        }
        client = MagicMock()
        collection_entry = MagicMock()
        collection_entry.name = "rag_docs"
        client.get_collections.return_value.collections = [collection_entry]
        client.scroll.side_effect = [([record], None)]
        client.create_payload_index = MagicMock()

        store = object.__new__(QdrantLocalStore)
        store.collection = "rag_docs"
        store.client = client

        rows = store.list_sources()
        assert rows[0]["total_chunks"] == 42

    def test_build_payload_omits_total_chunks_when_unknown(self):
        meta = resolve_metadata("data/a.pdf")
        payload = build_chunk_payload(_FakeChunk(), meta, 0)
        assert "total_chunks" not in payload


class TestCitationTotalChunks:
    def test_collect_citations_includes_total_chunks(self):
        from providers.rag_pipeline import collect_citations

        cites = collect_citations(
            [
                {
                    "source": "docs/a.pdf",
                    "file_name": "a.pdf",
                    "vendor": "pekat",
                    "page": 1,
                    "section": "S",
                    "chunk_index": 3,
                    "total_chunks": 10,
                }
            ]
        )
        assert cites[0]["total_chunks"] == 10
