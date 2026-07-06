"""Sprint C unit tests: Qdrant filter builder and list_sources aggregation."""
import sys
import types
from unittest.mock import MagicMock



def _install_fake_qdrant_client():
    fake_qdrant = types.ModuleType("qdrant_client")
    fake_models = types.ModuleType("qdrant_client.models")
    for name in (
        "Distance",
        "FieldCondition",
        "Filter",
        "MatchAny",
        "MatchValue",
        "PointStruct",
        "PayloadSchemaType",
        "QueryRequest",
        "VectorParams",
        "SparseVector",
        "SparseVectorParams",
        "Modifier",
    ):
        setattr(fake_models, name, MagicMock())
    fake_qdrant.QdrantClient = MagicMock()
    fake_qdrant.models = fake_models
    sys.modules["qdrant_client"] = fake_qdrant
    sys.modules["qdrant_client.models"] = fake_models
    return fake_models


_fake_models = _install_fake_qdrant_client()
from providers.qdrant_store import QdrantLocalStore  # noqa: E402


class TestQdrantFilterBuilder:
    def test_build_filter_vendor_only(self, monkeypatch):
        captured: list = []

        def fake_filter(*, must=None):
            captured.extend(must or [])
            return MagicMock(must=must)

        monkeypatch.setattr("providers.qdrant_store.Filter", fake_filter)
        filt = QdrantLocalStore._build_filter({"vendor": "pekat"})
        assert filt is not None
        assert len(captured) == 1

    def test_build_filter_empty(self):
        assert QdrantLocalStore._build_filter(None) is None
        assert QdrantLocalStore._build_filter({}) is None


class TestListSourcesAggregation:
    def test_list_sources_groups_metadata(self, monkeypatch):
        record = MagicMock()
        record.payload = {
            "source": "docs/pekat/a.pdf",
            "vendor": "pekat",
            "file_name": "a.pdf",
            "document_type": "manual",
            "ingestion_timestamp": "2026-01-01T00:00:00+00:00",
        }
        client = MagicMock()
        collection_entry = MagicMock()
        collection_entry.name = "rag_docs"
        client.get_collections.return_value.collections = [collection_entry]
        client.scroll.side_effect = [([record], None)]
        client.create_payload_index = MagicMock()

        monkeypatch.setenv("QDRANT_COLLECTION", "rag_docs")

        store = object.__new__(QdrantLocalStore)
        store.collection = "rag_docs"
        store.client = client
        rows = store.list_sources()

        assert len(rows) == 1
        assert rows[0]["vendor"] == "pekat"
        assert rows[0]["chunks"] == 1
        assert rows[0]["file_name"] == "a.pdf"
