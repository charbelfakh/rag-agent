"""Tests for sparse/hybrid retrieval (QDRANT_SPARSE_ENABLED)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from providers.sparse_text import (
    SPARSE_VECTOR_NAME,
    is_sparse_enabled,
    sparse_text_vector,
    tokenize,
)


@pytest.fixture(autouse=True)
def _reset_providers():
    from providers.factory import reset_providers

    reset_providers()
    yield
    reset_providers()


def _make_store():
    """QdrantLocalStore with a mocked client, bypassing __init__ network calls."""
    from providers.qdrant_store import QdrantLocalStore

    store = object.__new__(QdrantLocalStore)
    store.collection = "rag_docs"
    store.client = MagicMock()
    store._sparse_search_ok = True
    return store


def _points(*rows):
    return SimpleNamespace(
        points=[SimpleNamespace(payload=payload, score=score) for payload, score in rows]
    )


class TestSparseTextVector:
    def test_flag_default_off(self, monkeypatch):
        monkeypatch.delenv("QDRANT_SPARSE_ENABLED", raising=False)
        assert is_sparse_enabled() is False

    def test_tokenize_keeps_model_numbers(self):
        assert tokenize("Gocator 2120A NOHD values") == [
            "gocator", "2120a", "nohd", "values",
        ]

    def test_term_frequencies_counted(self):
        indices, values = sparse_text_vector("laser laser class")
        assert len(indices) == 2
        assert sorted(values) == [1.0, 2.0]

    def test_indices_stable_across_calls(self):
        first, _ = sparse_text_vector("XL250")
        second, _ = sparse_text_vector("the xl250 sensor")
        assert first[0] in second

    def test_empty_text_yields_empty_vector(self):
        assert sparse_text_vector("   ") == ([], [])


class TestCollectionCreation:
    def test_sparse_config_included_when_enabled(self, monkeypatch):
        monkeypatch.setenv("QDRANT_SPARSE_ENABLED", "true")
        monkeypatch.setenv("QDRANT_VECTOR_SIZE", "4")
        store = _make_store()
        store.client.get_collections.return_value = SimpleNamespace(collections=[])

        store._ensure_collection()

        kwargs = store.client.create_collection.call_args.kwargs
        assert SPARSE_VECTOR_NAME in kwargs["sparse_vectors_config"]

    def test_no_sparse_config_when_disabled(self, monkeypatch):
        monkeypatch.delenv("QDRANT_SPARSE_ENABLED", raising=False)
        monkeypatch.setenv("QDRANT_VECTOR_SIZE", "4")
        store = _make_store()
        store.client.get_collections.return_value = SimpleNamespace(collections=[])

        store._ensure_collection()

        kwargs = store.client.create_collection.call_args.kwargs
        assert "sparse_vectors_config" not in kwargs


class TestUpsert:
    def test_named_vectors_with_sparse_component(self, monkeypatch):
        monkeypatch.setenv("QDRANT_SPARSE_ENABLED", "true")
        store = _make_store()

        with pytest.MonkeyPatch.context() as mp:
            captured = []
            mp.setattr(
                "providers.qdrant_store.PointStruct",
                lambda **kw: captured.append(kw) or SimpleNamespace(**kw),
            )
            store.upsert(["id1"], [[0.1, 0.2]], [{"text": "Gocator 2120A laser"}])

        vector = captured[0]["vector"]
        assert vector[""] == [0.1, 0.2]
        assert SPARSE_VECTOR_NAME in vector

    def test_plain_vectors_when_disabled(self, monkeypatch):
        monkeypatch.delenv("QDRANT_SPARSE_ENABLED", raising=False)
        store = _make_store()

        with pytest.MonkeyPatch.context() as mp:
            captured = []
            mp.setattr(
                "providers.qdrant_store.PointStruct",
                lambda **kw: captured.append(kw) or SimpleNamespace(**kw),
            )
            store.upsert(["id1"], [[0.1, 0.2]], [{"text": "plain"}])

        assert captured[0]["vector"] == [0.1, 0.2]


class TestHybridSearch:
    def test_dense_only_without_query_text(self, monkeypatch):
        monkeypatch.setenv("QDRANT_SPARSE_ENABLED", "true")
        store = _make_store()
        store.client.query_points.return_value = _points(
            ({"chunk_id": "a", "text": "A"}, 0.9)
        )

        hits = store.search([0.1], top_k=5)

        assert [h["chunk_id"] for h in hits] == ["a"]
        assert store.client.query_points.call_count == 1

    def test_hybrid_fuses_and_preserves_dense_cosine(self, monkeypatch):
        monkeypatch.setenv("QDRANT_SPARSE_ENABLED", "true")
        store = _make_store()
        dense = _points(
            ({"chunk_id": "a", "text": "A"}, 0.9),
            ({"chunk_id": "b", "text": "B"}, 0.8),
        )
        sparse = _points(
            ({"chunk_id": "c", "text": "C 2120A"}, 12.5),
            ({"chunk_id": "a", "text": "A"}, 9.0),
        )
        store.client.query_points.side_effect = [dense, sparse]

        hits = store.search([0.1], top_k=3, query_text="2120A NOHD")

        assert store.client.query_points.call_count == 2
        by_id = {h["chunk_id"]: h for h in hits}
        # 'a' ranked in both lists → fused first, cosine preserved.
        assert hits[0]["chunk_id"] == "a"
        assert by_id["a"]["vector_score"] == pytest.approx(0.9)
        # Sparse-only hit present with backfilled (lowest dense) cosine.
        assert "c" in by_id
        assert by_id["c"]["vector_score"] == pytest.approx(0.8)

    def test_sparse_failure_falls_back_and_disables(self, monkeypatch):
        monkeypatch.setenv("QDRANT_SPARSE_ENABLED", "true")
        store = _make_store()
        dense = _points(({"chunk_id": "a", "text": "A"}, 0.9))
        store.client.query_points.side_effect = [dense, RuntimeError("no sparse index"), dense]

        first = store.search([0.1], top_k=3, query_text="q")
        assert [h["chunk_id"] for h in first] == ["a"]
        assert store._sparse_search_ok is False

        # Next search skips the sparse attempt entirely.
        second = store.search([0.1], top_k=3, query_text="q")
        assert [h["chunk_id"] for h in second] == ["a"]
        assert store.client.query_points.call_count == 3


class TestOrchestratorQueryTextForwarding:
    def test_supports_detection(self):
        from providers.query_orchestrator import _store_supports_query_text

        class Modern:
            def search(self, vector, top_k=5, filter_payload=None, *, query_text=None):
                return []

        class Legacy:
            def search(self, vector, top_k=5, filter_payload=None):
                return []

        class Kwargs:
            def search(self, vector, **kwargs):
                return []

        assert _store_supports_query_text(Modern()) is True
        assert _store_supports_query_text(Legacy()) is False
        assert _store_supports_query_text(Kwargs()) is True
