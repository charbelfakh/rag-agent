"""Tests for reingest_all stale completed-PDF guard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.ingest.reingest_all as reingest_all


def _pdf(name: str, vendor: str = "lmi") -> tuple[str, Path]:
    return vendor, Path("data") / vendor / name


@pytest.fixture
def fake_store():
    store = MagicMock()
    store.client = MagicMock()
    store.collection = "rag_docs"
    store.client.count.return_value = MagicMock(count=0)
    return store


def test_completed_pdf_with_text_points_stays_skipped(fake_store):
    pdfs = [_pdf("doc-a.pdf"), _pdf("doc-b.pdf")]
    completed = {"doc-a.pdf"}

    with (
        patch.object(reingest_all, "_store_collection_exists", return_value=True),
        patch.object(reingest_all, "_source_has_text_points", side_effect=lambda _s, name: name == "doc-a.pdf"),
    ):
        pending = reingest_all._pending_pdfs_for_text_ingest(
            pdfs,
            completed,
            store=fake_store,
            verify_completed=True,
        )

    assert pending == [_pdf("doc-b.pdf")]


def test_source_has_text_points_queries_text_content_type(fake_store):
    fake_store.client.count.return_value = MagicMock(count=1)

    with patch.object(reingest_all, "_store_collection_exists", return_value=True):
        assert reingest_all._source_has_text_points(fake_store, "doc-a.pdf") is True

    fake_store.client.count.assert_called_once_with(
        collection_name="rag_docs",
        count_filter=reingest_all._text_points_filter("doc-a.pdf"),
    )


def test_completed_pdf_with_only_image_captions_is_requeued(fake_store):
    fake_store.client.count.return_value = MagicMock(count=0)
    name = "15159-1.5.52.78_MANUAL_User_GoPxL_Line_Profile_Sensors_EN-US.pdf"
    pdfs = [_pdf(name)]
    completed = {name}

    with patch.object(reingest_all, "_store_collection_exists", return_value=True):
        pending = reingest_all._pending_pdfs_for_text_ingest(
            pdfs,
            completed,
            store=fake_store,
            verify_completed=True,
        )

    assert pending == [_pdf(name)]


def test_completed_pdf_requeued_when_collection_missing(fake_store):
    pdfs = [_pdf("doc-a.pdf"), _pdf("doc-b.pdf")]
    completed = {"doc-a.pdf"}

    with patch.object(reingest_all, "_store_collection_exists", return_value=False):
        pending = reingest_all._pending_pdfs_for_text_ingest(
            pdfs,
            completed,
            store=fake_store,
            verify_completed=True,
        )

    assert pending == pdfs
    fake_store.client.count.assert_not_called()


def test_verify_completed_disabled_uses_state_only(fake_store):
    name = "15159-1.5.52.78_MANUAL_User_GoPxL_Line_Profile_Sensors_EN-US.pdf"
    pdfs = [_pdf(name)]
    completed = {name}

    pending = reingest_all._pending_pdfs_for_text_ingest(
        pdfs,
        completed,
        store=fake_store,
        verify_completed=False,
    )

    assert pending == []
    fake_store.client.count.assert_not_called()


def test_source_has_text_points_false_when_collection_missing(fake_store):
    with patch.object(reingest_all, "_store_collection_exists", return_value=False):
        assert reingest_all._source_has_text_points(fake_store, "doc.pdf") is False
    fake_store.client.count.assert_not_called()


class TestStateFilePath:
    """REINGEST_STATE_PATH lets parallel per-vendor runs own separate state."""

    def test_env_override_sets_state_file(self, monkeypatch, tmp_path):
        import importlib

        override = tmp_path / "reingest_state_lmi.json"
        monkeypatch.setenv("REINGEST_STATE_PATH", str(override))
        try:
            importlib.reload(reingest_all)
            assert reingest_all.STATE_FILE == override
        finally:
            monkeypatch.delenv("REINGEST_STATE_PATH", raising=False)
            importlib.reload(reingest_all)

    def test_default_state_file_under_data_dir(self, monkeypatch):
        import importlib

        monkeypatch.delenv("REINGEST_STATE_PATH", raising=False)
        importlib.reload(reingest_all)
        assert reingest_all.STATE_FILE.name == "reingest_state.json"


class TestPreflightCollectionCreate:
    """check_qdrant_collection must honor QDRANT_SPARSE_ENABLED at creation."""

    def _create_kwargs(self, monkeypatch, sparse: str) -> dict:
        monkeypatch.setenv("QDRANT_COLLECTION", "rag_docs_test")
        monkeypatch.setenv("QDRANT_SPARSE_ENABLED", sparse)
        client = MagicMock()
        with (
            patch.object(reingest_all, "_qdrant_client", return_value=client),
            patch("providers.qdrant_store.collection_vector_size", return_value=None),
        ):
            reingest_all.check_qdrant_collection(1024)
        return client.create_collection.call_args.kwargs

    def test_sparse_enabled_creates_sparse_vectors_config(self, monkeypatch):
        kwargs = self._create_kwargs(monkeypatch, "true")
        assert "sparse_vectors_config" in kwargs

    def test_sparse_disabled_creates_dense_only(self, monkeypatch):
        kwargs = self._create_kwargs(monkeypatch, "false")
        assert "sparse_vectors_config" not in kwargs
