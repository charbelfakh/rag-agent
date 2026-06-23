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
