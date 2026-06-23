"""Tests for caption_worker queue helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.ingest.caption_worker import (
    build_payload,
    fetch_source_metadata,
    is_pending,
    reconcile_missing_captions,
)
from providers.caption_images import should_skip_image_src


def test_is_pending_respects_skipped():
    assert is_pending({"captioned": "skipped"}, retry_failed=False) is False
    assert is_pending({"captioned": "failed"}, retry_failed=True) is True


def test_should_skip_ico():
    assert should_skip_image_src("/favicon.ico")


def test_reconcile_missing_captions_resets_absent_points(tmp_path, monkeypatch):
    queue = tmp_path / "pending_captions.json"
    queue.write_text(
        """[
  {"source": "doc.pdf", "image_src": "images/a.png", "captioned": true},
  {"source": "doc.pdf", "image_src": "images/b.png", "captioned": true}
]""",
        encoding="utf-8",
    )

    from scripts.ingest.caption_worker import make_point_id

    present_id = make_point_id("doc.pdf", "images/a.png")

    class FakeClient:
        def scroll(self, **kwargs):
            record = type("R", (), {"id": present_id})()
            return [record], None

    class FakeStore:
        collection = "rag_docs"
        client = FakeClient()

    monkeypatch.setattr(
        "scripts.ingest.caption_worker._collection_for_vendor",
        lambda store, vendor: "rag_docs",
    )
    monkeypatch.setattr(
        "scripts.ingest.caption_worker._qdrant_client_from_store",
        lambda store: FakeStore.client,
    )

    reset = reconcile_missing_captions("lmi", queue, FakeStore())
    assert reset == 1


def test_fetch_source_metadata_includes_extended_fields(monkeypatch):
    payload = {
        "vendor": "mechmind",
        "product": "mech-eye 3d camera",
        "product_version": "2.5.4",
        "doc_type": "manual",
        "category": "mechmind",
        "device_family": "Mech-Eye",
        "device_model": None,
        "doc_version": "2.5.4",
        "language": "en",
        "source_type": "pdf",
        "video_path": "data/mechmind/videos/abc/source.mp4",
        "video_url": "https://example.com/video",
    }

    class FakeRecord:
        def __init__(self, payload):
            self.payload = payload

    class FakeClient:
        def scroll(self, **kwargs):
            return [FakeRecord(payload)], None

    class FakeStore:
        collection = "rag_docs"
        client = FakeClient()

    monkeypatch.setattr(
        "scripts.ingest.caption_worker._collection_for_vendor",
        lambda store, vendor: "rag_docs",
    )
    monkeypatch.setattr(
        "scripts.ingest.caption_worker._qdrant_client_from_store",
        lambda store: FakeStore.client,
    )

    meta = fetch_source_metadata(
        FakeStore(),
        source="eye-3d-camera-v2.5.4-en.pdf",
        vendor="mechmind",
        cache={},
    )
    assert meta["category"] == "mechmind"
    assert meta["device_family"] == "Mech-Eye"
    assert meta["doc_version"] == "2.5.4"
    assert meta["language"] == "en"
    assert meta["source_type"] == "pdf"
    assert meta["video_path"].endswith("source.mp4")
    assert meta["video_url"] == "https://example.com/video"


def test_build_payload_copies_extended_metadata():
    meta = {
        "vendor": "mechmind",
        "product": "mech-eye 3d camera",
        "product_version": "2.5.4",
        "doc_type": "manual",
        "category": "mechmind",
        "device_family": "Mech-Eye",
        "doc_version": "2.5.4",
        "language": "en",
        "source_type": "pdf",
        "video_path": "data/mechmind/videos/abc/source.mp4",
    }
    entry = {
        "source": "eye-3d-camera-v2.5.4-en.pdf",
        "page": 10,
        "url": None,
        "vendor": "mechmind",
        "section": "Working distance",
    }
    payload = build_payload(
        entry,
        caption="Camera setup screen with distance settings.",
        meta=meta,
        image_path=Path("data/mechmind/images/x.png"),
        ingested_at="2026-06-16T12:00:00Z",
    )
    assert payload["content_type"] == "image_caption"
    assert payload["category"] == "mechmind"
    assert payload["device_family"] == "Mech-Eye"
    assert payload["doc_version"] == "2.5.4"
    assert payload["language"] == "en"
    assert payload["source_type"] == "pdf"
    assert payload["video_path"].endswith("source.mp4")
