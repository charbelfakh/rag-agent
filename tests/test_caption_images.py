"""Tests for providers.caption_images."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from providers.caption_images import (
    is_skipped_caption_entry,
    prepare_caption_queue_entries,
    prepare_vision_image_bytes,
    should_skip_image_src,
)


def test_should_skip_ico_and_apiary():
    assert should_skip_image_src("https://static.apiary.io/assets/3aqvBs0H.ico")
    assert should_skip_image_src("https://chat.google.com/u/0/api/get_attachment_url?x=1")
    assert should_skip_image_src("images/foo/p1_0.png") is None


def test_prepare_vision_image_bytes_converts_webp_to_png():
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(buf, format="WEBP")
    out = prepare_vision_image_bytes(buf.getvalue())
    assert out[:4] == b"\x89PNG"


def test_prepare_caption_queue_entries_filters_and_materializes(tmp_path, monkeypatch):
    vendor = "pekat"
    vendor_root = tmp_path / "data" / vendor / "images" / "_ingest_cache"
    vendor_root.mkdir(parents=True)

    buf = BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="WEBP")
    webp = buf.getvalue()

    class FakeResponse:
        content = webp

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "providers.caption_images.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )
    monkeypatch.chdir(tmp_path)

    entries = [
        {"source": "a.html", "image_src": "https://example.com/x.webp", "url": "https://example.com/a.html"},
        {"source": "a.html", "image_src": "https://static.apiary.io/assets/x.ico"},
    ]
    prepared = prepare_caption_queue_entries(entries, vendor)
    assert len(prepared) == 1
    assert prepared[0]["image_src"].startswith("images/_ingest_cache/")
    assert (Path("data") / vendor / prepared[0]["image_src"]).is_file()
    assert is_skipped_caption_entry({"image_src": "https://static.apiary.io/x.ico"})
