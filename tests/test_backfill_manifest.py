"""Tests for scripts/backfill_manifest.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.modules.pop("scripts.ingest.ingest", None)
sys.modules.pop("ingest", None)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.ingest.ingest as ingest

from scripts.backfill_manifest import (
    SourceAggregate,
    aggregate_sources_from_scroll,
    apply_backfill_rows,
    is_backfill_candidate,
    plan_backfill_rows,
    run_backfill,
)


def _payload(
    *,
    source: str,
    vendor: str = "pekat",
    content_type: str = "text",
    source_type: str | None = None,
    video_id: str | None = None,
):
    payload = {
        "source": source,
        "vendor": vendor,
        "content_type": content_type,
    }
    if source_type:
        payload["source_type"] = source_type
    if video_id:
        payload["video_id"] = video_id
    return payload


def _scroll_batches(*batches):
    """Build scroll side_effect from (records, next_offset) batches."""
    calls = list(batches)

    def _scroll(**kwargs):
        if not calls:
            return [], None
        records, offset = calls.pop(0)
        return records, offset

    return _scroll


def test_is_backfill_candidate_video_and_html_url_only():
    video = SourceAggregate(
        source="Detector Tutorial",
        content_type="video_transcript",
        source_type="video",
    )
    html_url = SourceAggregate(
        source="https://example.com/article/setup.html",
        content_type="text",
        source_type="html",
    )
    pdf = SourceAggregate(
        source="manual.pdf",
        content_type="text",
        source_type="pdf",
    )

    assert is_backfill_candidate(video, "all")
    assert is_backfill_candidate(html_url, "all")
    assert not is_backfill_candidate(pdf, "all")
    assert is_backfill_candidate(video, "video_transcript")
    assert not is_backfill_candidate(html_url, "video_transcript")
    assert is_backfill_candidate(html_url, "html")
    assert not is_backfill_candidate(video, "html")


def test_aggregate_sources_from_scroll_counts_per_source():
    client = MagicMock()
    client.scroll.side_effect = _scroll_batches(
        (
            [
                SimpleNamespace(payload=_payload(source="Detector Tutorial", content_type="video_transcript", source_type="video", video_id="abc123")),
                SimpleNamespace(payload=_payload(source="Detector Tutorial", content_type="video_transcript", source_type="video", video_id="abc123")),
                SimpleNamespace(payload=_payload(source="https://example.com/a", source_type="html")),
            ],
            None,
        )
    )

    aggregated = aggregate_sources_from_scroll(client, "rag_docs")
    assert aggregated["Detector Tutorial"].chunk_count == 2
    assert aggregated["Detector Tutorial"].video_id == "abc123"
    assert aggregated["https://example.com/a"].chunk_count == 1


def test_plan_backfill_skips_pdf_and_existing_manifest_rows():
    aggregated = {
        "Detector Tutorial": SourceAggregate(
            source="Detector Tutorial",
            chunk_count=3,
            vendor="pekat",
            content_type="video_transcript",
            source_type="video",
            video_id="abc123",
        ),
        "https://support.example.com/page": SourceAggregate(
            source="https://support.example.com/page",
            chunk_count=5,
            vendor="pekat",
            content_type="text",
            source_type="html",
        ),
        "manual.pdf": SourceAggregate(
            source="manual.pdf",
            chunk_count=10,
            vendor="lmi",
            content_type="text",
            source_type="pdf",
        ),
    }
    manifest = {"Detector Tutorial": {"chunk_count": 3}}

    rows, skipped_existing, skipped_other = plan_backfill_rows(
        aggregated,
        manifest,
        scope="all",
        force=False,
    )

    assert [row.source for row in rows] == ["https://support.example.com/page"]
    assert skipped_existing == 1
    assert skipped_other == 1


def test_plan_backfill_force_includes_existing(tmp_path):
    aggregated = {
        "Detector Tutorial": SourceAggregate(
            source="Detector Tutorial",
            chunk_count=3,
            vendor="pekat",
            content_type="video_transcript",
            source_type="video",
        ),
    }
    rows, skipped_existing, _ = plan_backfill_rows(
        aggregated,
        {"Detector Tutorial": {}},
        scope="all",
        force=True,
    )
    assert len(rows) == 1
    assert skipped_existing == 0


def test_apply_writes_manifest_rows(tmp_path, monkeypatch):
    manifest_path = tmp_path / "ingest_manifest.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ingest, "manifest_path", lambda: manifest_path)

    rows = [
        SourceAggregate(
            source="Detector Tutorial",
            chunk_count=3,
            vendor="pekat",
            content_type="video_transcript",
            source_type="video",
            video_id="abc123",
        ),
        SourceAggregate(
            source="https://support.example.com/page",
            chunk_count=5,
            vendor="pekat",
            content_type="text",
            source_type="html",
        ),
    ]

    written = apply_backfill_rows(rows)
    assert written == 2

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["Detector Tutorial"]["chunk_count"] == 3
    assert data["Detector Tutorial"]["content_type"] == "video_transcript"
    assert data["Detector Tutorial"]["video_id"] == "abc123"
    assert data["Detector Tutorial"]["sha256"] == "backfill:video_transcript"
    assert data["https://support.example.com/page"]["chunk_count"] == 5
    assert data["https://support.example.com/page"]["sha256"] == "backfill:html"
    assert "manual.pdf" not in data


def test_run_backfill_dry_run_writes_nothing(tmp_path, monkeypatch):
    manifest_path = tmp_path / "ingest_manifest.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ingest, "manifest_path", lambda: manifest_path)

    client = MagicMock()
    client.scroll.side_effect = _scroll_batches(
        (
            [
                SimpleNamespace(
                    payload=_payload(
                        source="Detector Tutorial",
                        content_type="video_transcript",
                        source_type="video",
                        video_id="abc123",
                    )
                ),
            ],
            None,
        )
    )

    stats = run_backfill(
        client=client,
        collection="rag_docs",
        scope="all",
        force=False,
        apply=False,
    )

    assert stats["selected"] == 1
    assert stats["written"] == 0
    assert not manifest_path.exists()


def test_run_backfill_apply_writes_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "ingest_manifest.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ingest, "manifest_path", lambda: manifest_path)

    client = MagicMock()
    client.scroll.side_effect = _scroll_batches(
        (
            [
                SimpleNamespace(
                    payload=_payload(
                        source="https://example.com/docs/page",
                        source_type="html",
                    )
                ),
            ],
            None,
        )
    )

    stats = run_backfill(
        client=client,
        collection="rag_docs",
        scope="all",
        force=False,
        apply=True,
    )

    assert stats["written"] == 1
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "https://example.com/docs/page" in data
