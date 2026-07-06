"""Tests for yt-dlp VTT transcript ingest path in scripts/ingest/ingest.py."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
_lc_docs = types.ModuleType("langchain_core.documents")
_lc_docs.Document = MagicMock
sys.modules["langchain_core.documents"] = _lc_docs

_lc_splitters = types.ModuleType("langchain_text_splitters")
_lc_splitters.RecursiveCharacterTextSplitter = lambda **kwargs: MagicMock(
    split_text=lambda text: [text]
)
sys.modules["langchain_text_splitters"] = _lc_splitters

_INGEST_MODULE = "scripts.ingest.ingest"

sys.modules.pop(_INGEST_MODULE, None)
sys.modules.pop("ingest", None)
import scripts.ingest.ingest  # noqa: F401 — load real ingest (replaces conftest stub)

from providers.vtt_transcript import (
    build_video_transcript_metadata,
    discover_vtt_files,
    find_info_json_for_vtt,
    group_vtt_cues_into_chunks,
    parse_vtt,
    parse_vtt_cues,
)


@pytest.fixture(autouse=True)
def _reset_provider_singletons():
    from providers.factory import reset_providers

    reset_providers()
    yield
    reset_providers()


SAMPLE_VTT = """WEBVTT

NOTE
This is a comment block

00:00:01.000 --> 00:00:04.000 align:start position:0%
<c>Hello</c> <00:00:01.500>world

00:00:04.000 --> 00:00:07.000
Hello world
Hello world this is a test

STYLE
::cue { color: white; }
"""


def test_parse_vtt_strips_timestamps_tags_and_collapses_duplicates():
    text = parse_vtt_from_string(SAMPLE_VTT)
    assert "00:00" not in text
    assert "<c>" not in text
    assert "NOTE" not in text
    assert "Hello world this is a test" in text
    assert text.count("Hello world") == 1


ROLLING_VTT = """WEBVTT

00:00:00.480 --> 00:00:03.110 align:start position:0%
At MechMind, we put a mind into

00:00:03.110 --> 00:00:03.120 align:start position:0%
At MechMind, we put a mind into


00:00:03.120 --> 00:00:05.230 align:start position:0%
At MechMind, we put a mind into
machines, enabling them to see,

00:00:05.230 --> 00:00:05.240 align:start position:0%
machines, enabling them to see,


00:00:05.240 --> 00:00:07.150 align:start position:0%
machines, enabling them to see,
understand, reason, and act.
"""


def test_parse_vtt_cues_drops_rolling_repeats_across_cues():
    """YouTube auto-captions repeat each line across 2-3 cues; naive cue
    concatenation stutters ('At MechMind, we put a mind into At MechMind, …')."""
    cues = parse_vtt_cues_from_string(ROLLING_VTT)
    joined = " ".join(cue.text for cue in cues)
    assert joined.count("At MechMind, we put a mind into") == 1
    assert joined.count("machines, enabling them to see,") == 1
    assert joined.count("understand, reason, and act.") == 1


def test_group_vtt_cues_chunks_have_no_rolling_repeats():
    cues = parse_vtt_cues_from_string(ROLLING_VTT)
    windows = group_vtt_cues_into_chunks(cues, max_chars=1500, overlap_chars=150)
    assert len(windows) == 1
    assert windows[0].text == (
        "At MechMind, we put a mind into machines, enabling them to see, "
        "understand, reason, and act."
    )


def test_parse_vtt_cues_preserves_start_seconds():
    cues = parse_vtt_cues_from_string(SAMPLE_VTT)
    assert len(cues) >= 2
    assert cues[0].start == 1.0
    assert cues[0].end == 4.0
    assert "Hello world" in cues[0].text
    assert cues[1].start == 4.0
    assert "this is a test" in cues[1].text


def test_group_vtt_cues_into_chunks_carries_timestamp():
    cues = parse_vtt_cues_from_string(SAMPLE_VTT)
    windows = group_vtt_cues_into_chunks(cues, max_chars=200, overlap_chars=0)
    assert windows
    assert windows[0].start_seconds == 1.0
    assert windows[0].end_seconds >= 4.0
    assert "Hello world" in windows[0].text


def test_missing_info_json_is_skipped_without_upsert(tmp_path, caplog):
    vtt = tmp_path / "abc123.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")

    fake_store = MagicMock()
    fake_store.collection = "rag_docs"
    fake_store.client.count.return_value = MagicMock(count=0)

    with patch.object(sys.modules[_INGEST_MODULE], "get_vector_store", return_value=fake_store):
        counts = sys.modules[_INGEST_MODULE].ingest_video_transcripts(str(tmp_path), vendor="pekat")

    assert counts["skipped"] == 1
    assert counts["ingested"] == 0
    fake_store.upsert.assert_not_called()
    assert "missing sibling .info.json" in caplog.text


def test_build_payload_from_info_json_has_video_transcript_fields():
    info = {
        "id": "abc123",
        "title": "Detector Tutorial",
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "channel_id": "UC_PEKAT",
        "upload_date": "20240615",
        "language": "en",
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }
    config = {"youtube_channels": {"UC_PEKAT": "pekat"}, "youtube_channel_names": {}}
    meta = build_video_transcript_metadata(
        info,
        "abc123.en.vtt",
        vendors_config=config,
    )
    assert meta is not None
    assert meta["content_type"] == "video_transcript"
    assert meta["source_type"] == "video"
    assert meta["video_id"] == "abc123"
    assert meta["vendor"] == "pekat"
    assert meta["source"] == "Detector Tutorial"
    assert meta["url"] == "https://www.youtube.com/watch?v=abc123"
    assert meta["doc_version"] == "20240615"
    assert meta["transcript_source"] == "auto"


@pytest.fixture
def isolated_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "ingest_manifest.json"
    monkeypatch.setattr(sys.modules[_INGEST_MODULE], "manifest_path", lambda: manifest)
    return manifest


def test_ingest_video_transcripts_upserts_chunks(tmp_path, isolated_manifest):
    video_dir = tmp_path / "transcripts"
    video_dir.mkdir()
    vtt = video_dir / "abc123.en.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nZivid camera calibration steps.\n",
        encoding="utf-8",
    )
    info = {
        "id": "abc123",
        "title": "Zivid Calibration Tutorial",
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "channel_id": "UC_ZIVID",
        "upload_date": "20250101",
        "language": "en",
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }
    (video_dir / "abc123.info.json").write_text(json.dumps(info), encoding="utf-8")

    fake_store = MagicMock()
    fake_store.collection = "rag_docs"
    fake_store.client.count.return_value = MagicMock(count=0)

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1, 0.2]]

    with (
        patch.object(sys.modules[_INGEST_MODULE], "get_vector_store", return_value=fake_store),
        patch.object(sys.modules[_INGEST_MODULE], "get_embedder", return_value=fake_embedder),
        patch("scripts.ingest.ingest.tqdm.write"),
        patch("scripts.ingest.ingest.tqdm", side_effect=lambda *args, **kwargs: MagicMock()),
    ):
        counts = sys.modules[_INGEST_MODULE].ingest_video_transcripts(str(video_dir), vendor="zivid")

    assert counts["videos"] == 1
    assert counts["ingested"] >= 1
    fake_store.upsert.assert_called()
    payloads = fake_store.upsert.call_args[0][2]
    assert payloads[0]["content_type"] == "video_transcript"
    assert payloads[0]["video_id"] == "abc123"
    assert payloads[0]["source_type"] == "video"
    assert payloads[0]["vendor"] == "zivid"
    assert payloads[0]["start_seconds"] == 0.0
    assert payloads[0]["end_seconds"] == 2.0


def test_ingest_video_transcripts_writes_manifest_row(tmp_path, isolated_manifest):
    video_dir = tmp_path / "transcripts"
    video_dir.mkdir()
    title = "Zivid Calibration Tutorial"
    vtt = video_dir / "abc123.en.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nZivid camera calibration steps.\n",
        encoding="utf-8",
    )
    info = {
        "id": "abc123",
        "title": title,
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "channel_id": "UC_ZIVID",
        "upload_date": "20250101",
        "language": "en",
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }
    (video_dir / "abc123.info.json").write_text(json.dumps(info), encoding="utf-8")

    fake_store = MagicMock()
    fake_store.collection = "rag_docs"
    fake_store.client.count.return_value = MagicMock(count=0)
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1, 0.2]]

    with (
        patch.object(sys.modules[_INGEST_MODULE], "get_vector_store", return_value=fake_store),
        patch.object(sys.modules[_INGEST_MODULE], "get_embedder", return_value=fake_embedder),
        patch("scripts.ingest.ingest.tqdm.write"),
        patch("scripts.ingest.ingest.tqdm", side_effect=lambda *args, **kwargs: MagicMock()),
    ):
        sys.modules[_INGEST_MODULE].ingest_video_transcripts(str(video_dir), vendor="zivid")

    data = json.loads(isolated_manifest.read_text(encoding="utf-8"))
    assert title in data
    row = data[title]
    assert row["vendor"] == "zivid"
    assert row["chunk_count"] >= 1
    assert row["content_type"] == "video_transcript"
    assert row["source_type"] == "video"
    assert row["video_id"] == "abc123"
    assert row["sha256"]
    assert row["ingested_at"]


def test_ingest_video_transcripts_reingest_updates_manifest_in_place(
    tmp_path, isolated_manifest
):
    video_dir = tmp_path / "transcripts"
    video_dir.mkdir()
    title = "Detector Tutorial"
    vtt = video_dir / "abc123.en.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nDetector training overview.\n",
        encoding="utf-8",
    )
    info = {
        "id": "abc123",
        "title": title,
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "channel_id": "UC_PEKAT",
        "upload_date": "20240615",
        "language": "en",
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }
    (video_dir / "abc123.info.json").write_text(json.dumps(info), encoding="utf-8")

    fake_store = MagicMock()
    fake_store.collection = "rag_docs"
    fake_store.client.count.return_value = MagicMock(count=0)
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1, 0.2]]

    ingest = sys.modules[_INGEST_MODULE]
    with (
        patch.object(ingest, "get_vector_store", return_value=fake_store),
        patch.object(ingest, "get_embedder", return_value=fake_embedder),
        patch("scripts.ingest.ingest.tqdm.write"),
        patch("scripts.ingest.ingest.tqdm", side_effect=lambda *args, **kwargs: MagicMock()),
        patch.object(ingest, "_ingest_video_transcript_text", return_value=2),
    ):
        ingest_mod = sys.modules[_INGEST_MODULE]
        ingest_mod.ingest_video_transcripts(str(video_dir), vendor="pekat", force=True)
    first = json.loads(isolated_manifest.read_text(encoding="utf-8"))
    first_ingested_at = first[title]["ingested_at"]

    with (
        patch.object(ingest, "get_vector_store", return_value=fake_store),
        patch.object(ingest, "get_embedder", return_value=fake_embedder),
        patch("scripts.ingest.ingest.tqdm.write"),
        patch("scripts.ingest.ingest.tqdm", side_effect=lambda *args, **kwargs: MagicMock()),
        patch.object(ingest, "_purge_video_id_chunks", return_value=0),
        patch.object(ingest, "_purge_source_chunks", return_value=0),
        patch.object(ingest, "_ingest_video_transcript_text", return_value=3),
    ):
        ingest_mod = sys.modules[_INGEST_MODULE]
        ingest_mod.ingest_video_transcripts(str(video_dir), vendor="pekat", force=True)

    second = json.loads(isolated_manifest.read_text(encoding="utf-8"))
    assert list(second.keys()) == [title]
    assert second[title]["chunk_count"] == 3
    assert second[title]["ingested_at"] >= first_ingested_at


def test_glossary_fixes_new_observed_asr_errors():
    from providers.transcript_glossary import normalize_transcript_text

    fixed = normalize_transcript_text(
        "With cuttingedge embodied AI, from dual armed systems in homoids."
    )
    assert "cutting-edge" in fixed
    assert "humanoids" in fixed
    assert "homoids" not in fixed


def test_ingest_video_transcripts_applies_glossary(tmp_path, isolated_manifest):
    """VTT ingest must run the same ASR-glossary pass as the Whisper path."""
    video_dir = tmp_path / "transcripts"
    video_dir.mkdir()
    vtt = video_dir / "abc123.en.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n"
        "mac mine uses cuttingedge AI for homoids.\n",
        encoding="utf-8",
    )
    info = {
        "id": "abc123",
        "title": "Mech-Mind Overview",
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "channel_id": "UC_MM",
        "upload_date": "20250101",
        "language": "en",
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }
    (video_dir / "abc123.info.json").write_text(json.dumps(info), encoding="utf-8")

    fake_store = MagicMock()
    fake_store.collection = "rag_docs"
    fake_store.client.count.return_value = MagicMock(count=0)
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1, 0.2]]

    with (
        patch.object(sys.modules[_INGEST_MODULE], "get_vector_store", return_value=fake_store),
        patch.object(sys.modules[_INGEST_MODULE], "get_embedder", return_value=fake_embedder),
        patch("scripts.ingest.ingest.tqdm.write"),
        patch("scripts.ingest.ingest.tqdm", side_effect=lambda *args, **kwargs: MagicMock()),
    ):
        counts = sys.modules[_INGEST_MODULE].ingest_video_transcripts(
            str(video_dir), vendor="mechmind"
        )

    assert counts["ingested"] >= 1
    payloads = fake_store.upsert.call_args[0][2]
    text = payloads[0]["text"]
    assert "Mech-Mind" in text
    assert "cutting-edge" in text
    assert "humanoids" in text
    assert "homoids" not in text


def test_find_info_json_for_bracketed_vtt_filename(tmp_path):
    vtt = tmp_path / "My Title [abc123].en.vtt"
    vtt.write_text("WEBVTT\n", encoding="utf-8")
    (tmp_path / "abc123.info.json").write_text(
        json.dumps({"id": "abc123", "title": "My Title"}),
        encoding="utf-8",
    )
    found = find_info_json_for_vtt(vtt)
    assert found is not None
    assert found.name == "abc123.info.json"


def test_discover_vtt_files_prefers_en_over_en_orig(tmp_path):
    (tmp_path / "vid1.en.vtt").write_text("", encoding="utf-8")
    (tmp_path / "vid1.en-orig.vtt").write_text("", encoding="utf-8")
    found = discover_vtt_files(tmp_path)
    assert [p.name for p in found] == ["vid1.en.vtt"]


def test_discover_vtt_files_falls_back_to_en_orig_when_only_option(tmp_path):
    (tmp_path / "vid2.en-orig.vtt").write_text("", encoding="utf-8")
    found = discover_vtt_files(tmp_path)
    assert [p.name for p in found] == ["vid2.en-orig.vtt"]


def test_discover_vtt_files_rejects_auto_translation_in_favor_of_en_orig(tmp_path):
    (tmp_path / "vid3.en-ja.vtt").write_text("", encoding="utf-8")
    (tmp_path / "vid3.en-orig.vtt").write_text("", encoding="utf-8")
    found = discover_vtt_files(tmp_path)
    assert [p.name for p in found] == ["vid3.en-orig.vtt"]


def test_discover_vtt_files_yields_nothing_when_only_translated_tracks(tmp_path):
    (tmp_path / "vid4.en-ja.vtt").write_text("", encoding="utf-8")
    (tmp_path / "vid4.en-ko.vtt").write_text("", encoding="utf-8")
    found = discover_vtt_files(tmp_path)
    assert found == []


def test_discover_vtt_files_rejects_en_en_translation(tmp_path):
    (tmp_path / "vid5.en-en.vtt").write_text("", encoding="utf-8")
    (tmp_path / "vid5.en.vtt").write_text("", encoding="utf-8")
    found = discover_vtt_files(tmp_path)
    assert [p.name for p in found] == ["vid5.en.vtt"]


def parse_vtt_from_string(content: str) -> str:
    path = Path("_pytest_vtt_sample.vtt")
    path.write_text(content, encoding="utf-8")
    try:
        return parse_vtt(str(path))
    finally:
        path.unlink(missing_ok=True)


def parse_vtt_cues_from_string(content: str):
    path = Path("_pytest_vtt_sample.vtt")
    path.write_text(content, encoding="utf-8")
    try:
        return parse_vtt_cues(str(path))
    finally:
        path.unlink(missing_ok=True)
