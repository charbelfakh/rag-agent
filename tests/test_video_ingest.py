"""Tests for video transcript ingest (Stage 1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.ingest.ingest_video import (
    apply_glossary_to_segments,
    build_transcript_payload,
    ingest_local_mp4,
    preflight_ffmpeg,
    upsert_transcript_windows,
)
from providers.video_transcribe import (
    TranscriptSegment,
    build_transcript_index_text,
    group_segments_into_windows,
    reset_whisper_model,
    transcribe,
    transcribe_plain_text,
)
from providers.rag_pipeline import collect_citations


class TestTranscribe:
    def test_transcribe_returns_timestamped_segments(self):
        reset_whisper_model()
        fake_segment = MagicMock(start=1.5, end=4.2, text=" Gocator sensor setup ")
        fake_model = MagicMock()
        fake_model.transcribe.return_value = ([fake_segment], MagicMock())

        with patch("providers.video_transcribe._load_whisper_model", return_value=fake_model):
            segments = transcribe("/tmp/_audio.wav")

        assert len(segments) == 1
        assert segments[0].start == 1.5
        assert segments[0].end == 4.2
        assert segments[0].text == "Gocator sensor setup"

    def test_transcribe_plain_text_joins_segments(self):
        reset_whisper_model()
        fake_segments = [
            MagicMock(start=0.0, end=1.0, text=" Hello "),
            MagicMock(start=1.0, end=2.0, text=" world "),
        ]
        fake_model = MagicMock()
        fake_model.transcribe.return_value = (fake_segments, MagicMock())

        with patch("providers.video_transcribe._load_whisper_model", return_value=fake_model):
            text = transcribe_plain_text("/tmp/query.webm")

        assert text == "Hello world"


class TestWindowGrouping:
    def test_group_segments_respects_chunk_seconds(self):
        segments = [
            TranscriptSegment(0.0, 3.0, "Intro to Profinet."),
            TranscriptSegment(3.0, 6.0, "Configure the anomaly detector."),
            TranscriptSegment(6.0, 12.0, "Gocator alignment steps."),
            TranscriptSegment(12.0, 15.0, "Save the model."),
        ]
        windows = group_segments_into_windows(segments, 10.0)
        assert len(windows) == 2
        assert windows[0].start == 0.0
        assert windows[0].end == 6.0
        assert "Profinet" in windows[0].text
        assert windows[1].start == 6.0
        assert windows[1].end == 15.0
        assert "Gocator" in windows[1].text

    def test_empty_segments_returns_empty(self):
        assert group_segments_into_windows([], 45.0) == []

    def test_apply_glossary_to_segments(self):
        segments = [TranscriptSegment(0.0, 1.0, "Try Peacot Vision now.")]
        corrected = apply_glossary_to_segments(segments)
        assert corrected[0].text == "Try PEKAT Vision now."


class TestTranscriptPayload:
    def test_build_transcript_payload_schema(self):
        window = TranscriptSegment(12.5, 40.0, "Connect Profinet to the PLC.")
        payload = build_transcript_payload(
            source="tutorial.mp4",
            vendor="pekat",
            window=window,
            video_path="data/pekat/videos/abc/source.mp4",
            chunk_index=0,
            product=None,
            doc_type="tutorial",
            section="Setup",
            url=None,
            video_url=None,
            duration_seconds=120.0,
            ingested_at="2026-06-18T12:00:00Z",
        )
        assert payload["content_type"] == "video_transcript"
        assert payload["start_seconds"] == 12.5
        assert payload["end_seconds"] == 40.0
        assert payload["video_path"] == "data/pekat/videos/abc/source.mp4"
        assert payload["text"] == "Connect Profinet to the PLC."
        assert payload["page"] is None
        assert payload["schema_version"] == 2

    def test_index_text_format(self):
        text = build_transcript_index_text("demo.mp4", 12.5, "Gocator calibration.")
        assert text == "[Video transcript, demo.mp4, t=12.5s] Gocator calibration."


class TestUpsertTranscriptWindows:
    def test_upsert_embeds_index_text_and_stores_raw_text(self):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2]]
        store = MagicMock()
        windows = [TranscriptSegment(0.0, 5.0, "Profinet wiring overview.")]

        upserted, skipped = upsert_transcript_windows(
            windows=windows,
            source="demo.mp4",
            vendor="lmi",
            video_path="data/lmi/videos/id/source.mp4",
            product=None,
            doc_type="tutorial",
            section=None,
            url=None,
            video_url=None,
            duration_seconds=30.0,
            embedder=embedder,
            store=store,
            ingested_at="2026-06-18T12:00:00Z",
        )

        assert upserted == 1
        assert skipped == 0
        embedder.embed.assert_called_once_with(
            ["[Video transcript, demo.mp4, t=0.0s] Profinet wiring overview."]
        )
        payload = store.upsert.call_args[0][2][0]
        assert payload["text"] == "Profinet wiring overview."
        assert payload["content_type"] == "video_transcript"
        assert payload["start_seconds"] == 0.0

    def test_upsert_skips_none_embeddings(self):
        embedder = MagicMock()
        embedder.embed.return_value = [None]
        store = MagicMock()
        windows = [TranscriptSegment(1.0, 2.0, "Skipped chunk.")]

        upserted, skipped = upsert_transcript_windows(
            windows=windows,
            source="demo.mp4",
            vendor="lmi",
            video_path="data/lmi/videos/id/source.mp4",
            product=None,
            doc_type="other",
            section=None,
            url=None,
            video_url=None,
            duration_seconds=10.0,
            embedder=embedder,
            store=store,
        )

        assert upserted == 0
        assert skipped == 1
        store.upsert.assert_not_called()


class TestIngestLocalMp4:
    def test_ingest_pipeline_produces_video_transcript_points(self, tmp_path, monkeypatch):
        mp4 = tmp_path / "demo.mp4"
        mp4.write_bytes(b"fake-mp4")
        monkeypatch.setenv("VIDEO_DATA_DIR", str(tmp_path / "data"))

        segments = [
            TranscriptSegment(0.0, 20.0, "Gocator sensor overview."),
            TranscriptSegment(50.0, 55.0, "Profinet network setup."),
        ]
        embedder = MagicMock()
        embedder.embed.return_value = [[0.5], [0.6]]
        store = MagicMock()

        with (
            patch("scripts.ingest.ingest_video.run_preflight"),
            patch(
                "scripts.ingest.ingest_video.stage_source_mp4",
                side_effect=lambda src, dest: dest,
            ),
            patch("scripts.ingest.ingest_video.probe_duration_seconds", return_value=60.0),
            patch("scripts.ingest.ingest_video.extract_audio_wav"),
            patch("scripts.ingest.ingest_video.transcribe", return_value=segments),
        ):
            result = ingest_local_mp4(
                mp4,
                vendor="pekat",
                source="demo.mp4",
                embedder=embedder,
                store=store,
            )

        assert result["upserted"] == 2
        store.ensure_payload_indexes.assert_called_once()
        payloads = store.upsert.call_args[0][2]
        assert all(p["content_type"] == "video_transcript" for p in payloads)
        assert payloads[0]["start_seconds"] == 0.0
        assert payloads[1]["start_seconds"] == 50.0
        assert payloads[0]["video_path"].endswith("source.mp4")


class TestCitationDedupe:
    def test_video_chunks_with_different_timestamps_both_survive(self):
        chunks = [
            {
                "source": "tutorial.mp4",
                "section": "Setup",
                "content_type": "video_transcript",
                "start_seconds": 0.0,
                "vendor": "pekat",
            },
            {
                "source": "tutorial.mp4",
                "section": "Setup",
                "content_type": "video_transcript",
                "start_seconds": 45.0,
                "vendor": "pekat",
            },
        ]
        cites = collect_citations(chunks)
        assert len(cites) == 2

    def test_text_chunk_dedupe_unchanged(self):
        chunks = [
            {
                "source": "manual.pdf",
                "page": 2,
                "section": "Triggers",
                "content_type": "text",
                "vendor": "pekat",
            },
            {
                "source": "manual.pdf",
                "page": 2,
                "section": "Triggers",
                "content_type": "text",
                "vendor": "pekat",
            },
        ]
        cites = collect_citations(chunks)
        assert len(cites) == 1
        assert cites[0]["page"] == 2
        assert cites[0]["section"] == "Triggers"


class TestPreflight:
    def test_ffmpeg_missing_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr("scripts.ingest.ingest_video.shutil.which", lambda name: None)
        with pytest.raises(RuntimeError, match="ffmpeg"):
            preflight_ffmpeg()

    def test_ffmpeg_present_does_not_raise(self, monkeypatch):
        monkeypatch.setattr("scripts.ingest.ingest_video.shutil.which", lambda name: "/usr/bin/" + name)
        preflight_ffmpeg()

    def test_whisper_import_missing_raises_clear_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from scripts.ingest.ingest_video import preflight_whisper

        with pytest.raises(RuntimeError, match="faster-whisper"):
            preflight_whisper()
