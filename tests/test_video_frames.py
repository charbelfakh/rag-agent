from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from providers.video_frames import cap_frames_evenly
from scripts.ingest.video_frame_worker import (
    build_frame_index_text,
    build_video_frame_payload,
    process_video_frames,
)


def test_cap_frames_evenly_keeps_distributed_subset():
    frames = [{"start_seconds": float(i), "path": Path(f"f{i}.png")} for i in range(10)]
    kept = cap_frames_evenly(frames, 4)
    kept_secs = [f["start_seconds"] for f in kept]
    assert len(kept_secs) == 4
    assert kept_secs[0] == 0.0
    assert kept_secs[-1] == 9.0
    assert kept_secs == sorted(kept_secs)


def test_build_frame_payload_contains_video_frame_fields():
    payload = build_video_frame_payload(
        source="detector-and-classifier-cba3f5c6.html",
        vendor="pekat",
        doc_type="tutorial",
        section="Detector tutorial",
        url="https://example.com",
        product=None,
        product_version=None,
        video_path="data/pekat/videos/id/source.mp4",
        frame_path="data/pekat/videos/_frame_cache/id/0123.450.png",
        start_seconds=123.45,
        caption="Class manager panel with confidence threshold controls.",
        video_url="https://www.youtube.com/watch?v=ABzwi5n1fGA",
        ingested_at="2026-06-18T12:00:00Z",
    )
    assert payload["content_type"] == "video_frame"
    assert payload["start_seconds"] == 123.45
    assert payload["end_seconds"] == 123.45
    assert payload["frame_path"].endswith("0123.450.png")
    assert payload["image_path"] == payload["frame_path"]
    assert payload["video_path"].endswith("source.mp4")


def test_build_frame_index_text_format():
    text = build_frame_index_text(
        "detector-and-classifier-cba3f5c6.html",
        123.45,
        "Detector configuration panel with threshold slider.",
    )
    assert (
        text
        == "[Video frame, detector-and-classifier-cba3f5c6.html, t=123.5s] "
        "Detector configuration panel with threshold slider."
    )


def test_process_video_frames_upserts_expected_points(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake")
    frame_a = tmp_path / "0000.000.png"
    frame_b = tmp_path / "0123.450.png"
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")

    fake_store = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed.side_effect = [[[0.1]], [[0.2]]]
    fake_vision = MagicMock()
    fake_vision.describe_image.side_effect = [
        "Detector module panel with class manager open.",
        "Threshold slider near 75 percent on evaluation tab.",
    ]

    monkeypatch.setenv("VIDEO_FRAME_MAX_PER_VIDEO", "10")
    with (
        patch("scripts.ingest.video_frame_worker.preflight_ffmpeg"),
        patch(
            "scripts.ingest.video_frame_worker.extract_scene_frames",
            return_value=[
                {"path": frame_a, "start_seconds": 0.0},
                {"path": frame_b, "start_seconds": 123.45},
            ],
        ),
        patch(
            "scripts.ingest.video_frame_worker.fetch_source_metadata",
            return_value={
                "vendor": "pekat",
                "product": None,
                "product_version": None,
                "doc_type": "tutorial",
            },
        ),
    ):
        result = process_video_frames(
            source="detector-and-classifier-cba3f5c6.html",
            vendor="pekat",
            video_path=video,
            embedder=fake_embedder,
            store=fake_store,
            vision=fake_vision,
            doc_type="tutorial",
            section="Detector tutorial",
        )

    assert result["scenes_detected"] == 2
    assert result["scenes_kept"] == 2
    assert result["captioned"] == 2
    payloads = fake_store.upsert.call_args_list
    assert len(payloads) == 2
    first_payload = payloads[0].args[2][0]
    assert first_payload["content_type"] == "video_frame"
    assert first_payload["start_seconds"] == 0.0
    assert first_payload["image_path"] == first_payload["frame_path"]

