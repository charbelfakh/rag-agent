from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from providers.transcript_glossary import normalize_transcript_text
from providers.video_acquire import (
    download_video_with_yt_dlp,
    is_fetchable_single_video_url,
    normalize_video_url,
)


def test_glossary_peacot_to_pekat_case_insensitive():
    text = "Try Peacot Vision now at peacotvision.com."
    corrected = normalize_transcript_text(text)
    assert "PEKAT Vision" in corrected
    assert "pekatvision.com" in corrected


def test_glossary_does_not_corrupt_unrelated_words():
    text = "The detector classifies screws accurately."
    assert normalize_transcript_text(text) == text


def test_glossary_macmine_to_mech_mind():
    assert normalize_transcript_text("show me macmine videos") == "show me Mech-Mind videos"


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        ("mac mind camera", "Mech-Mind camera"),
        ("mech mind camera", "Mech-Mind camera"),
        ("mechmind camera", "Mech-Mind camera"),
        ("mech mine robot", "Mech-Mind robot"),
        ("meck mind sensor", "Mech-Mind sensor"),
        ("MAC MINE demo", "Mech-Mind demo"),
    ],
)
def test_glossary_mech_mind_variants(text, expected_fragment):
    assert normalize_transcript_text(text) == expected_fragment


def test_url_filter_accepts_embed_and_watch():
    assert is_fetchable_single_video_url("https://www.youtube.com/watch?v=ABzwi5n1fGA")
    assert is_fetchable_single_video_url("https://www.youtube.com/embed/ABzwi5n1fGA?si=foo")


def test_url_filter_rejects_channel_user_playlist():
    assert not is_fetchable_single_video_url("https://www.youtube.com/channel/UCvbx26TqHNgbIVAY-80XJ1A")
    assert not is_fetchable_single_video_url("https://www.youtube.com/user/foo")
    assert not is_fetchable_single_video_url("https://www.youtube.com/playlist?list=PL123")


def test_normalize_embed_to_watch():
    normalized = normalize_video_url("https://www.youtube.com/embed/ABzwi5n1fGA?si=foo")
    assert normalized == "https://www.youtube.com/watch?v=ABzwi5n1fGA"


def test_download_video_uses_android_client_and_mp4(tmp_path):
    calls = {}

    def fake_run(cmd, check, capture_output, text):
        calls["cmd"] = cmd
        out = Path(cmd[cmd.index("-o") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("providers.video_acquire.subprocess.run", side_effect=fake_run):
        path, error, normalized = download_video_with_yt_dlp(
            vendor="pekat",
            source="detector-and-classifier-cba3f5c6.html",
            video_url="https://www.youtube.com/embed/ABzwi5n1fGA?si=foo",
            data_root=tmp_path,
        )
    assert error is None
    assert path is not None and path.is_file()
    assert normalized == "https://www.youtube.com/watch?v=ABzwi5n1fGA"
    cmd = calls["cmd"]
    assert "--extractor-args" in cmd
    assert "youtube:player_client=android" in cmd
    assert "-f" in cmd and "mp4/best[ext=mp4]/best" in cmd


def test_download_video_gated_failure_is_graceful(tmp_path):
    def fail_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

    with patch("providers.video_acquire.subprocess.run", side_effect=fail_run):
        path, error, _ = download_video_with_yt_dlp(
            vendor="pekat",
            source="detector-and-classifier-cba3f5c6.html",
            video_url="https://www.youtube.com/watch?v=ABzwi5n1fGA",
            data_root=tmp_path,
        )
    assert path is None
    assert error == "yt-dlp-failed:1"

