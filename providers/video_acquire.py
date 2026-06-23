"""Video URL acquisition helpers (yt-dlp download + URL filtering/normalization)."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def normalize_video_url(video_url: str) -> str:
    """Canonicalize YouTube embed/short links to ``watch?v=`` form."""
    raw = (video_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path = parsed.path or ""

    if "youtube.com" in host and path.startswith("/embed/"):
        video_id = path.split("/embed/", 1)[1].split("/", 1)[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    if "youtu.be" in host:
        video_id = path.strip("/").split("/", 1)[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return raw


def is_fetchable_single_video_url(video_url: str) -> bool:
    """Return whether the URL targets one video (not a channel or playlist)."""
    normalized = normalize_video_url(video_url)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host = parsed.netloc.lower()
    path = (parsed.path or "").lower()

    if any(marker in host for marker in ("youtube.com", "youtu.be")):
        if "/channel/" in path or "/user/" in path:
            return False
        query = parse_qs(parsed.query)
        if "list" in query and "v" not in query:
            return False
        if path == "/playlist":
            return False
        if path == "/watch" and query.get("v"):
            return True
        if "youtu.be" in host and path.strip("/"):
            return True
        return False

    # Keep non-YouTube URLs permissive; fetchability is decided by yt-dlp runtime.
    return True


def make_video_id(source: str, video_url: str | None = None) -> str:
    """Return a stable 20-char id from source path and optional URL."""
    key = f"{source}|{video_url or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def download_video_with_yt_dlp(
    *,
    vendor: str,
    source: str,
    video_url: str,
    data_root: Path,
) -> tuple[Path | None, str | None, str]:
    """Download a single video URL to data/<vendor>/videos/<video-id>/source.mp4.

    Returns (path, error_reason, normalized_url).
    """
    normalized_url = normalize_video_url(video_url)
    if not is_fetchable_single_video_url(normalized_url):
        return None, "non-fetchable-url", normalized_url

    video_id = make_video_id(source, normalized_url)
    out_dir = data_root / vendor.lower() / "videos" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "source.mp4"

    cmd = [
        "yt-dlp",
        "--extractor-args",
        "youtube:player_client=android",
        "-f",
        "mp4/best[ext=mp4]/best",
        "-o",
        str(out_path),
        normalized_url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return None, f"yt-dlp-failed:{exc.returncode}", normalized_url
    if not out_path.is_file():
        return None, "download-missing-output", normalized_url
    return out_path, None, normalized_url

