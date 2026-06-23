"""Safe path resolution and content-type detection for staged video files."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_VENDOR_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$", re.I)


def video_data_dir_name() -> str:
    return os.getenv("VIDEO_DATA_DIR", "data").strip("/\\") or "data"


def detect_video_content_type(path: Path, header: bytes | None = None) -> str:
    """Detect video MIME type from extension and/or magic bytes."""
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".webm":
        return "video/webm"
    if suffix in (".ogv", ".ogg"):
        return "video/ogg"

    sample = header if header is not None else path.read_bytes()[:16]
    if len(sample) >= 8 and sample[4:8] == b"ftyp":
        return "video/mp4"
    if len(sample) >= 4 and sample[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    if len(sample) >= 4 and sample[:4] == b"OggS":
        return "video/ogg"
    return "application/octet-stream"


def _normalize_rel_path(path_param: str) -> str | None:
    if not path_param or not path_param.strip():
        return None
    normalized = path_param.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    data_dir = video_data_dir_name()
    if not normalized.startswith(f"{data_dir}/"):
        return None
    parts = [p for p in normalized.split("/") if p]
    if any(p in (".", "..") for p in parts):
        return None
    if len(parts) < 4 or parts[0] != data_dir or parts[2] != "videos":
        return None
    if not _VENDOR_SEGMENT.match(parts[1]):
        logger.warning("Blocked video path (bad vendor): %s", path_param)
        return None
    return "/".join(parts)


def resolve_video_candidate(path_param: str, repo_root: Path) -> Path | None:
    """Resolve path under data/<vendor>/videos/ without requiring the file to exist."""
    rel = _normalize_rel_path(path_param)
    if rel is None:
        return None
    root = repo_root.resolve()
    full = (root / rel).resolve()
    try:
        full.relative_to(root)
    except ValueError:
        logger.warning("Blocked video path traversal outside repo: %s", path_param)
        return None

    parts = rel.split("/")
    videos_root = (root / parts[0] / parts[1] / "videos").resolve()
    try:
        full.relative_to(videos_root)
    except ValueError:
        logger.warning("Blocked video path outside videos subtree: %s", path_param)
        return None

    if full.is_symlink():
        real = Path(os.path.realpath(full))
        try:
            real.relative_to(videos_root)
        except ValueError:
            logger.warning("Blocked video symlink escape: %s", path_param)
            return None
    return full


def resolve_video_path(path_param: str, repo_root: Path) -> Path | None:
    """Resolve a repo-relative video path under data/<vendor>/videos/."""
    full = resolve_video_candidate(path_param, repo_root)
    if full is None or not full.is_file():
        return None
    return full


def parse_range_header(range_header: str | None, file_size: int) -> tuple[int, int] | str | None:
    """Parse a single HTTP Range header into (start, end) inclusive byte indices.

    Returns None when no Range header is present.
    Returns "unsatisfiable" when the range cannot be satisfied (caller → 416).
    """
    if not range_header or not range_header.strip():
        return None
    if file_size <= 0:
        return "unsatisfiable"

    value = range_header.strip()
    if not value.lower().startswith("bytes="):
        return "unsatisfiable"

    spec = value[6:].strip()
    if not spec:
        return "unsatisfiable"
    if "," in spec:
        spec = spec.split(",", 1)[0].strip()

    try:
        if spec.startswith("-"):
            suffix_len = int(spec[1:])
            if suffix_len <= 0:
                return "unsatisfiable"
            start = max(0, file_size - suffix_len)
            end = file_size - 1
        else:
            start_text, end_text = spec.split("-", 1)
            start = int(start_text) if start_text else 0
            end = file_size - 1 if end_text == "" else int(end_text)
    except ValueError:
        return "unsatisfiable"

    if start < 0 or start >= file_size or end < start:
        return "unsatisfiable"
    end = min(end, file_size - 1)
    return start, end


def iter_file_bytes(path: Path, start: int, end: int, chunk_size: int = 64 * 1024):
    """Yield bytes [start, end] inclusive from a file."""
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            data = handle.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data
