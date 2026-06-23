"""Safe path resolution and content-type detection for caption-source images."""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_VENDOR_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$", re.I)


def detect_image_content_type(data: bytes) -> str:
    """Detect PNG/JPEG/WEBP/GIF from magic bytes (extensionless cache files)."""
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "application/octet-stream"


def _normalize_rel_path(path_param: str) -> str | None:
    if not path_param or not path_param.strip():
        return None
    normalized = path_param.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    if not normalized.startswith("data/"):
        return None
    parts = [p for p in normalized.split("/") if p]
    if any(p in (".", "..") for p in parts):
        return None
    if len(parts) < 4 or parts[0] != "data" or parts[2] != "images":
        return None
    if not _VENDOR_SEGMENT.match(parts[1]):
        logger.warning("Blocked caption image path (bad vendor): %s", path_param)
        return None
    return "/".join(parts)


def resolve_caption_image_candidate(path_param: str, repo_root: Path) -> Path | None:
    """Resolve path under data/<vendor>/images/ without requiring the file to exist."""
    rel = _normalize_rel_path(path_param)
    if rel is None:
        return None
    root = repo_root.resolve()
    full = (root / rel).resolve()
    try:
        full.relative_to(root)
    except ValueError:
        logger.warning("Blocked caption image path traversal: %s", path_param)
        return None
    return full


def resolve_caption_image_path(path_param: str, repo_root: Path) -> Path | None:
    """Resolve a repo-relative caption image path under data/<vendor>/images/."""
    full = resolve_caption_image_candidate(path_param, repo_root)
    if full is None or not full.is_file():
        return None
    return full
