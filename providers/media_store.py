"""Local blob store for multimodal media (images, thumbnails) — Sprint L M0."""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_media_store = None
_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9._-]+")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def is_media_storage_enabled() -> bool:
    return _env_bool("MEDIA_STORAGE_ENABLED", "true")


def media_root() -> Path:
    return Path(os.getenv("MEDIA_STORAGE_PATH", "data/media"))


def normalize_media_uri(rel_path: str) -> str:
    """Public URI served by ``GET /media/{path}``."""
    cleaned = rel_path.replace("\\", "/").lstrip("/")
    return f"/media/{cleaned}"


def _safe_relative_path(rel_path: str) -> str:
    parts = []
    for segment in rel_path.replace("\\", "/").split("/"):
        if not segment or segment in (".", ".."):
            continue
        parts.append(_SAFE_SEGMENT.sub("_", segment))
    return "/".join(parts)


class MediaStore:
    """Write and resolve image blobs under ``data/media`` for multimodal ingest."""

    def __init__(self, root: str | Path | None = None):
        self.enabled = is_media_storage_enabled()
        self.root = Path(root or media_root())
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def source_hash(self, source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    def put_bytes(self, rel_path: str, data: bytes) -> str:
        if not self.enabled:
            return ""
        safe = _safe_relative_path(rel_path)
        path = self.root / safe
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return normalize_media_uri(safe)

    def resolve_path(self, media_uri: str) -> Path | None:
        if not media_uri:
            return None
        rel = media_uri
        if rel.startswith("/media/"):
            rel = rel[len("/media/") :]
        elif rel.startswith("media/"):
            rel = rel[len("media/") :]
        safe = _safe_relative_path(rel)
        path = (self.root / safe).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            logger.warning("Blocked media path traversal: %s", media_uri)
            return None
        return path if path.exists() else None

    def media_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


def get_media_store() -> MediaStore:
    global _media_store
    if _media_store is None:
        _media_store = MediaStore()
    return _media_store


def reset_media_store() -> None:
    global _media_store
    _media_store = None
