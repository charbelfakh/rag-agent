"""External chunk text storage (Qdrant payload holds pointer + preview)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_blob_store = None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def is_blob_storage_enabled() -> bool:
    return _env_bool("BLOB_STORAGE_ENABLED")


def _preview_chars() -> int:
    return int(os.getenv("BLOB_PAYLOAD_PREVIEW_CHARS", "200"))


class BlobStore:
    """Store full chunk text on disk; Qdrant payloads keep previews and ``text_uri``."""

    def __init__(self, root: str | None = None):
        self.enabled = is_blob_storage_enabled()
        self.root = Path(root or os.getenv("BLOB_STORAGE_PATH", "data/blobs"))
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def put_text(self, key: str, text: str) -> str:
        if not self.enabled:
            return ""
        safe_key = key.replace("\\", "/").replace("..", "_")
        path = self.root / safe_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return f"blob://{safe_key}"

    def get_text(self, uri: str) -> str:
        if not uri.startswith("blob://"):
            return ""
        rel = uri[len("blob://") :]
        path = self.root / rel
        if not path.exists():
            logger.warning("Missing blob object: %s", uri)
            return ""
        return path.read_text(encoding="utf-8")

    def externalize_payload_text(self, payload: dict) -> dict:
        """Move full chunk text to blob storage; keep preview in payload."""
        if not self.enabled:
            return payload
        text = payload.get("text") or payload.get("text_full") or ""
        if not text:
            return payload
        chunk_id = payload.get("chunk_id") or payload.get("source", "chunk")
        uri = self.put_text(f"{chunk_id}.txt", text)
        preview_len = _preview_chars()
        preview = text if len(text) <= preview_len else text[:preview_len].rstrip() + "…"
        updated = dict(payload)
        updated["text_uri"] = uri
        updated["text"] = preview
        updated.pop("text_full", None)
        return updated

    def hydrate_payload(self, payload: dict) -> dict:
        if not payload.get("text_uri"):
            if payload.get("text_full"):
                payload = dict(payload)
                payload["text"] = payload["text_full"]
            return payload
        full = self.get_text(payload["text_uri"])
        if not full:
            return payload
        hydrated = dict(payload)
        hydrated["text"] = full
        return hydrated


def get_blob_store() -> BlobStore:
    global _blob_store
    if _blob_store is None:
        _blob_store = BlobStore()
    return _blob_store


def reset_blob_store() -> None:
    global _blob_store
    _blob_store = None
