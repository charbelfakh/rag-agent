"""Document and chunk metadata helpers for ingest and retrieval."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROCEDURE_LINE = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)
_SOFTWARE_VERSION_PATTERNS = (
    re.compile(r"_v(\d+)\.(\d+)(?:\.(\d+))?_", re.IGNORECASE),
    re.compile(
        r"(?:^|[^A-Za-z0-9])V(\d+)\.(\d+)(?:\.(\d+))?(?:[^A-Za-z0-9]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|[^A-Za-z0-9])V(\d{2,3})(?:[^A-Za-z0-9]|$)", re.IGNORECASE),
    re.compile(r"_(\d)(\d{2})(\d?)_"),
)
_PRODUCT_LINE_BY_VENDOR: dict[str, tuple[tuple[str, str], ...]] = {
    "lmi": (
        ("gocator", "Gocator"),
        ("hdrstudio", "HDR Studio"),
    ),
}


@dataclass
class DocumentMetadata:
    source: str
    file_name: str
    file_extension: str
    vendor: str
    document_type: str = "manual"
    language: str = "en"
    content_type: str = "text"
    ingestion_timestamp: str = ""
    content_hash: str = ""
    product_line: str = ""
    software_version: str = ""

    def to_payload_fields(self) -> dict:
        fields = {
            "source": self.source,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "vendor": self.vendor,
            "document_type": self.document_type,
            "language": self.language,
            "content_type": self.content_type,
            "ingestion_timestamp": self.ingestion_timestamp,
        }
        if self.content_hash:
            fields["content_hash"] = self.content_hash
        if self.product_line:
            fields["product_line"] = self.product_line
        if self.software_version:
            fields["software_version"] = self.software_version
        return fields


def compute_file_content_hash(file_path: str) -> str:
    """SHA-256 of raw file bytes for incremental ingest skip/replace."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_incremental_ingest_enabled() -> bool:
    import os

    return os.getenv("INCREMENTAL_INGEST_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def incremental_ingest_precheck(
    file_path: str,
    normalized_path: str,
    store,
) -> tuple[str, str]:
    """Return ``(action, content_hash)`` where action is ``skip``, ``replace``, or ``ingest``."""
    if not is_incremental_ingest_enabled():
        return "ingest", ""
    content_hash = compute_file_content_hash(file_path)
    stored_hash = store.get_source_content_hash(normalized_path)
    if stored_hash and stored_hash == content_hash:
        return "skip", content_hash
    if stored_hash:
        return "replace", content_hash
    return "ingest", content_hash


def normalize_source(file_path: str) -> str:
    """Canonical repo-relative POSIX path for ``source``."""
    return file_path.replace("\\", "/")


def infer_vendor_from_source(source: str) -> str:
    """Infer vendor slug from ``docs/<vendor>/…`` or ``data/…`` upload paths."""
    parts = normalize_source(source).split("/")
    if len(parts) >= 2 and parts[0] == "docs":
        return parts[1].lower()
    if len(parts) >= 2 and parts[0] == "data":
        return "uploads"
    return "unknown"


def infer_vendor_from_filename(file_name: str) -> str | None:
    """Match known vendor slugs in a filename (for upload modal pre-fill)."""
    lowered = file_name.lower()
    for slug in (
        "pekat",
        "mechmind",
        "zivid",
        "lmi",
        "basler",
        "photoneo",
    ):
        if slug in lowered:
            return slug
    return None


def infer_software_version_from_filename(file_name: str) -> str:
    match = _SOFTWARE_VERSION_PATTERNS[0].search(file_name)
    if match:
        major, minor, patch = match.group(1), match.group(2), match.group(3)
        if patch:
            return f"{major}.{minor}.{patch}"
        return f"{major}.{minor}"
    match = _SOFTWARE_VERSION_PATTERNS[1].search(file_name)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    match = _SOFTWARE_VERSION_PATTERNS[2].search(file_name)
    if match:
        return match.group(1)
    match = _SOFTWARE_VERSION_PATTERNS[3].search(file_name)
    if match:
        major, minor, patch = match.group(1), match.group(2), match.group(3) or "0"
        return f"{major}.{minor}.{patch}"
    return ""


def infer_product_line_from_filename(vendor: str, file_name: str) -> str:
    lowered = file_name.lower()
    vendor_key = vendor.lower()
    for needle, label in _PRODUCT_LINE_BY_VENDOR.get(vendor_key, ()):
        if needle in lowered:
            return label
    return ""


def detect_is_procedure(text: str) -> bool:
    return len(_PROCEDURE_LINE.findall(text)) >= 3


def detect_has_code(text: str) -> bool:
    if any(token in text for token in ("{", "}", "def ", "function", "import", "#include")):
        return True
    return bool(re.search(r"^(?:\s{2,}|\t)\S", text, re.MULTILINE))


def resolve_metadata(
    file_path: str,
    *,
    vendor: str | None = None,
    document_type: str | None = None,
    language: str | None = None,
    product_line: str | None = None,
    software_version: str | None = None,
) -> DocumentMetadata:
    """Build ``DocumentMetadata`` from path with filename-based inference fallbacks."""
    source = normalize_source(file_path)
    path = Path(source)
    inferred_vendor = vendor.strip().lower() if vendor else infer_vendor_from_source(source)
    file_name = path.name
    resolved_product_line = (product_line or "").strip()
    if not resolved_product_line:
        resolved_product_line = infer_product_line_from_filename(inferred_vendor, file_name)
    resolved_software_version = (software_version or "").strip()
    if not resolved_software_version:
        resolved_software_version = infer_software_version_from_filename(file_name)
    return DocumentMetadata(
        source=source,
        file_name=file_name,
        file_extension=path.suffix.lower(),
        vendor=inferred_vendor,
        document_type=document_type or "manual",
        language=language or "en",
        ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
        product_line=resolved_product_line,
        software_version=resolved_software_version,
    )


def make_chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic point ID for idempotent upserts."""
    digest = hashlib.sha256(f"{normalize_source(source)}:{chunk_index}".encode()).hexdigest()
    return digest[:32]


def _payload_text_max_chars() -> int:
    import os

    return int(os.getenv("QDRANT_PAYLOAD_TEXT_MAX_CHARS", "0"))


def slim_payload_text(text: str, max_chars: int | None = None) -> tuple[str, str | None]:
    """Return ``(stored_text, text_full)`` with optional preview slimming."""
    limit = max_chars if max_chars is not None else _payload_text_max_chars()
    if limit <= 0 or len(text) <= limit:
        return text, None
    preview = text[:limit].rstrip() + "…"
    return preview, text


def build_image_chunk_payload(
    *,
    metadata: DocumentMetadata,
    chunk_index: int,
    page: int,
    media_uri: str,
    media_hash: str,
    ocr_text: str = "",
    image_class: str = "unknown",
    width: int | None = None,
    height: int | None = None,
    section: str = "",
    parent_chunk_id: str | None = None,
) -> dict:
    """Build Qdrant payload for an image chunk (M0/M1 multimodal)."""
    payload = {
        "text": ocr_text or f"[image {Path(media_uri).name}]",
        "page": page,
        "section": section,
        "chunk_index": chunk_index,
        "chunk_id": make_chunk_id(metadata.source, chunk_index),
        "content_type": "image",
        "media_uri": media_uri,
        "media_hash": media_hash,
        "image_class": image_class,
        **metadata.to_payload_fields(),
    }
    payload["content_type"] = "image"
    if width is not None:
        payload["width"] = width
    if height is not None:
        payload["height"] = height
    if parent_chunk_id:
        payload["parent_chunk_id"] = parent_chunk_id
    if ocr_text:
        payload["ocr_text"] = ocr_text
    return payload


def build_chunk_payload(
    item: Any,
    metadata: DocumentMetadata,
    chunk_index: int,
    *,
    total_chunks: int | None = None,
) -> dict:
    """Assemble a Qdrant text-chunk payload with procedure/code flags."""
    stored_text, text_full = slim_payload_text(item.text)
    payload = {
        "text": stored_text,
        "page": item.page,
        "section": item.section,
        "chunk_index": chunk_index,
        "chunk_id": make_chunk_id(metadata.source, chunk_index),
        **metadata.to_payload_fields(),
    }
    if total_chunks is not None:
        payload["total_chunks"] = total_chunks
    if text_full is not None:
        payload["text_full"] = text_full
    payload["is_procedure"] = detect_is_procedure(item.text)
    payload["has_code"] = detect_has_code(item.text)
    return payload
