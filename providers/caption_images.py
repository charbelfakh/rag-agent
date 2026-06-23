"""Shared caption-queue image rules: skip filters, remote materialization, vision prep."""
from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import requests
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

CAPTION_SKIP_SRC_MARKERS = (
    "avatar",
    "aa-avatar",
    "profilepics",
    "gravatar",
    "logo",
    "favicon",
    "icon",
    "emoticon",
    "badge",
    "spacer",
    "tracking",
    "pixel",
)

DOWNLOAD_TIMEOUT = 20


def should_skip_image_src(image_src: str) -> str | None:
    """Return a skip reason for non-captionable image refs (favicons, etc.)."""
    lowered = image_src.strip().lower()
    if not lowered:
        return "empty image_src"
    if lowered.endswith(".ico"):
        return "ico favicon"
    if "static.apiary.io" in lowered or "/apiary.io/assets/" in lowered:
        return "apiary favicon"
    if "chat.google.com" in lowered:
        return "google chat attachment URL"
    return None


def is_skipped_caption_image_src(src: str) -> bool:
    """True if ``src`` should never enter ``pending_captions.json``."""
    if should_skip_image_src(src):
        return True
    lowered = src.strip().lower()
    if not lowered or lowered.startswith("data:"):
        return True
    if lowered.endswith(".svg"):
        return True
    return any(marker in lowered for marker in CAPTION_SKIP_SRC_MARKERS)


def is_skipped_caption_entry(entry: dict) -> bool:
    src = (entry.get("image_src") or "").strip()
    return is_skipped_caption_image_src(src)


def is_remote_image_src(image_src: str) -> bool:
    lowered = image_src.strip().lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return True
    if image_src.strip().startswith("/"):
        return True
    normalized = image_src.replace("\\", "/")
    return not normalized.startswith("images/")


def resolve_image_url(image_src: str, page_url: str | None) -> str | None:
    src = image_src.strip()
    if src.lower().startswith(("http://", "https://")):
        return src
    if page_url:
        return urljoin(page_url, src)
    return None


def prepare_vision_image_bytes(data: bytes) -> bytes:
    """Normalize image bytes for Ollama vision (llava rejects WEBP; convert to PNG)."""
    if data[:4] == b"\x89PNG" or data[:3] == b"\xff\xd8\xff":
        return data
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        with Image.open(BytesIO(data)) as im:
            rgb = im.convert("RGB")
            out = BytesIO()
            rgb.save(out, format="PNG")
            return out.getvalue()
    try:
        with Image.open(BytesIO(data)) as im:
            rgb = im.convert("RGB")
            out = BytesIO()
            rgb.save(out, format="PNG")
            return out.getvalue()
    except UnidentifiedImageError:
        return data


def ingest_cache_path(vendor: str, image_src: str) -> Path:
    digest = hashlib.sha256(image_src.encode("utf-8")).hexdigest()[:20]
    return Path("data") / vendor.lower() / "images" / "_ingest_cache" / f"{digest}.png"


def materialize_remote_image_src(
    entry: dict,
    vendor: str,
    *,
    timeout: float = DOWNLOAD_TIMEOUT,
) -> dict:
    """Download remote ``image_src``, save vision-safe PNG locally; update entry path."""
    image_src = (entry.get("image_src") or "").strip()
    if not image_src or not is_remote_image_src(image_src):
        return entry
    if should_skip_image_src(image_src):
        return entry

    path = ingest_cache_path(vendor, image_src)
    vendor_root = Path("data") / vendor.lower()
    try:
        local_rel = path.relative_to(vendor_root).as_posix()
    except ValueError:
        local_rel = f"images/_ingest_cache/{path.name}"

    if path.is_file():
        updated = dict(entry)
        updated["image_src"] = local_rel
        return updated

    url = resolve_image_url(image_src, entry.get("url"))
    if not url:
        return entry

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        png_bytes = prepare_vision_image_bytes(response.content)
    except requests.RequestException as exc:
        logger.debug("Ingest image materialize failed (%s): %s", image_src, exc)
        return entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
    updated = dict(entry)
    updated["image_src"] = local_rel
    return updated


def prepare_caption_queue_entries(
    entries: list[dict],
    vendor: str,
    *,
    materialize_remote: bool = True,
) -> list[dict]:
    """Filter skip rules and optionally materialize remote images for the caption queue."""
    prepared: list[dict] = []
    for entry in entries:
        if is_skipped_caption_entry(entry):
            continue
        if materialize_remote:
            entry = materialize_remote_image_src(entry, vendor)
        prepared.append(entry)
    return prepared
