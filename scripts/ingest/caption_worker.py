#!/usr/bin/env python3
"""Caption queued images with Ollama vision and upsert to Qdrant.

Run: ``python -m scripts.ingest.caption_worker``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from providers.caption_images import (
    is_remote_image_src,
    prepare_vision_image_bytes,
    resolve_image_url,
    should_skip_image_src,
)
from providers.factory import get_embedder, get_vector_store
from providers.ollama_vision import OllamaVision
from providers.qdrant_sharded_store import VendorShardedQdrantStore, is_vendor_sharding_enabled

load_dotenv()

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
CHECKPOINT_EVERY = 10
DOWNLOAD_TIMEOUT = 20
MIN_CAPTION_LEN = 15
MAX_CAPTION_LEN = 600
_PASSTHROUGH_META_KEYS = (
    "category",
    "device_family",
    "device_model",
    "doc_version",
    "language",
    "source_type",
    "video_path",
    "video_url",
)

_BOILERPLATE_PATTERNS = (
    r"^this (image|figure|photo|screenshot|diagram|picture) (shows|depicts|displays|contains|is)\s*:?\s*",
    r"^the (image|figure|photo|screenshot|diagram|picture) (shows|depicts|displays|contains|is)\s*:?\s*",
    r"^(an? )?(image|figure|photo|screenshot|diagram|picture)\s+(shows|depicts|displays)\s*:?\s*",
)
_IDS_PREFIX_RE = re.compile(r"^\s*ids?\s*[=:]", re.IGNORECASE)
_DECIMAL_ARRAY_RE = re.compile(
    r"\[\s*(?:\d+\.\d+\s*,\s*)+\d+\.\d+\s*\]",
)
_COORD_CHAR_SET = frozenset("0123456789.,[] \t")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pending_captions_path(vendor: str) -> Path:
    return Path("data") / vendor.lower() / "pending_captions.json"


def discover_queues(vendor: str | None = None) -> list[tuple[str, Path]]:
    root = Path("data")
    if not root.is_dir():
        return []
    queues: list[tuple[str, Path]] = []
    for path in sorted(root.glob("*/pending_captions.json")):
        name = path.parent.name.lower()
        if vendor and name != vendor.lower():
            continue
        queues.append((name, path))
    return queues


def load_queue(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return []
    return data if isinstance(data, list) else []


def save_queue_atomic(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def is_pending(entry: dict, *, retry_failed: bool) -> bool:
    captioned = entry.get("captioned")
    if captioned is True:
        return False
    if captioned in ("skipped", "skip"):
        return False
    if captioned == "failed":
        return retry_failed
    return captioned is False or captioned is None


def sanitize_caption(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw.strip())
    for pattern in _BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def truncate_caption(caption: str, max_len: int = MAX_CAPTION_LEN) -> str:
    if len(caption) <= max_len:
        return caption
    window = caption[:max_len]
    boundary = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    if boundary > 0:
        return window[: boundary + 1].strip()
    space = window.rfind(" ")
    if space > 0:
        return window[:space].strip()
    return window.strip()


def reject_caption(caption: str) -> str | None:
    """Return a rejection reason, or None if the caption is acceptable."""
    if len(caption) < MIN_CAPTION_LEN:
        return "too short"
    if _IDS_PREFIX_RE.match(caption):
        return "ids prefix"
    if _DECIMAL_ARRAY_RE.search(caption):
        return "decimal coordinate array"
    if caption:
        coord_chars = sum(1 for ch in caption if ch in _COORD_CHAR_SET)
        if coord_chars / len(caption) > 0.5:
            return "mostly coordinates/numbers"
    return None


def build_caption_prompt(entry: dict) -> str:
    vendor = entry.get("vendor") or ""
    source = entry.get("source") or ""
    header = f"This is a figure from industrial technical documentation ({vendor}, {source}"
    page = entry.get("page")
    if page is not None:
        header += f", page {page}"
    section = entry.get("section")
    if section:
        header += f", section '{section}'"
    header += ")."
    return (
        f"{header} Describe what it shows in 2-4 sentences for a search index: "
        "type of figure (photo, wiring diagram, screenshot, chart, mechanical drawing), "
        "the components or UI elements visible, and any readable labels or values. "
        "Be factual; do not speculate. "
        "Only mention labels, numbers, or IP addresses you can clearly read. "
        "If text is too small or unclear, say 'labels not legible' instead of guessing."
    )


def build_index_text(source: str, page: int | None, caption: str) -> str:
    if page is not None:
        return f"[Figure, {source}, p.{page}] {caption}"
    return f"[Figure, {source}] {caption}"


def make_point_id(source: str, image_src: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|img|{image_src}"))


def local_image_path(vendor: str, image_src: str) -> Path:
    normalized = image_src.replace("\\", "/").lstrip("/")
    return Path("data") / vendor.lower() / normalized


def cache_image_path(vendor: str, image_src: str) -> Path:
    digest = hashlib.sha256(image_src.encode("utf-8")).hexdigest()[:20]
    return Path("data") / vendor.lower() / "images" / "_caption_cache" / f"{digest}.img"


def load_image_bytes(entry: dict, vendor: str) -> tuple[bytes | None, Path | None, str | None]:
    image_src = (entry.get("image_src") or "").strip()
    if not image_src:
        return None, None, "missing image_src"

    if is_remote_image_src(image_src):
        url = resolve_image_url(image_src, entry.get("url"))
        if not url:
            return None, None, f"cannot resolve URL for {image_src!r}"
        cache_path = cache_image_path(vendor, image_src)
        if cache_path.is_file():
            return prepare_vision_image_bytes(cache_path.read_bytes()), cache_path, None
        try:
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            return None, None, f"download failed: {exc}"
        data = prepare_vision_image_bytes(response.content)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        return data, cache_path, None

    path = local_image_path(vendor, image_src)
    if not path.is_file():
        return None, None, f"local image not found: {path}"
    return prepare_vision_image_bytes(path.read_bytes()), path, None


def _qdrant_client_from_store(store) -> QdrantClient:
    client = getattr(store, "client", None)
    if client is None:
        raise RuntimeError("Vector store has no Qdrant client")
    return client


def _collection_for_vendor(store, vendor: str) -> str:
    if isinstance(store, VendorShardedQdrantStore) and is_vendor_sharding_enabled():
        return store._shard_collection(vendor)
    return store.collection


def fetch_source_metadata(
    store,
    *,
    source: str,
    vendor: str,
    cache: dict[str, dict],
) -> dict:
    if source in cache:
        return cache[source]

    client = _qdrant_client_from_store(store)
    collection = _collection_for_vendor(store, vendor)
    source_filter = Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=source))]
    )
    records, _ = client.scroll(
        collection_name=collection,
        limit=32,
        scroll_filter=source_filter,
        with_payload=True,
        with_vectors=False,
    )

    chosen: dict | None = None
    fallback: dict | None = None
    for record in records:
        payload = dict(record.payload or {})
        if payload.get("content_type") == "text":
            chosen = payload
            break
        if fallback is None:
            fallback = payload

    payload = chosen or fallback or {}
    meta = {
        "vendor": (payload.get("vendor") or vendor or "").strip().lower(),
        "product": payload.get("product"),
        "product_version": payload.get("product_version"),
        "doc_type": payload.get("doc_type") or payload.get("document_type") or "other",
    }
    for key in _PASSTHROUGH_META_KEYS:
        value = payload.get(key)
        if value is not None:
            meta[key] = value
    cache[source] = meta
    return meta


def build_payload(
    entry: dict,
    *,
    caption: str,
    meta: dict,
    image_path: Path,
    ingested_at: str,
) -> dict:
    source = entry["source"]
    page = entry.get("page")
    text = build_index_text(source, page, caption)
    rel_image_path = image_path.as_posix()
    payload = {
        "text": text,
        "source": source,
        "page": page,
        "url": entry.get("url"),
        "vendor": meta.get("vendor") or entry.get("vendor"),
        "product": meta.get("product"),
        "product_version": meta.get("product_version"),
        "content_type": "image_caption",
        "doc_type": meta.get("doc_type") or "other",
        "section": entry.get("section"),
        "timestamp": None,
        "schema_version": SCHEMA_VERSION,
        "ingested_at": ingested_at,
        "image_path": rel_image_path,
    }
    for key in _PASSTHROUGH_META_KEYS:
        value = meta.get(key)
        if value is not None:
            payload[key] = value
    return payload


def process_entry(
    entry: dict,
    *,
    vendor: str,
    vision: OllamaVision,
    embedder,
    store,
    meta_cache: dict[str, dict],
) -> str:
    image_src = (entry.get("image_src") or "").strip()
    source = (entry.get("source") or "").strip()
    if not image_src or not source:
        return "skipped"

    skip_reason = should_skip_image_src(image_src)
    if skip_reason:
        logger.info("Caption skipped (%s): %s", image_src, skip_reason)
        return "skipped"

    image_bytes, image_path, err = load_image_bytes(entry, vendor)
    if err or image_bytes is None or image_path is None:
        logger.warning("Image load failed (%s): %s", image_src, err)
        return "failed"

    prompt = build_caption_prompt(entry)
    try:
        raw_caption = vision.describe_image(prompt, image_bytes)
    except Exception as exc:
        logger.warning("Vision caption failed (%s): %s", image_src, exc)
        return "failed"

    caption = truncate_caption(sanitize_caption(raw_caption))
    reject_reason = reject_caption(caption)
    if reject_reason:
        logger.warning("Caption rejected (%s): %s — %r", source, reject_reason, caption)
        return "failed"

    meta = fetch_source_metadata(store, source=source, vendor=vendor, cache=meta_cache)
    payload = build_payload(
        entry,
        caption=caption,
        meta=meta,
        image_path=image_path,
        ingested_at=_utc_now(),
    )
    point_id = make_point_id(source, image_src)
    vector = embedder.embed([payload["text"]])[0]
    store.upsert([point_id], [vector], [payload])
    return "captioned"


def process_queue(
    vendor: str,
    path: Path,
    *,
    limit: int | None,
    retry_failed: bool,
    vision: OllamaVision,
    embedder,
    store,
) -> Counter[str]:
    entries = load_queue(path)
    if not entries:
        return Counter()

    pending_indices = [i for i, e in enumerate(entries) if is_pending(e, retry_failed=retry_failed)]
    if limit is not None:
        pending_indices = pending_indices[: max(0, limit)]

    stats: Counter[str] = Counter()
    meta_cache: dict[str, dict] = {}
    dirty = 0

    label = f"{vendor}"
    for idx in tqdm(pending_indices, desc=label, unit="img"):
        entry = entries[idx]
        outcome = process_entry(
            entry,
            vendor=vendor,
            vision=vision,
            embedder=embedder,
            store=store,
            meta_cache=meta_cache,
        )
        stats[outcome] += 1
        if outcome == "captioned":
            entries[idx]["captioned"] = True
        elif outcome == "failed":
            entries[idx]["captioned"] = "failed"
        elif outcome == "skipped":
            entries[idx]["captioned"] = "skipped"
        else:
            continue

        dirty += 1
        if dirty >= CHECKPOINT_EVERY:
            save_queue_atomic(path, entries)
            dirty = 0

    if dirty:
        save_queue_atomic(path, entries)

    return stats


def _qdrant_image_caption_ids(store, vendor: str) -> set[str]:
    client = _qdrant_client_from_store(store)
    collection = _collection_for_vendor(store, vendor)
    ids: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="vendor", match=MatchValue(value=vendor)),
                    FieldCondition(key="content_type", match=MatchValue(value="image_caption")),
                ]
            ),
            with_payload=False,
            with_vectors=False,
        )
        if not records:
            break
        for record in records:
            ids.add(str(record.id))
        if offset is None:
            break
    return ids


def reconcile_missing_captions(
    vendor: str,
    path: Path,
    store,
) -> int:
    """Reset captioned=true queue entries that are absent from Qdrant (re-queue)."""
    entries = load_queue(path)
    if not entries:
        return 0

    qdrant_ids = _qdrant_image_caption_ids(store, vendor)
    reset = 0
    for entry in entries:
        if entry.get("captioned") is not True:
            continue
        image_src = (entry.get("image_src") or "").strip()
        source = (entry.get("source") or "").strip()
        if not image_src or not source:
            continue
        point_id = make_point_id(source, image_src)
        if point_id not in qdrant_ids:
            entry["captioned"] = False
            entry.pop("caption", None)
            reset += 1

    if reset:
        save_queue_atomic(path, entries)
    return reset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caption queued images and upsert to Qdrant")
    parser.add_argument("--vendor", help="Process only this vendor queue")
    parser.add_argument("--limit", type=int, help="Max images to process per vendor")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-process entries marked captioned='failed'",
    )
    parser.add_argument(
        "--reconcile-missing",
        action="store_true",
        help="Reset captioned=true entries missing from Qdrant back to pending",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    queues = discover_queues(args.vendor)
    if not queues:
        target = args.vendor or "any vendor"
        print(f"No pending_captions.json found for {target}.", file=sys.stderr)
        return 1

    embedder = get_embedder()
    store = get_vector_store()
    store.ensure_payload_indexes()

    if args.reconcile_missing:
        for vendor, path in queues:
            reset = reconcile_missing_captions(vendor, path, store)
            print(f"Reconcile {vendor}: reset {reset} entries to pending")

    vision = OllamaVision()
    totals: Counter[str] = Counter()
    for vendor, path in queues:
        stats = process_queue(
            vendor,
            path,
            limit=args.limit,
            retry_failed=args.retry_failed,
            vision=vision,
            embedder=embedder,
            store=store,
        )
        totals.update(stats)

    captioned = totals.get("captioned", 0)
    failed = totals.get("failed", 0)
    skipped = totals.get("skipped", 0)
    print(f"Done: captioned={captioned} failed={failed} skipped={skipped}")
    return 0 if failed == 0 or captioned > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
