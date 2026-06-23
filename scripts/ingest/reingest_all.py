#!/usr/bin/env python3
"""Resumable full-corpus re-ingest (text → images → captions → audit).

Run: ``python -m scripts.ingest.reingest_all``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

load_dotenv()

from providers.config_paths import DATA_DIR

STATE_FILE = DATA_DIR / "reingest_state.json"
LOG_DIR = Path("logs")

logger = logging.getLogger("reingest_all")


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@dataclass
class SourceMetadata:
    vendor: str
    category: str
    device_family: str | None
    device_model: str | None
    doc_type: str
    doc_version: str | None
    language: str

    def ingest_kwargs(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "product": (self.device_family or "").lower() or None,
            "product_version": self.doc_version,
            "doc_type": self.doc_type,
        }

    def payload_extra(self, *, source_type: str) -> dict[str, Any]:
        return {
            "category": self.category,
            "device_family": self.device_family,
            "device_model": self.device_model,
            "doc_version": self.doc_version,
            "language": self.language,
            "source_type": source_type,
        }


@dataclass
class RunStats:
    phase_seconds: dict[str, float] = field(default_factory=dict)
    text_chunks: int = 0
    html_chunks: int = 0
    images_extracted: int = 0
    html_images_collected: int = 0
    captions_new: int = 0
    captions_reembedded: int = 0
    captions_failed: int = 0
    skipped_sources: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    failed_urls: list[dict[str, str]] = field(default_factory=list)
    embed_model: str = ""
    embed_dim: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def setup_logging(run_ts: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"reingest_{run_ts}.log"
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def save_state_atomic(state: dict) -> None:
    """Atomically write ``reingest_state.json`` so a crash mid-run can resume."""
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def load_state() -> dict:
    if not STATE_FILE.is_file():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s: %s", STATE_FILE, exc)
        return {}


def load_vendors_config() -> dict:
    from providers.config_paths import VENDORS_JSON

    path = VENDORS_JSON
    if not path.is_file():
        return {"vendors": {}, "custom_folders": [], "keywords": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"vendors": {}, "custom_folders": [], "keywords": {}}
    data.setdefault("keywords", {})
    data.setdefault("custom_folders", [])
    data.setdefault("vendors", {})
    return data


def resolve_vendors(cli_vendors: str | None, config: dict) -> list[str]:
    from scripts.data.download_docs import load_vendors_json

    _, custom = load_vendors_json()
    candidates: set[str] = set(custom)
    candidates.update(config.get("custom_folders", []))
    candidates.update(config.get("vendors", {}).keys())
    data_root = Path("data")
    if data_root.is_dir():
        for folder in data_root.iterdir():
            if folder.is_dir() and any(folder.glob("*.pdf")):
                candidates.add(folder.name.lower())
    vendors = sorted(v for v in candidates if v)
    if cli_vendors:
        want = {v.strip().lower() for v in cli_vendors.split(",") if v.strip()}
        vendors = [v for v in vendors if v in want]
    return vendors


def list_pdfs(vendors: list[str]) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for vendor in vendors:
        folder = Path("data") / vendor
        if not folder.is_dir():
            continue
        for pdf in sorted(folder.glob("*.pdf")):
            rows.append((vendor, pdf))
    return rows


def discover_url_lists(vendor: str) -> list[Path]:
    folder = Path("data") / vendor
    if not folder.is_dir():
        return []
    return sorted(folder.glob("urls*.txt"))


def list_html_urls(vendors: list[str]) -> list[tuple[str, str, Path]]:
    """Return (vendor, url, list_file) for every URL in vendor url list files."""
    from scripts.ingest.ingest import _load_url_list

    rows: list[tuple[str, str, Path]] = []
    seen: set[tuple[str, str]] = set()
    for vendor in vendors:
        for list_path in discover_url_lists(vendor):
            for url in _load_url_list(str(list_path)):
                key = (vendor, url)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((vendor, url, list_path))
    return rows


def infer_language(filename: str) -> str:
    upper = filename.upper()
    if "_EN-US" in upper or "_EN_US" in upper or "-EN-US" in upper or "-EN-" in upper:
        return "en"
    if "_DE" in upper or "-DE-" in upper:
        return "de"
    return "en"


def infer_doc_type(filename: str) -> str:
    from scripts.ingest.ingest import normalize_doc_type

    lowered = filename.lower()
    if lowered.endswith((".html", ".htm")):
        return "article"
    if "datasheet" in lowered or "data_sheet" in lowered or "data-sheet" in lowered:
        return "datasheet"
    if "release_notes" in lowered or "release-notes" in lowered:
        return "release_notes"
    if "tutorial" in lowered or "quick_start" in lowered or "quick-start" in lowered:
        return "tutorial"
    if "manual" in lowered or "_man_" in lowered:
        return "manual"
    if lowered.endswith(".pdf"):
        return "manual"
    return normalize_doc_type(None)


def infer_device_model(filename: str, vendor: str, vendor_keywords: dict) -> str | None:
    lowered = filename.lower()
    for family in vendor_keywords.get("device_families", []):
        for model in family.get("models", []):
            if str(model).lower() in lowered:
                return str(model)
    match = re.search(r"\b([a-z]{0,3}\d{3,5}[a-z]?)\b", lowered)
    if match and vendor in ("lmi", "pekat", "mechmind"):
        return match.group(1)
    return None


def derive_html_metadata(vendor: str, url: str, config: dict) -> SourceMetadata:
    vendor_keywords = config.get("keywords", {}).get(vendor, {})
    category = str(vendor_keywords.get("category") or vendor)
    device_family = None
    for family in vendor_keywords.get("device_families", []):
        keywords = [str(k).lower() for k in family.get("keywords", [])]
        if any(kw in url.lower() for kw in keywords):
            device_family = str(family.get("family") or family.get("name") or "")
            break
    return SourceMetadata(
        vendor=vendor,
        category=category,
        device_family=device_family or None,
        device_model=None,
        doc_type="article",
        doc_version=None,
        language=infer_language(url),
    )


def derive_metadata(pdf: Path, vendor: str, config: dict) -> SourceMetadata:
    from providers.metadata import (
        infer_product_line_from_filename,
        infer_software_version_from_filename,
    )

    vendor_keywords = config.get("keywords", {}).get(vendor, {})
    category = str(vendor_keywords.get("category") or vendor)
    device_family: str | None = None
    device_model: str | None = None

    for family in vendor_keywords.get("device_families", []):
        keywords = [str(k).lower() for k in family.get("keywords", [])]
        if any(kw in pdf.name.lower() for kw in keywords):
            device_family = str(family.get("family") or family.get("name") or "")
            device_model = infer_device_model(pdf.name, vendor, {"device_families": [family]})
            break

    if not device_family:
        line = infer_product_line_from_filename(vendor, pdf.name)
        device_family = line or None

    if not device_model:
        device_model = infer_device_model(pdf.name, vendor, vendor_keywords)

    doc_version = infer_software_version_from_filename(pdf.name) or None
    return SourceMetadata(
        vendor=vendor,
        category=category,
        device_family=device_family,
        device_model=device_model,
        doc_type=infer_doc_type(pdf.name),
        doc_version=doc_version,
        language=infer_language(pdf.name),
    )


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _model_available(tags: set[str], model: str) -> bool:
    if model in tags:
        return True
    base = model.split(":")[0]
    return any(tag == base or tag.startswith(f"{base}:") for tag in tags)


def check_ollama_up() -> dict[str, list[str]]:
    try:
        response = requests.get(f"{_ollama_base_url()}/api/tags", timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Preflight failed: Ollama is not reachable at {_ollama_base_url()} ({exc})") from exc
    models = response.json().get("models", [])
    tags = {m.get("name", "") for m in models if m.get("name")}
    required = [
        ("OLLAMA_LLM_MODEL", os.getenv("OLLAMA_LLM_MODEL", "")),
        ("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")),
        ("OLLAMA_VISION_MODEL", os.getenv("OLLAMA_VISION_MODEL", "moondream")),
    ]
    missing: list[str] = []
    for label, model in required:
        if not model:
            missing.append(f"{label} is not set")
            continue
        if not _model_available(tags, model):
            missing.append(f"{label}={model!r} not found in Ollama /api/tags")
    if missing:
        raise SystemExit("Preflight failed: missing Ollama models:\n  - " + "\n  - ".join(missing))
    return {"tags": sorted(tags)}


def _qdrant_client() -> QdrantClient:
    kwargs: dict[str, Any] = {"url": os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")}
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if api_key:
        kwargs["api_key"] = api_key
    grpc_port = os.getenv("QDRANT_GRPC_PORT", "").strip()
    if grpc_port:
        kwargs["grpc_port"] = int(grpc_port)
    return QdrantClient(**kwargs)


def check_qdrant_collection(embed_dim: int) -> None:
    from providers.qdrant_store import collection_vector_size

    collection = os.getenv("QDRANT_COLLECTION", "rag_docs")
    try:
        client = _qdrant_client()
        client.get_collections()
    except Exception as exc:
        raise SystemExit(f"Preflight failed: Qdrant is not reachable ({exc})") from exc

    existing = collection_vector_size(client, collection)
    if existing is None:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )
        logger.info("Created collection %s with dimension %s", collection, embed_dim)
        return
    if existing != embed_dim:
        raise SystemExit(
            f"Preflight failed: collection {collection!r} has dimension {existing}, "
            f"but the embedder produces {embed_dim}. Delete the collection and rerun."
        )
    logger.info("Collection %s exists with matching dimension %s", collection, embed_dim)


def _text_points_filter(source: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="source", match=MatchValue(value=source)),
            FieldCondition(key="content_type", match=MatchValue(value="text")),
        ]
    )


def _store_collection_exists(store) -> bool:
    from providers.qdrant_store import collection_vector_size

    client = getattr(store, "client", None)
    collection = getattr(store, "collection", None)
    if client is None or not collection:
        return False
    try:
        return collection_vector_size(client, collection) is not None
    except Exception:
        return False


def _source_has_text_points(store, source_filename: str) -> bool:
    """Return True when at least one ``content_type=text`` point exists for ``source``."""
    client = getattr(store, "client", None)
    collection = getattr(store, "collection", None)
    if client is None or not collection:
        return False
    if not _store_collection_exists(store):
        return False
    try:
        return (
            client.count(
                collection_name=collection,
                count_filter=_text_points_filter(source_filename),
            ).count
            >= 1
        )
    except Exception as exc:
        logger.warning(
            "Could not verify text points for %s: %s — treating as missing",
            source_filename,
            exc,
        )
        return False


def _pending_pdfs_for_text_ingest(
    pdfs: list[tuple[str, Path]],
    completed_pdfs: set[str],
    *,
    store,
    verify_completed: bool,
) -> list[tuple[str, Path]]:
    """Build phase-1 PDF work list, re-queuing stale completed entries when needed."""
    pending = [(vendor, pdf) for vendor, pdf in pdfs if pdf.name not in completed_pdfs]
    if not verify_completed:
        return pending

    if not _store_collection_exists(store):
        for _vendor, pdf in pdfs:
            if pdf.name in completed_pdfs:
                logger.warning(
                    "Source %s marked completed but Qdrant collection is missing — re-queuing",
                    pdf.name,
                )
        return list(pdfs)

    for vendor, pdf in pdfs:
        if pdf.name not in completed_pdfs:
            continue
        if _source_has_text_points(store, pdf.name):
            continue
        logger.warning(
            "Source %s marked completed but has no text points in Qdrant — re-queuing",
            pdf.name,
        )
        pending.append((vendor, pdf))
    return pending


def flush_semantic_cache() -> None:
    try:
        import redis

        client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), socket_timeout=3)
        deleted = client.delete("semantic_cache:entries")
        logger.info("Flushed Redis semantic_cache:entries (deleted=%s)", deleted)
    except Exception as exc:
        logger.warning("Redis flush skipped: %s", exc)


def probe_embedder() -> tuple[Any, int, str]:
    from providers.factory import get_embedder, reset_providers

    reset_providers()
    embedder = get_embedder()
    model = getattr(embedder, "model", os.getenv("OLLAMA_EMBED_MODEL", ""))
    vectors = embedder.embed(["dimension probe"])
    dim = len(vectors[0])
    os.environ["QDRANT_VECTOR_SIZE"] = str(dim)
    return embedder, dim, model


def patch_embedder_keep_alive(embedder, keep_alive: str = "120m") -> None:
    if not hasattr(embedder, "_request_embed"):
        return
    original = embedder._request_embed

    def _patched(client, batch):
        response = client.post(
            f"{embedder.base_url}/api/embed",
            json={
                "model": embedder.model,
                "input": batch,
                "keep_alive": keep_alive,
            },
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    embedder._request_embed = _patched  # type: ignore[method-assign]


@contextmanager
def extended_v2_payload(extra_by_source: dict[str, dict]):
    """Temporarily merge per-source metadata into ``build_v2_payload`` during ingest."""
    import scripts.ingest.ingest as ingest_mod

    original = ingest_mod.build_v2_payload

    def _wrapped(item, ctx):
        payload = original(item, ctx)
        payload.update(extra_by_source.get(ctx.source, {}))
        return payload

    ingest_mod.build_v2_payload = _wrapped
    try:
        yield
    finally:
        ingest_mod.build_v2_payload = original


@contextmanager
def html_source_as_url():
    """Use the article URL as Qdrant ``source`` for HTML ingests."""
    import scripts.ingest.ingest as ingest_mod

    original = ingest_mod._build_ingest_context

    def _wrapped(file_path: str, **kwargs):
        ctx = original(file_path, **kwargs)
        page_url = (kwargs.get("url") or "").strip()
        if page_url and str(file_path).lower().endswith((".html", ".htm")):
            return ingest_mod.IngestContext(
                source=page_url,
                vendor=ctx.vendor,
                product=ctx.product,
                product_version=ctx.product_version,
                doc_type=ctx.doc_type,
                url=page_url,
                ingested_at=ctx.ingested_at,
            )
        return ctx

    ingest_mod._build_ingest_context = _wrapped
    try:
        yield
    finally:
        ingest_mod._build_ingest_context = original


@contextmanager
def defer_html_media_collection():
    """Skip pending_captions/videos sidecars during HTML text ingest (phase 2 collects)."""
    import scripts.ingest.ingest as ingest_mod

    original_captions = ingest_mod._append_pending_captions
    original_videos = ingest_mod._append_pending_videos
    ingest_mod._append_pending_captions = lambda *args, **kwargs: 0
    ingest_mod._append_pending_videos = lambda *args, **kwargs: None
    try:
        yield
    finally:
        ingest_mod._append_pending_captions = original_captions
        ingest_mod._append_pending_videos = original_videos


def fetch_html_with_retry(url: str, *, attempts: int = 2, backoff: float = 5.0) -> str:
    from scripts.ingest.ingest import _fetch_url_html

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return _fetch_url_html(url)
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def run_dry_run(vendors: list[str], config: dict) -> int:
    pdfs = list_pdfs(vendors)
    urls = list_html_urls(vendors)
    if not pdfs and not urls:
        print("No PDFs or HTML URLs found.")
        return 1

    print(
        f"DRY RUN — {len(pdfs)} PDF(s), {len(urls)} HTML URL(s) "
        f"across {len(vendors)} vendor(s)\n"
    )
    if pdfs:
        print("## PDFs\n")
        for vendor, pdf in pdfs:
            meta = derive_metadata(pdf, vendor, config)
            print(
                f"{pdf.name}\n"
                f"  vendor={meta.vendor} source_type=pdf category={meta.category} "
                f"device_family={meta.device_family} device_model={meta.device_model}\n"
                f"  doc_type={meta.doc_type} doc_version={meta.doc_version} language={meta.language}\n"
            )
    if urls:
        print("## HTML URLs\n")
        for vendor, url, list_path in urls:
            meta = derive_html_metadata(vendor, url, config)
            print(
                f"{url}\n"
                f"  list={list_path.as_posix()} vendor={meta.vendor} source_type=html "
                f"category={meta.category} device_family={meta.device_family}\n"
                f"  doc_type={meta.doc_type} language={meta.language}\n"
            )
    print(f"TOTAL: {len(pdfs)} PDF(s), {len(urls)} HTML URL(s)")
    return 0


def _migrate_image_extract_state(state: dict) -> None:
    legacy = state.get("image_extract_completed_vendors")
    if legacy and "pdf_image_extract_completed_vendors" not in state:
        state["pdf_image_extract_completed_vendors"] = legacy


def ingest_html_url(
    url: str,
    *,
    vendor: str,
    config: dict,
    extra_by_source: dict[str, dict],
    stats: RunStats,
) -> bool:
    """Fetch + ingest one HTML URL. Returns True on success."""
    import scripts.ingest.ingest as ingest_mod
    from providers.factory import get_vector_store

    meta = derive_html_metadata(vendor, url, config)
    filename = ingest_mod._filename_from_url(url, vendor)
    dest_path = Path("data") / vendor / filename
    extra_by_source[url] = meta.payload_extra(source_type="html")

    store = get_vector_store()
    removed = store.delete_by_source(url)
    if removed:
        logger.info("Removed %s existing chunks for HTML source %s", removed, url)
    legacy_removed = store.delete_by_source(filename)
    if legacy_removed:
        logger.info("Removed %s legacy chunks for %s", legacy_removed, filename)

    try:
        html = fetch_html_with_retry(url)
    except Exception as exc:
        msg = str(exc)
        logger.error("HTML fetch failed for %s: %s", url, msg)
        stats.failed_urls.append({"url": url, "vendor": vendor, "error": msg})
        stats.errors.append({"phase": "1", "source": url, "error": f"fetch: {msg}"})
        return False

    try:
        extracted = ingest_mod._extract_html_clean_text_from_raw(html)
        if len(extracted) < ingest_mod.MIN_URL_EXTRACTED_CHARS:
            msg = "extracted text too short (may need JS rendering)"
            logger.error("HTML ingest skipped for %s: %s", url, msg)
            stats.failed_urls.append({"url": url, "vendor": vendor, "error": msg})
            stats.errors.append({"phase": "1", "source": url, "error": msg})
            return False

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(html, encoding="utf-8")

        kwargs = meta.ingest_kwargs()
        kwargs["doc_type"] = "article"
        chunks = ingest_mod.ingest(
            str(dest_path.resolve()),
            force=True,
            url=url,
            **kwargs,
        )
        stats.html_chunks += chunks
        stats.text_chunks += chunks
        ingest_mod.upsert_manifest_source(
            url,
            file_hash=ingest_mod._hash_text_content(extracted),
            vendor=vendor,
            product=kwargs.get("product"),
            chunk_count=chunks,
            extra={"content_type": "text", "source_type": "html"},
        )
        return True
    except Exception as exc:
        msg = str(exc)
        logger.error("HTML ingest failed for %s: %s", url, msg)
        stats.failed_urls.append({"url": url, "vendor": vendor, "error": msg})
        stats.errors.append({"phase": "1", "source": url, "error": msg})
        return False


def phase_text_ingest(
    vendors: list[str],
    config: dict,
    state: dict,
    stats: RunStats,
    embedder,
    *,
    verify_completed: bool = True,
) -> None:
    import scripts.ingest.ingest as ingest_mod

    from providers.factory import get_vector_store, reset_providers

    completed_pdfs = set(state.get("text_ingest_completed", []))
    completed_html = set(state.get("html_ingest_completed", []))
    extra_by_source: dict[str, dict] = {}
    pdfs = list_pdfs(vendors)
    store = get_vector_store()
    pending_pdfs = _pending_pdfs_for_text_ingest(
        pdfs,
        completed_pdfs,
        store=store,
        verify_completed=verify_completed,
    )
    pending_html = [
        (v, url, lp)
        for v, url, lp in list_html_urls(vendors)
        if url not in completed_html
    ]

    if not pending_pdfs and not pending_html:
        logger.info("Phase 1: all PDF and HTML sources already ingested — skipping")
        return

    os.environ["MULTIMODAL_IMAGE_INGEST_ENABLED"] = "false"
    patch_embedder_keep_alive(embedder, "120m")
    reset_providers()

    with extended_v2_payload(extra_by_source), html_source_as_url(), defer_html_media_collection():
        if pending_pdfs:
            with tqdm(pending_pdfs, desc="Phase 1 PDF ingest", unit="pdf") as bar:
                for vendor, pdf in bar:
                    bar.set_postfix_str(pdf.name[:40])
                    meta = derive_metadata(pdf, vendor, config)
                    extra_by_source[pdf.name] = meta.payload_extra(source_type="pdf")
                    try:
                        chunks = ingest_mod.ingest(
                            str(pdf.resolve()), force=True, **meta.ingest_kwargs()
                        )
                        stats.text_chunks += chunks
                        completed_pdfs.add(pdf.name)
                        state["text_ingest_completed"] = sorted(completed_pdfs)
                        save_state_atomic(state)
                    except Exception as exc:
                        msg = str(exc)
                        logger.error("PDF ingest failed for %s: %s", pdf.name, msg)
                        stats.errors.append(
                            {"phase": "1", "source": pdf.name, "error": msg}
                        )
        else:
            logger.info("Phase 1 PDFs: all completed")

        if pending_html:
            with tqdm(pending_html, desc="Phase 1 HTML ingest", unit="url") as bar:
                for vendor, url, _list_path in bar:
                    bar.set_postfix_str(url[-48:])
                    if ingest_html_url(
                        url,
                        vendor=vendor,
                        config=config,
                        extra_by_source=extra_by_source,
                        stats=stats,
                    ):
                        completed_html.add(url)
                        state["html_ingest_completed"] = sorted(completed_html)
                        save_state_atomic(state)
        else:
            logger.info("Phase 1 HTML: all completed")


def collect_html_images_for_vendor(vendor: str, stats: RunStats) -> int:
    """Run ingest's HTML image collector on saved article HTML files."""
    from bs4 import BeautifulSoup
    from scripts.ingest.ingest import (
        _append_pending_captions,
        _collect_html_images,
        _filename_from_url,
        _load_url_list,
    )

    collected = 0
    for list_path in discover_url_lists(vendor):
        for url in _load_url_list(str(list_path)):
            filename = _filename_from_url(url, vendor)
            html_path = Path("data") / vendor / filename
            if not html_path.is_file():
                continue
            try:
                raw = html_path.read_text(encoding="utf-8", errors="replace")
                soup = BeautifulSoup(raw, "lxml")
                images = _collect_html_images(
                    soup,
                    source=url,
                    vendor=vendor,
                    page_url=url,
                )
                _append_pending_captions(vendor, images)
                collected += len(images)
            except Exception as exc:
                msg = str(exc)
                logger.error("HTML image collect failed for %s: %s", url, msg)
                stats.errors.append(
                    {"phase": "2", "source": url, "error": f"html images: {msg}"}
                )
    return collected


def phase_image_extract(vendors: list[str], state: dict, stats: RunStats) -> None:
    from scripts.ingest.extract_pdf_images import extract_pdf_images

    _migrate_image_extract_state(state)
    pdf_done = set(state.get("pdf_image_extract_completed_vendors", []))
    html_done = set(state.get("html_image_collect_completed_vendors", []))

    for vendor in vendors:
        if vendor not in pdf_done:
            folder = Path("data") / vendor
            pdfs = sorted(folder.glob("*.pdf"))
            for pdf in tqdm(pdfs, desc=f"Phase 2 PDF images {vendor}", unit="pdf"):
                try:
                    counts, _ = extract_pdf_images(pdf)
                    stats.images_extracted += counts.get("extracted", 0)
                except Exception as exc:
                    msg = str(exc)
                    logger.error("Image extract failed for %s: %s", pdf.name, msg)
                    stats.errors.append(
                        {"phase": "2", "source": pdf.name, "error": msg}
                    )
            pdf_done.add(vendor)
            state["pdf_image_extract_completed_vendors"] = sorted(pdf_done)
            save_state_atomic(state)

        if vendor not in html_done and discover_url_lists(vendor):
            try:
                added = collect_html_images_for_vendor(vendor, stats)
                stats.html_images_collected += added
            except Exception as exc:
                msg = str(exc)
                logger.error("HTML image collection failed for %s: %s", vendor, msg)
                stats.errors.append(
                    {"phase": "2", "source": vendor, "error": f"html collect: {msg}"}
                )
            html_done.add(vendor)
            state["html_image_collect_completed_vendors"] = sorted(html_done)
            save_state_atomic(state)


def _caption_entry_needs_work(entry: dict) -> bool:
    captioned = entry.get("captioned")
    if captioned is True:
        return False
    if (entry.get("caption") or "").strip():
        return True
    return captioned is False or captioned is None or captioned == "failed"


def fetch_extended_source_metadata(
    store,
    *,
    source: str,
    vendor: str,
    cache: dict[str, dict],
) -> dict:
    if source in cache:
        return cache[source]

    from scripts.ingest.caption_worker import _collection_for_vendor, _qdrant_client_from_store
    from qdrant_client.models import FieldCondition, Filter, MatchValue

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
        "product_version": payload.get("product_version") or payload.get("doc_version"),
        "doc_type": payload.get("doc_type") or payload.get("document_type") or "other",
        "category": payload.get("category"),
        "device_family": payload.get("device_family") or payload.get("product"),
        "device_model": payload.get("device_model"),
        "doc_version": payload.get("doc_version") or payload.get("product_version"),
        "language": payload.get("language"),
    }
    cache[source] = meta
    return meta


def process_caption_entry_reingest(
    entry: dict,
    *,
    vendor: str,
    vision,
    embedder,
    store,
    meta_cache: dict[str, dict],
) -> str:
    from scripts.ingest.caption_worker import (
        _utc_now as caption_utc_now,
        build_caption_prompt,
        build_payload,
        load_image_bytes,
        make_point_id,
        reject_caption,
        sanitize_caption,
        truncate_caption,
    )
    from providers.caption_images import should_skip_image_src

    image_src = (entry.get("image_src") or "").strip()
    source = (entry.get("source") or "").strip()
    if not image_src or not source:
        return "skipped"

    skip_reason = should_skip_image_src(image_src)
    if skip_reason:
        logger.info("Caption skipped (%s): %s", image_src, skip_reason)
        return "skipped"

    existing_text = (entry.get("caption") or "").strip()
    use_existing = bool(existing_text) and entry.get("captioned") != "failed"

    if use_existing:
        caption = truncate_caption(sanitize_caption(existing_text))
    else:
        if entry.get("captioned") is True:
            return "skipped"
        image_bytes, image_path, err = load_image_bytes(entry, vendor)
        if err or image_bytes is None or image_path is None:
            logger.warning("Image load failed (%s): %s", image_src, err)
            return "failed"
        try:
            raw_caption = vision.describe_image(build_caption_prompt(entry), image_bytes)
        except Exception as exc:
            logger.warning("Vision caption failed (%s): %s", image_src, exc)
            return "failed"
        caption = truncate_caption(sanitize_caption(raw_caption))

    reject_reason = reject_caption(caption)
    if reject_reason:
        logger.warning("Caption rejected (%s): %s — %r", source, reject_reason, caption)
        return "failed"

    _, image_path, err = load_image_bytes(entry, vendor)
    if err or image_path is None:
        logger.warning("Image path missing for upsert (%s): %s", image_src, err)
        return "failed"

    meta = fetch_extended_source_metadata(
        store, source=source, vendor=vendor, cache=meta_cache
    )
    payload = build_payload(
        entry,
        caption=caption,
        meta=meta,
        image_path=image_path,
        ingested_at=caption_utc_now(),
    )
    for key in ("category", "device_family", "device_model", "doc_version", "language"):
        value = meta.get(key)
        if value is not None:
            payload[key] = value

    point_id = make_point_id(source, image_src)
    vector = embedder.embed([payload["text"]])[0]
    store.upsert([point_id], [vector], [payload])
    entry["caption"] = caption
    return "reembedded" if use_existing else "captioned"


def phase_captions(
    vendors: list[str],
    state: dict,
    stats: RunStats,
    embedder,
) -> None:
    from scripts.ingest.caption_worker import (
        CHECKPOINT_EVERY,
        discover_queues,
        load_queue,
        save_queue_atomic,
    )
    from providers.factory import get_vector_store, reset_providers
    from providers.ollama_vision import OllamaVision

    reset_providers()
    store = get_vector_store()
    store.ensure_payload_indexes()
    patch_embedder_keep_alive(embedder, "120m")
    vision = OllamaVision()

    done_vendors = set(state.get("caption_completed_vendors", []))
    for vendor, path in discover_queues():
        if vendor not in vendors or vendor in done_vendors:
            continue
        entries = load_queue(path)
        if not entries:
            done_vendors.add(vendor)
            state["caption_completed_vendors"] = sorted(done_vendors)
            save_state_atomic(state)
            continue

        indices = [i for i, e in enumerate(entries) if _caption_entry_needs_work(e)]
        meta_cache: dict[str, dict] = {}
        dirty = 0

        for idx in tqdm(indices, desc=f"Phase 3 caption {vendor}", unit="img"):
            outcome = process_caption_entry_reingest(
                entries[idx],
                vendor=vendor,
                vision=vision,
                embedder=embedder,
                store=store,
                meta_cache=meta_cache,
            )
            if outcome == "captioned":
                entries[idx]["captioned"] = True
                stats.captions_new += 1
                dirty += 1
            elif outcome == "reembedded":
                entries[idx]["captioned"] = True
                stats.captions_reembedded += 1
                dirty += 1
            elif outcome == "failed":
                entries[idx]["captioned"] = "failed"
                stats.captions_failed += 1
                dirty += 1
            elif outcome == "skipped":
                entries[idx]["captioned"] = "skipped"
                dirty += 1
            if dirty >= CHECKPOINT_EVERY:
                save_queue_atomic(path, entries)
                dirty = 0

        if dirty:
            save_queue_atomic(path, entries)
        done_vendors.add(vendor)
        state["caption_completed_vendors"] = sorted(done_vendors)
        save_state_atomic(state)


def write_report(
    run_ts: str,
    stats: RunStats,
    vendors: list[str],
    log_path: Path,
    *,
    skip_captions: bool,
) -> Path:
    # Explicit package import; bare ``from audit import`` only worked via accidental cwd.
    from scripts.ops.audit import (
        _aggregate,
        _build_warnings,
        _connect,
        _load_env,
        _scroll_payloads,
    )

    url, collection = _load_env()
    client = _connect(url)
    payloads = _scroll_payloads(client, collection)
    audit = _aggregate(payloads, collection)
    audit.warnings = _build_warnings(audit, None)

    vendor_rows: dict[str, Counter] = defaultdict(Counter)
    vendor_source_type: dict[str, Counter] = defaultdict(Counter)
    failed_captions = stats.captions_failed
    for payload in payloads:
        vendor = (payload.get("vendor") or "unknown").strip().lower()
        if vendors and vendor not in vendors:
            continue
        ct = payload.get("content_type") or "unknown"
        vendor_rows[vendor][str(ct)] += 1
        if ct == "text":
            st = str(payload.get("source_type") or "unknown")
            vendor_source_type[vendor][st] += 1

    total_points = sum(audit.vendor_chunks.get(v, 0) for v in vendors) if vendors else audit.total_chunks
    report_path = Path(f"reingest_report_{run_ts}.md")
    lines = [
        f"# Re-ingest report ({run_ts})",
        "",
        f"- Log file: `{log_path.as_posix()}`",
        f"- Collection: `{collection}`",
        f"- Embedder: `{stats.embed_model}` (dimension {stats.embed_dim})",
        f"- Vendors: {', '.join(vendors) if vendors else 'all'}",
        f"- Captions skipped: {'yes' if skip_captions else 'no'}",
        "",
        "## Phase timings (seconds)",
        "",
    ]
    for phase, seconds in stats.phase_seconds.items():
        lines.append(f"- {phase}: {seconds:.1f}s")
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Text chunks ingested this run: {stats.text_chunks} "
            f"(pdf+html; html-only={stats.html_chunks})",
            f"- PDF images extracted: {stats.images_extracted}",
            f"- HTML images queued: {stats.html_images_collected}",
            f"- New captions: {stats.captions_new}",
            f"- Re-embedded captions: {stats.captions_reembedded}",
            f"- Failed captions this run: {failed_captions}",
            f"- Qdrant points (selected vendors): {total_points}",
            "",
            "## Per-vendor content types",
            "",
            "| vendor | text | image_caption | other |",
            "|--------|------|---------------|-------|",
        ]
    )
    for vendor in sorted(vendor_rows.keys()):
        counts = vendor_rows[vendor]
        text_n = counts.get("text", 0)
        cap_n = counts.get("image_caption", 0)
        other = sum(v for k, v in counts.items() if k not in ("text", "image_caption"))
        lines.append(f"| {vendor} | {text_n} | {cap_n} | {other} |")

    lines.extend(
        [
            "",
            "## Per-vendor text chunks by source_type",
            "",
            "| vendor | pdf | html | other |",
            "|--------|-----|------|-------|",
        ]
    )
    for vendor in sorted(vendor_source_type.keys()):
        counts = vendor_source_type[vendor]
        pdf_n = counts.get("pdf", 0)
        html_n = counts.get("html", 0)
        other = sum(v for k, v in counts.items() if k not in ("pdf", "html"))
        lines.append(f"| {vendor} | {pdf_n} | {html_n} | {other} |")

    if stats.failed_urls:
        lines.extend(["", "## Failed HTML URLs", ""])
        for row in stats.failed_urls:
            lines.append(f"- `{row['url']}` ({row['vendor']}): {row['error']}")

    if stats.errors:
        lines.extend(["", "## Errors", ""])
        for err in stats.errors:
            lines.append(f"- Phase {err['phase']} `{err['source']}`: {err['error']}")

    if audit.warnings:
        lines.extend(["", "## Audit warnings", ""])
        for warning in audit.warnings[:30]:
            lines.append(f"- {warning}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def init_state(run_ts: str, vendors: list[str], resume: bool) -> dict:
    """Load ``reingest_state.json`` when resuming; otherwise start a fresh phase checklist."""
    if resume and STATE_FILE.is_file():
        state = load_state()
        state.setdefault("text_ingest_completed", [])
        state.setdefault("html_ingest_completed", [])
        state.setdefault("pdf_image_extract_completed_vendors", [])
        state.setdefault("html_image_collect_completed_vendors", [])
        state.setdefault("caption_completed_vendors", [])
        _migrate_image_extract_state(state)
        state["resumed_at"] = _utc_now()
        return state
    return {
        "run_id": run_ts,
        "started_at": _utc_now(),
        "vendors": vendors,
        "text_ingest_completed": [],
        "html_ingest_completed": [],
        "pdf_image_extract_completed_vendors": [],
        "html_image_collect_completed_vendors": [],
        "caption_completed_vendors": [],
        "errors": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full corpus re-ingest (resumable)")
    parser.add_argument("--vendors", help="Comma-separated vendor subset (e.g. lmi,mechmind)")
    parser.add_argument("--skip-captions", action="store_true", help="Skip phase 3 captioning")
    parser.add_argument("--dry-run", action="store_true", help="List PDFs and metadata only")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore reingest_state.json and start a new run",
    )
    parser.add_argument(
        "--verify-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Re-queue PDFs marked completed in state when text points are missing "
            "from Qdrant (default: on)"
        ),
    )
    args = parser.parse_args(argv)

    run_ts = _run_timestamp()
    log_path = setup_logging(run_ts)
    logger.info("Re-ingest run %s — log %s", run_ts, log_path)

    config = load_vendors_config()
    vendors = resolve_vendors(args.vendors, config)
    if not vendors:
        logger.error("No vendors to process.")
        return 1

    if args.dry_run:
        return run_dry_run(vendors, config)

    t0 = time.perf_counter()
    check_ollama_up()
    embedder, embed_dim, embed_model = probe_embedder()
    stats = RunStats(embed_model=embed_model, embed_dim=embed_dim)
    check_qdrant_collection(embed_dim)
    flush_semantic_cache()
    stats.phase_seconds["preflight"] = time.perf_counter() - t0

    state = init_state(run_ts, vendors, resume=not args.fresh)
    save_state_atomic(state)

    t1 = time.perf_counter()
    phase_text_ingest(
        vendors,
        config,
        state,
        stats,
        embedder,
        verify_completed=args.verify_completed,
    )
    stats.phase_seconds["phase1_text"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    phase_image_extract(vendors, state, stats)
    stats.phase_seconds["phase2_images"] = time.perf_counter() - t2

    if not args.skip_captions:
        t3 = time.perf_counter()
        phase_captions(vendors, state, stats, embedder)
        stats.phase_seconds["phase3_captions"] = time.perf_counter() - t3
    else:
        stats.phase_seconds["phase3_captions"] = 0.0

    t4 = time.perf_counter()
    report_path = write_report(
        run_ts, stats, vendors, log_path, skip_captions=args.skip_captions
    )
    stats.phase_seconds["phase4_report"] = time.perf_counter() - t4

    state["completed_at"] = _utc_now()
    state["errors"] = stats.errors
    save_state_atomic(state)

    print(
        f"\nDone. text_chunks={stats.text_chunks} images={stats.images_extracted} "
        f"captions={stats.captions_new} reembedded={stats.captions_reembedded} "
        f"failed={stats.captions_failed}\nReport: {report_path}"
    )
    return 0 if not stats.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
