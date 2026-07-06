"""Ingest PDFs, HTML, and text into Qdrant. Run: ``python -m scripts.ingest.ingest``."""
from __future__ import annotations

import scripts._bootstrap  # noqa: F401 — project root on sys.path for direct script runs

import argparse
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import threading
import statistics
import uuid
from urllib.parse import urlparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue

ProgressCallback = Callable[[str, int, int], None]

import pymupdf as fitz
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from providers.doc_registry import get_doc_registry
from providers.caption_images import (
    is_skipped_caption_entry,
    is_skipped_caption_image_src,
    prepare_caption_queue_entries,
)
from providers.factory import get_embedder, get_vector_store
from providers.metadata import DocumentMetadata, make_chunk_id
from providers.transcript_glossary import normalize_transcript_text
from providers.vtt_transcript import (
    build_video_transcript_metadata,
    discover_vtt_files,
    find_info_json_for_vtt,
    group_vtt_cues_into_chunks,
    load_info_json,
    load_vendors_config,
    parse_vtt,
    parse_vtt_cues,
)
from providers.section_chunking import (
    is_section_aware_chunking_enabled,
    split_procedure_steps,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
MANIFEST_FILENAME = "ingest_manifest.json"
DOC_TYPES = (
    "manual",
    "datasheet",
    "tutorial",
    "release_notes",
    "article",
    "other",
)
DOC_TYPE_ALIASES = {
    "user_manual": "manual",
    "quick_start": "tutorial",
    "api_reference": "manual",
    "integration_guide": "tutorial",
    "troubleshooting": "other",
    "specification": "datasheet",
}
HTML_STRIP_TAGS = frozenset(
    {"script", "style", "nav", "header", "footer", "iframe", "noscript"}
)
HTML_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4"})
VIDEO_HOST_MARKERS = ("youtube.com", "youtu.be", "vimeo.com", "wistia.com")

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "512"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "500"))
CHUNK_QUEUE_SIZE = int(os.getenv("CHUNK_QUEUE_SIZE", str(EMBED_BATCH_SIZE * 4)))
SECTION_MAX_CHARS = 1500
SECTION_OVERLAP = 150
MIN_URL_EXTRACTED_CHARS = 200


@dataclass
class ChunkItem:
    text: str
    page: int | None
    section: str | None = None
    chunk_index: int = 0
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass
class IngestContext:
    source: str
    vendor: str
    product: str | None
    product_version: str | None
    doc_type: str
    url: str | None
    ingested_at: str
    content_type: str = "text"
    video_id: str | None = None
    source_type: str | None = None
    transcript_source: str | None = None
    language: str | None = None
    doc_version: str | None = None
    device_family: str | None = None
    point_id_seed: str | None = None


@dataclass
class _ChunkIndexState:
    per_page: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    global_index: int = 0


class _ChunkCounter:
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def add(self, amount: int = 1) -> None:
        with self._lock:
            self._count += amount

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


def compute_file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_path() -> Path:
    from providers.config_paths import INGEST_MANIFEST_PATH

    return INGEST_MANIFEST_PATH


def load_manifest() -> dict:
    path = manifest_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Malformed %s: %s", path, exc)
        return {}


def save_manifest(manifest: dict) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _manifest_entry(
    *,
    file_hash: str,
    vendor: str,
    product: str | None,
    chunk_count: int,
    extra: dict | None = None,
) -> dict:
    entry = {
        "sha256": file_hash,
        "vendor": vendor,
        "product": product,
        "chunk_count": chunk_count,
        "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        entry.update(extra)
    return entry


def upsert_manifest_source(
    source: str,
    *,
    file_hash: str,
    vendor: str,
    product: str | None,
    chunk_count: int,
    extra: dict | None = None,
) -> None:
    """Record or update one manifest row keyed by Qdrant ``source``."""
    manifest = load_manifest()
    manifest[source] = _manifest_entry(
        file_hash=file_hash,
        vendor=vendor,
        product=product,
        chunk_count=chunk_count,
        extra=extra,
    )
    save_manifest(manifest)


def _should_skip_manifest(
    filename: str,
    file_hash: str,
    manifest: dict,
    *,
    force: bool,
) -> bool:
    return (
        not force
        and filename in manifest
        and manifest[filename].get("sha256") == file_hash
    )


def _filename_from_url(url: str, vendor: str) -> str:
    """Derive a collision-proof local HTML filename from a page URL."""
    url_hash8 = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    segment = path.rsplit("/", 1)[-1] if path else ""
    if not path or not segment:
        slug = f"{vendor.lower()}-index"
    else:
        slug = segment.lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        if not slug:
            slug = f"{vendor.lower()}-index"
    return f"{slug}-{url_hash8}.html"


_DEFAULT_FETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _fetch_url_html(url: str) -> str:
    import requests

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": _DEFAULT_FETCH_USER_AGENT},
    )
    response.raise_for_status()
    return response.text


def _load_url_list(path: str) -> list[str]:
    urls: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        urls.append(cleaned)
    return urls


def _process_url_ingest(
    url: str,
    *,
    vendor: str,
    product: str | None,
    product_version: str | None,
    doc_type: str,
    force: bool,
) -> str:
    """Fetch a URL, save HTML under ``data/<vendor>/``, and ingest. Returns status."""
    filename = _filename_from_url(url, vendor)
    dest_path = Path("data") / vendor.lower() / filename
    manifest = load_manifest()

    if not force and dest_path.is_file():
        file_hash = compute_ingest_hash(str(dest_path))
        if _should_skip_manifest(filename, file_hash, manifest, force=force):
            print(f"Skipping {filename} (unchanged)")
            return "skipped"

    try:
        html = _fetch_url_html(url)
    except Exception as exc:
        logger.warning("Failed to fetch URL %s: %s", url, exc)
        return "failed"

    try:
        extracted = _extract_html_clean_text_from_raw(html)
        if len(extracted) < MIN_URL_EXTRACTED_CHARS:
            logger.warning(
                "URL %s: page may require JavaScript rendering or is blocked",
                url,
            )
            return "failed"

        file_hash = _hash_text_content(extracted)
        if _should_skip_manifest(filename, file_hash, manifest, force=force):
            print(f"Skipping {filename} (unchanged)")
            return "skipped"

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(html, encoding="utf-8")

        total = ingest(
            str(dest_path),
            vendor=vendor,
            product=product,
            product_version=product_version,
            doc_type=doc_type,
            url=url,
            force=force,
        )
        upsert_manifest_source(
            filename,
            file_hash=file_hash,
            vendor=vendor,
            product=product,
            chunk_count=total,
            extra={"content_type": "text", "source_type": "html"},
        )
        return "ingested"
    except Exception as exc:
        logger.warning("Failed to ingest URL %s: %s", url, exc)
        return "failed"


def infer_vendor_from_path(file_path: str) -> str | None:
    # Normalize backslashes up front so a Windows-style path resolves the same
    # on any OS — on Linux ``\`` is a literal filename char, not a separator,
    # so ``Path(...).as_posix()`` alone would not split it.
    parts = Path(file_path.replace("\\", "/")).as_posix().split("/")
    try:
        data_idx = parts.index("data")
    except ValueError:
        return None
    if data_idx + 2 < len(parts):
        return parts[data_idx + 1].lower()
    return None


def normalize_doc_type(value: str | None) -> str:
    if not value:
        return "manual"
    cleaned = value.strip().lower()
    if cleaned in DOC_TYPES:
        return cleaned
    return DOC_TYPE_ALIASES.get(cleaned, "other")


def lookup_url_from_sources(vendor: str, filename: str) -> str | None:
    sources_path = Path("data") / vendor / "sources.json"
    if not sources_path.is_file():
        return None
    try:
        data = json.loads(sources_path.read_text(encoding="utf-8"))
        url = data.get(filename)
        return str(url) if url else None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", sources_path, exc)
        return None


def resolve_ingest_url(
    vendor: str,
    filename: str,
    cli_url: str | None,
) -> str | None:
    if cli_url and cli_url.strip():
        return cli_url.strip()
    return lookup_url_from_sources(vendor, filename)


def make_point_id(source: str, page: int | None, chunk_index: int) -> str:
    page_key = "none" if page is None else str(page)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{page_key}|{chunk_index}"))


def build_v2_payload(item: ChunkItem, ctx: IngestContext) -> dict:
    section = item.section if item.section else None
    payload = {
        "text": item.text,
        "source": ctx.source,
        "page": item.page,
        "url": ctx.url,
        "vendor": ctx.vendor,
        "product": ctx.product,
        "product_version": ctx.product_version,
        "content_type": ctx.content_type,
        "doc_type": ctx.doc_type,
        "section": section,
        "timestamp": None,
        "schema_version": SCHEMA_VERSION,
        "ingested_at": ctx.ingested_at,
    }
    if ctx.language:
        payload["language"] = ctx.language
    if ctx.doc_version:
        payload["doc_version"] = ctx.doc_version
    if ctx.device_family:
        payload["device_family"] = ctx.device_family
    if ctx.video_id:
        payload["video_id"] = ctx.video_id
    if ctx.source_type:
        payload["source_type"] = ctx.source_type
    if ctx.transcript_source:
        payload["transcript_source"] = ctx.transcript_source
    if item.start_seconds is not None:
        payload["start_seconds"] = float(item.start_seconds)
    if item.end_seconds is not None:
        payload["end_seconds"] = float(item.end_seconds)
    return payload


def _assign_chunk_index(state: _ChunkIndexState, page: int | None) -> int:
    if page is None:
        idx = state.global_index
        state.global_index += 1
        return idx
    idx = state.per_page[page]
    state.per_page[page] += 1
    return idx


def _put_chunk(
    chunk_queue: Queue,
    item: ChunkItem,
    counter: _ChunkCounter | None,
) -> None:
    chunk_queue.put(item)
    if counter is not None:
        counter.add()


def _resolve_stage(
    producer: threading.Thread,
    embedder_thread: threading.Thread,
    producer_done: bool,
    chunks_stored: int,
) -> str:
    if not producer_done:
        if chunks_stored == 0:
            return "reading"
        return "embedding"
    if embedder_thread.is_alive():
        return "embedding"
    return "storing"


def _median_font_size(page_dict: dict) -> float | None:
    sizes: list[float] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0)
                if text and size > 0:
                    sizes.append(float(size))
    if not sizes:
        return None
    return float(statistics.median(sizes))


def _extract_page_segments(page: fitz.Page) -> list[tuple[str, bool]]:
    try:
        page_dict = page.get_text("dict")
        median = _median_font_size(page_dict)
        if median is None:
            plain = page.get_text().strip()
            return [(plain, False)] if plain else []

        segments: list[tuple[str, bool]] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                parts: list[str] = []
                line_max_size = 0.0
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text:
                        parts.append(text)
                        line_max_size = max(line_max_size, float(span.get("size", 0)))
                line_text = "".join(parts).strip()
                if not line_text:
                    continue
                segments.append((line_text, line_max_size > median))
        if segments:
            return segments
    except Exception:
        pass

    plain = page.get_text().strip()
    return [(plain, False)] if plain else []


def _build_pdf_sections(pdf: fitz.Document) -> list[tuple[str, str, int]]:
    sections: list[tuple[str, str, int]] = []
    current_heading = ""
    current_body: list[str] = []
    current_page = 0

    def flush() -> None:
        nonlocal current_heading, current_body
        body = "\n".join(current_body).strip()
        if body or current_heading:
            sections.append((current_heading, body, current_page))
        current_heading = ""
        current_body = []

    for page_num in range(pdf.page_count):
        for text, is_heading in _extract_page_segments(pdf[page_num]):
            if is_heading:
                if current_body or current_heading:
                    flush()
                current_heading = text
                current_page = page_num
            else:
                if not current_body and not current_heading:
                    current_page = page_num
                current_body.append(text)

    flush()
    return sections


def _html_section_root(soup):
    """Prefer Confluence ``.ak-renderer-document`` content; else full ``<body>``."""
    renderer = soup.select_one(".ak-renderer-document")
    if renderer is not None:
        return renderer
    return soup.body or soup


def _build_html_sections(soup) -> list[tuple[str | None, str]]:
    for tag_name in HTML_STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    body = _html_section_root(soup)
    title_text = ""
    if soup.title and soup.title.string:
        title_text = soup.title.string.strip()

    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = title_text or None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_body
        body_text = "\n\n".join(current_body).strip()
        if body_text or current_heading:
            sections.append((current_heading, body_text))
        current_body = []

    def walk(node) -> None:
        nonlocal current_heading
        from bs4 import NavigableString, Tag

        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                current_body.append(text)
            return
        if not isinstance(node, Tag):
            return
        if node.name in HTML_HEADING_TAGS:
            flush()
            current_heading = node.get_text(" ", strip=True)
            return
        if node.name in HTML_STRIP_TAGS or node.name == "img":
            return
        for child in node.children:
            walk(child)

    walk(body)
    flush()
    return sections


def _extract_html_clean_text_from_raw(raw_html: str) -> str:
    """Return cleaned extracted text from HTML (post script/style/nav removal)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "lxml")
    sections = _build_html_sections(soup)
    parts: list[str] = []
    for heading, body in sections:
        if heading and body:
            parts.append(f"{heading}\n\n{body}")
        elif heading:
            parts.append(heading)
        elif body:
            parts.append(body)
    return "\n\n".join(part.strip() for part in parts if part.strip())


def compute_ingest_hash(file_path: str) -> str:
    """Manifest hash: extracted text for HTML, raw bytes for other files."""
    path = Path(file_path)
    if path.suffix.lower() in (".html", ".htm"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        return _hash_text_content(_extract_html_clean_text_from_raw(raw))
    return compute_file_sha256(file_path)


def _purge_source_chunks(source: str) -> int:
    """Remove all Qdrant points (and registry row) for a source filename."""
    store = get_vector_store()
    removed = store.delete_by_source(source)
    if removed:
        get_doc_registry().delete(source)
        tqdm.write(f"Removed {removed} existing chunks for {source}")
    return removed


def _pending_captions_path(vendor: str) -> Path:
    return Path("data") / vendor.lower() / "pending_captions.json"


_CAPTION_MIN_DIMENSION = 80


def _parse_img_dimension(value) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    return int(match.group())


def _is_skipped_caption_image_src(src: str) -> bool:
    return is_skipped_caption_image_src(src)


def _is_skipped_caption_img_tag(node) -> bool:
    src = (node.get("src") or "").strip()
    if not src or src.lower().startswith("data:"):
        return True
    if _is_skipped_caption_image_src(src):
        return True
    for attr in ("width", "height"):
        dim = _parse_img_dimension(node.get(attr))
        if dim is not None and dim < _CAPTION_MIN_DIMENSION:
            return True
    return False


def _is_skipped_caption_entry(entry: dict) -> bool:
    return is_skipped_caption_entry(entry)


def _collect_html_images(
    soup,
    *,
    source: str,
    vendor: str,
    page_url: str | None,
) -> list[dict]:
    """Collect non-data-URI ``<img>`` tags with nearest preceding h1–h4 section."""
    from bs4 import Tag

    body = soup.body or soup
    current_heading: str | None = None
    entries: list[dict] = []

    def walk(node) -> None:
        nonlocal current_heading
        if not isinstance(node, Tag):
            return
        if node.name in HTML_HEADING_TAGS:
            heading = node.get_text(" ", strip=True)
            current_heading = heading or None
            return
        if node.name == "img":
            if _is_skipped_caption_img_tag(node):
                return
            src = (node.get("src") or "").strip()
            entries.append(
                {
                    "source": source,
                    "vendor": vendor,
                    "section": current_heading,
                    "image_src": src,
                    "url": page_url,
                    "captioned": False,
                }
            )
            return
        for child in node.children:
            walk(child)

    walk(body)
    return entries


def _append_pending_captions(vendor: str, new_entries: list[dict]) -> None:
    """Merge image refs into ``data/<vendor>/pending_captions.json`` (best-effort)."""
    try:
        path = _pending_captions_path(vendor)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        existing = [e for e in existing if not _is_skipped_caption_entry(e)]
        filtered_new = prepare_caption_queue_entries(new_entries, vendor)

        seen = {(e.get("source"), e.get("image_src")) for e in existing}
        for entry in filtered_new:
            key = (entry["source"], entry["image_src"])
            if key in seen:
                continue
            existing.append(entry)
            seen.add(key)

        existing = [e for e in existing if not _is_skipped_caption_entry(e)]

        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("Failed to write pending captions sidecar: %s", exc)


def _pending_videos_path(vendor: str) -> Path:
    return Path("data") / vendor.lower() / "pending_videos.json"


def _is_video_host_url(url: str) -> bool:
    lowered = url.lower()
    return any(marker in lowered for marker in VIDEO_HOST_MARKERS)


def _is_collectible_media_url(url: str) -> bool:
    cleaned = url.strip()
    return bool(cleaned) and not cleaned.lower().startswith("data:")


def _collect_html_videos(
    soup,
    *,
    source: str,
    vendor: str,
    page_url: str | None,
) -> list[dict]:
    """Collect embedded video refs with nearest preceding h1–h4 section."""
    from bs4 import Tag

    body = soup.body or soup
    current_heading: str | None = None
    entries: list[dict] = []

    def add_video(video_url: str) -> None:
        if not _is_collectible_media_url(video_url):
            return
        entries.append(
            {
                "source": source,
                "vendor": vendor,
                "section": current_heading,
                "video_url": video_url.strip(),
                "url": page_url,
                "transcribed": False,
            }
        )

    def walk(node) -> None:
        nonlocal current_heading
        if not isinstance(node, Tag):
            return
        if node.name in HTML_HEADING_TAGS:
            heading = node.get_text(" ", strip=True)
            current_heading = heading or None
            return
        if node.name == "video":
            src = (node.get("src") or "").strip()
            if src:
                add_video(src)
            for child in node.children:
                walk(child)
            return
        if node.name == "source":
            parent = node.parent
            if isinstance(parent, Tag) and parent.name == "video":
                src = (node.get("src") or "").strip()
                if src:
                    add_video(src)
            return
        if node.name == "iframe":
            src = (node.get("src") or "").strip()
            if src and _is_video_host_url(src):
                add_video(src)
            return
        if node.name == "a":
            href = (node.get("href") or "").strip()
            if href and _is_video_host_url(href):
                add_video(href)
            return
        for child in node.children:
            walk(child)

    walk(body)
    return entries


def _append_pending_videos(vendor: str, new_entries: list[dict]) -> None:
    """Merge video refs into ``data/<vendor>/pending_videos.json`` (best-effort)."""
    if not new_entries:
        return
    try:
        path = _pending_videos_path(vendor)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        seen = {(e.get("source"), e.get("video_url")) for e in existing}
        for entry in new_entries:
            key = (entry["source"], entry["video_url"])
            if key in seen:
                continue
            existing.append(entry)
            seen.add(key)

        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("Failed to write pending videos sidecar: %s", exc)


def _section_full_text(heading: str | None, body: str) -> str:
    if heading and body:
        return f"{heading}\n\n{body}"
    return heading or body


def _chunks_from_section(
    heading: str | None,
    body: str,
    page: int | None,
    splitter: RecursiveCharacterTextSplitter,
    chunk_queue: Queue,
    index_state: _ChunkIndexState,
    counter: _ChunkCounter | None = None,
) -> None:
    full_text = _section_full_text(heading, body).strip()
    if not full_text:
        return

    section_label = heading or None

    if len(full_text) <= SECTION_MAX_CHARS:
        _put_chunk(
            chunk_queue,
            ChunkItem(
                text=full_text,
                page=page,
                section=section_label,
                chunk_index=_assign_chunk_index(index_state, page),
            ),
            counter,
        )
        return

    if is_section_aware_chunking_enabled() and section_label:
        steps = split_procedure_steps(full_text)
        if len(steps) > 1:
            for step in steps:
                if len(step) <= SECTION_MAX_CHARS:
                    _put_chunk(
                        chunk_queue,
                        ChunkItem(
                            text=step,
                            page=page,
                            section=section_label,
                            chunk_index=_assign_chunk_index(index_state, page),
                        ),
                        counter,
                    )
                else:
                    doc = Document(
                        page_content=step,
                        metadata={"page": page, "section": section_label},
                    )
                    for chunk in splitter.split_documents([doc]):
                        content = chunk.page_content.strip()
                        if content:
                            _put_chunk(
                                chunk_queue,
                                ChunkItem(
                                    text=content,
                                    page=page,
                                    section=section_label,
                                    chunk_index=_assign_chunk_index(index_state, page),
                                ),
                                counter,
                            )
            return

    doc = Document(
        page_content=full_text,
        metadata={"page": page, "section": section_label},
    )
    for chunk in splitter.split_documents([doc]):
        content = chunk.page_content.strip()
        if not content:
            continue
        _put_chunk(
            chunk_queue,
            ChunkItem(
                text=content,
                page=page,
                section=section_label,
                chunk_index=_assign_chunk_index(index_state, page),
            ),
            counter,
        )


def _produce_pdf_chunks(
    file_path: str,
    chunk_queue: Queue,
    splitter: RecursiveCharacterTextSplitter,
    error: list,
    index_state: _ChunkIndexState,
    counter: _ChunkCounter | None = None,
) -> None:
    from providers.markdown_ingest import is_markdown_ingest_enabled

    if is_markdown_ingest_enabled():
        try:
            from providers.markdown_ingest import pdf_markdown_sections

            md_sections = pdf_markdown_sections(file_path)
        except ImportError as exc:
            logger.warning(
                "MARKDOWN_INGEST_ENABLED but pymupdf4llm unavailable (%s); "
                "falling back to legacy PDF extraction",
                exc,
            )
        else:
            for heading, body, page_num in tqdm(
                md_sections,
                desc="Read/split (md)",
                unit="section",
                position=0,
                leave=True,
            ):
                _chunks_from_section(
                    heading or None,
                    body,
                    page_num,
                    splitter,
                    chunk_queue,
                    index_state,
                    counter,
                )
            return

    with fitz.open(file_path) as pdf:
        sections = _build_pdf_sections(pdf)
        if not sections:
            for page_num in tqdm(
                range(pdf.page_count),
                desc="Read/split",
                unit="page",
                position=0,
                leave=True,
            ):
                text = pdf[page_num].get_text().strip()
                if not text:
                    continue
                _chunks_from_section(
                    None, text, page_num, splitter, chunk_queue, index_state, counter
                )
            return

        for heading, body, page_num in tqdm(
            sections,
            desc="Read/split",
            unit="section",
            position=0,
            leave=True,
        ):
            _chunks_from_section(
                heading or None,
                body,
                page_num,
                splitter,
                chunk_queue,
                index_state,
                counter,
            )


def _produce_html_chunks(
    file_path: str,
    chunk_queue: Queue,
    splitter: RecursiveCharacterTextSplitter,
    error: list,
    index_state: _ChunkIndexState,
    ctx: IngestContext,
    counter: _ChunkCounter | None = None,
) -> None:
    from bs4 import BeautifulSoup

    raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    from providers.markdown_ingest import is_markdown_ingest_enabled

    if is_markdown_ingest_enabled():
        from providers.markdown_ingest import html_markdown_sections

        sections = html_markdown_sections(raw)
    else:
        sections = _build_html_sections(soup)

    for heading, body in tqdm(
        sections,
        desc="Read/split",
        unit="section",
        position=0,
        leave=True,
    ):
        _chunks_from_section(
            heading,
            body,
            None,
            splitter,
            chunk_queue,
            index_state,
            counter,
        )

    try:
        images = _collect_html_images(
            soup,
            source=ctx.source,
            vendor=ctx.vendor,
            page_url=ctx.url,
        )
        _append_pending_captions(ctx.vendor, images)
    except Exception as exc:
        logger.warning("Failed to collect HTML images for pending captions: %s", exc)

    try:
        # Fresh parse so <iframe> tags remain (stripped during text extraction).
        video_soup = BeautifulSoup(raw, "lxml")
        videos = _collect_html_videos(
            video_soup,
            source=ctx.source,
            vendor=ctx.vendor,
            page_url=ctx.url,
        )
        _append_pending_videos(ctx.vendor, videos)
    except Exception as exc:
        logger.warning("Failed to collect HTML videos for pending videos: %s", exc)


def _load_text_documents(file_path: str) -> list[Document]:
    text = Path(file_path).read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": file_path})]


def _produce_text_chunks(
    file_path: str,
    chunk_queue: Queue,
    splitter: RecursiveCharacterTextSplitter,
    index_state: _ChunkIndexState,
    counter: _ChunkCounter | None = None,
) -> None:
    docs = _load_text_documents(file_path)
    for doc in tqdm(docs, desc="Read/split", unit="page", position=0):
        for chunk in splitter.split_documents([doc]):
            content = chunk.page_content.strip()
            if not content:
                continue
            _put_chunk(
                chunk_queue,
                ChunkItem(
                    text=content,
                    page=None,
                    section=None,
                    chunk_index=_assign_chunk_index(index_state, None),
                ),
                counter,
            )


def _produce_chunks(
    file_path: str,
    chunk_queue: Queue,
    splitter,
    error: list,
    index_state: _ChunkIndexState,
    counter: _ChunkCounter | None = None,
    ctx: IngestContext | None = None,
):
    try:
        lower = file_path.lower()
        if lower.endswith(".pdf"):
            _produce_pdf_chunks(
                file_path, chunk_queue, splitter, error, index_state, counter
            )
        elif lower.endswith(".html") or lower.endswith(".htm"):
            if ctx is None:
                raise ValueError("ingest context is required for HTML files")
            _produce_html_chunks(
                file_path,
                chunk_queue,
                splitter,
                error,
                index_state,
                ctx,
                counter,
            )
        else:
            _produce_text_chunks(
                file_path, chunk_queue, splitter, index_state, counter
            )
    except Exception as exc:
        error.append(exc)
    finally:
        chunk_queue.put(None)


def _embed_chunks(
    chunk_queue: Queue,
    result_queue: Queue,
    embedder,
    embed_pbar: tqdm,
    error: list,
    source: str,
):
    buffer: list[ChunkItem] = []
    try:
        while True:
            item = chunk_queue.get()
            if item is None:
                break
            buffer.append(item)
            if len(buffer) >= EMBED_BATCH_SIZE:
                _flush_embed(buffer, embedder, result_queue, embed_pbar, source)
                buffer = []
        if buffer:
            _flush_embed(buffer, embedder, result_queue, embed_pbar, source)
    except Exception as exc:
        error.append(exc)
    finally:
        result_queue.put(None)


def _flush_embed(
    buffer: list[ChunkItem],
    embedder,
    result_queue: Queue,
    embed_pbar: tqdm,
    source: str,
):
    texts = [item.text for item in buffer]
    embed_kwargs: dict = {}
    try:
        params = inspect.signature(embedder.embed).parameters
    except (TypeError, ValueError):
        params = {}
    if "sources" in params:
        embed_kwargs["sources"] = [source] * len(buffer)
    if "pages" in params:
        embed_kwargs["pages"] = [item.page for item in buffer]
    vectors = embedder.embed(texts, **embed_kwargs)
    for item, vector in zip(buffer, vectors):
        if vector is None:
            logger.warning(
                "Skipping chunk with no embedding (source=%s page=%s)",
                source,
                item.page,
            )
            continue
        result_queue.put((vector, item))
    embed_pbar.update(len(buffer))


def _store_results(
    result_queue: Queue,
    store,
    store_pbar: tqdm,
    error: list,
    ctx: IngestContext,
    progress_callback: ProgressCallback | None = None,
    chunk_counter: _ChunkCounter | None = None,
    producer: threading.Thread | None = None,
    embedder_thread: threading.Thread | None = None,
    producer_done: list[bool] | None = None,
) -> int:
    buffer: list[tuple] = []
    total = 0

    def report() -> None:
        if progress_callback is None:
            return
        chunks_produced = chunk_counter.count if chunk_counter else total
        producer_finished = producer_done[0] if producer_done else True
        chunks_total = chunks_produced if producer_finished else max(chunks_produced, total)
        stage = _resolve_stage(
            producer,
            embedder_thread,
            producer_finished,
            total,
        )
        progress_callback(stage, total, chunks_total)

    try:
        while True:
            item = result_queue.get()
            if item is None:
                break
            buffer.append(item)
            if len(buffer) >= UPSERT_BATCH_SIZE:
                total += _flush_store(buffer, store, ctx)
                store_pbar.update(len(buffer))
                buffer = []
                report()
        if buffer:
            total += _flush_store(buffer, store, ctx)
            store_pbar.update(len(buffer))
            report()
    except Exception as exc:
        error.append(exc)
    return total


def _flush_store(
    buffer: list[tuple],
    store,
    ctx: IngestContext,
) -> int:
    ids: list[str] = []
    vectors = [vector for vector, _ in buffer]
    payloads = []
    for _, item in buffer:
        payload = build_v2_payload(item, ctx)
        if ctx.point_id_seed is not None:
            if item.start_seconds is not None:
                ids.append(
                    str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{ctx.point_id_seed}|{item.start_seconds:.3f}|{item.chunk_index}",
                        )
                    )
                )
            else:
                ids.append(make_chunk_id(ctx.point_id_seed, item.chunk_index))
        else:
            ids.append(make_point_id(ctx.source, item.page, item.chunk_index))
        payloads.append(payload)
    store.upsert(ids, vectors, payloads)
    return len(buffer)


def _build_ingest_context(
    file_path: str,
    *,
    vendor: str | None = None,
    product: str | None = None,
    product_version: str | None = None,
    doc_type: str | None = None,
    url: str | None = None,
    document_type: str | None = None,
    product_line: str | None = None,
    software_version: str | None = None,
) -> IngestContext:
    path = Path(file_path)
    source = path.name
    resolved_vendor = (vendor or infer_vendor_from_path(file_path) or "").strip().lower()
    if not resolved_vendor:
        raise ValueError(
            "vendor is required (use --vendor or place the file under data/<vendor>/...)"
        )

    resolved_product = (product or product_line or "").strip().lower() or None
    resolved_version = (product_version or software_version or "").strip() or None
    resolved_doc_type = normalize_doc_type(doc_type or document_type)
    resolved_url = resolve_ingest_url(resolved_vendor, source, url)
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return IngestContext(
        source=source,
        vendor=resolved_vendor,
        product=resolved_product,
        product_version=resolved_version,
        doc_type=resolved_doc_type,
        url=resolved_url,
        ingested_at=ingested_at,
    )


def ingest(
    file_path: str,
    progress_callback: ProgressCallback | None = None,
    *,
    vendor: str | None = None,
    document_type: str | None = None,
    product_line: str | None = None,
    software_version: str | None = None,
    product: str | None = None,
    product_version: str | None = None,
    doc_type: str | None = None,
    url: str | None = None,
    force: bool = False,
):
    """Ingest one file (PDF/HTML/text/VTT); return chunk count or ``0`` when skipped."""
    ctx = _build_ingest_context(
        file_path,
        vendor=vendor,
        product=product,
        product_version=product_version,
        doc_type=doc_type,
        url=url,
        document_type=document_type,
        product_line=product_line,
        software_version=software_version,
    )

    store = get_vector_store()
    if hasattr(store, "ensure_payload_indexes"):
        store.ensure_payload_indexes()

    _purge_source_chunks(ctx.source)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SECTION_MAX_CHARS,
        chunk_overlap=SECTION_OVERLAP,
    )
    embedder = get_embedder()
    error: list[Exception] = []
    chunk_counter = _ChunkCounter() if progress_callback else None
    producer_done = [False]
    index_state = _ChunkIndexState()

    chunk_queue: Queue = Queue(maxsize=CHUNK_QUEUE_SIZE)
    result_queue: Queue = Queue()

    tqdm.write("Pipeline: read/split || embed || store (parallel)")
    embed_pbar = tqdm(desc="Embedding", unit="chunk", position=1, leave=True)
    store_pbar = tqdm(desc="Storing", unit="chunk", position=2, leave=True)

    def producer_target() -> None:
        try:
            _produce_chunks(
                file_path,
                chunk_queue,
                splitter,
                error,
                index_state,
                chunk_counter,
                ctx=ctx,
            )
        finally:
            producer_done[0] = True

    producer = threading.Thread(target=producer_target, daemon=True)
    embedder_thread = threading.Thread(
        target=_embed_chunks,
        args=(chunk_queue, result_queue, embedder, embed_pbar, error, ctx.source),
        daemon=True,
    )

    if progress_callback:
        progress_callback("reading", 0, 0)

    producer.start()
    embedder_thread.start()

    total = _store_results(
        result_queue,
        store,
        store_pbar,
        error,
        ctx,
        progress_callback=progress_callback,
        chunk_counter=chunk_counter,
        producer=producer,
        embedder_thread=embedder_thread,
        producer_done=producer_done,
    )

    producer.join()
    embedder_thread.join()
    embed_pbar.close()
    store_pbar.close()

    if error:
        raise error[0]

    if progress_callback:
        progress_callback("storing", total, total)

    content_hash = compute_ingest_hash(file_path)
    registry_meta = _registry_meta_from_context(ctx, file_path)
    registry_meta.content_hash = content_hash
    if total > 0:
        get_doc_registry().upsert(
            registry_meta,
            chunk_count=total,
            content_hash=content_hash,
        )

    if os.getenv("MULTIMODAL_IMAGE_INGEST_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    ) and file_path.lower().endswith(".pdf"):
        total += _ingest_pdf_images(
            file_path,
            registry_meta,
            store,
            chunk_index_start=total,
        )

    print(f"Ingested {total} chunks into vector store")
    return total


def _video_id_has_points(store, video_id: str) -> bool:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = getattr(store, "client", None)
    collection = getattr(store, "collection", None)
    if client is None or not collection:
        return False
    video_filter = Filter(
        must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
    )
    return client.count(collection_name=collection, count_filter=video_filter).count > 0


def _purge_video_id_chunks(store, video_id: str) -> int:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = getattr(store, "client", None)
    collection = getattr(store, "collection", None)
    if client is None or not collection:
        return 0
    video_filter = Filter(
        must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
    )
    count = client.count(collection_name=collection, count_filter=video_filter).count
    if count == 0:
        return 0
    client.delete(collection_name=collection, points_selector=video_filter)
    tqdm.write(f"Removed {count} existing chunks for video_id={video_id}")
    return count


def _produce_vtt_timestamped_chunks(
    windows: list,
    *,
    chunk_queue: Queue,
    index_state: _ChunkIndexState,
    chunk_counter: _ChunkCounter | None,
    first_section: str | None,
) -> None:
    for idx, window in enumerate(windows):
        cleaned = window.text.strip()
        if not cleaned:
            continue
        section = first_section if idx == 0 else None
        chunk_index = _assign_chunk_index(index_state, None)
        _put_chunk(
            chunk_queue,
            ChunkItem(
                text=cleaned,
                page=None,
                section=section,
                chunk_index=chunk_index,
                start_seconds=float(window.start_seconds),
                end_seconds=float(window.end_seconds),
            ),
            chunk_counter,
        )
    chunk_queue.put(None)


def _produce_split_text_chunks(
    text: str,
    *,
    chunk_queue: Queue,
    splitter: RecursiveCharacterTextSplitter,
    index_state: _ChunkIndexState,
    chunk_counter: _ChunkCounter | None,
    first_section: str | None,
) -> None:
    parts = splitter.split_text(text)
    for idx, part in enumerate(parts):
        cleaned = part.strip()
        if not cleaned:
            continue
        section = first_section if idx == 0 else None
        chunk_index = _assign_chunk_index(index_state, None)
        _put_chunk(
            chunk_queue,
            ChunkItem(
                text=cleaned,
                page=None,
                section=section,
                chunk_index=chunk_index,
            ),
            chunk_counter,
        )
    chunk_queue.put(None)


def _ingest_video_transcript_text(
    text: str | None,
    meta: dict,
    *,
    vtt_windows: list | None = None,
) -> int:
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ctx = IngestContext(
        source=meta["source"],
        vendor=meta["vendor"],
        product=meta.get("product"),
        product_version=meta.get("product_version"),
        doc_type=meta.get("doc_type", "tutorial"),
        url=meta.get("url"),
        ingested_at=ingested_at,
        content_type=meta.get("content_type", "video_transcript"),
        video_id=meta["video_id"],
        source_type=meta.get("source_type", "video"),
        transcript_source=meta.get("transcript_source"),
        language=meta.get("language"),
        doc_version=meta.get("doc_version"),
        device_family=meta.get("device_family"),
        point_id_seed=meta["video_id"],
    )

    store = get_vector_store()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SECTION_MAX_CHARS,
        chunk_overlap=SECTION_OVERLAP,
    )
    embedder = get_embedder()
    error: list[Exception] = []
    chunk_counter = _ChunkCounter()
    producer_done = [False]
    index_state = _ChunkIndexState()
    chunk_queue: Queue = Queue(maxsize=CHUNK_QUEUE_SIZE)
    result_queue: Queue = Queue()

    tqdm.write(f"Video transcript: {ctx.source} ({ctx.video_id})")
    embed_pbar = tqdm(desc="Embedding", unit="chunk", position=1, leave=True)
    store_pbar = tqdm(desc="Storing", unit="chunk", position=2, leave=True)

    def producer_target() -> None:
        try:
            if vtt_windows:
                _produce_vtt_timestamped_chunks(
                    vtt_windows,
                    chunk_queue=chunk_queue,
                    index_state=index_state,
                    chunk_counter=chunk_counter,
                    first_section=meta["source"],
                )
            else:
                _produce_split_text_chunks(
                    text or "",
                    chunk_queue=chunk_queue,
                    splitter=splitter,
                    index_state=index_state,
                    chunk_counter=chunk_counter,
                    first_section=meta["source"],
                )
        except Exception as exc:
            error.append(exc)
        finally:
            producer_done[0] = True

    producer = threading.Thread(target=producer_target, daemon=True)
    embedder_thread = threading.Thread(
        target=_embed_chunks,
        args=(chunk_queue, result_queue, embedder, embed_pbar, error, ctx.source),
        daemon=True,
    )

    producer.start()
    embedder_thread.start()

    total = _store_results(
        result_queue,
        store,
        store_pbar,
        error,
        ctx,
        chunk_counter=chunk_counter,
        producer=producer,
        embedder_thread=embedder_thread,
        producer_done=producer_done,
    )

    producer.join()
    embedder_thread.join()
    embed_pbar.close()
    store_pbar.close()

    if error:
        raise error[0]
    return total


def ingest_video_transcripts(
    video_dir: str,
    *,
    vendor: str | None = None,
    product: str | None = None,
    product_version: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Ingest yt-dlp ``.vtt`` transcripts from a directory."""
    store = get_vector_store()
    if hasattr(store, "ensure_payload_indexes"):
        store.ensure_payload_indexes()

    vendors_config = load_vendors_config()
    counts = {"ingested": 0, "skipped": 0, "failed": 0, "videos": 0}

    for vtt_path in discover_vtt_files(video_dir):
        info_path = find_info_json_for_vtt(vtt_path)
        if info_path is None:
            logger.warning("Skipping %s: missing sibling .info.json", vtt_path.name)
            counts["skipped"] += 1
            continue

        info = load_info_json(str(info_path))
        meta = build_video_transcript_metadata(
            info,
            vtt_path,
            cli_vendor=vendor,
            cli_product=product,
            cli_product_version=product_version,
            vendors_config=vendors_config,
        )
        if meta is None:
            logger.warning("Skipping %s: incomplete info.json metadata", vtt_path.name)
            counts["skipped"] += 1
            continue

        if not meta.get("vendor"):
            logger.warning(
                "Skipping %s: could not resolve vendor (pass --vendor)",
                vtt_path.name,
            )
            counts["skipped"] += 1
            continue

        video_id = meta["video_id"]
        if not force and _video_id_has_points(store, video_id):
            logger.info("Skipping video %s (already ingested)", video_id)
            counts["skipped"] += 1
            continue

        if force:
            _purge_video_id_chunks(store, video_id)
            _purge_source_chunks(meta["source"])

        try:
            cues = parse_vtt_cues(str(vtt_path))
            if not cues:
                logger.warning("Skipping %s: no cues after VTT parse", vtt_path.name)
                counts["skipped"] += 1
                continue
            # Fix observed ASR mistranscriptions (same glossary as the
            # Whisper path) before windowing and hashing.
            cues = [
                type(cue)(
                    start=cue.start,
                    end=cue.end,
                    text=normalize_transcript_text(cue.text),
                )
                for cue in cues
            ]
            transcript = normalize_transcript_text(parse_vtt(str(vtt_path)))
            if not transcript.strip():
                logger.warning("Skipping %s: empty transcript after VTT parse", vtt_path.name)
                counts["skipped"] += 1
                continue
            vtt_windows = group_vtt_cues_into_chunks(
                cues,
                max_chars=SECTION_MAX_CHARS,
                overlap_chars=SECTION_OVERLAP,
            )
            if not vtt_windows:
                logger.warning("Skipping %s: no timestamped chunks produced", vtt_path.name)
                counts["skipped"] += 1
                continue
            chunk_count = _ingest_video_transcript_text(
                transcript,
                meta,
                vtt_windows=vtt_windows,
            )
            counts["ingested"] += chunk_count
            counts["videos"] += 1
            upsert_manifest_source(
                meta["source"],
                file_hash=_hash_text_content(transcript),
                vendor=meta["vendor"],
                product=meta.get("product"),
                chunk_count=chunk_count,
                extra={
                    "content_type": "video_transcript",
                    "source_type": "video",
                    "video_id": video_id,
                },
            )
        except Exception as exc:
            logger.warning("Failed to ingest %s: %s", vtt_path.name, exc)
            counts["failed"] += 1

    return counts


def _registry_meta_from_context(ctx: IngestContext, file_path: str) -> DocumentMetadata:
    path = Path(file_path)
    return DocumentMetadata(
        source=ctx.source,
        file_name=ctx.source,
        file_extension=path.suffix.lower(),
        vendor=ctx.vendor,
        document_type=ctx.doc_type,
        ingestion_timestamp=ctx.ingested_at,
        product_line=ctx.product or "",
        software_version=ctx.product_version or "",
    )


def _ingest_pdf_images(
    file_path: str,
    doc_metadata,
    store,
    *,
    chunk_index_start: int,
) -> int:
    from providers.factory import get_image_embedder
    from providers.media_store import get_media_store
    from providers.metadata import build_image_chunk_payload
    from providers.pdf_images import extract_pdf_images

    images = extract_pdf_images(file_path)
    if not images:
        return 0

    media = get_media_store()
    embedder = get_image_embedder()
    source_hash = media.source_hash(doc_metadata.source)
    ids: list[str] = []
    vectors: list[list[float]] = []
    payloads: list[dict] = []
    chunk_index = chunk_index_start

    for image in images:
        rel = f"{source_hash}/{image.page}_{image.index}.{image.ext}"
        media_uri = media.put_bytes(rel, image.data)
        if not media_uri:
            continue
        vector = embedder.embed_bytes([image.data])[0]
        payload = build_image_chunk_payload(
            metadata=doc_metadata,
            chunk_index=chunk_index,
            page=image.page,
            media_uri=media_uri,
            media_hash=media.media_hash(image.data),
            image_class="diagram",
            width=image.width,
            height=image.height,
        )
        ids.append(payload["chunk_id"])
        vectors.append(vector)
        payloads.append(payload)
        chunk_index += 1

    if ids:
        store.upsert(ids, vectors, payloads)
    return len(ids)


def _cli_main(argv: list[str] | None = None) -> int:
    # Force UTF-8 stdio: on Windows the default cp1252 console codec raises
    # UnicodeEncodeError on CJK/full-width characters (e.g. Mech-Mind video
    # titles), which would abort ingest of those sources on a status print.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Ingest a document into Qdrant (schema v2)")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Path to PDF, HTML, or text file",
    )
    parser.add_argument(
        "--url-ingest",
        default=None,
        metavar="URL",
        help="Fetch and ingest a web page from URL",
    )
    parser.add_argument(
        "--url-list",
        default=None,
        metavar="PATH",
        help="Plain text file with one URL per line (# comments allowed)",
    )
    parser.add_argument("--vendor", help="Vendor slug (required unless data/<vendor>/...)")
    parser.add_argument("--product", default=None, help="Product name, e.g. 'pekat vision'")
    parser.add_argument(
        "--version",
        default=None,
        dest="product_version",
        help="Product version, e.g. 3.17",
    )
    parser.add_argument(
        "--doc-type",
        choices=DOC_TYPES,
        default="manual",
        dest="doc_type",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Original source URL metadata for local files",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        metavar="PATH",
        help="Ingest yt-dlp .vtt transcripts from a directory (requires sibling .info.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even when the file hash is unchanged",
    )
    args = parser.parse_args(argv)

    mode_count = sum(
        1
        for value in (args.file_path, args.url_ingest, args.url_list, args.video_dir)
        if value
    )
    if mode_count != 1:
        parser.error(
            "Provide exactly one of: file_path, --url-ingest, --url-list, or --video-dir"
        )

    product = args.product.strip().lower() if args.product else None
    product_version = args.product_version.strip() if args.product_version else None

    if args.video_dir:
        vendor = (args.vendor or "").strip().lower() or None
        try:
            counts = ingest_video_transcripts(
                args.video_dir,
                vendor=vendor,
                product=product,
                product_version=product_version,
                force=args.force,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(
            "Videos ingested: {videos} | Chunks: {ingested} | Skipped: {skipped} | Failed: {failed}".format(
                **counts
            )
        )
        return 0 if counts["failed"] == 0 else 1

    if args.url_ingest or args.url_list:
        vendor = (args.vendor or "").strip().lower()
        if not vendor:
            print("Error: --vendor is required for URL ingestion", file=sys.stderr)
            return 1

        if args.url_ingest:
            result = _process_url_ingest(
                args.url_ingest,
                vendor=vendor,
                product=product,
                product_version=product_version,
                doc_type=args.doc_type,
                force=args.force,
            )
            return 1 if result == "failed" else 0

        list_path = Path(args.url_list)
        if not list_path.is_file():
            print(f"Error: URL list not found: {args.url_list}", file=sys.stderr)
            return 1

        counts = {"ingested": 0, "skipped": 0, "failed": 0}
        for url in _load_url_list(str(list_path)):
            result = _process_url_ingest(
                url,
                vendor=vendor,
                product=product,
                product_version=product_version,
                doc_type=args.doc_type,
                force=args.force,
            )
            counts[result] = counts.get(result, 0) + 1

        print(
            "Ingested: {ingested} | Skipped (unchanged): {skipped} | Failed: {failed}".format(
                **counts
            )
        )
        return 0

    file_path = args.file_path
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1

    vendor = (args.vendor or infer_vendor_from_path(file_path) or "").strip().lower()
    if not vendor:
        print(
            "Error: vendor is required. Use --vendor or place the file under data/<vendor>/...",
            file=sys.stderr,
        )
        return 1

    filename = path.name
    file_hash = compute_ingest_hash(file_path)
    manifest = load_manifest()

    if _should_skip_manifest(filename, file_hash, manifest, force=args.force):
        print(f"Skipping {filename} (unchanged)")
        return 0

    try:
        total = ingest(
            file_path,
            vendor=vendor,
            product=product,
            product_version=product_version,
            doc_type=args.doc_type,
            url=args.url,
            force=args.force,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    manifest[filename] = _manifest_entry(
        file_hash=file_hash,
        vendor=vendor,
        product=product,
        chunk_count=total,
    )
    save_manifest(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
