#!/usr/bin/env python3
"""Caption scene-change video frames and upsert ``video_frame`` points.

Run: ``python -m scripts.ingest.video_frame_worker``
"""
from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from scripts.ingest.caption_worker import (
    fetch_source_metadata,
    reject_caption,
    sanitize_caption,
    truncate_caption,
)
from scripts.ingest.ingest_video import make_video_id, preflight_ffmpeg, video_data_root
from providers.factory import get_embedder, get_vector_store
from providers.ollama_vision import OllamaVision
from providers.video_frames import cap_frames_evenly, extract_scene_frames, max_frames_per_video

load_dotenv()

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_frame_prompt(source: str, section: str | None, start_seconds: float) -> str:
    section_text = f", section '{section}'" if section else ""
    return (
        f"This is a scene-change frame from video source {source}{section_text}, "
        f"captured at {start_seconds:.1f} seconds. "
        "Describe UI panels, labels, controls, and visible workflow state in 1-3 sentences. "
        "Use exact text only when clearly legible."
    )


def build_frame_index_text(source: str, start_seconds: float, caption: str) -> str:
    return f"[Video frame, {source}, t={start_seconds:.1f}s] {caption}"


def make_video_frame_point_id(source: str, start_seconds: float) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|video_frame|{start_seconds:.3f}"))


def build_video_frame_payload(
    *,
    source: str,
    vendor: str,
    doc_type: str,
    section: str | None,
    url: str | None,
    product: str | None,
    product_version: str | None,
    video_path: str,
    frame_path: str,
    start_seconds: float,
    caption: str,
    video_url: str | None,
    ingested_at: str,
) -> dict:
    return {
        "text": caption,
        "source": source,
        "page": None,
        "url": url,
        "vendor": vendor,
        "product": product,
        "product_version": product_version,
        "content_type": "video_frame",
        "doc_type": doc_type,
        "section": section,
        "timestamp": None,
        "schema_version": SCHEMA_VERSION,
        "ingested_at": ingested_at,
        "start_seconds": float(start_seconds),
        "end_seconds": float(start_seconds),
        "video_path": video_path,
        "video_url": video_url,
        "frame_path": frame_path,
        "image_path": frame_path,
    }


def process_video_frames(
    *,
    source: str,
    vendor: str,
    video_path: Path,
    doc_type: str = "tutorial",
    section: str | None = None,
    url: str | None = None,
    video_url: str | None = None,
    embedder=None,
    store=None,
    vision: OllamaVision | None = None,
) -> dict:
    preflight_ffmpeg()
    vendor = vendor.lower().strip()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    store = store or get_vector_store()
    embedder = embedder or get_embedder()
    vision = vision or OllamaVision()
    store.ensure_payload_indexes()
    meta_cache: dict[str, dict] = {}
    meta = fetch_source_metadata(store, source=source, vendor=vendor, cache=meta_cache)
    product = meta.get("product")
    product_version = meta.get("product_version")
    resolved_doc_type = meta.get("doc_type") or doc_type
    resolved_section = section if section is not None else None
    resolved_url = url if url is not None else None

    video_id = make_video_id(source, video_url)
    frame_dir = video_data_root() / vendor / "videos" / "_frame_cache" / video_id
    detected = extract_scene_frames(video_path, frame_dir)
    capped = cap_frames_evenly(detected, max_frames_per_video())

    captioned = 0
    skipped = 0
    ingested_at = _utc_now()
    rel_video_path = video_path.as_posix()

    for frame in capped:
        frame_path = Path(frame["path"])
        start_seconds = float(frame["start_seconds"])
        prompt = build_frame_prompt(source, resolved_section, start_seconds)
        raw = vision.describe_image(prompt, frame_path.read_bytes())
        caption = truncate_caption(sanitize_caption(raw))
        reject_reason = reject_caption(caption)
        if reject_reason:
            logger.warning(
                "Frame caption rejected (source=%s t=%.3f): %s",
                source,
                start_seconds,
                reject_reason,
            )
            skipped += 1
            continue
        index_text = build_frame_index_text(source, start_seconds, caption)
        vector = embedder.embed([index_text])[0]
        if vector is None:
            logger.warning(
                "Skipping video frame with no embedding (source=%s start_seconds=%.3f)",
                source,
                start_seconds,
            )
            skipped += 1
            continue
        rel_frame_path = frame_path.as_posix()
        payload = build_video_frame_payload(
            source=source,
            vendor=vendor,
            doc_type=resolved_doc_type,
            section=resolved_section,
            url=resolved_url,
            product=product,
            product_version=product_version,
            video_path=rel_video_path,
            frame_path=rel_frame_path,
            start_seconds=start_seconds,
            caption=caption,
            video_url=video_url,
            ingested_at=ingested_at,
        )
        point_id = make_video_frame_point_id(source, start_seconds)
        store.upsert([point_id], [vector], [payload])
        captioned += 1

    return {
        "source": source,
        "vendor": vendor,
        "video_path": rel_video_path,
        "video_id": video_id,
        "scenes_detected": len(detected),
        "scenes_kept": len(capped),
        "captioned": captioned,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caption scene-change video frames and upsert")
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--doc-type", default="tutorial", dest="doc_type")
    parser.add_argument("--section")
    parser.add_argument("--url")
    parser.add_argument("--video-url")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        result = process_video_frames(
            source=args.source,
            vendor=args.vendor,
            video_path=Path(args.video_path),
            doc_type=args.doc_type,
            section=args.section,
            url=args.url,
            video_url=args.video_url,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1

    print(
        "Video frame captioning done: "
        f"detected={result['scenes_detected']} kept={result['scenes_kept']} "
        f"captioned={result['captioned']} skipped={result['skipped']} "
        f"source={result['source']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
