#!/usr/bin/env python3
"""Transcribe local MP4 videos and upsert transcript chunks to Qdrant.

Run: ``python -m scripts.ingest.ingest_video``
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from providers.factory import get_embedder, get_vector_store
from providers.transcript_glossary import normalize_transcript_text
from providers.video_transcribe import (
    TranscriptSegment,
    build_transcript_index_text,
    group_segments_into_windows,
    transcribe,
    transcript_chunk_seconds,
)

load_dotenv()

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "512"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def video_data_root() -> Path:
    return Path(os.getenv("VIDEO_DATA_DIR", "data"))


def make_video_id(source: str, video_url: str | None = None) -> str:
    key = f"{source}|{video_url or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def apply_glossary_to_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    corrected: list[TranscriptSegment] = []
    for segment in segments:
        corrected.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=normalize_transcript_text(segment.text),
            )
        )
    return corrected


def video_storage_dir(vendor: str, video_id: str) -> Path:
    return video_data_root() / vendor.lower() / "videos" / video_id


def preflight_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{joined} not found on PATH. Install ffmpeg (https://ffmpeg.org/download.html) "
            "and ensure ffprobe is on PATH, then retry."
        )


def run_preflight() -> None:
    preflight_ffmpeg()
    preflight_whisper()


def preflight_whisper() -> None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc


def probe_duration_seconds(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def extract_audio_wav(video_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ],
        capture_output=True,
        check=True,
    )


def stage_source_mp4(source_mp4: Path, dest_mp4: Path) -> Path:
    dest_mp4.parent.mkdir(parents=True, exist_ok=True)
    if source_mp4.resolve() != dest_mp4.resolve():
        shutil.copy2(source_mp4, dest_mp4)
    return dest_mp4


def make_point_id(source: str, start_seconds: float, chunk_index: int) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{source}|video_transcript|{start_seconds:.3f}|{chunk_index}",
        )
    )


def build_transcript_payload(
    *,
    source: str,
    vendor: str,
    window: TranscriptSegment,
    video_path: str,
    chunk_index: int,
    product: str | None,
    doc_type: str,
    section: str | None,
    url: str | None,
    video_url: str | None,
    duration_seconds: float | None,
    ingested_at: str,
) -> dict:
    return {
        "text": window.text,
        "source": source,
        "page": None,
        "url": url,
        "vendor": vendor.lower(),
        "product": product,
        "product_version": None,
        "content_type": "video_transcript",
        "doc_type": doc_type,
        "section": section,
        "timestamp": None,
        "schema_version": SCHEMA_VERSION,
        "ingested_at": ingested_at,
        "start_seconds": float(window.start),
        "end_seconds": float(window.end),
        "video_path": video_path,
        "video_url": video_url,
        "duration_seconds": duration_seconds,
    }


def upsert_transcript_windows(
    *,
    windows: list[TranscriptSegment],
    source: str,
    vendor: str,
    video_path: str,
    product: str | None,
    doc_type: str,
    section: str | None,
    url: str | None,
    video_url: str | None,
    duration_seconds: float | None,
    embedder,
    store,
    ingested_at: str | None = None,
) -> tuple[int, int]:
    """Embed index text and upsert transcript windows. Returns (upserted, skipped)."""
    if not windows:
        return 0, 0

    ingested_at = ingested_at or _utc_now()
    upserted = 0
    skipped = 0

    for batch_start in range(0, len(windows), EMBED_BATCH_SIZE):
        batch = windows[batch_start : batch_start + EMBED_BATCH_SIZE]
        index_texts = [
            build_transcript_index_text(source, window.start, window.text)
            for window in batch
        ]
        vectors = embedder.embed(index_texts)
        ids: list[str] = []
        vector_batch: list[list[float]] = []
        payload_batch: list[dict] = []

        for chunk_index, (window, vector) in enumerate(
            zip(batch, vectors), start=batch_start
        ):
            if vector is None:
                skipped += 1
                logger.warning(
                    "Skipping video transcript chunk with no embedding "
                    "(source=%s start_seconds=%.3f)",
                    source,
                    window.start,
                )
                continue
            payload = build_transcript_payload(
                source=source,
                vendor=vendor,
                window=window,
                video_path=video_path,
                chunk_index=chunk_index,
                product=product,
                doc_type=doc_type,
                section=section,
                url=url,
                video_url=video_url,
                duration_seconds=duration_seconds,
                ingested_at=ingested_at,
            )
            ids.append(make_point_id(source, window.start, chunk_index))
            vector_batch.append(vector)
            payload_batch.append(payload)

        if ids:
            store.upsert(ids, vector_batch, payload_batch)
            upserted += len(ids)

    return upserted, skipped


def ingest_local_mp4(
    mp4_path: Path,
    *,
    vendor: str,
    source: str | None = None,
    product: str | None = None,
    doc_type: str = "other",
    section: str | None = None,
    url: str | None = None,
    video_url: str | None = None,
    embedder=None,
    store=None,
) -> dict:
    run_preflight()

    mp4_path = mp4_path.resolve()
    if not mp4_path.is_file():
        raise FileNotFoundError(f"Video not found: {mp4_path}")

    source = (source or mp4_path.name).strip()
    vendor = vendor.lower().strip()
    video_id = make_video_id(source, video_url)
    storage_dir = video_storage_dir(vendor, video_id)
    dest_mp4 = storage_dir / "source.mp4"
    wav_path = storage_dir / "_audio.wav"
    rel_video_path = dest_mp4.as_posix()

    stage_source_mp4(mp4_path, dest_mp4)
    duration_seconds = probe_duration_seconds(dest_mp4)

    if not wav_path.is_file():
        extract_audio_wav(dest_mp4, wav_path)

    segments = apply_glossary_to_segments(transcribe(str(wav_path)))
    windows = group_segments_into_windows(segments, transcript_chunk_seconds())

    embedder = embedder or get_embedder()
    store = store or get_vector_store()
    store.ensure_payload_indexes()

    upserted, skipped = upsert_transcript_windows(
        windows=windows,
        source=source,
        vendor=vendor,
        video_path=rel_video_path,
        product=product,
        doc_type=doc_type,
        section=section,
        url=url,
        video_url=video_url,
        duration_seconds=duration_seconds,
        embedder=embedder,
        store=store,
    )

    return {
        "source": source,
        "vendor": vendor,
        "video_id": video_id,
        "video_path": rel_video_path,
        "duration_seconds": duration_seconds,
        "segments": len(segments),
        "windows": len(windows),
        "upserted": upserted,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a local MP4: transcribe with timestamps and upsert to Qdrant",
    )
    parser.add_argument("mp4_path", help="Path to local MP4 file")
    parser.add_argument("--vendor", required=True, help="Vendor slug (e.g. pekat, lmi)")
    parser.add_argument(
        "--source",
        help="Stable source id (default: MP4 filename)",
    )
    parser.add_argument("--product", help="Optional product metadata")
    parser.add_argument("--doc-type", default="other", dest="doc_type")
    parser.add_argument("--section", help="Optional section heading")
    parser.add_argument("--url", help="Optional parent page URL")
    parser.add_argument("--video-url", dest="video_url", help="Optional provenance URL")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        result = ingest_local_mp4(
            Path(args.mp4_path),
            vendor=args.vendor,
            source=args.source,
            product=args.product,
            doc_type=args.doc_type,
            section=args.section,
            url=args.url,
            video_url=args.video_url,
        )
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Ingested {result['upserted']} transcript windows "
        f"({result['segments']} segments -> {result['windows']} windows) "
        f"for {result['source']} [{result['video_path']}]"
    )
    if result["skipped"]:
        print(f"Warning: skipped {result['skipped']} chunks (no embedding)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
