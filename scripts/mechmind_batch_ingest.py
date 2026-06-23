#!/usr/bin/env python3
"""One-off batch ingest for remaining Mech-Mind PDFs (pilot excluded).

Run: ``python -m scripts.mechmind_batch_ingest``
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import scripts.ingest.ingest as ingest  # noqa: F401
import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs
from scripts.ingest.reingest_all import extended_v2_payload

PILOT = "eye-3d-camera-v2.5.4-en.pdf"
VENDOR = "mechmind"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("data") / VENDOR
LOG_PATH = DATA_DIR / "_batch_ingest_log.jsonl"

MANUALS: list[dict] = [
    {
        "file": "3d-measurement-manual-v2.1.1-en.pdf",
        "product": "Mech-Vision 3D Measurement",
        "device_family": "Mech-Vision",
        "doc_type": "manual",
        "doc_version": "2.1.1",
    },
    {
        "file": "dlk-sdk-user-manual-v2.1.0-en.pdf",
        "product": "Mech-DLK SDK",
        "device_family": "Mech-DLK",
        "doc_type": "manual",
        "doc_version": "2.1.0",
    },
    {
        "file": "dlk-software-manual-v2.6.0-en.pdf",
        "product": "Mech-DLK Software",
        "device_family": "Mech-DLK",
        "doc_type": "manual",
        "doc_version": "2.6.0",
    },
    {
        "file": "eye-3d-profiler-v2.5.4-en.pdf",
        "product": "Mech-Eye 3D Profiler",
        "device_family": "Mech-Eye",
        "doc_type": "manual",
        "doc_version": "2.5.4",
    },
    {
        "file": "ipc-adv-en.pdf",
        "product": "Mech-Eye IPC Advanced",
        "device_family": "Mech-Eye",
        "doc_type": "manual",
        "doc_version": None,
    },
    {
        "file": "ipc-en.pdf",
        "product": "Mech-Eye IPC",
        "device_family": "Mech-Eye",
        "doc_type": "manual",
        "doc_version": None,
    },
    {
        "file": "ipc-pro-en.pdf",
        "product": "Mech-Eye IPC Pro",
        "device_family": "Mech-Eye",
        "doc_type": "manual",
        "doc_version": None,
    },
    {
        "file": "ipc-std-2022-en.pdf",
        "product": "Mech-Eye IPC Std",
        "device_family": "Mech-Eye",
        "doc_type": "manual",
        "doc_version": "2022",
    },
    {
        "file": "ipc-std-en.pdf",
        "product": "Mech-Eye IPC Std",
        "device_family": "Mech-Eye",
        "doc_type": "manual",
        "doc_version": None,
    },
    {
        "file": "robot-communication-and-integration-v2.1.2-en.pdf",
        "product": "Mech-Vision Robot Communication",
        "device_family": "Mech-Vision",
        "doc_type": "manual",
        "doc_version": "2.1.2",
    },
    {
        "file": "suite-best-practice-v2.1.2-en.pdf",
        "product": "Mech-Vision Suite",
        "device_family": "Mech-Vision",
        "doc_type": "manual",
        "doc_version": "2.1.2",
    },
    {
        "file": "vision-system-service-manual-v2.1.2-en.pdf",
        "product": "Mech-Vision System",
        "device_family": "Mech-Vision",
        "doc_type": "manual",
        "doc_version": "2.1.2",
    },
    {
        "file": "vision-system-software-manual-v2.1.2-en.pdf",
        "product": "Mech-Vision System Software",
        "device_family": "Mech-Vision",
        "doc_type": "manual",
        "doc_version": "2.1.2",
    },
    {
        "file": "vision-system-tutorial-v2.1.2-en.pdf",
        "product": "Mech-Vision System",
        "device_family": "Mech-Vision",
        "doc_type": "tutorial",
        "doc_version": "2.1.2",
    },
]


def log_event(event: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def extra_for(meta: dict) -> dict:
    extra = {
        "category": VENDOR,
        "device_family": meta["device_family"],
        "language": "en",
        "source_type": "pdf",
    }
    if meta.get("doc_version"):
        extra["doc_version"] = meta["doc_version"]
    return extra


def ingest_text(pdf: Path, meta: dict) -> int:
    source = pdf.name
    extra_by_source = {source: extra_for(meta)}
    with extended_v2_payload(extra_by_source):
        return ingest.ingest(
            str(pdf.resolve()),
            vendor=VENDOR,
            product=meta["product"],
            product_version=meta.get("doc_version"),
            doc_type=meta["doc_type"],
            force=False,
        )


def extract_images(pdf: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ingest" / "extract_pdf_images.py"), str(pdf)],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def backfill_pilot_captions() -> int:
    """Patch pilot image_caption points missing extended metadata."""
    from dotenv import load_dotenv
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    load_dotenv()
    import os

    client = QdrantClient(url=os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333"))
    coll = os.getenv("QDRANT_COLLECTION", "rag_docs")
    source = PILOT
    text_filter = Filter(
        must=[
            FieldCondition(key="source", match=MatchValue(value=source)),
            FieldCondition(key="content_type", match=MatchValue(value="text")),
        ]
    )
    records, _ = client.scroll(
        collection_name=coll,
        scroll_filter=text_filter,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return 0
    text_payload = dict(records[0].payload or {})
    patch = {
        "category": text_payload.get("category"),
        "device_family": text_payload.get("device_family"),
        "doc_version": text_payload.get("doc_version"),
        "language": text_payload.get("language"),
        "source_type": text_payload.get("source_type"),
    }
    patch = {k: v for k, v in patch.items() if v is not None}
    if not patch:
        return 0
    client.set_payload(
        collection_name=coll,
        payload=patch,
        points=Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=source)),
                FieldCondition(key="content_type", match=MatchValue(value="image_caption")),
            ]
        ),
        wait=True,
    )
    offset = None
    count = 0
    while True:
        records, offset = client.scroll(
            collection_name=coll,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source)),
                    FieldCondition(key="content_type", match=MatchValue(value="image_caption")),
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        count += len(records)
        if offset is None:
            break
    return count


def main() -> int:
    import os

    os.chdir(ROOT)
    results: list[dict] = []
    for meta in MANUALS:
        pdf = DATA_DIR / meta["file"]
        if not pdf.is_file():
            log_event({"phase": "skip", "file": meta["file"], "reason": "missing"})
            continue
        row = {"file": meta["file"]}
        try:
            t0 = time.time()
            chunks = ingest_text(pdf, meta)
            row["text_chunks"] = chunks
            row["text_seconds"] = round(time.time() - t0, 1)
            t1 = time.time()
            proc = extract_images(pdf)
            row["extract_rc"] = proc.returncode
            row["extract_stdout"] = (proc.stdout or "").strip().splitlines()[-1:] or []
            row["extract_stderr"] = (proc.stderr or "").strip()[-500:]
            row["extract_seconds"] = round(time.time() - t1, 1)
            row["status"] = "ok" if proc.returncode == 0 else "extract_failed"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        log_event({"phase": "pdf_done", **row})
        results.append(row)

    patched = backfill_pilot_captions()
    log_event({"phase": "pilot_caption_backfill", "points": patched})

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ingest" / "caption_worker.py"), "--vendor", VENDOR],
        cwd=str(ROOT),
        check=False,
    )
    log_event(
        {
            "phase": "captions_done",
            "returncode": proc.returncode,
            "seconds": round(time.time() - t0, 1),
        }
    )
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
