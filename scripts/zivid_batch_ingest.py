#!/usr/bin/env python3
"""Batch ingest Zivid PDFs: text, image extract, and captions.

Run: ``python -m scripts.zivid_batch_ingest``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import scripts.ingest.ingest as ingest  # noqa: F401
import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs
from scripts.ingest.reingest_all import extended_v2_payload

VENDOR = "zivid"
DATA_DIR = Path("data") / VENDOR
LOG_PATH = DATA_DIR / "_batch_ingest_log.jsonl"

DOCS: list[dict] = [
    {
        "file": "Zivid 2 User Guide 1.11 - English.pdf",
        "product": "Zivid 2",
        "device_family": "Zivid 2",
        "device_model": None,
        "doc_type": "manual",
        "doc_version": "1.11",
    },
    {
        "file": "Zivid 2+ User Guide 1.11 - English.pdf",
        "product": "Zivid 2+",
        "device_family": "Zivid 2+",
        "device_model": None,
        "doc_type": "manual",
        "doc_version": "1.11",
    },
    {
        "file": "Zivid 3 User Guide 1.0 - English.pdf",
        "product": "Zivid 3",
        "device_family": "Zivid 3",
        "device_model": None,
        "doc_type": "manual",
        "doc_version": "1.0",
    },
    {
        "file": "Zivid Studio User Guide SDK 2.17 - English.pdf",
        "product": "Zivid Studio",
        "device_family": "Zivid Studio",
        "device_model": None,
        "doc_type": "manual",
        "doc_version": "2.17",
    },
    {
        "file": "Zivid 2+ L110 Datasheet.pdf",
        "product": "Zivid 2+",
        "device_family": "Zivid 2+",
        "device_model": "L110",
        "doc_type": "datasheet",
        "doc_version": None,
    },
    {
        "file": "Zivid 2+ LR110 Datasheet.pdf",
        "product": "Zivid 2+",
        "device_family": "Zivid 2+",
        "device_model": "LR110",
        "doc_type": "datasheet",
        "doc_version": None,
    },
    {
        "file": "Zivid 2+ M130 Datasheet.pdf",
        "product": "Zivid 2+",
        "device_family": "Zivid 2+",
        "device_model": "M130",
        "doc_type": "datasheet",
        "doc_version": None,
    },
    {
        "file": "Zivid 2+ M60 Datasheet.pdf",
        "product": "Zivid 2+",
        "device_family": "Zivid 2+",
        "device_model": "M60",
        "doc_type": "datasheet",
        "doc_version": None,
    },
    {
        "file": "Zivid 2+ MR130 Datasheet.pdf",
        "product": "Zivid 2+",
        "device_family": "Zivid 2+",
        "device_model": "MR130",
        "doc_type": "datasheet",
        "doc_version": None,
    },
    {
        "file": "Zivid 2+ MR60 Datasheet.pdf",
        "product": "Zivid 2+",
        "device_family": "Zivid 2+",
        "device_model": "MR60",
        "doc_type": "datasheet",
        "doc_version": None,
    },
    {
        "file": "Zivid 3 XL250 Datasheet.pdf",
        "product": "Zivid 3",
        "device_family": "Zivid 3",
        "device_model": "XL250",
        "doc_type": "datasheet",
        "doc_version": None,
    },
    {
        "file": "Zivid Two L100 Datasheet.pdf",
        "product": "Zivid Two",
        "device_family": "Zivid Two",
        "device_model": "L100",
        "doc_type": "datasheet",
        "doc_version": None,
    },
    {
        "file": "Zivid Two M70 Datasheet.pdf",
        "product": "Zivid Two",
        "device_family": "Zivid Two",
        "device_model": "M70",
        "doc_type": "datasheet",
        "doc_version": None,
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
    if meta.get("device_model"):
        extra["device_model"] = meta["device_model"]
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


def main() -> int:
    os.chdir(ROOT)
    for meta in DOCS:
        pdf = DATA_DIR / meta["file"]
        if not pdf.is_file():
            log_event({"phase": "skip", "file": meta["file"], "reason": "missing"})
            continue
        row = {"file": meta["file"]}
        try:
            t0 = time.time()
            row["text_chunks"] = ingest_text(pdf, meta)
            row["text_seconds"] = round(time.time() - t0, 1)
            t1 = time.time()
            proc = extract_images(pdf)
            row["extract_rc"] = proc.returncode
            lines = (proc.stdout or "").strip().splitlines()
            row["extract_stdout"] = lines[-1:] if lines else []
            row["extract_stderr"] = (proc.stderr or "").strip()[-500:]
            row["extract_seconds"] = round(time.time() - t1, 1)
            row["status"] = "ok" if proc.returncode == 0 else "extract_failed"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        log_event({"phase": "pdf_done", **row})

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
