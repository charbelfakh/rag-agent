#!/usr/bin/env python3
"""Extract embedded PDF images into vendor folders and ``pending_captions.json``.

Run: ``python -m scripts.ingest.extract_pdf_images``
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

import pymupdf as fitz

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from providers.caption_images import is_skipped_caption_entry, prepare_caption_queue_entries

MIN_DIMENSION = 150
MIN_AREA = 40_000
MAX_ASPECT_RATIO = 8.0
SOLID_FILL_VARIANCE = 25.0
THUMB_MAX_PX = 48
THUMB_SAMPLE_STRIDE = 4
NEAR_BLANK_THUMB_PX = 64
NEAR_WHITE = 245
NEAR_BLACK = 10
NEAR_BLANK_RATIO = 0.90


def _pending_captions_path(vendor: str) -> Path:
    return Path("data") / vendor.lower() / "pending_captions.json"


def _append_pending_captions(vendor: str, new_entries: list[dict]) -> int:
    """Merge entries into pending_captions.json; dedupe on (source, image_src)."""
    path = _pending_captions_path(vendor)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: failed to load {path}: {exc}", file=sys.stderr)

    seen = {(e.get("source"), e.get("image_src")) for e in existing}
    added = 0
    for entry in prepare_caption_queue_entries(new_entries, vendor, materialize_remote=False):
        key = (entry["source"], entry["image_src"])
        if key in seen:
            continue
        existing.append(entry)
        seen.add(key)
        added += 1

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return added


def _infer_vendor(pdf_path: Path) -> str:
    parts = pdf_path.resolve().parts
    try:
        data_idx = parts.index("data")
        return parts[data_idx + 1].lower()
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot infer vendor from path (expected data/<vendor>/...): {pdf_path}") from exc


def _aspect_ratio(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return float("inf")
    return max(width / height, height / width)


def _solid_fill_variance_low(image_bytes: bytes) -> bool:
    """Cheap thumbnail sample: near-zero variance means flat color fills."""
    try:
        pix = fitz.Pixmap(image_bytes)
        if pix.n >= 4 and pix.colorspace and pix.colorspace.n >= 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        elif pix.alpha:
            pix = fitz.Pixmap(pix, 0)

        width, height = pix.width, pix.height
        if width <= 0 or height <= 0:
            return True

        scale = min(THUMB_MAX_PX / width, THUMB_MAX_PX / height, 1.0)
        if scale < 1.0:
            matrix = fitz.Matrix(scale, scale)
            pix = fitz.Pixmap(pix, 0, 0, width, height, matrix)

        samples = pix.samples
        n = pix.n
        if not samples or n < 1:
            return False

        values: list[float] = []
        pixel_stride = n * THUMB_SAMPLE_STRIDE
        for offset in range(0, len(samples) - n + 1, pixel_stride):
            if n >= 3:
                values.append((samples[offset] + samples[offset + 1] + samples[offset + 2]) / 3.0)
            else:
                values.append(float(samples[offset]))

        if len(values) < 2:
            return False
        return statistics.pvariance(values) < SOLID_FILL_VARIANCE
    except Exception:
        return False


def _grayscale_thumbnail(image_bytes: bytes, *, max_px: int) -> fitz.Pixmap | None:
    try:
        pix = fitz.Pixmap(image_bytes)
        if pix.n >= 4 and pix.colorspace and pix.colorspace.n >= 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        elif pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        gray = fitz.Pixmap(fitz.csGRAY, pix)
        width, height = gray.width, gray.height
        if width <= 0 or height <= 0:
            return None
        scale = min(max_px / width, max_px / height, 1.0)
        if scale < 1.0:
            matrix = fitz.Matrix(scale, scale)
            gray = fitz.Pixmap(gray, 0, 0, width, height, matrix)
        return gray
    except Exception:
        return None


def _near_blank(image_bytes: bytes) -> bool:
    """Skip mostly white/black layers (blank pages, callout overlays)."""
    thumb = _grayscale_thumbnail(image_bytes, max_px=NEAR_BLANK_THUMB_PX)
    if thumb is None:
        return False
    samples = thumb.samples
    if not samples:
        return False
    extreme = sum(1 for value in samples if value > NEAR_WHITE or value < NEAR_BLACK)
    return extreme / len(samples) > NEAR_BLANK_RATIO


def _skip_reason(width: int, height: int, image_bytes: bytes, seen_xrefs: set[int], xref: int) -> str | None:
    if xref in seen_xrefs:
        return "duplicate_xref"
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        return "too_small_dim"
    if width * height < MIN_AREA:
        return "too_small_area"
    if _aspect_ratio(width, height) > MAX_ASPECT_RATIO:
        return "aspect_ratio"
    if _solid_fill_variance_low(image_bytes):
        return "solid_fill"
    if _near_blank(image_bytes):
        return "near-blank"
    return None


def _verify_source_in_qdrant(source: str, vendor: str) -> bool | None:
    """Optional read-only check via Qdrant REST (stdlib). Returns None if skipped."""
    import urllib.error
    import urllib.request

    base_url = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333").rstrip("/")
    collection = os.getenv("QDRANT_V2_COLLECTION") or os.getenv("QDRANT_COLLECTION", "rag_docs_v2")
    body = json.dumps(
        {
            "filter": {
                "must": [
                    {"key": "source", "match": {"value": source}},
                    {"key": "vendor", "match": {"value": vendor}},
                ]
            },
            "limit": 1,
            "with_payload": False,
            "with_vector": False,
        }
    ).encode("utf-8")
    url = f"{base_url}/collections/{collection}/points/scroll"
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        points = payload.get("result", {}).get("points", [])
        return bool(points)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def extract_pdf_images(
    pdf_path: Path,
    *,
    verify_qdrant: bool = False,
) -> tuple[Counter[str], int]:
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    vendor = _infer_vendor(pdf_path)
    source = pdf_path.name
    pdfstem = pdf_path.stem
    out_dir = Path("data") / vendor / "images" / pdfstem
    out_dir.mkdir(parents=True, exist_ok=True)

    if verify_qdrant:
        found = _verify_source_in_qdrant(source, vendor)
        if found is False:
            print(f"Warning: {source} not found in Qdrant for vendor {vendor}", file=sys.stderr)
        elif found is None:
            print(f"Note: Qdrant source check skipped (unavailable)", file=sys.stderr)

    counts: Counter[str] = Counter()
    queue_entries: list[dict] = []
    seen_xrefs: set[int] = set()

    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            page_one = page_num + 1
            kept_idx = 0
            for img in page.get_images(full=True):
                xref = int(img[0])
                try:
                    base = pdf.extract_image(xref)
                except Exception:
                    counts["extract_error"] += 1
                    continue

                image_bytes = base.get("image") or b""
                width = int(base.get("width") or 0)
                height = int(base.get("height") or 0)
                if not image_bytes:
                    counts["empty_image"] += 1
                    continue

                reason = _skip_reason(width, height, image_bytes, seen_xrefs, xref)
                if reason:
                    counts[reason] += 1
                    continue

                seen_xrefs.add(xref)
                filename = f"p{page_one}_{kept_idx}.png"
                rel_image_src = (Path("images") / pdfstem / filename).as_posix()
                out_file = out_dir / filename

                try:
                    pix = fitz.Pixmap(image_bytes)
                    if pix.alpha:
                        pix = fitz.Pixmap(pix, 0)
                    if pix.n >= 4 and pix.colorspace and pix.colorspace.n >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(str(out_file))
                except Exception:
                    out_file.write_bytes(image_bytes)

                queue_entries.append(
                    {
                        "source": source,
                        "vendor": vendor,
                        "section": None,
                        "image_src": rel_image_src,
                        "page": page_one,
                        "url": None,
                        "captioned": False,
                    }
                )
                counts["extracted"] += 1
                kept_idx += 1

    added = _append_pending_captions(vendor, queue_entries)
    counts["queue_added"] = added
    counts["queue_skipped_dup"] = len(queue_entries) - added
    return counts, len(queue_entries)


def _print_summary(pdf_path: Path, counts: Counter[str]) -> None:
    extracted = counts.get("extracted", 0)
    skipped = sum(counts[k] for k in counts if k not in ("extracted", "queue_added", "queue_skipped_dup"))
    parts = [f"{pdf_path.name}: extracted {extracted}", f"skipped {skipped}"]
    skip_details = [
        (k, counts[k])
        for k in (
            "too_small_dim",
            "too_small_area",
            "aspect_ratio",
            "solid_fill",
            "near-blank",
            "duplicate_xref",
            "empty_image",
            "extract_error",
        )
        if counts.get(k)
    ]
    if skip_details:
        parts.append("(" + ", ".join(f"{k}={v}" for k, v in skip_details) + ")")
    if counts.get("queue_added") or counts.get("queue_skipped_dup"):
        parts.append(
            f"queue +{counts.get('queue_added', 0)}"
            f" (dup {counts.get('queue_skipped_dup', 0)})"
        )
    print(" ".join(parts))


def _collect_pdfs(args: argparse.Namespace) -> list[Path]:
    pdfs: list[Path] = []
    if args.all:
        for folder in args.paths:
            root = Path(folder)
            if not root.is_dir():
                raise NotADirectoryError(folder)
            pdfs.extend(sorted(root.glob("*.pdf")))
    else:
        for path in args.paths:
            p = Path(path)
            if p.is_dir():
                pdfs.extend(sorted(p.glob("*.pdf")))
            else:
                pdfs.append(p)
    return pdfs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract embedded PDF images into pending_captions queue",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="PDF file(s), or folder with --all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every *.pdf in each folder path",
    )
    parser.add_argument(
        "--verify-qdrant",
        action="store_true",
        help="Warn when PDF source is missing from Qdrant (read-only REST check)",
    )
    args = parser.parse_args(argv)

    try:
        pdfs = _collect_pdfs(args)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not pdfs:
        print("No PDF files found.", file=sys.stderr)
        return 1

    exit_code = 0
    for pdf_path in pdfs:
        try:
            counts, _ = extract_pdf_images(pdf_path, verify_qdrant=args.verify_qdrant)
            _print_summary(pdf_path, counts)
        except Exception as exc:
            print(f"Error processing {pdf_path}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
