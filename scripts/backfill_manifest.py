#!/usr/bin/env python3
"""Backfill ``ingest_manifest.json`` rows for legacy video/URL Qdrant sources.

Run: ``python -m scripts.backfill_manifest`` (``--apply`` to write; default dry-run).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Same role as scripts._bootstrap for direct ``python scripts/backfill_manifest.py`` runs.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ingest.ingest import load_manifest, upsert_manifest_source

SCROLL_LIMIT = int(os.getenv("QDRANT_SCROLL_PAGE_SIZE", "500"))


@dataclass
class SourceAggregate:
    source: str
    chunk_count: int = 0
    vendor: str = ""
    content_type: str = ""
    source_type: str = ""
    video_id: str | None = None
    product: str | None = None
    content_types: set[str] = field(default_factory=set)
    source_types: set[str] = field(default_factory=set)

    def observe(self, payload: dict) -> None:
        self.chunk_count += 1
        vendor = (payload.get("vendor") or "").strip().lower()
        if vendor and not self.vendor:
            self.vendor = vendor
        content_type = (payload.get("content_type") or "").strip()
        if content_type:
            self.content_types.add(content_type)
            if not self.content_type:
                self.content_type = content_type
        source_type = (payload.get("source_type") or "").strip()
        if source_type:
            self.source_types.add(source_type)
            if not self.source_type:
                self.source_type = source_type
        video_id = payload.get("video_id")
        if video_id and not self.video_id:
            self.video_id = str(video_id)
        product = payload.get("product")
        if product and not self.product:
            self.product = str(product)


def is_http_url(source: str) -> bool:
    lowered = source.strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def is_backfill_candidate(agg: SourceAggregate, scope: str) -> bool:
    if agg.content_type == "video_transcript":
        return scope in ("all", "video_transcript")
    if agg.source_type == "html" and is_http_url(agg.source):
        return scope in ("all", "html")
    return False


def backfill_hash_marker(agg: SourceAggregate) -> str:
    if agg.content_type == "video_transcript":
        return "backfill:video_transcript"
    if agg.source_type == "html":
        return "backfill:html"
    return f"backfill:{agg.content_type or 'unknown'}"


def aggregate_sources_from_scroll(client, collection: str) -> dict[str, SourceAggregate]:
    aggregated: dict[str, SourceAggregate] = {}
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break
        for record in records:
            payload = dict(record.payload or {})
            source = (payload.get("source") or "").strip()
            if not source:
                continue
            agg = aggregated.get(source)
            if agg is None:
                agg = SourceAggregate(source=source)
                aggregated[source] = agg
            agg.observe(payload)
        if offset is None:
            break
    return aggregated


def plan_backfill_rows(
    aggregated: dict[str, SourceAggregate],
    manifest: dict,
    *,
    scope: str,
    force: bool,
) -> tuple[list[SourceAggregate], int, int]:
    """Return rows to write, skipped-existing count, and non-candidate count."""
    selected: list[SourceAggregate] = []
    skipped_existing = 0
    skipped_other = 0

    for agg in sorted(aggregated.values(), key=lambda row: row.source):
        if not is_backfill_candidate(agg, scope):
            skipped_other += 1
            continue
        if agg.source in manifest and not force:
            skipped_existing += 1
            continue
        selected.append(agg)

    return selected, skipped_existing, skipped_other


def apply_backfill_rows(rows: list[SourceAggregate]) -> int:
    written = 0
    for agg in rows:
        extra: dict[str, str] = {
            "content_type": agg.content_type,
            "source_type": agg.source_type,
        }
        if agg.video_id:
            extra["video_id"] = agg.video_id
        upsert_manifest_source(
            agg.source,
            file_hash=backfill_hash_marker(agg),
            vendor=agg.vendor or "unknown",
            product=agg.product,
            chunk_count=agg.chunk_count,
            extra=extra,
        )
        written += 1
    return written


def run_backfill(
    *,
    client,
    collection: str,
    scope: str = "all",
    force: bool = False,
    apply: bool = False,
) -> dict[str, int]:
    aggregated = aggregate_sources_from_scroll(client, collection)
    manifest = load_manifest()
    rows, skipped_existing, skipped_other = plan_backfill_rows(
        aggregated,
        manifest,
        scope=scope,
        force=force,
    )

    written = 0
    if apply:
        written = apply_backfill_rows(rows)

    return {
        "sources_scanned": len(aggregated),
        "selected": len(rows),
        "written": written,
        "skipped_existing": skipped_existing,
        "skipped_other": skipped_other,
    }


def _print_plan(rows: list[SourceAggregate], *, limit: int = 20) -> None:
    if not rows:
        print("No manifest rows to write.")
        return
    print(f"Would write {len(rows)} manifest row(s):")
    for agg in rows[:limit]:
        kind = (
            "video_transcript"
            if agg.content_type == "video_transcript"
            else "html-url"
        )
        video = f" video_id={agg.video_id}" if agg.video_id else ""
        print(
            f"  [{kind}] {agg.source!r} "
            f"chunks={agg.chunk_count} vendor={agg.vendor}{video}"
        )
    if len(rows) > limit:
        print(f"  ... and {len(rows) - limit} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill ingest_manifest.json for legacy video/URL Qdrant sources"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write manifest rows (default is dry-run only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite manifest rows that already exist for a source",
    )
    parser.add_argument(
        "--content-type",
        choices=("all", "video_transcript", "html"),
        default="all",
        help="Limit backfill scope (default: all)",
    )
    args = parser.parse_args(argv)

    from providers.factory import get_vector_store, reset_providers

    reset_providers()
    store = get_vector_store()
    client = store.client
    collection = store.collection

    aggregated = aggregate_sources_from_scroll(client, collection)
    manifest = load_manifest()
    rows, skipped_existing, skipped_other = plan_backfill_rows(
        aggregated,
        manifest,
        scope=args.content_type,
        force=args.force,
    )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode} | collection={collection!r} | scope={args.content_type}")
    if not args.apply:
        _print_plan(rows)

    written = apply_backfill_rows(rows) if args.apply else 0

    print(
        "Summary: "
        f"sources_scanned={len(aggregated)} "
        f"selected={len(rows)} "
        f"written={written} "
        f"skipped_existing={skipped_existing} "
        f"skipped_other={skipped_other}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
