#!/usr/bin/env python3
"""Backfill extended metadata on existing Qdrant points from ``source`` paths.

Run: ``python -m scripts.ops.migrate_metadata`` (``--dry-run`` supported).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from providers.factory import get_vector_store
from providers.metadata import (
    infer_vendor_from_source,
    make_chunk_id,
    normalize_source,
)


def migrate(*, dry_run: bool = False) -> int:
    store = get_vector_store()
    client = store.client
    collection = store.collection
    offset = None
    updated = 0
    by_source: dict[str, list] = {}

    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break
        for record in records:
            payload = dict(record.payload or {})
            source = normalize_source(payload.get("source", ""))
            if source:
                by_source.setdefault(source, []).append((record, payload))
        if offset is None:
            break

    for source, items in by_source.items():
        items.sort(
            key=lambda item: (
                int(item[1].get("page", 0)),
                item[1].get("section", ""),
                item[1].get("text", "")[:80],
            )
        )
        total_chunks = len(items)
        path = Path(source)
        for index, (record, payload) in enumerate(items):
            payload["source"] = source
            payload.setdefault("vendor", infer_vendor_from_source(source))
            payload.setdefault("file_name", path.name)
            payload.setdefault("file_extension", path.suffix.lower())
            payload.setdefault("document_type", "manual")
            payload.setdefault("language", "en")
            payload.setdefault("content_type", "text")
            payload["chunk_index"] = index
            payload["total_chunks"] = total_chunks
            payload["chunk_id"] = make_chunk_id(source, index)
            if not dry_run:
                from qdrant_client.models import PointStruct

                client.upsert(
                    collection_name=collection,
                    points=[
                        PointStruct(
                            id=record.id,
                            vector=record.vector,
                            payload=payload,
                        )
                    ],
                )
            updated += 1

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Qdrant chunk metadata")
    parser.add_argument("--dry-run", action="store_true", help="Count updates only")
    args = parser.parse_args()
    count = migrate(dry_run=args.dry_run)
    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {count} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
