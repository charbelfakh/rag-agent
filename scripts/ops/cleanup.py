#!/usr/bin/env python3
"""Delete or retag Qdrant sources without re-ingestion. Run: ``python -m scripts.ops.cleanup``

Operations:
  - Delete all points matching payload.source (and drop manifest entries).
  - Set payload.product for matching points (manifest unchanged).

Config is read from ``.env`` (``QDRANT_LOCAL_URL``, ``QDRANT_COLLECTION``) like audit.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from providers.config_paths import INGEST_MANIFEST_PATH

MANIFEST_FILENAME = INGEST_MANIFEST_PATH.name


def _load_env() -> tuple[str, str]:
    load_dotenv()
    url = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "rag_docs")
    return url, collection


def _connect(url: str) -> QdrantClient:
    kwargs: dict = {"url": url}
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if api_key:
        kwargs["api_key"] = api_key
    grpc_port = os.getenv("QDRANT_GRPC_PORT", "").strip()
    if grpc_port:
        kwargs["grpc_port"] = int(grpc_port)
    return QdrantClient(**kwargs)


def _source_filter(source: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=source))]
    )


def _read_source_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Source list not found: {path}")
    sources: list[str] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        sources.append(line)
    if not sources:
        raise ValueError(f"No sources in {path} (after skipping blanks and # comments)")
    return sources


def _count_by_source(client: QdrantClient, collection: str, sources: list[str]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for source in sources:
        count = client.count(
            collection_name=collection,
            count_filter=_source_filter(source),
        ).count
        rows.append((source, count))
    return rows


def _print_plan(action: str, rows: list[tuple[str, int]], *, dry_run: bool) -> int:
    total_points = sum(count for _, count in rows)
    print(f"\n{action}")
    print("-" * 60)
    for source, count in rows:
        print(f"  {source}: {count} points")
    print("-" * 60)
    print(f"Sources: {len(rows)} | Points: {total_points}")
    if dry_run:
        print("\nDry run — no changes made.")
    return total_points


def _confirm(action: str, rows: list[tuple[str, int]], *, assume_yes: bool) -> bool:
    _print_plan(action, rows, dry_run=False)
    if assume_yes:
        return True
    try:
        answer = input("\nProceed? [y/N] ").strip().lower()
    except EOFError:
        print("\nCancelled (no input).")
        return False
    return answer in ("y", "yes")


def _load_manifest() -> dict:
    path = INGEST_MANIFEST_PATH
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _save_manifest(manifest: dict) -> None:
    path = INGEST_MANIFEST_PATH
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _delete_sources(
    client: QdrantClient,
    collection: str,
    sources: list[str],
    *,
    dry_run: bool,
    assume_yes: bool,
) -> int:
    rows = _count_by_source(client, collection, sources)
    if dry_run:
        _print_plan("DELETE (dry run)", rows, dry_run=True)
        return 0

    if not _confirm("DELETE — affected sources", rows, assume_yes=assume_yes):
        print("Cancelled.")
        return 0

    manifest = _load_manifest()
    points_deleted = 0
    manifest_removed = 0

    for source, count in rows:
        if count > 0:
            client.delete(
                collection_name=collection,
                points_selector=_source_filter(source),
            )
            points_deleted += count
        if source in manifest:
            del manifest[source]
            manifest_removed += 1

    if manifest_removed:
        _save_manifest(manifest)

    print(
        "\nSummary: "
        f"sources={len(sources)} | "
        f"points_deleted={points_deleted} | "
        f"manifest_entries_removed={manifest_removed}"
    )
    return 0


def _retag_sources(
    client: QdrantClient,
    collection: str,
    sources: list[str],
    product: str,
    *,
    dry_run: bool,
    assume_yes: bool,
) -> int:
    rows = _count_by_source(client, collection, sources)
    action = f'RETAG product="{product}" (dry run)' if dry_run else f'RETAG product="{product}"'

    if dry_run:
        _print_plan(action, rows, dry_run=True)
        return 0

    if not _confirm(action, rows, assume_yes=assume_yes):
        print("Cancelled.")
        return 0

    points_retagged = 0
    for source, count in rows:
        if count == 0:
            continue
        client.set_payload(
            collection_name=collection,
            payload={"product": product},
            points=_source_filter(source),
            wait=True,
        )
        points_retagged += count

    print(
        "\nSummary: "
        f"sources={len(sources)} | "
        f"points_retagged={points_retagged} | "
        f"manifest_entries_removed=0"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete or retag Qdrant sources by payload.source (no re-ingestion)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--delete-sources",
        metavar="FILE",
        help="Newline-delimited source filenames to delete (# comments allowed)",
    )
    mode.add_argument(
        "--retag",
        metavar="FILE",
        help="Newline-delimited source filenames to retag payload.product",
    )
    parser.add_argument(
        "--product",
        default=None,
        help='New product value for --retag (e.g. "hexsight")',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-source point counts only; make no changes",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation",
    )
    args = parser.parse_args(argv)

    if args.retag and not args.product:
        parser.error("--product is required with --retag")
    if args.delete_sources and args.product:
        parser.error("--product applies only to --retag")

    list_path = Path(args.delete_sources or args.retag)
    try:
        sources = _read_source_list(list_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    url, collection = _load_env()
    client = _connect(url)

    names = {c.name for c in client.get_collections().collections}
    if collection not in names:
        print(f"Collection '{collection}' does not exist.", file=sys.stderr)
        return 1

    if args.delete_sources:
        return _delete_sources(
            client,
            collection,
            sources,
            dry_run=args.dry_run,
            assume_yes=args.yes,
        )

    return _retag_sources(
        client,
        collection,
        sources,
        args.product.strip(),
        dry_run=args.dry_run,
        assume_yes=args.yes,
    )


if __name__ == "__main__":
    sys.exit(main())
