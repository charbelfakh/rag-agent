#!/usr/bin/env python3
"""Read-only Qdrant coverage audit against the ingest manifest.

Run: ``python -m scripts.ops.audit``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from providers.config_paths import INGEST_MANIFEST_PATH

MANIFEST_FILENAME = INGEST_MANIFEST_PATH.name
SCHEMA_VERSION = 2
SCROLL_LIMIT = 500


@dataclass
class SourceStats:
    vendor: str = ""
    doc_type: str = ""
    product: str | None = None
    chunks: int = 0
    sections: set[str] = field(default_factory=set)
    has_url: bool = False
    ingested_at: str = ""
    legacy_points: int = 0

    def observe(self, payload: dict) -> None:
        self.chunks += 1
        vendor = (payload.get("vendor") or "").strip().lower()
        if vendor:
            self.vendor = vendor
        doc_type = payload.get("doc_type") or payload.get("document_type") or ""
        if doc_type and not self.doc_type:
            self.doc_type = str(doc_type)
        product = payload.get("product")
        if product and not self.product:
            self.product = str(product)
        section = payload.get("section")
        if section:
            self.sections.add(str(section))
        if payload.get("url"):
            self.has_url = True
        ingested_at = payload.get("ingested_at") or payload.get("ingestion_timestamp") or ""
        if ingested_at and ingested_at > self.ingested_at:
            self.ingested_at = str(ingested_at)
        schema = payload.get("schema_version")
        if schema is None or int(schema) != SCHEMA_VERSION:
            self.legacy_points += 1


@dataclass
class AuditData:
    collection: str
    sources: dict[str, SourceStats] = field(default_factory=dict)
    vendor_chunks: Counter[str] = field(default_factory=Counter)
    vendor_sources: Counter[str] = field(default_factory=Counter)
    content_types: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    doc_types: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(s.chunks for s in self.sources.values())

    @property
    def total_sources(self) -> int:
        return len(self.sources)

    @property
    def total_vendors(self) -> int:
        return len(self.vendor_sources)


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


def _collection_exists(client: QdrantClient, collection: str) -> bool:
    names = {c.name for c in client.get_collections().collections}
    return collection in names


def _scroll_payloads(client: QdrantClient, collection: str) -> list[dict]:
    payloads: list[dict] = []
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
            payloads.append(dict(record.payload or {}))
        if offset is None:
            break
    return payloads


def _load_manifest() -> tuple[dict | None, str | None]:
    path = INGEST_MANIFEST_PATH
    if not path.is_file():
        return None, f"Notice: {MANIFEST_FILENAME} not found — skipping manifest checks."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data, None
    except json.JSONDecodeError as exc:
        return None, f"Notice: malformed {MANIFEST_FILENAME} ({exc}) — skipping manifest checks."
    return None, f"Notice: unexpected {MANIFEST_FILENAME} format — skipping manifest checks."


def _aggregate(payloads: list[dict], collection: str) -> AuditData:
    data = AuditData(collection=collection)
    for payload in payloads:
        source = (payload.get("source") or "").strip()
        if not source:
            continue
        stats = data.sources.setdefault(source, SourceStats())
        stats.observe(payload)
        vendor = stats.vendor or "unknown"
        data.vendor_chunks[vendor] += 1
        content_type = (payload.get("content_type") or "unknown").strip() or "unknown"
        doc_type = (
            payload.get("doc_type")
            or payload.get("document_type")
            or "unknown"
        )
        data.content_types[vendor][str(content_type)] += 1
        data.doc_types[vendor][str(doc_type)] += 1

    for stats in data.sources.values():
        data.vendor_sources[stats.vendor or "unknown"] += 1
    return data


def _build_warnings(data: AuditData, manifest: dict | None) -> list[str]:
    warnings: list[str] = []

    for source, stats in sorted(data.sources.items()):
        if stats.chunks < 3:
            warnings.append(
                f"WARN: {source} — possibly empty/shell page ({stats.chunks} chunks)"
            )
        if stats.chunks > 0 and not stats.sections:
            warnings.append(f"WARN: {source} — no headings captured")
        if stats.legacy_points:
            warnings.append(
                f"WARN: {source} — legacy/malformed points ({stats.legacy_points})"
            )

    if manifest is not None:
        qdrant_sources = set(data.sources)
        manifest_sources = set(manifest.keys())

        for source in sorted(manifest_sources - qdrant_sources):
            warnings.append(
                f"WARN: {source} — manifest/DB mismatch (deleted from DB?)"
            )

        for source in sorted(qdrant_sources - manifest_sources):
            warnings.append(f"WARN: {source} — untracked source")

        for source in sorted(qdrant_sources & manifest_sources):
            manifest_chunks = manifest[source].get("chunk_count")
            actual = data.sources[source].chunks
            if manifest_chunks is not None and int(manifest_chunks) != actual:
                warnings.append(
                    f"WARN: {source} — chunk count drift "
                    f"(manifest={manifest_chunks}, qdrant={actual})"
                )

    return warnings


def _format_counter(counter: Counter) -> str:
    return ", ".join(f"{k}: {v}" for k, v in sorted(counter.items()))


def _print_vendor_summary(data: AuditData) -> None:
    vendors = sorted(data.vendor_chunks.keys())
    if not vendors:
        print("No points found.")
        return

    rows: list[tuple[str, str, str, str, str]] = []
    for vendor in vendors:
        rows.append(
            (
                vendor,
                str(data.vendor_sources[vendor]),
                str(data.vendor_chunks[vendor]),
                _format_counter(data.content_types[vendor]),
                _format_counter(data.doc_types[vendor]),
            )
        )

    headers = ("vendor", "sources", "chunks", "content_types", "doc_types")
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: tuple[str, ...]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("PER-VENDOR SUMMARY")
    print(line(headers))
    print(line(tuple("-" * widths[i] for i in range(len(headers)))))
    for row in rows:
        print(line(row))
    print()


def _print_source_detail(data: AuditData, vendor_filter: str | None) -> None:
    rows: list[tuple[str, ...]] = []
    for source, stats in sorted(data.sources.items()):
        vendor = stats.vendor or "unknown"
        if vendor_filter and vendor != vendor_filter.lower():
            continue
        rows.append(
            (
                source,
                vendor,
                stats.doc_type or "-",
                stats.product or "-",
                str(stats.chunks),
                str(len(stats.sections)),
                "yes" if stats.has_url else "no",
                stats.ingested_at or "-",
            )
        )

    if not rows:
        if vendor_filter:
            print(f"No sources for vendor '{vendor_filter}'.")
        return

    headers = (
        "source",
        "vendor",
        "doc_type",
        "product",
        "chunks",
        "sections",
        "url",
        "ingested_at",
    )
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: tuple[str, ...]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    title = "PER-SOURCE DETAIL"
    if vendor_filter:
        title += f" ({vendor_filter})"
    print(title)
    print(line(headers))
    print(line(tuple("-" * widths[i] for i in range(len(headers)))))
    for row in rows:
        print(line(row))
    print()


def _to_json(data: AuditData) -> dict:
    vendors: dict[str, dict] = {}
    for source, stats in data.sources.items():
        vendor = stats.vendor or "unknown"
        bucket = vendors.setdefault(
            vendor,
            {"chunks": 0, "sources": {}},
        )
        bucket["chunks"] += stats.chunks
        bucket["sources"][source] = {
            "chunks": stats.chunks,
            "doc_type": stats.doc_type,
            "product": stats.product,
            "distinct_sections": len(stats.sections),
            "has_url": stats.has_url,
            "ingested_at": stats.ingested_at,
        }

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collection": data.collection,
        "vendors": vendors,
        "warnings": data.warnings,
    }


def run_audit(
    *,
    detail: bool,
    vendor: str | None,
    json_path: str | None,
) -> int:
    url, collection = _load_env()
    client = _connect(url)

    if not _collection_exists(client, collection):
        print(f"Collection '{collection}' does not exist — nothing to audit.")
        print(
            f"TOTALS: vendors: 0 | sources: 0 | chunks: 0 | collection: {collection}"
        )
        return 0

    payloads = _scroll_payloads(client, collection)
    if not payloads:
        print(f"Collection '{collection}' is empty — nothing to audit.")
        print(
            f"TOTALS: vendors: 0 | sources: 0 | chunks: 0 | collection: {collection}"
        )
        return 0

    data = _aggregate(payloads, collection)
    manifest, manifest_notice = _load_manifest()
    if manifest_notice:
        print(manifest_notice)
    data.warnings = _build_warnings(data, manifest)

    _print_vendor_summary(data)
    if detail or vendor:
        _print_source_detail(data, vendor.lower() if vendor else None)
    if data.warnings:
        print("HEALTH WARNINGS")
        for warning in data.warnings:
            print(warning)
        print()

    print(
        "TOTALS: "
        f"vendors: {data.total_vendors} | "
        f"sources: {data.total_sources} | "
        f"chunks: {data.total_chunks} | "
        f"collection: {data.collection}"
    )

    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(
            json.dumps(_to_json(data), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report to {json_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only knowledge base coverage audit (Qdrant)"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show per-source table for all vendors",
    )
    parser.add_argument(
        "--vendor",
        default=None,
        help="Show per-source table for one vendor",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        metavar="PATH",
        help="Write full aggregation JSON to PATH",
    )
    args = parser.parse_args(argv)
    return run_audit(
        detail=args.detail,
        vendor=args.vendor,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    sys.exit(main())
