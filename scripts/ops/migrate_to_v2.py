#!/usr/bin/env python3
"""Copy a Qdrant collection to ``rag_docs_v2`` with ``schema_version=2`` and stable IDs.

Run: ``python -m scripts.ops.migrate_to_v2`` (``--dry-run`` supported). After success,
set ``QDRANT_COLLECTION=rag_docs_v2`` in ``.env`` and restart the API.
"""
from __future__ import annotations

import argparse
import os

import scripts._bootstrap  # noqa: F401 — ``providers.*`` on direct script runs

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from providers.metadata import make_chunk_id, normalize_source

PAYLOAD_INDEX_FIELDS = ("vendor", "document_type", "content_type", "source")
SCROLL_LIMIT = int(os.getenv("QDRANT_SCROLL_PAGE_SIZE", "500"))


def _ensure_target_collection(client: QdrantClient, target: str, vector_size: int) -> None:
    names = [c.name for c in client.get_collections().collections]
    if target not in names:
        client.create_collection(
            collection_name=target,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    for field in PAYLOAD_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=target,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def _point_id(payload: dict, record_id) -> str:
    chunk_id = payload.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    source = normalize_source(payload.get("source", ""))
    index = int(payload.get("chunk_index", 0))
    if source:
        return make_chunk_id(source, index)
    return str(record_id)


def migrate_collection(
    *,
    source: str,
    target: str,
    dry_run: bool = False,
    delete_target_first: bool = False,
) -> dict:
    url = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
    client = QdrantClient(url=url)

    info = client.get_collection(source)
    vector_size = info.config.params.vectors.size

    if dry_run:
        count = client.count(collection_name=source).count
        return {
            "source": source,
            "target": target,
            "points": count,
            "dry_run": True,
        }

    if delete_target_first:
        names = [c.name for c in client.get_collections().collections]
        if target in names:
            client.delete_collection(target)

    _ensure_target_collection(client, target, vector_size)

    offset = None
    copied = 0
    while True:
        records, offset = client.scroll(
            collection_name=source,
            limit=SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break
        points = []
        for record in records:
            payload = dict(record.payload or {})
            payload["schema_version"] = 2
            source_path = normalize_source(payload.get("source", ""))
            if source_path:
                payload["source"] = source_path
                payload.setdefault(
                    "chunk_id",
                    make_chunk_id(source_path, int(payload.get("chunk_index", 0))),
                )
            points.append(
                PointStruct(
                    id=_point_id(payload, record.id),
                    vector=record.vector,
                    payload=payload,
                )
            )
        if points:
            client.upsert(collection_name=target, points=points)
            copied += len(points)
        if offset is None:
            break

    return {
        "source": source,
        "target": target,
        "points": copied,
        "dry_run": False,
        "vector_size": vector_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=os.getenv("QDRANT_COLLECTION", "rag_docs"),
        help="Source collection name",
    )
    parser.add_argument(
        "--target",
        default=os.getenv("QDRANT_V2_COLLECTION", "rag_docs_v2"),
        help="Target v2 collection name",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count only")
    parser.add_argument(
        "--replace-target",
        action="store_true",
        help="Delete target collection before copy",
    )
    args = parser.parse_args()

    result = migrate_collection(
        source=args.source,
        target=args.target,
        dry_run=args.dry_run,
        delete_target_first=args.replace_target,
    )
    if result["dry_run"]:
        print(
            f"Would copy {result['points']} points from {result['source']} "
            f"to {result['target']}"
        )
    else:
        print(
            f"Copied {result['points']} points from {result['source']} "
            f"to {result['target']}"
        )
        print(f"Set QDRANT_COLLECTION={args.target} in .env and restart the API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
