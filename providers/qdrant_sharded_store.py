"""Vendor-sharded Qdrant collections for multi-tenant scale (Sprint K)."""
from __future__ import annotations

import logging
import os
import re

from providers.qdrant_store import QdrantLocalStore

logger = logging.getLogger(__name__)

_VENDOR_SAFE = re.compile(r"[^a-z0-9_]+")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def is_vendor_sharding_enabled() -> bool:
    return _env_bool("QDRANT_VENDOR_SHARDING")


def normalize_vendor_shard(vendor: str) -> str:
    value = vendor.strip().lower()
    value = _VENDOR_SAFE.sub("_", value)
    return value or "unknown"


class VendorShardedQdrantStore(QdrantLocalStore):
    """Routes reads/writes to per-vendor collections when sharding is enabled."""

    def __init__(self):
        self.shard_prefix = os.getenv("QDRANT_COLLECTION_PREFIX", "rag_docs")
        self.default_collection = os.getenv("QDRANT_COLLECTION", "rag_docs")
        super().__init__()

    def _shard_collection(self, vendor: str | None) -> str:
        if not vendor:
            return self.default_collection
        return f"{self.shard_prefix}_{normalize_vendor_shard(vendor)}"

    def _all_shard_collections(self) -> list[str]:
        names = self._collection_names()
        prefix = f"{self.shard_prefix}_"
        shards = [name for name in names if name.startswith(prefix)]
        if self.default_collection in names:
            shards.append(self.default_collection)
        return sorted(set(shards))

    def upsert(self, ids, vectors, payloads) -> None:
        if not is_vendor_sharding_enabled():
            return super().upsert(ids, vectors, payloads)

        batches: dict[str, tuple[list, list, list]] = {}
        for point_id, vector, payload in zip(ids, vectors, payloads):
            collection = self._shard_collection(payload.get("vendor"))
            bucket = batches.setdefault(collection, ([], [], []))
            bucket[0].append(point_id)
            bucket[1].append(vector)
            bucket[2].append(payload)

        for collection, (batch_ids, batch_vectors, batch_payloads) in batches.items():
            previous = self.collection
            self.collection = collection
            try:
                self._ensure_collection()
                self._ensure_payload_indexes()
                super().upsert(batch_ids, batch_vectors, batch_payloads)
            finally:
                self.collection = previous

    def search(self, vector, top_k: int = 5, filter_payload: dict | None = None) -> list[dict]:
        if not is_vendor_sharding_enabled():
            return super().search(vector, top_k=top_k, filter_payload=filter_payload)

        vendor = (filter_payload or {}).get("vendor")
        if vendor:
            previous = self.collection
            self.collection = self._shard_collection(vendor)
            try:
                self._ensure_collection()
                return super().search(vector, top_k=top_k, filter_payload=filter_payload)
            finally:
                self.collection = previous

        merged: list[dict] = []
        for collection in self._all_shard_collections():
            previous = self.collection
            self.collection = collection
            try:
                hits = super().search(vector, top_k=top_k, filter_payload=filter_payload)
                merged.extend(hits)
            except Exception as exc:
                logger.debug("Shard search %s skipped: %s", collection, exc)
            finally:
                self.collection = previous

        merged.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return merged[:top_k]

    def delete_by_source(self, source: str) -> int:
        if not is_vendor_sharding_enabled():
            return super().delete_by_source(source)

        deleted = 0
        for collection in self._all_shard_collections():
            previous = self.collection
            self.collection = collection
            try:
                deleted += super().delete_by_source(source)
            finally:
                self.collection = previous
        return deleted

    def list_sources(self) -> list[dict]:
        if not is_vendor_sharding_enabled():
            return super().list_sources()

        merged: dict[str, dict] = {}
        for collection in self._all_shard_collections():
            previous = self.collection
            self.collection = collection
            try:
                for row in super().list_sources():
                    merged[row["source"]] = row
            finally:
                self.collection = previous
        return sorted(merged.values(), key=lambda row: (row["vendor"], row["source"]))

    def get_source_content_hash(self, source: str) -> str | None:
        if not is_vendor_sharding_enabled():
            return super().get_source_content_hash(source)

        for collection in self._all_shard_collections():
            previous = self.collection
            self.collection = collection
            try:
                value = super().get_source_content_hash(source)
                if value:
                    return value
            finally:
                self.collection = previous
        return None

    def patch_total_chunks(self, source: str, total_chunks: int) -> int:
        if not is_vendor_sharding_enabled():
            return super().patch_total_chunks(source, total_chunks)

        updated = 0
        for collection in self._all_shard_collections():
            previous = self.collection
            self.collection = collection
            try:
                updated += super().patch_total_chunks(source, total_chunks)
            finally:
                self.collection = previous
        return updated
