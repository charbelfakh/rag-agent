"""RedisVL HNSW backend for semantic cache (O(1) vector lookup vs legacy O(n) scan)."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

from providers.semantic_cache import CACHE_STATS_KEY, cosine_similarity

logger = logging.getLogger(__name__)

CACHE_ORDER_KEY = "semantic_cache:redisvl:order"
LEGACY_CACHE_KEY = "semantic_cache:entries"


def _vector_dims() -> int:
    return int(os.getenv("SEMANTIC_CACHE_VECTOR_DIMS", "768"))


def _index_schema(dims: int) -> dict:
    return {
        "index": {
            "name": os.getenv("SEMANTIC_CACHE_INDEX_NAME", "rag_semantic_cache"),
            "prefix": "semantic_cache:entry:",
            "storage_type": "json",
        },
        "fields": [
            {"name": "question", "type": "text"},
            {"name": "answer", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": dims,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32",
                },
            },
        ],
    }


class RedisVLSemanticCacheBackend:
    """HNSW vector index backend; migrates legacy list entries on first connect."""

    def __init__(self, redis_client, *, threshold: float, max_size: int):
        from redisvl.index import SearchIndex
        from redisvl.query import VectorQuery

        self._redis = redis_client
        self.threshold = threshold
        self.max_size = max_size
        self._VectorQuery = VectorQuery
        dims = _vector_dims()
        self._index = SearchIndex.from_dict(
            _index_schema(dims),
            redis_client=redis_client,
            validate_on_load=True,
        )
        self._ensure_index()
        if os.getenv("SEMANTIC_CACHE_MIGRATE_LEGACY", "true").lower() in (
            "true",
            "1",
            "yes",
        ):
            self._migrate_legacy_entries()

    def _ensure_index(self) -> None:
        try:
            self._index.create(overwrite=False)
        except Exception as exc:
            logger.debug("Semantic cache index create: %s", exc)

    def _migrate_legacy_entries(self) -> None:
        raw_entries = self._redis.lrange(LEGACY_CACHE_KEY, 0, -1)
        if not raw_entries:
            return
        imported = 0
        for raw in raw_entries:
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            vector = entry.get("vector")
            answer = entry.get("answer")
            question = entry.get("question", "")
            if not vector or not answer:
                continue
            self.store(question, vector, answer)
            imported += 1
        if imported:
            self._redis.delete(LEGACY_CACHE_KEY)
            logger.info("Migrated %d legacy semantic cache entries to RedisVL", imported)

    def _incr_stat(self, field: str) -> None:
        try:
            self._redis.hincrby(CACHE_STATS_KEY, field, 1)
        except Exception as exc:
            logger.debug("Cache stat increment failed: %s", exc)

    def _trim_to_max_size(self) -> None:
        while self._redis.llen(CACHE_ORDER_KEY) > self.max_size:
            entry_id = self._redis.lpop(CACHE_ORDER_KEY)
            if not entry_id:
                break
            try:
                self._index.delete([entry_id])
            except Exception as exc:
                logger.debug("Cache trim delete failed for %s: %s", entry_id, exc)

    def lookup(self, vector: list[float]) -> tuple[str, float] | None:
        t0 = time.perf_counter()
        try:
            query = self._VectorQuery(
                vector=vector,
                vector_field_name="embedding",
                return_fields=["question", "answer", "embedding"],
                num_results=3,
                return_score=True,
            )
            results = self._index.query(query)
        except Exception as exc:
            logger.warning("RedisVL semantic cache lookup failed: %s", exc)
            return None

        best_sim = 0.0
        best_answer = None
        for doc in results:
            row = dict(doc) if not isinstance(doc, dict) else doc
            answer = row.get("answer")
            if not answer:
                continue
            stored_vector = row.get("embedding")
            if stored_vector:
                sim = cosine_similarity(vector, stored_vector)
            else:
                distance = float(row.get("vector_distance", row.get("score", 1.0)))
                sim = max(0.0, 1.0 - distance)
            if sim >= self.threshold and sim > best_sim:
                best_sim = sim
                best_answer = answer

        if best_answer is not None:
            self._incr_stat("hits")
            return best_answer, best_sim
        self._incr_stat("misses")
        return None

    def store(self, question: str, vector: list[float], answer: str) -> None:
        entry_id = uuid.uuid4().hex
        record = {
            "id": entry_id,
            "question": question,
            "answer": answer,
            "embedding": vector,
        }
        try:
            self._index.load([record])
            self._redis.rpush(CACHE_ORDER_KEY, entry_id)
            self._trim_to_max_size()
        except Exception as exc:
            logger.warning("RedisVL semantic cache store failed: %s", exc)
