"""Redis-backed ingest work queue for horizontal API + worker deployments."""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

INGEST_QUEUE_KEY = os.getenv("INGEST_QUEUE_KEY", "ingest:queue")
INGEST_CLAIM_PREFIX = "ingest:claim:"

_store = None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def is_ingest_queue_enabled() -> bool:
    return _env_bool("INGEST_QUEUE_ENABLED")


class IngestQueue:
    """Redis list queue for decoupled ingest workers."""

    def __init__(self):
        self._redis = None
        self.claim_ttl_seconds = int(os.getenv("INGEST_CLAIM_TTL_SECONDS", "7200"))
        if is_ingest_queue_enabled():
            self._connect()

    def _connect(self) -> None:
        try:
            import redis

            url = os.getenv("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis = client
        except Exception as exc:
            logger.warning("Ingest queue disabled: Redis unreachable (%s)", exc)
            self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def enqueue(self, payload: dict) -> bool:
        if self._redis is None:
            return False
        body = json.dumps(payload)
        self._redis.lpush(INGEST_QUEUE_KEY, body)
        return True

    def dequeue(self, *, block_timeout: int = 5) -> dict | None:
        if self._redis is None:
            return None
        item = self._redis.brpop(INGEST_QUEUE_KEY, timeout=block_timeout)
        if not item:
            return None
        _, raw = item
        job = json.loads(raw)
        job_id = job.get("job_id")
        if job_id:
            # Claim TTL lets another worker reclaim jobs after a crash.
            self._redis.setex(
                f"{INGEST_CLAIM_PREFIX}{job_id}",
                self.claim_ttl_seconds,
                str(time.time()),
            )
        return job

    def queue_depth(self) -> int:
        if self._redis is None:
            return 0
        return int(self._redis.llen(INGEST_QUEUE_KEY))


def get_ingest_queue() -> IngestQueue:
    global _store
    if _store is None:
        _store = IngestQueue()
    return _store


def reset_ingest_queue() -> None:
    global _store
    _store = None
