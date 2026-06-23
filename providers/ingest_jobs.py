"""Persist ingest job status in Redis (fallback: in-memory) across API restarts."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

JOB_KEY_PREFIX = "ingest_job:"
JOB_INDEX_KEY = "ingest_job:index"

_store: "IngestJobStore | None" = None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


class IngestJobStore:
    """Track ingest job status in Redis with in-memory fallback."""

    def __init__(self):
        self.retention_seconds = int(os.getenv("INGEST_JOB_RETENTION_SECONDS", "600"))
        self.stale_seconds = int(os.getenv("INGEST_JOB_STALE_SECONDS", "7200"))
        self.use_redis = _env_bool("INGEST_JOBS_REDIS_ENABLED", "true")
        self._memory: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._redis = None
        if self.use_redis:
            self._connect()

    def _connect(self) -> None:
        try:
            import redis

            url = os.getenv("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis = client
        except Exception as exc:
            logger.warning("Ingest jobs using in-memory store: Redis unreachable (%s)", exc)
            self._redis = None

    def _redis_key(self, job_id: str) -> str:
        return f"{JOB_KEY_PREFIX}{job_id}"

    def _save(self, job: dict) -> None:
        job_id = job["job_id"]
        if self._redis is None:
            with self._lock:
                self._memory[job_id] = dict(job)
            return
        payload = json.dumps(job)
        pipe = self._redis.pipeline()
        pipe.set(self._redis_key(job_id), payload)
        pipe.sadd(JOB_INDEX_KEY, job_id)
        pipe.execute()

    def create(self, job: dict) -> dict:
        self._save(job)
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        if self._redis is None:
            with self._lock:
                job = self._memory.get(job_id)
                return dict(job) if job else None
        raw = self._redis.get(self._redis_key(job_id))
        if not raw:
            return None
        return json.loads(raw)

    def update(self, job_id: str, **fields) -> dict | None:
        job = self.get(job_id)
        if not job:
            return None
        job.update(fields)
        self._save(job)
        return job

    def prune(self) -> None:
        now = time.time()
        if self._redis is None:
            with self._lock:
                expired = [
                    job_id
                    for job_id, job in self._memory.items()
                    if job.get("status") in ("done", "error", "interrupted")
                    and now - job.get("finished_at", job["started_at"])
                    > self.retention_seconds
                ]
                for job_id in expired:
                    del self._memory[job_id]
            return

        job_ids = self._redis.smembers(JOB_INDEX_KEY)
        for job_id in job_ids:
            job = self.get(job_id)
            if not job:
                self._redis.srem(JOB_INDEX_KEY, job_id)
                continue
            if job.get("status") not in ("done", "error", "interrupted"):
                continue
            finished = job.get("finished_at", job["started_at"])
            if now - finished > self.retention_seconds:
                pipe = self._redis.pipeline()
                pipe.delete(self._redis_key(job_id))
                pipe.srem(JOB_INDEX_KEY, job_id)
                pipe.execute()

    def mark_stale_ingesting_as_interrupted(self) -> int:
        """On API startup, flag jobs left ``ingesting`` after a restart."""
        now = time.time()
        updated = 0
        if self._redis is None:
            with self._lock:
                for job in self._memory.values():
                    if job.get("status") != "ingesting":
                        continue
                    if now - job["started_at"] <= self.stale_seconds:
                        continue
                    job.update(
                        {
                            "status": "interrupted",
                            "error": "Ingest interrupted by API restart",
                            "finished_at": now,
                        }
                    )
                    updated += 1
            return updated

        for job_id in self._redis.smembers(JOB_INDEX_KEY):
            job = self.get(job_id)
            if not job or job.get("status") != "ingesting":
                continue
            if now - job["started_at"] <= self.stale_seconds:
                continue
            self.update(
                job_id,
                status="interrupted",
                error="Ingest interrupted by API restart",
                finished_at=now,
            )
            updated += 1
        return updated


def get_ingest_job_store() -> IngestJobStore:
    global _store
    if _store is None:
        _store = IngestJobStore()
    return _store


def reset_ingest_job_store() -> None:
    global _store
    _store = None
