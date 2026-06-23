#!/usr/bin/env python3
"""Consume ingest jobs from the Redis queue. Run: ``python -m scripts.ingest.ingest_worker``."""
from __future__ import annotations

import logging
import sys
import time

# Replaces broken ``parent.parent`` ROOT hack after scripts/ reorganisation.
import scripts._bootstrap  # noqa: F401

from scripts.ingest.ingest import ingest
from providers.ingest_jobs import get_ingest_job_store
from providers.ingest_queue import get_ingest_queue, is_ingest_queue_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_worker")


def _run_job(job: dict) -> None:
    job_id = job["job_id"]
    store = get_ingest_job_store()

    def callback(stage: str, chunks_done: int, chunks_total: int) -> None:
        current = store.get(job_id)
        if not current:
            return
        now = time.time()
        elapsed = now - current["started_at"]
        eta = 0
        if chunks_done > 0 and chunks_total > chunks_done:
            eta = int((elapsed / chunks_done) * (chunks_total - chunks_done))
        store.update(
            job_id,
            status="ingesting",
            stage=stage,
            chunks_done=chunks_done,
            chunks_total=chunks_total,
            elapsed_seconds=int(elapsed),
            estimated_remaining_seconds=eta,
        )

    store.update(job_id, status="ingesting", stage="queued")
    try:
        total = ingest(
            job["ingest_path"],
            progress_callback=callback,
            vendor=job.get("vendor"),
            document_type=job.get("document_type"),
            product_line=job.get("product_line"),
            software_version=job.get("software_version"),
        )
        now = time.time()
        current = store.get(job_id) or {}
        store.update(
            job_id,
            status="done",
            stage="skipped" if total == 0 else "storing",
            chunks_done=total,
            chunks_total=total,
            elapsed_seconds=int(now - current.get("started_at", now)),
            estimated_remaining_seconds=0,
            finished_at=now,
        )
        logger.info("Job %s finished (%d chunks)", job_id, total)
    except Exception as exc:
        now = time.time()
        current = store.get(job_id) or {}
        store.update(
            job_id,
            status="error",
            error=str(exc),
            elapsed_seconds=int(now - current.get("started_at", now)),
            finished_at=now,
        )
        logger.exception("Job %s failed: %s", job_id, exc)


def main() -> int:
    if not is_ingest_queue_enabled():
        logger.error("INGEST_QUEUE_ENABLED is not true; worker exiting")
        return 1

    queue = get_ingest_queue()
    if not queue.available:
        logger.error("Redis ingest queue unavailable")
        return 1

    logger.info("Ingest worker started (queue=%s)", queue.queue_depth())
    while True:
        job = queue.dequeue(block_timeout=5)
        if not job:
            continue
        logger.info("Claimed job %s for %s", job.get("job_id"), job.get("ingest_path"))
        _run_job(job)


if __name__ == "__main__":
    raise SystemExit(main())
