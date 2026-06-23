"""Privacy-safe production query sampling for drift detection (Sprint P rank 55)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def is_production_sampling_enabled() -> bool:
    return os.getenv("PRODUCTION_EVAL_SAMPLING_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _sample_rate() -> float:
    return float(os.getenv("PRODUCTION_EVAL_SAMPLE_RATE", "0.05"))


def _sample_path() -> Path:
    return Path(os.getenv("PRODUCTION_EVAL_SAMPLE_PATH", "data/eval_samples.jsonl"))


def should_sample(question: str) -> bool:
    if not is_production_sampling_enabled():
        return False
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < _sample_rate()


def redact_question(question: str) -> str:
    max_len = int(os.getenv("PRODUCTION_EVAL_QUESTION_MAX_CHARS", "240"))
    return question.strip()[:max_len]


def record_sample(
    *,
    question: str,
    answer_preview: str,
    meta: dict | None = None,
) -> None:
    if not should_sample(question):
        return
    path = _sample_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.time(),
        "question_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "question_redacted": redact_question(question),
        "answer_preview": answer_preview[:240],
        "meta": meta or {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
