"""Section- and procedure-aware chunking helpers (Sprint L rank 53)."""
from __future__ import annotations

import os
import re

PROCEDURE_STEP_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Step\s+)?(\d{1,2})[\.\):]\s+",
    re.IGNORECASE,
)


def is_section_aware_chunking_enabled() -> bool:
    return os.getenv("SECTION_AWARE_CHUNKING_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def split_procedure_steps(body: str) -> list[str]:
    """Split procedural body text on numbered steps when present."""
    text = body.strip()
    if not text:
        return []
    matches = list(PROCEDURE_STEP_PATTERN.finditer(text))
    if len(matches) < 2:
        return [text]
    parts: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if segment:
            parts.append(segment)
    return parts or [text]
