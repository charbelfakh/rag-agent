"""Deterministic transcript corrections for observed domain mistranscriptions."""
from __future__ import annotations

import re

# Keep this list to observed errors only. Extend as new errors are observed.
GLOSSARY_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bpeacot\s+vision\b", re.IGNORECASE), "PEKAT Vision"),
    (re.compile(r"\bpeacott\s+vision\b", re.IGNORECASE), "PEKAT Vision"),
    (re.compile(r"\bpeacotvision\.com\b", re.IGNORECASE), "pekatvision.com"),
    (re.compile(r"\bpeacottvision\.com\b", re.IGNORECASE), "pekatvision.com"),
    (re.compile(r"\bpeacott\b", re.IGNORECASE), "PEKAT"),
    (re.compile(r"\bpeacot\b", re.IGNORECASE), "PEKAT"),
    (re.compile(r"\bmac\s*mine\b", re.IGNORECASE), "Mech-Mind"),
    (re.compile(r"\bmac\s*mind\b", re.IGNORECASE), "Mech-Mind"),
    (re.compile(r"\bmech\s*mine\b", re.IGNORECASE), "Mech-Mind"),
    (re.compile(r"\bmeck\s*mind\b", re.IGNORECASE), "Mech-Mind"),
    (re.compile(r"\bmechmind\b", re.IGNORECASE), "Mech-Mind"),
    (re.compile(r"\bmech\s*mind\b", re.IGNORECASE), "Mech-Mind"),
)


def normalize_transcript_text(text: str) -> str:
    """Apply observed ASR mistranscription fixes before chunking."""
    corrected = text
    for pattern, replacement in GLOSSARY_REPLACEMENTS:
        corrected = pattern.sub(replacement, corrected)
    return corrected

