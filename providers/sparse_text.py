"""Lexical sparse vectors for hybrid dense+sparse retrieval (``QDRANT_SPARSE_ENABLED``).

Sparse vectors carry raw term frequencies; Qdrant applies IDF server-side
(``Modifier.IDF``), giving BM25-style lexical matching without an external
sparse-embedding model. Exact tokens like model numbers ("2120A", "XL250")
match literally — precisely where dense embeddings are weakest.

Indices are stable CRC32 hashes of lowercased alphanumeric tokens, so query
and document tokenization always agree.
"""
from __future__ import annotations

import os
import re
import zlib
from collections import Counter

SPARSE_VECTOR_NAME = "sparse_text"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def is_sparse_enabled() -> bool:
    return os.getenv("QDRANT_SPARSE_ENABLED", "false").lower() in ("true", "1", "yes")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens; '2120A' → '2120a', 'GoPxL' → 'gopxl'."""
    return _TOKEN_RE.findall((text or "").lower())


def sparse_text_vector(text: str) -> tuple[list[int], list[float]]:
    """Return ``(indices, values)`` — CRC32 token hashes with term frequencies."""
    counts = Counter(tokenize(text))
    indices: list[int] = []
    values: list[float] = []
    for token, tf in counts.items():
        indices.append(zlib.crc32(token.encode("utf-8")))
        values.append(float(tf))
    return indices, values
