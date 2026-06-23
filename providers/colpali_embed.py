"""ColPali / page-level embedder stub for diagram-heavy PDFs (Sprint N M4)."""
from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)


class ColPaliEmbedder:
    """Placeholder page embedder — returns deterministic vectors until model wired."""

    def __init__(self):
        self.model_name = os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3")
        self.dimensions = int(os.getenv("COLPALI_EMBED_DIMENSIONS", "128"))

    def embed_page_images(self, images: list[bytes]) -> list[list[float]]:
        vectors = []
        for data in images:
            digest = hashlib.sha256(data).digest()
            repeats = (self.dimensions + len(digest) - 1) // len(digest)
            raw = (digest * repeats)[: self.dimensions]
            vectors.append([b / 255.0 for b in raw])
        return vectors
