"""SigLIP / CLIP image embedder for hybrid retrieval (Sprint M M2)."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class SigLIPEmbedder:
    """Image embedder; uses sentence-transformers when available."""

    def __init__(self):
        self.model_name = os.getenv(
            "IMAGE_EMBED_MODEL", "google/siglip-base-patch16-384"
        )
        self._model = None
        self._dimensions = int(os.getenv("IMAGE_EMBED_DIMENSIONS", "768"))

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_sentence_embedding_dimension()
            if dim:
                self._dimensions = dim
        except Exception as exc:
            logger.warning("SigLIP embedder unavailable (%s) — using hash fallback", exc)
            self._model = None

    @property
    def dimensions(self) -> int:
        self._load()
        return self._dimensions

    def embed_bytes(self, images: list[bytes]) -> list[list[float]]:
        self._load()
        if self._model is None:
            return [self._hash_fallback(data) for data in images]
        try:
            from PIL import Image
            import io

            pil_images = [Image.open(io.BytesIO(data)).convert("RGB") for data in images]
            vectors = self._model.encode(pil_images, normalize_embeddings=True)
            return [list(map(float, row)) for row in vectors]
        except Exception as exc:
            logger.warning("SigLIP encode failed: %s", exc)
            return [self._hash_fallback(data) for data in images]

    def _hash_fallback(self, data: bytes) -> list[float]:
        import hashlib

        digest = hashlib.sha256(data).digest()
        repeats = (self._dimensions + len(digest) - 1) // len(digest)
        raw = (digest * repeats)[: self._dimensions]
        return [b / 255.0 for b in raw]
