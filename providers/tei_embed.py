"""HTTP embedder for Hugging Face Text Embeddings Inference (TEI)."""
import os

import httpx

from providers.base import EmbedProvider


class TEIEmbedder(EmbedProvider):
    """Batch-embed texts via a Hugging Face Text Embeddings Inference server."""

    def __init__(self):
        self.base_url = os.getenv("TEI_BASE_URL", "http://localhost:8081").rstrip("/")
        self.batch_size = int(os.getenv("EMBED_ENCODE_BATCH_SIZE", "32"))
        self._client = httpx.Client(timeout=300.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        cleaned = [t if t.strip() else " " for t in texts]
        vectors: list[list[float]] = []
        for index in range(0, len(cleaned), self.batch_size):
            batch = cleaned[index : index + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._client.post(
            f"{self.base_url}/embed",
            json={"inputs": batch},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return payload
        return payload["embeddings"]
