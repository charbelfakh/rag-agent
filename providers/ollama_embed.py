"""Ollama ``/api/embed`` client with batching and transient-crash recovery."""
import logging
import os
import time

import httpx

from providers.base import EmbedProvider

logger = logging.getLogger(__name__)

_TRANSIENT_RUNNER_MARKERS = ("dial tcp", "connectex", "connection refused")
_TRANSIENT_MAX_RETRIES = 5
_TRANSIENT_BACKOFF_BASE_S = 3.0


class OllamaEmbedder(EmbedProvider):
    """Batch embed texts via Ollama; split and retry on runner crashes or 400s."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.batch_size = int(os.getenv("EMBED_ENCODE_BATCH_SIZE", "128"))
        self._client = httpx.Client(timeout=300.0)

    def embed(
        self,
        texts: list[str],
        *,
        sources: list[str | None] | None = None,
        pages: list[int | None] | None = None,
    ) -> list[list[float] | None]:
        """Return one embedding per input; ``None`` marks skipped empty or failed texts."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        for index, raw in enumerate(texts):
            text = (raw or "").strip()
            if not text:
                source = sources[index] if sources and index < len(sources) else None
                page = pages[index] if pages and index < len(pages) else None
                logger.warning(
                    "Skipping empty embed input%s%s",
                    f" source={source!r}" if source else "",
                    f" page={page}" if page is not None else "",
                )
                continue
            to_embed.append((index, text))

        if not to_embed:
            return results

        embedded_vectors: list[list[float] | None] = []
        indices = [index for index, _ in to_embed]
        embed_texts = [text for _, text in to_embed]

        for start in range(0, len(embed_texts), self.batch_size):
            end = start + self.batch_size
            batch_texts = embed_texts[start:end]
            batch_indices = indices[start:end]
            batch_sources = (
                [sources[i] if sources and i < len(sources) else None for i in batch_indices]
                if sources
                else None
            )
            batch_pages = (
                [pages[i] if pages and i < len(pages) else None for i in batch_indices]
                if pages
                else None
            )
            embedded_vectors.extend(
                self._embed_batch(
                    self._client,
                    batch_texts,
                    sources=batch_sources,
                    pages=batch_pages,
                )
            )

        for index, vector in zip(indices, embedded_vectors):
            results[index] = vector
        return results

    @staticmethod
    def _is_transient_runner_crash(error_body: str) -> bool:
        lowered = error_body.lower()
        return any(marker in lowered for marker in _TRANSIENT_RUNNER_MARKERS)

    def _retry_transient_batch(
        self,
        client: httpx.Client,
        batch: list[str],
        exc: httpx.HTTPStatusError,
    ) -> tuple[list[list[float]] | None, httpx.HTTPStatusError]:
        """Retry the same batch after runner crashes."""
        last_exc = exc
        for attempt in range(1, _TRANSIENT_MAX_RETRIES + 1):
            delay = _TRANSIENT_BACKOFF_BASE_S * (2 ** (attempt - 1))
            logger.warning(
                "Ollama runner crash (transient), retry %s/%s in %ss",
                attempt,
                _TRANSIENT_MAX_RETRIES,
                int(delay),
            )
            time.sleep(delay)
            try:
                return self._request_embed(client, batch), last_exc
            except httpx.HTTPStatusError as retry_exc:
                if retry_exc.response.status_code != 400:
                    raise
                last_exc = retry_exc
                if not self._is_transient_runner_crash(retry_exc.response.text):
                    return None, last_exc
        return None, last_exc

    def _embed_batch(
        self,
        client: httpx.Client,
        batch: list[str],
        *,
        sources: list[str | None] | None = None,
        pages: list[int | None] | None = None,
    ) -> list[list[float] | None]:
        try:
            return self._request_embed(client, batch)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            if self._is_transient_runner_crash(exc.response.text):
                recovered, exc = self._retry_transient_batch(client, batch, exc)
                if recovered is not None:
                    return recovered
            logger.warning(
                "Ollama /api/embed 400 (%s item(s)): %s",
                len(batch),
                exc.response.text,
            )
            if len(batch) == 1:
                self._log_skip_item(
                    batch[0],
                    sources[0] if sources else None,
                    pages[0] if pages else None,
                )
                return [None]
            # Bisect to isolate a single bad chunk without dropping the whole batch.
            mid = len(batch) // 2
            return self._embed_batch(
                client,
                batch[:mid],
                sources=sources[:mid] if sources else None,
                pages=pages[:mid] if pages else None,
            ) + self._embed_batch(
                client,
                batch[mid:],
                sources=sources[mid:] if sources else None,
                pages=pages[mid:] if pages else None,
            )

    def _log_skip_item(
        self,
        text: str,
        source: str | None,
        page: int | None,
    ) -> None:
        preview = text[:200]
        logger.warning(
            "Skipping embed after 400; text=%r%s%s",
            preview,
            f" source={source!r}" if source else "",
            f" page={page}" if page is not None else "",
        )

    def _request_embed(
        self, client: httpx.Client, batch: list[str]
    ) -> list[list[float]]:
        response = client.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": batch,
                "truncate": True,
                "options": {"num_ctx": 8192},
                "keep_alive": "60m",
            },
        )
        response.raise_for_status()
        return response.json()["embeddings"]
