"""Abstract provider interfaces for LLM, embedding, vector store, and reranker backends."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Text generation backend (Ollama, OpenAI, Anthropic, etc.)."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a complete response for the given prompt."""
        ...


class EmbedProvider(ABC):
    """Embedding backend used for retrieval and semantic cache lookups."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class VectorStore(ABC):
    """Vector database for chunk storage, search, and source management."""
    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_payload: dict | None = None,
    ) -> list[dict]:
        ...

    @abstractmethod
    def delete_by_source(self, source: str) -> int:
        ...

    @abstractmethod
    def list_sources(self) -> list[dict]:
        ...

    def patch_total_chunks(self, source: str, total_chunks: int) -> int:
        """Update ``total_chunks`` on all points for a source (optional override)."""
        return 0

    def get_source_content_hash(self, source: str) -> str | None:
        """Return stored ``content_hash`` for a source, if any (optional override)."""
        return None

    def ensure_payload_indexes(self) -> None:
        """Create payload indexes idempotently (optional override)."""
        return None


class Reranker(ABC):
    """Re-scores retrieved chunks for relevance before LLM context assembly."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        """Return the top_n most relevant chunks for the query."""
        ...