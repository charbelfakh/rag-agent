"""Managed Qdrant Cloud vector store (Sprint Q rank 59)."""
from __future__ import annotations

import os

from providers.qdrant_store import QdrantLocalStore


class QdrantCloudStore(QdrantLocalStore):
    """Qdrant Cloud / cluster URL with API key authentication."""

    def __init__(self):
        url = os.getenv("QDRANT_CLOUD_URL") or os.getenv("QDRANT_LOCAL_URL", "")
        if not url:
            raise ValueError("QDRANT_CLOUD_URL is required when VECTOR_STORE=qdrant_cloud")
        api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if not api_key:
            raise ValueError("QDRANT_API_KEY is required for Qdrant Cloud")
        self.collection = os.getenv("QDRANT_COLLECTION", "rag_docs")
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection()
        self._ensure_payload_indexes()
