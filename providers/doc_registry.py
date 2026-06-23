"""SQLite document registry for fast ``GET /documents`` without Qdrant scroll."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path

from providers.metadata import DocumentMetadata

logger = logging.getLogger(__name__)

_registry: "DocumentRegistry | None" = None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


class DocumentRegistry:
    """SQLite mirror of ingested documents for fast ``GET /documents`` listing."""

    def __init__(self, db_path: str | None = None):
        self.enabled = _env_bool("DOC_REGISTRY_ENABLED", "true")
        raw_path = db_path or os.getenv("DOC_REGISTRY_PATH", "data/doc_registry.db")
        self.db_path = Path(raw_path)
        self._lock = threading.Lock()
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        source TEXT PRIMARY KEY,
                        vendor TEXT NOT NULL DEFAULT '',
                        file_name TEXT NOT NULL DEFAULT '',
                        document_type TEXT NOT NULL DEFAULT 'manual',
                        content_hash TEXT NOT NULL DEFAULT '',
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        total_chunks INTEGER NOT NULL DEFAULT 0,
                        ingestion_timestamp TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_vendor ON documents(vendor)"
                )
                conn.commit()

    def upsert(
        self,
        metadata: DocumentMetadata,
        *,
        chunk_count: int,
        content_hash: str = "",
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO documents (
                        source, vendor, file_name, document_type,
                        content_hash, chunk_count, total_chunks, ingestion_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source) DO UPDATE SET
                        vendor = excluded.vendor,
                        file_name = excluded.file_name,
                        document_type = excluded.document_type,
                        content_hash = excluded.content_hash,
                        chunk_count = excluded.chunk_count,
                        total_chunks = excluded.total_chunks,
                        ingestion_timestamp = excluded.ingestion_timestamp
                    """,
                    (
                        metadata.source,
                        metadata.vendor,
                        metadata.file_name,
                        metadata.document_type,
                        content_hash or metadata.content_hash,
                        chunk_count,
                        chunk_count,
                        metadata.ingestion_timestamp,
                    ),
                )
                conn.commit()

    def delete(self, source: str) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM documents WHERE source = ?",
                    (source,),
                )
                conn.commit()
                return cursor.rowcount > 0

    def list_documents(self) -> list[dict]:
        if not self.enabled:
            return []
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT source, vendor, file_name, document_type,
                           content_hash, chunk_count, total_chunks, ingestion_timestamp
                    FROM documents
                    ORDER BY vendor, source
                    """
                ).fetchall()
        return [
            {
                "source": row["source"],
                "vendor": row["vendor"],
                "file_name": row["file_name"],
                "document_type": row["document_type"],
                "chunks": row["chunk_count"],
                "total_chunks": row["total_chunks"] or row["chunk_count"],
                "ingestion_timestamp": row["ingestion_timestamp"],
                "content_hash": row["content_hash"] or "",
            }
            for row in rows
        ]

    def backfill_from_vector_store(self, store) -> int:
        """Populate registry from a full Qdrant ``list_sources()`` scroll."""
        if not self.enabled:
            return 0
        sources = store.list_sources()
        count = 0
        for entry in sources:
            meta = DocumentMetadata(
                source=entry["source"],
                file_name=entry.get("file_name") or Path(entry["source"]).name,
                file_extension=Path(entry["source"]).suffix.lower(),
                vendor=entry.get("vendor") or "unknown",
                document_type=entry.get("document_type") or "manual",
                ingestion_timestamp=entry.get("ingestion_timestamp") or "",
            )
            stored_hash = store.get_source_content_hash(entry["source"]) or ""
            self.upsert(
                meta,
                chunk_count=int(entry.get("chunks") or entry.get("total_chunks") or 0),
                content_hash=stored_hash,
            )
            count += 1
        logger.info("Backfilled %d documents into registry", count)
        return count


def get_doc_registry() -> DocumentRegistry:
    """Return the process-wide document registry singleton."""
    global _registry
    if _registry is None:
        _registry = DocumentRegistry()
    return _registry


def reset_doc_registry() -> None:
    global _registry
    _registry = None
