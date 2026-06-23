"""Local Qdrant vector store: collection lifecycle, search, and source management."""
import logging
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    PayloadSchemaType,
    VectorParams,
)
from providers.base import VectorStore

logger = logging.getLogger(__name__)

SCROLL_PAGE_SIZE = int(os.getenv("QDRANT_SCROLL_PAGE_SIZE", "500"))
PAYLOAD_INDEX_FIELDS = ("vendor", "content_type", "doc_type", "product")


def probe_embedding_dimension() -> int:
    """Return embedding vector size from the configured embedder (or ``QDRANT_VECTOR_SIZE``)."""
    explicit = os.getenv("QDRANT_VECTOR_SIZE", "").strip()
    if explicit:
        return int(explicit)
    # Lazy import breaks factory ↔ store init cycle during dimension probe.
    from providers.factory import get_embedder

    vectors = get_embedder().embed(["dimension probe"])
    if not vectors or not vectors[0]:
        raise RuntimeError("Embedder returned an empty vector during dimension probe")
    return len(vectors[0])


def collection_vector_size(client: QdrantClient, collection: str) -> int | None:
    """Read the configured vector dimension for a collection, if it exists."""
    names = {c.name for c in client.get_collections().collections}
    if collection not in names:
        return None
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        return int(first.size)
    return None


class QdrantLocalStore(VectorStore):
    """Qdrant collection backed by ``QDRANT_LOCAL_URL`` with optional int8 quantization."""

    def __init__(self):
        url = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
        self.collection = os.getenv("QDRANT_COLLECTION", "rag_docs")
        client_kwargs: dict = {"url": url}
        api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if api_key:
            client_kwargs["api_key"] = api_key
        grpc_port = os.getenv("QDRANT_GRPC_PORT", "").strip()
        if grpc_port:
            client_kwargs["grpc_port"] = int(grpc_port)
        self.client = QdrantClient(**client_kwargs)
        self._ensure_collection()
        self._ensure_payload_indexes()

    def _collection_names(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def _vector_params(self) -> VectorParams:
        size = probe_embedding_dimension()
        if os.getenv("QDRANT_QUANTIZATION", "").lower() == "int8":
            try:
                from qdrant_client.models import (
                    ScalarQuantization,
                    ScalarQuantizationConfig,
                    ScalarType,
                )

                return VectorParams(
                    size=size,
                    distance=Distance.COSINE,
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                )
            except ImportError:
                logger.warning(
                    "QDRANT_QUANTIZATION=int8 requested but this qdrant-client "
                    "build lacks ScalarQuantization support"
                )
        return VectorParams(size=size, distance=Distance.COSINE)

    def _ensure_collection(self):
        if self.collection not in self._collection_names():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self._vector_params(),
            )

    def _ensure_payload_indexes(self) -> None:
        for field in PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" in message or "already exist" in message:
                    continue
                logger.debug("Payload index %s: %s", field, exc)

    def ensure_payload_indexes(self) -> None:
        """Create v2 payload indexes idempotently (safe to call at ingest start)."""
        self._ensure_payload_indexes()

    def _require_collection(self) -> None:
        if self.collection not in self._collection_names():
            raise ValueError(f"Collection '{self.collection}' does not exist")

    @staticmethod
    def _build_filter(filter_payload: dict | None) -> Filter | None:
        if not filter_payload:
            return None
        must = []
        vendors = filter_payload.get("vendors")
        if isinstance(vendors, list):
            normalized = [str(v).strip().lower() for v in vendors if str(v).strip()]
            if len(normalized) == 1:
                must.append(
                    FieldCondition(
                        key="vendor",
                        match=MatchValue(value=normalized[0]),
                    )
                )
            elif len(normalized) > 1:
                must.append(
                    FieldCondition(
                        key="vendor",
                        match=MatchAny(any=normalized),
                    )
                )
        for key in ("vendor", "product", "document_type", "content_type", "source"):
            value = filter_payload.get(key)
            if value:
                must.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=must) if must else None

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        from providers.blob_store import get_blob_store

        blob = get_blob_store()
        stored_payloads = [blob.externalize_payload_text(dict(p)) for p in payloads]
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, stored_payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_payload: dict | None = None,
        *,
        vendor: str | None = None,
        product: str | None = None,
    ) -> list[dict]:
        payload = dict(filter_payload or {})
        if vendor:
            payload["vendor"] = vendor.strip().lower()
        if product:
            payload["product"] = product.strip().lower()
        query_filter = self._build_filter(payload or None)
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
        )
        from providers.blob_store import get_blob_store

        blob = get_blob_store()
        hydrated = []
        for hit in results.points:
            payload = dict(hit.payload or {})
            if payload.get("text_full"):
                payload["text"] = payload["text_full"]
            payload = blob.hydrate_payload(payload)
            hydrated.append({**payload, "score": hit.score})
        return hydrated

    def delete_by_source(self, source: str) -> int:
        self._require_collection()

        source_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            ]
        )
        count = self.client.count(
            collection_name=self.collection,
            count_filter=source_filter,
        ).count
        if count == 0:
            return 0

        self.client.delete(
            collection_name=self.collection,
            points_selector=source_filter,
        )
        return count

    def list_sources(self) -> list[dict]:
        self._require_collection()

        aggregates: dict[str, dict] = {}
        offset = None
        payload_fields = [
            "source",
            "vendor",
            "file_name",
            "document_type",
            "ingestion_timestamp",
            "total_chunks",
        ]

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=payload_fields,
                with_vectors=False,
            )
            if not records:
                break
            for record in records:
                payload = record.payload or {}
                source = payload.get("source", "")
                if not source:
                    continue
                entry = aggregates.setdefault(
                    source,
                    {
                        "source": source,
                        "chunks": 0,
                        "vendor": payload.get("vendor") or "",
                        "file_name": payload.get("file_name") or "",
                        "document_type": payload.get("document_type") or "",
                        "ingestion_timestamp": payload.get("ingestion_timestamp") or "",
                    },
                )
                entry["chunks"] += 1
                for key in (
                    "vendor",
                    "file_name",
                    "document_type",
                    "ingestion_timestamp",
                ):
                    if not entry[key] and payload.get(key):
                        entry[key] = payload[key]
                total = payload.get("total_chunks")
                if total and not entry.get("total_chunks"):
                    entry["total_chunks"] = int(total)
            if offset is None:
                break

        for entry in aggregates.values():
            if "total_chunks" not in entry:
                entry["total_chunks"] = entry["chunks"]

        return sorted(aggregates.values(), key=lambda row: (row["vendor"], row["source"]))

    def get_source_content_hash(self, source: str) -> str | None:
        self._require_collection()
        source_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        )
        records, _ = self.client.scroll(
            collection_name=self.collection,
            limit=1,
            scroll_filter=source_filter,
            with_payload=["content_hash"],
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        value = payload.get("content_hash")
        return str(value) if value else None

    def patch_total_chunks(self, source: str, total_chunks: int) -> int:
        self._require_collection()
        source_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        )
        offset = None
        updated = 0

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
                scroll_filter=source_filter,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break
            points = []
            for record in records:
                payload = dict(record.payload or {})
                payload["total_chunks"] = total_chunks
                points.append(
                    PointStruct(id=record.id, vector=record.vector, payload=payload)
                )
            if points:
                self.client.upsert(collection_name=self.collection, points=points)
                updated += len(points)
            if offset is None:
                break
        return updated
