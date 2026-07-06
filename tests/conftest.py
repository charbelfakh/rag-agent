"""Shared pytest helpers."""
import sys
import types
from unittest.mock import MagicMock

import pytest

# Ingest pulls optional/heavy deps at import time; API tests import api.main → ingest.
for _optional in ("pymupdf", "tqdm"):
    if _optional not in sys.modules:
        sys.modules[_optional] = MagicMock()

_INGEST_MODULE = "scripts.ingest.ingest"
if _INGEST_MODULE not in sys.modules:
    _ingest_stub = types.ModuleType(_INGEST_MODULE)
    _ingest_stub.ingest = MagicMock(return_value=0)
    sys.modules[_INGEST_MODULE] = _ingest_stub

if "qdrant_client" not in sys.modules:
    _fake_qdrant = types.ModuleType("qdrant_client")
    _fake_models = types.ModuleType("qdrant_client.models")
    for _name in (
        "Distance",
        "FieldCondition",
        "Filter",
        "MatchAny",
        "MatchValue",
        "PointStruct",
        "PayloadSchemaType",
        "QueryRequest",
        "ScalarQuantization",
        "ScalarQuantizationConfig",
        "ScalarType",
        "VectorParams",
        "SparseVector",
        "SparseVectorParams",
        "Modifier",
    ):
        setattr(_fake_models, _name, MagicMock())
    _fake_qdrant.QdrantClient = MagicMock
    _fake_qdrant.models = _fake_models
    sys.modules["qdrant_client"] = _fake_qdrant
    sys.modules["qdrant_client.models"] = _fake_models


@pytest.fixture(autouse=True)
def _isolated_app_data_dir(tmp_path, monkeypatch):
    """Keep tests away from the real per-user secret store.

    App startup loads persisted API keys / provider state, and the default
    paths resolve to %APPDATA%\\rag-agent (with one-time migration from
    data/) — a test run must never read or move real credentials.
    """
    monkeypatch.setenv("RAG_AGENT_APP_DIR", str(tmp_path / "app-data"))


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)


def patch_retrieval_pipeline(
    monkeypatch,
    *,
    embedder=None,
    store=None,
    llm=None,
    cache=None,
    reranker=None,
) -> None:
    """Patch deps used by ``QueryOrchestrator`` (and legacy ``rag_pipeline`` imports)."""
    import providers.query_orchestrator as orchestrator
    import providers.rag_pipeline as rag_pipeline

    pairs = (
        ("get_embedder", embedder),
        ("get_vector_store", store),
        ("get_llm", llm),
        # HyDE + sufficiency use the fast tier; in tests it is the same fake as
        # the main llm (matches the Ollama path where get_fast_llm is get_llm).
        ("get_fast_llm", llm),
        ("get_semantic_cache", cache),
        ("get_reranker", reranker),
    )
    for attr, value in pairs:
        if value is None:
            continue
        if hasattr(orchestrator, attr):
            monkeypatch.setattr(orchestrator, attr, lambda v=value: v)
        if hasattr(rag_pipeline, attr):
            monkeypatch.setattr(rag_pipeline, attr, lambda v=value: v)
