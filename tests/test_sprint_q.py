"""Sprint Q tests: Qdrant Cloud store."""
from unittest.mock import MagicMock, patch

import pytest

from providers import factory


class TestQdrantCloudStore:
    def test_requires_cloud_url_and_api_key(self, monkeypatch):
        monkeypatch.delenv("QDRANT_CLOUD_URL", raising=False)
        monkeypatch.delenv("QDRANT_LOCAL_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        from providers.qdrant_cloud_store import QdrantCloudStore

        with pytest.raises(ValueError, match="QDRANT_CLOUD_URL"):
            QdrantCloudStore()

        monkeypatch.setenv("QDRANT_CLOUD_URL", "https://cloud.qdrant.test")
        with pytest.raises(ValueError, match="QDRANT_API_KEY"):
            QdrantCloudStore()

    def test_factory_selects_cloud_store(self, monkeypatch):
        from providers.qdrant_cloud_store import QdrantCloudStore

        factory.reset_providers()
        monkeypatch.setenv("VECTOR_STORE", "qdrant_cloud")
        monkeypatch.setenv("QDRANT_CLOUD_URL", "https://cloud.qdrant.test")
        monkeypatch.setenv("QDRANT_API_KEY", "secret")

        with patch("qdrant_client.QdrantClient") as client_ctor:
            client_ctor.return_value = MagicMock()
            with patch.object(QdrantCloudStore, "_ensure_collection", lambda self: None):
                with patch.object(
                    QdrantCloudStore,
                    "_ensure_payload_indexes",
                    lambda self: None,
                ):
                    store = factory.get_vector_store()
        assert store.__class__.__name__ == "QdrantCloudStore"
        factory.reset_providers()
