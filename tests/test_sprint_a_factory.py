"""Sprint A unit tests: singleton provider instances."""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from providers import factory


@pytest.fixture(autouse=True)
def reset_singletons():
    factory.reset_providers()
    yield
    factory.reset_providers()


@pytest.fixture
def fake_qdrant_store_module(monkeypatch):
    """Inject a lightweight qdrant_store module (avoids qdrant_client import)."""
    mock_store = MagicMock()
    mock_cls = MagicMock(return_value=mock_store)
    fake_module = types.ModuleType("providers.qdrant_store")
    fake_module.QdrantLocalStore = mock_cls
    monkeypatch.setitem(sys.modules, "providers.qdrant_store", fake_module)
    return mock_store, mock_cls


class TestProviderSingletons:
    def test_get_vector_store_returns_same_instance(
        self, monkeypatch, fake_qdrant_store_module
    ):
        monkeypatch.setenv("VECTOR_STORE", "qdrant_local")
        mock_store, mock_cls = fake_qdrant_store_module

        first = factory.get_vector_store()
        second = factory.get_vector_store()

        assert first is second
        mock_cls.assert_called_once()

    def test_get_embedder_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("EMBED_PROVIDER", "ollama")
        mock_embedder = MagicMock()

        with patch(
            "providers.ollama_embed.OllamaEmbedder",
            return_value=mock_embedder,
        ) as constructor:
            first = factory.get_embedder()
            second = factory.get_embedder()

        assert first is second
        constructor.assert_called_once()

    def test_get_llm_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        mock_llm = MagicMock()

        with patch(
            "providers.ollama_llm.OllamaLLM",
            return_value=mock_llm,
        ) as constructor:
            first = factory.get_llm()
            second = factory.get_llm()

        assert first is second
        constructor.assert_called_once()

    def test_reset_providers_clears_cache(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        first_mock = MagicMock(name="llm_first")
        second_mock = MagicMock(name="llm_second")

        with patch(
            "providers.ollama_llm.OllamaLLM",
            side_effect=[first_mock, second_mock],
        ):
            first = factory.get_llm()
            factory.reset_providers()
            second = factory.get_llm()

        assert first is first_mock
        assert second is second_mock
        assert first is not second
