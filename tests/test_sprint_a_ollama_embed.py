"""Sprint A unit tests: pooled httpx client in OllamaEmbedder."""
from unittest.mock import MagicMock, patch

from providers.ollama_embed import OllamaEmbedder


class TestOllamaEmbedderClientReuse:
    def test_embed_reuses_instance_client_across_batches(self, monkeypatch):
        monkeypatch.setenv("EMBED_ENCODE_BATCH_SIZE", "1")
        embedder = OllamaEmbedder()
        mock_client = MagicMock()
        embedder._client = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_client.post.return_value = mock_response

        result = embedder.embed(["first", "second"])

        assert result == [[0.1, 0.2], [0.1, 0.2]]
        assert mock_client.post.call_count == 2
        assert embedder._client is mock_client

    def test_constructor_creates_persistent_client(self):
        with patch("providers.ollama_embed.httpx.Client") as client_cls:
            client_cls.return_value = MagicMock()
            embedder = OllamaEmbedder()
            client_cls.assert_called_once_with(timeout=300.0)
            assert embedder._client is client_cls.return_value
