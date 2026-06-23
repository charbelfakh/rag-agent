"""Ollama LLM request shaping for thinking models."""
from unittest.mock import MagicMock, patch

from providers.ollama_llm import OllamaLLM


class TestOllamaLLMThinkFlag:
    def test_generate_disables_thinking_by_default(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_THINK_ENABLED", raising=False)
        llm = OllamaLLM()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "hello"}
        mock_response.raise_for_status = MagicMock()

        with patch("providers.ollama_llm.requests.post", return_value=mock_response) as post:
            assert llm.generate("prompt") == "hello"
            body = post.call_args.kwargs["json"]
            assert body["think"] is False

    def test_generate_respects_think_enabled(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_THINK_ENABLED", "true")
        llm = OllamaLLM()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "hello"}
        mock_response.raise_for_status = MagicMock()

        with patch("providers.ollama_llm.requests.post", return_value=mock_response) as post:
            llm.generate("prompt")
            body = post.call_args.kwargs["json"]
            assert "think" not in body
