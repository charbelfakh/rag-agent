"""Sprint F tests: cooperative cancel and Ollama stream stats."""
import json
import threading
from unittest.mock import MagicMock, patch


from providers.ollama_llm import OllamaLLM


class TestOllamaCancel:
    def test_generate_stream_stops_when_cancel_set(self):
        cancel = threading.Event()
        lines = [
            json.dumps({"response": "Hi", "done": False}).encode(),
            json.dumps(
                {
                    "response": "",
                    "done": True,
                    "prompt_eval_count": 12,
                    "eval_count": 3,
                }
            ).encode(),
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)
        mock_response.raise_for_status = MagicMock()

        llm = OllamaLLM()
        with patch("providers.ollama_llm.requests.post", return_value=mock_response):
            tokens = list(llm.generate_stream("prompt", cancel_event=cancel))
        assert tokens == ["Hi"]
        mock_response.close.assert_called()

    def test_cancel_before_first_token(self):
        cancel = threading.Event()
        cancel.set()
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(
            [json.dumps({"response": "X", "done": False}).encode()]
        )
        mock_response.raise_for_status = MagicMock()

        llm = OllamaLLM()
        with patch("providers.ollama_llm.requests.post", return_value=mock_response):
            tokens = list(llm.generate_stream("prompt", cancel_event=cancel))
        assert tokens == []
