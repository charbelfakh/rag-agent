"""Ollama LLM provider: synchronous and streaming text generation via /api/generate."""
import json
import logging
import os
import threading
from collections.abc import Iterator

import requests
from providers.base import LLMProvider

logger = logging.getLogger(__name__)


def _request_timeout() -> tuple[float, float]:
    read_timeout = float(os.getenv("OLLAMA_LLM_TIMEOUT", "120"))
    return (10.0, read_timeout)


def _think_enabled() -> bool:
    return os.getenv("OLLAMA_THINK_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


class OllamaLLM(LLMProvider):
    """Calls a local or remote Ollama instance for chat-style generation."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_LLM_MODEL", "mistral:7b-instruct-q4_K_M")
        self.last_stream_stats: dict = {}

    def _request_body(self, prompt: str, *, stream: bool) -> dict:
        body = {"model": self.model, "prompt": prompt, "stream": stream}
        if not _think_enabled():
            # Top-level only: think inside options is ignored (qwen3.5).
            body["think"] = False
        return body

    @staticmethod
    def _answer_from_chunk(chunk: dict) -> str:
        text = (chunk.get("response") or "").strip()
        if text:
            return chunk.get("response", "")
        thinking = (chunk.get("thinking") or "").strip()
        if thinking:
            logger.warning(
                "Ollama returned empty response with %d thinking chars; "
                "set OLLAMA_THINK_ENABLED=false or use a non-thinking model",
                len(thinking),
            )
            return chunk.get("thinking", "")
        return ""

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=self._request_body(prompt, stream=False),
            timeout=_request_timeout(),
        )
        response.raise_for_status()
        return self._answer_from_chunk(response.json())

    def generate_stream(
        self,
        prompt: str,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Yield response tokens as they arrive from Ollama's streaming API."""
        self.last_stream_stats = {}
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=self._request_body(prompt, stream=True),
            stream=True,
            timeout=_request_timeout(),
        )
        response.raise_for_status()
        thinking_parts: list[str] = []
        saw_response = False
        try:
            for line in response.iter_lines():
                if cancel_event and cancel_event.is_set():
                    break
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    saw_response = True
                    yield token
                elif chunk.get("thinking"):
                    thinking_parts.append(chunk["thinking"])
                if chunk.get("done"):
                    self.last_stream_stats = {
                        "prompt_eval_count": chunk.get("prompt_eval_count"),
                        "eval_count": chunk.get("eval_count"),
                    }
                    if not saw_response and thinking_parts:
                        fallback = "".join(thinking_parts)
                        logger.warning(
                            "Ollama stream had no response tokens (%d thinking chars); "
                            "enable OLLAMA_THINK_ENABLED=false (default) for qwen3.x",
                            len(fallback),
                        )
                        yield fallback
                    break
        finally:
            response.close()
