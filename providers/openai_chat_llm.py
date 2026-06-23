"""OpenAI-compatible chat API for vLLM / TGI inference pools (Sprint K)."""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator

import httpx

from providers.base import LLMProvider


class OpenAIChatLLM(LLMProvider):
    """Chat-completions client for vLLM, TGI, and other OpenAI-compatible pools."""

    def __init__(self):
        self.base_url = os.getenv(
            "OPENAI_COMPAT_BASE_URL",
            os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1"),
        ).rstrip("/")
        self.model = os.getenv(
            "OPENAI_COMPAT_MODEL",
            os.getenv("VLLM_MODEL", "mistral-7b-instruct"),
        )
        self.api_key = os.getenv("OPENAI_COMPAT_API_KEY", "EMPTY")
        self.timeout = float(os.getenv("OPENAI_COMPAT_TIMEOUT", "300"))
        self._client = httpx.Client(timeout=self.timeout)
        self.last_stream_stats: dict = {}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, prompt: str, *, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": float(os.getenv("OPENAI_COMPAT_TEMPERATURE", "0.2")),
        }

    def generate(self, prompt: str) -> str:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(prompt, stream=False),
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        self.last_stream_stats = {
            "prompt_eval_count": usage.get("prompt_tokens"),
            "eval_count": usage.get("completion_tokens"),
        }
        return data["choices"][0]["message"]["content"]

    def generate_stream(
        self,
        prompt: str,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Yield completion tokens from an OpenAI-compatible streaming response."""
        self.last_stream_stats = {}
        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(prompt, stream=True),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if cancel_event and cancel_event.is_set():
                    break
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    self.last_stream_stats = {
                        "prompt_eval_count": usage.get("prompt_tokens"),
                        "eval_count": usage.get("completion_tokens"),
                    }
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token
