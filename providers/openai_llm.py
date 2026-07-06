"""OpenAI (ChatGPT) API provider (key-based, ``LLM_PROVIDER=openai``).

Reuses the OpenAI-compatible chat client but reads the official ``OPENAI_*``
env vars and requires a real API key.
"""
from __future__ import annotations

import os

import httpx

from providers.openai_chat_llm import OpenAIChatLLM


class OpenAILLM(OpenAIChatLLM):
    """Chat-completions client for api.openai.com (requires ``OPENAI_API_KEY``)."""

    def __init__(self):
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "300"))
        self._client = httpx.Client(timeout=self.timeout)
        self.last_stream_stats: dict = {}

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _payload(self, prompt: str, *, stream: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        }
        if stream:
            # Ask OpenAI to append a final usage chunk to the stream.
            payload["stream_options"] = {"include_usage": True}
        return payload
