"""Google Gemini API provider (key-based, ``LLM_PROVIDER=gemini``)."""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator

import httpx

from providers.base import LLMProvider


class GeminiLLM(LLMProvider):
    """generateContent client for the Gemini API (requires ``GEMINI_API_KEY``)."""

    def __init__(self):
        self.base_url = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        self.timeout = float(os.getenv("GEMINI_TIMEOUT", "300"))
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
        self._client = httpx.Client(timeout=self.timeout)
        self.last_stream_stats: dict = {}

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def _payload(self, prompt: str) -> dict:
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }

    @staticmethod
    def _extract_text(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts)

    def _record_usage(self, data: dict) -> None:
        usage = data.get("usageMetadata") or {}
        if usage.get("promptTokenCount") is not None:
            self.last_stream_stats["prompt_eval_count"] = usage["promptTokenCount"]
        if usage.get("candidatesTokenCount") is not None:
            self.last_stream_stats["eval_count"] = usage["candidatesTokenCount"]

    def generate(self, prompt: str) -> str:
        response = self._client.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            headers=self._headers(),
            json=self._payload(prompt),
        )
        response.raise_for_status()
        data = response.json()
        self.last_stream_stats = {}
        self._record_usage(data)
        return self._extract_text(data)

    def generate_stream(
        self,
        prompt: str,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Yield text chunks from a streamGenerateContent SSE response."""
        self.last_stream_stats = {}
        with self._client.stream(
            "POST",
            f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse",
            headers=self._headers(),
            json=self._payload(prompt),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if cancel_event and cancel_event.is_set():
                    break
                if not line or not line.startswith("data:"):
                    continue
                try:
                    chunk = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                self._record_usage(chunk)
                text = self._extract_text(chunk)
                if text:
                    yield text
