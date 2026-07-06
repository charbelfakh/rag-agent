"""Anthropic Claude API provider (key-based, ``LLM_PROVIDER=anthropic``)."""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator

import httpx

from providers.base import LLMProvider

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicLLM(LLMProvider):
    """Claude Messages API client (requires ``ANTHROPIC_API_KEY``)."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        # ``model`` / ``max_tokens`` let the factory build a cheaper "fast tier"
        # instance (e.g. Haiku for HyDE) without a second env var per stage;
        # ``None`` keeps the current env-driven defaults (backward compatible).
        self.base_url = os.getenv(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        ).rstrip("/")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.timeout = float(os.getenv("ANTHROPIC_TIMEOUT", "300"))
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048"))
        )
        # Newer Claude models (Sonnet 5+, Fable) reject `temperature` outright
        # ("deprecated for this model"); send it only when explicitly configured.
        temperature_env = os.getenv("ANTHROPIC_TEMPERATURE", "").strip()
        self.temperature: float | None = float(temperature_env) if temperature_env else None
        self._client = httpx.Client(timeout=self.timeout)
        self.last_stream_stats: dict = {}

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _payload(self, prompt: str, *, stream: bool, system: str | None = None) -> dict:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if system:
            # Stable rules in a cached prefix; variable content stays in the user
            # message. Caching only kicks in once the block exceeds the model's
            # minimum cacheable prefix (a no-op, not an error, below it).
            payload["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    @staticmethod
    def _raise_for_status_with_body(response: httpx.Response) -> None:
        """Raise with the API's own error message — bare 4xx codes hide the reason."""
        if response.status_code < 400:
            return
        body = response.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(
            f"Anthropic API error {response.status_code}: {body}"
        )

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = self._client.post(
            f"{self.base_url}/v1/messages",
            headers=self._headers(),
            json=self._payload(prompt, stream=False, system=system),
        )
        self._raise_for_status_with_body(response)
        data = response.json()
        usage = data.get("usage") or {}
        self.last_stream_stats = {
            "prompt_eval_count": usage.get("input_tokens"),
            "eval_count": usage.get("output_tokens"),
        }
        return "".join(
            block.get("text", "")
            for block in data.get("content") or []
            if block.get("type") == "text"
        )

    def generate_stream(
        self,
        prompt: str,
        cancel_event: threading.Event | None = None,
        system: str | None = None,
    ) -> Iterator[str]:
        """Yield text deltas from a Messages API SSE stream."""
        self.last_stream_stats = {}
        with self._client.stream(
            "POST",
            f"{self.base_url}/v1/messages",
            headers=self._headers(),
            json=self._payload(prompt, stream=True, system=system),
        ) as response:
            self._raise_for_status_with_body(response)
            for line in response.iter_lines():
                if cancel_event and cancel_event.is_set():
                    break
                if not line or not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "message_start":
                    usage = (event.get("message") or {}).get("usage") or {}
                    if usage.get("input_tokens") is not None:
                        self.last_stream_stats["prompt_eval_count"] = usage["input_tokens"]
                elif event_type == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield delta["text"]
                elif event_type == "message_delta":
                    usage = event.get("usage") or {}
                    if usage.get("output_tokens") is not None:
                        self.last_stream_stats["eval_count"] = usage["output_tokens"]
                elif event_type == "message_stop":
                    break
