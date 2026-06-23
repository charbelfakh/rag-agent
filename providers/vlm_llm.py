"""Vision-language model via OpenAI-compatible multimodal chat API (Sprint N M3)."""
from __future__ import annotations

import json
import os

import httpx

from providers.base import LLMProvider


class VLMLLM(LLMProvider):
    """Multimodal chat-completions client for vision-language models."""

    def __init__(self):
        self.base_url = os.getenv(
            "VLM_BASE_URL",
            os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8002/v1"),
        ).rstrip("/")
        self.model = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.api_key = os.getenv("VLM_API_KEY", os.getenv("OPENAI_COMPAT_API_KEY", "EMPTY"))
        self.timeout = float(os.getenv("VLM_TIMEOUT", "300"))
        self._client = httpx.Client(timeout=self.timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, prompt: str) -> str:
        return self.generate_with_images(prompt, image_uris=[])

    def generate_with_images(self, prompt: str, *, image_uris: list[str]) -> str:
        """Generate from text plus up to five ``image_url`` parts."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for uri in image_uris[:5]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": uri},
                }
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(os.getenv("VLM_TEMPERATURE", "0.2")),
        }
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
