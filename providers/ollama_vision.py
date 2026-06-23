"""Ollama vision model: describe a single image via /api/chat."""
from __future__ import annotations

import base64
import os

import requests


class OllamaVision:
    """Describe images via Ollama ``/api/chat`` with base64-encoded frames."""

    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_VISION_MODEL", "moondream")
        self.timeout = float(os.getenv("OLLAMA_VISION_TIMEOUT", "120"))

    def describe_image(self, prompt: str, image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content", "")
        return str(content).strip()
