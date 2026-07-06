"""Anthropic Message Batches API — offline bulk inference at 50% cost.

For latency-insensitive workloads (e.g. LLM-as-judge grading of the whole golden
set): many independent prompts submitted as one batch, billed at half the normal
per-token rate. Uses the existing ``httpx``-based auth (``ANTHROPIC_API_KEY`` +
``ANTHROPIC_BASE_URL``) — no SDK dependency, mirroring ``providers.anthropic_llm``.

A single ``system`` prefix (optional) is sent as a cached block on every request,
so a shared rubric/instruction set is written once and read cheaply across the
batch (the classic "stable rules cached, variable content after" pattern).
"""
from __future__ import annotations

import json
import os
import time

import httpx

from providers.anthropic_llm import ANTHROPIC_VERSION, AnthropicLLM

BATCHES_PATH = "/v1/messages/batches"


class AnthropicBatchClient:
    """Create → poll → collect a Message Batches job. Results keyed by ``custom_id``."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        self.base_url = os.getenv(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        ).rstrip("/")
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048"))
        )
        self.timeout = float(os.getenv("ANTHROPIC_TIMEOUT", "300"))
        self.poll_interval = float(os.getenv("ANTHROPIC_BATCH_POLL_SECONDS", "10"))
        self.poll_timeout = float(os.getenv("ANTHROPIC_BATCH_TIMEOUT_SECONDS", "86400"))
        self._client = httpx.Client(timeout=self.timeout)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _params(self, prompt: str, system: str | None) -> dict:
        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            params["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        return params

    def create(self, prompts: dict[str, str], system: str | None = None) -> str:
        """Submit ``{custom_id: prompt}`` as one batch; return the batch id."""
        requests = [
            {"custom_id": custom_id, "params": self._params(prompt, system)}
            for custom_id, prompt in prompts.items()
        ]
        response = self._client.post(
            f"{self.base_url}{BATCHES_PATH}",
            headers=self._headers(),
            json={"requests": requests},
        )
        AnthropicLLM._raise_for_status_with_body(response)
        return response.json()["id"]

    def poll(self, batch_id: str) -> dict:
        """Block until the batch's ``processing_status`` is ``ended``; return the object."""
        deadline = time.monotonic() + self.poll_timeout
        while True:
            response = self._client.get(
                f"{self.base_url}{BATCHES_PATH}/{batch_id}",
                headers=self._headers(),
            )
            AnthropicLLM._raise_for_status_with_body(response)
            data = response.json()
            if data.get("processing_status") == "ended":
                return data
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Batch {batch_id} did not finish within {self.poll_timeout}s"
                )
            time.sleep(self.poll_interval)

    def results(self, batch: dict) -> dict[str, str]:
        """Fetch the JSONL results of an ended batch → ``{custom_id: answer_text}``.

        Errored/expired/canceled requests map to an empty string (results arrive
        in arbitrary order — always key by ``custom_id``, never by position).
        """
        results_url = batch.get("results_url")
        if not results_url:
            return {}
        response = self._client.get(results_url, headers=self._headers())
        AnthropicLLM._raise_for_status_with_body(response)
        answers: dict[str, str] = {}
        for line in response.text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            result = row.get("result") or {}
            if result.get("type") == "succeeded":
                message = result.get("message") or {}
                answers[custom_id] = "".join(
                    block.get("text", "")
                    for block in message.get("content") or []
                    if block.get("type") == "text"
                )
            else:
                answers[custom_id] = ""
        return answers

    def run(self, prompts: dict[str, str], system: str | None = None) -> dict[str, str]:
        """create → poll → results in one call."""
        batch_id = self.create(prompts, system=system)
        return self.results(self.poll(batch_id))


def submit_prompts(
    prompts: dict[str, str],
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, str]:
    """Convenience: run a one-off batch and return ``{custom_id: answer_text}``."""
    return AnthropicBatchClient(model=model, max_tokens=max_tokens).run(
        prompts, system=system
    )
