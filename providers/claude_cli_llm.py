"""Claude subscription provider via the Claude Code CLI (``LLM_PROVIDER=claude_cli``).

Uses the locally installed ``claude`` CLI in headless mode (``claude -p``), which
authenticates with the user's Claude subscription sign-in — no API key needed.
Sign in once with ``claude`` → ``/login``; this provider reuses that session.

The prompt is passed on stdin (RAG prompts exceed Windows argv limits).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator

from providers.base import LLMProvider


class ClaudeCLIError(RuntimeError):
    """Raised when the claude CLI is missing or a call fails."""


def _candidate_cli_paths() -> list[str]:
    """Well-known install locations, for processes whose PATH lacks them."""
    candidates: list[str] = []
    appdata = os.getenv("APPDATA", "")
    if appdata:
        candidates += [
            os.path.join(appdata, "npm", "claude.cmd"),
            os.path.join(appdata, "npm", "claude.exe"),
        ]
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, ".local", "bin", "claude.exe"))
    candidates.append(os.path.join(home, ".local", "bin", "claude"))
    localappdata = os.getenv("LOCALAPPDATA", "")
    if localappdata:
        candidates.append(
            os.path.join(localappdata, "Programs", "claude", "claude.exe")
        )
    return candidates


def find_claude_cli() -> str | None:
    """Locate the claude CLI: CLAUDE_CLI_PATH, then PATH, then known locations."""
    configured = os.getenv("CLAUDE_CLI_PATH", "").strip()
    if configured:
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        if os.path.isfile(configured):
            return configured
    resolved = shutil.which("claude")
    if resolved:
        return resolved
    for candidate in _candidate_cli_paths():
        if os.path.isfile(candidate):
            return candidate
    return None


class ClaudeCLILLM(LLMProvider):
    """Headless ``claude -p`` client backed by the user's subscription sign-in."""

    def __init__(self):
        cli_path = find_claude_cli()
        if cli_path is None:
            raise ClaudeCLIError(
                "claude CLI not found. Install Claude Code "
                "(npm install -g @anthropic-ai/claude-code) or set CLAUDE_CLI_PATH "
                "to the binary."
            )
        self.cli_path = cli_path
        self.model = os.getenv("CLAUDE_CLI_MODEL", "").strip() or None
        self.timeout = float(os.getenv("CLAUDE_CLI_TIMEOUT", "300"))
        self.last_stream_stats: dict = {}

    def _base_args(self, output_format: str) -> list[str]:
        args = [
            self.cli_path,
            "-p",
            "--output-format",
            output_format,
            # Single answer turn: no tool loops, just text generation.
            "--max-turns",
            "1",
        ]
        if self.model:
            args += ["--model", self.model]
        return args

    def _record_usage(self, usage: dict) -> None:
        if usage.get("input_tokens") is not None:
            self.last_stream_stats["prompt_eval_count"] = usage["input_tokens"]
        if usage.get("output_tokens") is not None:
            self.last_stream_stats["eval_count"] = usage["output_tokens"]

    def generate(self, prompt: str) -> str:
        self.last_stream_stats = {}
        try:
            completed = subprocess.run(
                self._base_args("json"),
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(f"claude CLI timed out after {self.timeout}s") from exc
        # Error results (e.g. "Not logged in") arrive as JSON on stdout with a
        # nonzero exit code — prefer that message over (often empty) stderr.
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError:
            data = None
        if data is not None and data.get("is_error"):
            raise ClaudeCLIError(f"claude CLI error: {data.get('result')!r}")
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise ClaudeCLIError(
                f"claude CLI exited with {completed.returncode}: {stderr[:500]}"
            )
        if data is None:
            raise ClaudeCLIError(
                f"claude CLI returned non-JSON output: {completed.stdout[:200]!r}"
            )
        self._record_usage(data.get("usage") or {})
        return str(data.get("result") or "")

    def generate_stream(
        self,
        prompt: str,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Yield text deltas from ``--output-format stream-json``."""
        self.last_stream_stats = {}
        args = self._base_args("stream-json") + [
            "--include-partial-messages",
            "--verbose",
        ]
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
            for line in proc.stdout:
                if cancel_event and cancel_event.is_set():
                    proc.kill()
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = self._delta_text(event)
                if text:
                    yield text
                if event.get("type") == "result":
                    self._record_usage(event.get("usage") or {})
            returncode = proc.wait(timeout=30)
            if returncode not in (0, None) and not (cancel_event and cancel_event.is_set()):
                stderr = (proc.stderr.read() if proc.stderr else "").strip()
                raise ClaudeCLIError(
                    f"claude CLI stream exited with {returncode}: {stderr[:500]}"
                )
        finally:
            if proc.poll() is None:
                proc.kill()

    @staticmethod
    def _delta_text(event: dict) -> str:
        """Extract a text delta from a stream-json event, if present."""
        if event.get("type") != "stream_event":
            return ""
        inner = event.get("event") or {}
        if inner.get("type") != "content_block_delta":
            return ""
        delta = inner.get("delta") or {}
        if delta.get("type") != "text_delta":
            return ""
        return delta.get("text") or ""
