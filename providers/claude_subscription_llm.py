"""Claude subscription provider — direct Messages API with the OAuth token.

Ported from the ForgeStation assistant: after the in-app OAuth sign-in
(:mod:`providers.claude_oauth`), requests go straight to the Anthropic
Messages API with ``Authorization: Bearer <token>`` plus the OAuth beta
header. No Claude Code CLI involved; tokens auto-refresh before expiry.

Anthropic gates subscription (Bearer) tokens to Claude Code: the API only
accepts them when the system prompt's FIRST block is exactly the Claude Code
identity string below — without it the API returns a disguised 429. This makes
requests identify as Claude Code, which is outside Anthropic's intended use of
subscription tokens. Acceptable for personal use of your own account; do NOT
redistribute an app relying on this — use the ``anthropic`` API-key provider
instead.
"""
from __future__ import annotations

import os

from providers import claude_oauth
from providers.anthropic_llm import AnthropicLLM, ANTHROPIC_VERSION

CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


class ClaudeSubscriptionLLM(AnthropicLLM):
    """Messages API client authenticated by the subscription OAuth token."""

    def __init__(self):
        super().__init__()
        self.model = os.getenv("CLAUDE_SUBSCRIPTION_MODEL", "claude-sonnet-5")

    def _headers(self) -> dict[str, str]:
        token = claude_oauth.valid_access_token()
        if not token:
            raise RuntimeError(
                "Not signed in — connect Claude (subscription) in Settings first."
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": claude_oauth.OAUTH_BETA_HEADER,
        }

    def _payload(self, prompt: str, *, stream: bool, system: str | None = None) -> dict:
        payload = super()._payload(prompt, stream=stream, system=system)
        # Required first system block for subscription tokens (see module docstring);
        # any cached rules block from the parent follows it so caching still applies.
        identity = {"type": "text", "text": CLAUDE_CODE_IDENTITY}
        payload["system"] = [identity, *(payload.get("system") or [])]
        return payload
