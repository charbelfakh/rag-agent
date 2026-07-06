"""In-app OAuth for the Claude subscription (PKCE, paste-back code, own token store).

Ported from the ForgeStation assistant's self-contained flow: the user approves
at claude.ai in a browser tab, copies the one-time ``code#state`` string from
the callback page, and pastes it back. Tokens are stored in this app's own
token file and refreshed transparently (``grant_type=refresh_token``) shortly
before expiry — no Claude Code CLI involved anywhere.

The access token is consumed by :mod:`providers.claude_subscription_llm`, which
calls the Anthropic Messages API directly with Bearer auth.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

import httpx

# Public OAuth client used by the "Login with Claude" flow. This client only
# supports the console copy-paste redirect, so we use the manual paste flow.
CLIENT_ID = os.getenv("CLAUDE_OAUTH_CLIENT_ID", "9d1c250a-e61b-44d9-88ed-5944d1962f5e")
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
SCOPES = "org:create_api_key user:profile user:inference"

# OAuth (Bearer) tokens require this beta header on every Messages API request.
OAUTH_BETA_HEADER = "oauth-2025-04-20"

# Refresh a little early so a request never goes out with a just-expired token.
_REFRESH_SKEW_S = 60.0
_TOKEN_HTTP_TIMEOUT_S = 30.0
# A default python UA is commonly rejected (403) by the edge; identify explicitly.
_HTTP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "rag-agent-assistant/1.0",
}

# Single pending login per process — fine for a local-first app.
_pending: dict | None = None


class ClaudeOAuthError(RuntimeError):
    """Raised when the OAuth flow cannot start or complete."""


def tokens_path() -> Path:
    configured = os.getenv("CLAUDE_OAUTH_TOKENS_PATH", "").strip()
    if configured:
        return Path(configured)
    from providers.app_paths import secret_file

    return secret_file(
        "claude_oauth_tokens.json",
        legacy_path=Path("data") / "claude_oauth_tokens.json",
    )


def credentials_path() -> Path:
    """Claude Code CLI credentials file (used only by the legacy claude_cli path)."""
    config_dir = os.getenv("CLAUDE_CONFIG_DIR", "").strip()
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / ".credentials.json"


def _load_tokens() -> dict | None:
    try:
        record = json.loads(tokens_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _store_tokens(payload: dict) -> dict:
    access_token = payload.get("access_token")
    if not access_token:
        raise ClaudeOAuthError(
            f"Token endpoint returned no access_token: {str(payload)[:200]}"
        )
    record = {
        "access_token": access_token,
        "refresh_token": payload.get("refresh_token"),
        "expires_at": time.time() + float(payload.get("expires_in", 3600)),
    }
    path = tokens_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    try:  # best-effort: restrict to the owner on POSIX (no-op on Windows)
        os.chmod(path, 0o600)
    except OSError:
        pass
    return record


def is_signed_in() -> bool:
    record = _load_tokens()
    return bool(record and record.get("access_token"))


def logout() -> None:
    """Forget the stored subscription tokens."""
    try:
        tokens_path().unlink()
    except OSError:
        pass


def _post_token_endpoint(payload: dict) -> dict:
    try:
        response = httpx.post(
            TOKEN_URL, json=payload, headers=_HTTP_HEADERS, timeout=_TOKEN_HTTP_TIMEOUT_S
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:300]
        raise ClaudeOAuthError(
            f"Token exchange failed ({exc.response.status_code}): {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ClaudeOAuthError(f"Token exchange failed: {exc}") from exc
    return response.json()


def _refresh_tokens(record: dict) -> dict:
    return _store_tokens(
        _post_token_endpoint(
            {
                "grant_type": "refresh_token",
                "refresh_token": record["refresh_token"],
                "client_id": CLIENT_ID,
            }
        )
    )


def valid_access_token() -> str | None:
    """Stored access token, refreshed in place when it is near expiry."""
    record = _load_tokens()
    if not record or not record.get("access_token"):
        return None
    if record.get("expires_at", 0) - time.time() > _REFRESH_SKEW_S:
        return record["access_token"]
    if record.get("refresh_token"):
        try:
            record = _refresh_tokens(record)
        except ClaudeOAuthError:
            # Fall back to the (possibly stale) token; the API call will 401
            # with a clearer message if it is truly dead.
            pass
    return record.get("access_token")


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def start_login() -> dict:
    """Begin a PKCE login; returns the authorization URL to open in a browser."""
    global _pending
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)
    _pending = {"verifier": verifier, "state": state, "started_at": time.time()}
    from urllib.parse import urlencode

    query = urlencode(
        {
            "code": "true",  # callback page displays a copyable code
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )
    return {"url": f"{AUTHORIZE_URL}?{query}"}


def _parse_pasted_code(raw: str) -> tuple[str, str | None]:
    """The callback page shows ``<code>#<state>``; accept either form."""
    cleaned = (raw or "").strip()
    if not cleaned:
        raise ClaudeOAuthError("Empty code — paste the code shown after approving.")
    if "#" in cleaned:
        code, state = cleaned.split("#", 1)
        return code.strip(), state.strip() or None
    return cleaned, None


def finish_login(pasted_code: str) -> dict:
    """Exchange the pasted code for tokens and store them in the token file."""
    global _pending
    if _pending is None:
        raise ClaudeOAuthError("No sign-in in progress — start the sign-in first.")
    code, pasted_state = _parse_pasted_code(pasted_code)
    if pasted_state and pasted_state != _pending["state"]:
        raise ClaudeOAuthError("State mismatch — restart the sign-in and try again.")

    data = _post_token_endpoint(
        {
            "grant_type": "authorization_code",
            "code": code,
            "state": pasted_state or _pending["state"],
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": _pending["verifier"],
        }
    )
    _store_tokens(data)
    _pending = None
    return {"signed_in": True, "subscription_type": data.get("subscription_type")}


def reset_pending() -> None:
    """Clear any in-flight login (for tests)."""
    global _pending
    _pending = None
