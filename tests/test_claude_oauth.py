"""Tests for the in-app Claude subscription OAuth flow (providers/claude_oauth.py)."""
import base64
import hashlib
import json
import time
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from providers import claude_oauth


@pytest.fixture(autouse=True)
def _isolated_flow(tmp_path, monkeypatch):
    """Fresh pending state + token store under a temp path."""
    monkeypatch.setenv("CLAUDE_OAUTH_TOKENS_PATH", str(tmp_path / "tokens.json"))
    claude_oauth.reset_pending()
    yield
    claude_oauth.reset_pending()


def _token_response(payload: dict):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class TestStartLogin:
    def test_authorize_url_has_pkce_and_state(self):
        result = claude_oauth.start_login()
        parsed = urlparse(result["url"])
        query = parse_qs(parsed.query)

        assert parsed.scheme == "https" and parsed.netloc == "claude.ai"
        assert query["client_id"] == [claude_oauth.CLIENT_ID]
        assert query["response_type"] == ["code"]
        assert query["code_challenge_method"] == ["S256"]
        assert query["code"] == ["true"]
        assert query["code_challenge"][0]
        assert query["state"][0]

    def test_challenge_matches_verifier(self):
        result = claude_oauth.start_login()
        query = parse_qs(urlparse(result["url"]).query)
        derived = (
            base64.urlsafe_b64encode(
                hashlib.sha256(
                    claude_oauth._pending["verifier"].encode("ascii")
                ).digest()
            )
            .rstrip(b"=")
            .decode("ascii")
        )
        assert query["code_challenge"] == [derived]


class TestParsePastedCode:
    def test_code_with_state_fragment(self):
        code, state = claude_oauth._parse_pasted_code("abc123#mystate")
        assert code == "abc123"
        assert state == "mystate"

    def test_bare_code(self):
        code, state = claude_oauth._parse_pasted_code("  abc123  ")
        assert code == "abc123"
        assert state is None

    def test_empty_raises(self):
        with pytest.raises(claude_oauth.ClaudeOAuthError, match="Empty code"):
            claude_oauth._parse_pasted_code("   ")


class TestFinishLogin:
    def test_requires_started_flow(self):
        with pytest.raises(claude_oauth.ClaudeOAuthError, match="No sign-in in progress"):
            claude_oauth.finish_login("abc")

    def test_state_mismatch_rejected(self):
        claude_oauth.start_login()
        with pytest.raises(claude_oauth.ClaudeOAuthError, match="State mismatch"):
            claude_oauth.finish_login("code#wrong-state")

    def test_exchanges_code_and_stores_tokens(self):
        claude_oauth.start_login()
        state = claude_oauth._pending["state"]
        verifier = claude_oauth._pending["verifier"]
        token_payload = {
            "access_token": "sk-ant-oat01-test",
            "refresh_token": "sk-ant-ort01-test",
            "expires_in": 3600,
            "subscription_type": "max",
        }
        with patch(
            "providers.claude_oauth.httpx.post",
            return_value=_token_response(token_payload),
        ) as post:
            result = claude_oauth.finish_login(f"authcode#{state}")

        assert result == {"signed_in": True, "subscription_type": "max"}
        body = post.call_args.kwargs["json"]
        assert body["grant_type"] == "authorization_code"
        assert body["code"] == "authcode"
        assert body["code_verifier"] == verifier
        assert body["client_id"] == claude_oauth.CLIENT_ID

        stored = json.loads(claude_oauth.tokens_path().read_text(encoding="utf-8"))
        assert stored["access_token"] == "sk-ant-oat01-test"
        assert stored["refresh_token"] == "sk-ant-ort01-test"
        assert stored["expires_at"] > time.time()
        assert claude_oauth.is_signed_in() is True
        assert claude_oauth._pending is None

    def test_missing_access_token_raises(self):
        claude_oauth.start_login()
        state = claude_oauth._pending["state"]
        with patch(
            "providers.claude_oauth.httpx.post",
            return_value=_token_response({"error": "denied"}),
        ):
            with pytest.raises(claude_oauth.ClaudeOAuthError, match="no access_token"):
                claude_oauth.finish_login(f"c#{state}")


class TestValidAccessToken:
    def _store(self, *, expires_in: float, refresh: str | None = "sk-ant-ort01-r"):
        claude_oauth._store_tokens(
            {
                "access_token": "sk-ant-oat01-live",
                "refresh_token": refresh,
                "expires_in": expires_in,
            }
        )

    def test_fresh_token_returned_without_refresh(self):
        self._store(expires_in=3600)
        with patch("providers.claude_oauth.httpx.post") as post:
            assert claude_oauth.valid_access_token() == "sk-ant-oat01-live"
        post.assert_not_called()

    def test_near_expiry_token_refreshed(self):
        self._store(expires_in=10)  # inside the 60s skew
        refreshed = {
            "access_token": "sk-ant-oat01-new",
            "refresh_token": "sk-ant-ort01-new",
            "expires_in": 3600,
        }
        with patch(
            "providers.claude_oauth.httpx.post",
            return_value=_token_response(refreshed),
        ) as post:
            assert claude_oauth.valid_access_token() == "sk-ant-oat01-new"

        body = post.call_args.kwargs["json"]
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "sk-ant-ort01-r"
        stored = json.loads(claude_oauth.tokens_path().read_text(encoding="utf-8"))
        assert stored["access_token"] == "sk-ant-oat01-new"

    def test_refresh_failure_falls_back_to_stale_token(self):
        self._store(expires_in=10)
        with patch(
            "providers.claude_oauth.httpx.post",
            side_effect=claude_oauth.httpx.ConnectError("offline"),
        ):
            assert claude_oauth.valid_access_token() == "sk-ant-oat01-live"

    def test_no_tokens_returns_none(self):
        assert claude_oauth.valid_access_token() is None


class TestSignedInAndLogout:
    def test_reflects_token_file(self):
        assert claude_oauth.is_signed_in() is False
        claude_oauth._store_tokens({"access_token": "tok", "expires_in": 60})
        assert claude_oauth.is_signed_in() is True

    def test_logout_forgets_tokens(self):
        claude_oauth._store_tokens({"access_token": "tok", "expires_in": 60})
        claude_oauth.logout()
        assert claude_oauth.is_signed_in() is False
        claude_oauth.logout()  # idempotent
