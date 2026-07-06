"""Tests for hosted LLM providers: Anthropic/OpenAI/Gemini APIs + Claude CLI sign-in."""
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from providers.factory import get_llm, reset_providers


@pytest.fixture(autouse=True)
def _reset():
    reset_providers()
    yield
    reset_providers()


def _fake_response(payload: dict):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _fake_sse_stream(lines: list[str]):
    """Context manager mimicking httpx.Client.stream over SSE lines."""
    stream_response = MagicMock()
    stream_response.status_code = 200
    stream_response.iter_lines.return_value = iter(lines)
    stream_response.raise_for_status.return_value = None
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=stream_response)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Factory selection
# ---------------------------------------------------------------------------


class TestFactorySelection:
    def test_anthropic_provider_selected(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from providers.anthropic_llm import AnthropicLLM

        assert isinstance(get_llm(), AnthropicLLM)

    def test_openai_provider_selected(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from providers.openai_llm import OpenAILLM

        assert isinstance(get_llm(), OpenAILLM)

    def test_gemini_provider_selected(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "g-test")
        from providers.gemini_llm import GeminiLLM

        assert isinstance(get_llm(), GeminiLLM)

    def test_claude_cli_provider_selected(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "claude_cli")
        with patch("shutil.which", return_value="C:/fake/claude.cmd"):
            from providers.claude_cli_llm import ClaudeCLILLM

            assert isinstance(get_llm(), ClaudeCLILLM)

    def test_claude_subscription_provider_selected(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "claude_subscription")
        from providers.claude_subscription_llm import ClaudeSubscriptionLLM

        assert isinstance(get_llm(), ClaudeSubscriptionLLM)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "nonsense")
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm()


# ---------------------------------------------------------------------------
# Anthropic API
# ---------------------------------------------------------------------------


class TestAnthropicLLM:
    def _llm(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("ANTHROPIC_TEMPERATURE", raising=False)
        from providers.anthropic_llm import AnthropicLLM

        return AnthropicLLM()

    def test_temperature_omitted_by_default(self, monkeypatch):
        """Sonnet 5+ rejects `temperature` ('deprecated for this model')."""
        llm = self._llm(monkeypatch)
        assert "temperature" not in llm._payload("q", stream=False)

    def test_temperature_sent_when_configured(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.2")
        from providers.anthropic_llm import AnthropicLLM

        assert AnthropicLLM()._payload("q", stream=False)["temperature"] == 0.2

    def test_api_error_body_surfaced(self, monkeypatch):
        llm = self._llm(monkeypatch)
        bad = MagicMock(status_code=400)
        bad.read.return_value = (
            b'{"error":{"message":"`temperature` is deprecated for this model."}}'
        )
        llm._client.post = MagicMock(return_value=bad)

        with pytest.raises(RuntimeError, match="deprecated for this model"):
            llm.generate("q")

    def test_generate_parses_text_and_usage(self, monkeypatch):
        llm = self._llm(monkeypatch)
        llm._client.post = MagicMock(
            return_value=_fake_response(
                {
                    "content": [{"type": "text", "text": "Laser class 2."}],
                    "usage": {"input_tokens": 12, "output_tokens": 4},
                }
            )
        )

        assert llm.generate("q") == "Laser class 2."
        assert llm.last_stream_stats == {"prompt_eval_count": 12, "eval_count": 4}
        body = llm._client.post.call_args.kwargs["json"]
        assert body["stream"] is False
        assert body["messages"] == [{"role": "user", "content": "q"}]

    def test_generate_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from providers.anthropic_llm import AnthropicLLM

        llm = AnthropicLLM()
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            llm.generate("q")

    def test_generate_stream_yields_text_deltas(self, monkeypatch):
        llm = self._llm(monkeypatch)
        lines = [
            'data: {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}}',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}',
            'data: {"type": "message_delta", "usage": {"output_tokens": 2}}',
            'data: {"type": "message_stop"}',
        ]
        llm._client.stream = MagicMock(return_value=_fake_sse_stream(lines))

        tokens = list(llm.generate_stream("q"))

        assert tokens == ["Hel", "lo"]
        assert llm.last_stream_stats == {"prompt_eval_count": 9, "eval_count": 2}

    def test_generate_stream_stops_on_cancel(self, monkeypatch):
        llm = self._llm(monkeypatch)
        lines = [
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}}',
        ] * 50
        llm._client.stream = MagicMock(return_value=_fake_sse_stream(lines))
        cancel = threading.Event()
        cancel.set()

        assert list(llm.generate_stream("q", cancel_event=cancel)) == []


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------


class TestOpenAILLM:
    def _llm(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from providers.openai_llm import OpenAILLM

        return OpenAILLM()

    def test_generate_parses_choice(self, monkeypatch):
        llm = self._llm(monkeypatch)
        llm._client.post = MagicMock(
            return_value=_fake_response(
                {
                    "choices": [{"message": {"content": "NOHD is 10 cm."}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 6},
                }
            )
        )

        assert llm.generate("q") == "NOHD is 10 cm."
        assert llm.last_stream_stats == {"prompt_eval_count": 20, "eval_count": 6}
        url = llm._client.post.call_args.args[0]
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_stream_payload_requests_usage(self, monkeypatch):
        llm = self._llm(monkeypatch)
        payload = llm._payload("q", stream=True)
        assert payload["stream_options"] == {"include_usage": True}

    def test_generate_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from providers.openai_llm import OpenAILLM

        llm = OpenAILLM()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm.generate("q")


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------


class TestGeminiLLM:
    def _llm(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-test")
        from providers.gemini_llm import GeminiLLM

        return GeminiLLM()

    def test_generate_parses_candidates_and_usage(self, monkeypatch):
        llm = self._llm(monkeypatch)
        llm._client.post = MagicMock(
            return_value=_fake_response(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "Zivid 2+ is "}, {"text": "structured light."}]}}
                    ],
                    "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 7},
                }
            )
        )

        assert llm.generate("q") == "Zivid 2+ is structured light."
        assert llm.last_stream_stats == {"prompt_eval_count": 15, "eval_count": 7}
        url = llm._client.post.call_args.args[0]
        assert ":generateContent" in url and "gemini-2.5-flash" in url

    def test_generate_stream_yields_chunks(self, monkeypatch):
        llm = self._llm(monkeypatch)
        lines = [
            'data: {"candidates": [{"content": {"parts": [{"text": "Bin "}]}}]}',
            'data: {"candidates": [{"content": {"parts": [{"text": "picking."}]}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}}',
        ]
        llm._client.stream = MagicMock(return_value=_fake_sse_stream(lines))

        tokens = list(llm.generate_stream("q"))

        assert tokens == ["Bin ", "picking."]
        assert llm.last_stream_stats == {"prompt_eval_count": 5, "eval_count": 2}

    def test_generate_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from providers.gemini_llm import GeminiLLM

        llm = GeminiLLM()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            llm.generate("q")

    def test_google_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "g-fallback")
        from providers.gemini_llm import GeminiLLM

        assert GeminiLLM().api_key == "g-fallback"


# ---------------------------------------------------------------------------
# Claude subscription (direct Messages API with OAuth token)
# ---------------------------------------------------------------------------


class TestClaudeSubscriptionLLM:
    @pytest.fixture(autouse=True)
    def _token_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_OAUTH_TOKENS_PATH", str(tmp_path / "tokens.json"))
        monkeypatch.delenv("CLAUDE_SUBSCRIPTION_MODEL", raising=False)

    def _llm(self):
        from providers.claude_subscription_llm import ClaudeSubscriptionLLM

        return ClaudeSubscriptionLLM()

    def _sign_in(self):
        from providers import claude_oauth

        claude_oauth._store_tokens(
            {"access_token": "sk-ant-oat01-live", "expires_in": 3600}
        )

    def test_headers_use_bearer_token_and_beta(self):
        self._sign_in()
        headers = self._llm()._headers()
        assert headers["Authorization"] == "Bearer sk-ant-oat01-live"
        assert headers["anthropic-beta"] == "oauth-2025-04-20"
        assert "x-api-key" not in headers

    def test_not_signed_in_raises_clean_error(self):
        llm = self._llm()
        with pytest.raises(RuntimeError, match="Not signed in"):
            llm._headers()

    def test_payload_leads_with_claude_code_identity_block(self):
        from providers.claude_subscription_llm import CLAUDE_CODE_IDENTITY

        payload = self._llm()._payload("What is NOHD?", stream=False)
        assert payload["system"][0] == {"type": "text", "text": CLAUDE_CODE_IDENTITY}
        assert payload["messages"] == [{"role": "user", "content": "What is NOHD?"}]

    def test_default_model_and_env_override(self, monkeypatch):
        assert self._llm().model == "claude-sonnet-5"
        monkeypatch.setenv("CLAUDE_SUBSCRIPTION_MODEL", "claude-haiku-4-5")
        assert self._llm().model == "claude-haiku-4-5"

    def test_generate_parses_text_like_api_provider(self):
        self._sign_in()
        llm = self._llm()
        llm._client.post = MagicMock(
            return_value=_fake_response(
                {
                    "content": [{"type": "text", "text": "NOHD is 10 cm."}],
                    "usage": {"input_tokens": 12, "output_tokens": 4},
                }
            )
        )

        assert llm.generate("q") == "NOHD is 10 cm."
        body = llm._client.post.call_args.kwargs["json"]
        assert body["system"][0]["text"].startswith("You are Claude Code")


# ---------------------------------------------------------------------------
# Claude CLI (legacy subscription path via the Claude Code CLI)
# ---------------------------------------------------------------------------


class TestClaudeCLILLM:
    def _llm(self, monkeypatch, model: str | None = None):
        if model is not None:
            monkeypatch.setenv("CLAUDE_CLI_MODEL", model)
        else:
            monkeypatch.delenv("CLAUDE_CLI_MODEL", raising=False)
        with patch("shutil.which", return_value="C:/fake/claude.cmd"):
            from providers.claude_cli_llm import ClaudeCLILLM

            return ClaudeCLILLM()

    def test_missing_cli_raises_with_install_hint(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CLI_PATH", raising=False)
        with (
            patch("shutil.which", return_value=None),
            patch("providers.claude_cli_llm.os.path.isfile", return_value=False),
        ):
            from providers.claude_cli_llm import ClaudeCLIError, ClaudeCLILLM

            with pytest.raises(ClaudeCLIError, match="CLAUDE_CLI_PATH"):
                ClaudeCLILLM()

    def test_find_claude_cli_falls_back_to_known_locations(self, monkeypatch, tmp_path):
        """A server started with a PATH lacking %APPDATA%\\npm must still find the CLI."""
        monkeypatch.delenv("CLAUDE_CLI_PATH", raising=False)
        monkeypatch.setenv("APPDATA", str(tmp_path))
        npm_dir = tmp_path / "npm"
        npm_dir.mkdir()
        cli = npm_dir / "claude.cmd"
        cli.write_text("@echo off", encoding="utf-8")

        with patch("shutil.which", return_value=None):
            from providers.claude_cli_llm import find_claude_cli

            assert find_claude_cli() == str(cli)

    def test_find_claude_cli_honors_explicit_path(self, monkeypatch, tmp_path):
        cli = tmp_path / "custom-claude.exe"
        cli.write_text("", encoding="utf-8")
        monkeypatch.setenv("CLAUDE_CLI_PATH", str(cli))

        with patch("shutil.which", return_value=None):
            from providers.claude_cli_llm import find_claude_cli

            assert find_claude_cli() == str(cli)

    def test_generate_passes_prompt_on_stdin_and_parses_result(self, monkeypatch):
        llm = self._llm(monkeypatch)
        completed = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "result": "Hand-eye calibration aligns camera and robot frames.",
                    "usage": {"input_tokens": 30, "output_tokens": 10},
                }
            ),
            stderr="",
        )
        with patch("providers.claude_cli_llm.subprocess.run", return_value=completed) as run:
            answer = llm.generate("What is hand-eye calibration?")

        assert answer == "Hand-eye calibration aligns camera and robot frames."
        assert llm.last_stream_stats == {"prompt_eval_count": 30, "eval_count": 10}
        args = run.call_args.args[0]
        assert args[:2] == ["C:/fake/claude.cmd", "-p"]
        assert "--output-format" in args and "json" in args
        assert "--max-turns" in args
        assert run.call_args.kwargs["input"] == "What is hand-eye calibration?"

    def test_generate_includes_model_flag_when_set(self, monkeypatch):
        llm = self._llm(monkeypatch, model="claude-sonnet-5")
        completed = MagicMock(returncode=0, stdout='{"result": "ok"}', stderr="")
        with patch("providers.claude_cli_llm.subprocess.run", return_value=completed) as run:
            llm.generate("q")
        args = run.call_args.args[0]
        assert "--model" in args and "claude-sonnet-5" in args

    def test_generate_raises_on_nonzero_exit(self, monkeypatch):
        llm = self._llm(monkeypatch)
        completed = MagicMock(returncode=1, stdout="", stderr="not logged in")
        from providers.claude_cli_llm import ClaudeCLIError

        with patch("providers.claude_cli_llm.subprocess.run", return_value=completed):
            with pytest.raises(ClaudeCLIError, match="not logged in"):
                llm.generate("q")

    def test_generate_raises_on_error_result(self, monkeypatch):
        llm = self._llm(monkeypatch)
        completed = MagicMock(
            returncode=0,
            stdout='{"is_error": true, "result": "usage limit reached"}',
            stderr="",
        )
        from providers.claude_cli_llm import ClaudeCLIError

        with patch("providers.claude_cli_llm.subprocess.run", return_value=completed):
            with pytest.raises(ClaudeCLIError, match="usage limit"):
                llm.generate("q")

    def test_generate_surfaces_json_error_over_exit_code(self, monkeypatch):
        """'Not logged in' arrives as JSON on stdout with exit code 1 and empty
        stderr — the JSON message must win (observed live on 2.1.201)."""
        llm = self._llm(monkeypatch)
        completed = MagicMock(
            returncode=1,
            stdout='{"type": "result", "is_error": true, "result": "Not logged in · Please run /login"}',
            stderr="",
        )
        from providers.claude_cli_llm import ClaudeCLIError

        with patch("providers.claude_cli_llm.subprocess.run", return_value=completed):
            with pytest.raises(ClaudeCLIError, match="Not logged in"):
                llm.generate("q")

    def test_delta_text_extracts_stream_event_text(self, monkeypatch):
        llm = self._llm(monkeypatch)
        event = {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "chunk"},
            },
        }
        assert llm._delta_text(event) == "chunk"
        assert llm._delta_text({"type": "result"}) == ""
        assert llm._delta_text({"type": "stream_event", "event": {"type": "message_start"}}) == ""
