"""Tests for providers/app_paths.py — per-user secret file locations."""

from pathlib import Path

import pytest

from providers import app_paths


@pytest.fixture()
def app_dir(tmp_path, monkeypatch):
    """Point the app-data dir at a temp folder."""
    target = tmp_path / "appdata"
    monkeypatch.setenv("RAG_AGENT_APP_DIR", str(target))
    return target


class TestAppDataDir:
    def test_env_override_wins(self, app_dir):
        assert app_paths.app_data_dir() == app_dir
        assert app_dir.is_dir()

    def test_appdata_used_on_windows(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RAG_AGENT_APP_DIR", raising=False)
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert app_paths.app_data_dir() == tmp_path / "rag-agent"

    def test_home_fallback_without_appdata(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RAG_AGENT_APP_DIR", raising=False)
        monkeypatch.setenv("APPDATA", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        assert app_paths.app_data_dir() == tmp_path / ".rag-agent"


class TestSecretFile:
    def test_returns_app_dir_path(self, app_dir):
        assert app_paths.secret_file("tokens.json") == app_dir / "tokens.json"

    def test_migrates_legacy_file(self, app_dir, tmp_path):
        legacy = tmp_path / "data" / "tokens.json"
        legacy.parent.mkdir()
        legacy.write_text('{"k": 1}', encoding="utf-8")
        resolved = app_paths.secret_file("tokens.json", legacy_path=legacy)
        assert resolved == app_dir / "tokens.json"
        assert resolved.read_text(encoding="utf-8") == '{"k": 1}'
        assert not legacy.exists()

    def test_existing_target_not_overwritten_by_legacy(self, app_dir, tmp_path):
        (app_dir / "").mkdir(parents=True, exist_ok=True)
        target = app_dir / "tokens.json"
        target.write_text("new", encoding="utf-8")
        legacy = tmp_path / "old.json"
        legacy.write_text("old", encoding="utf-8")
        resolved = app_paths.secret_file("tokens.json", legacy_path=legacy)
        assert resolved.read_text(encoding="utf-8") == "new"
        assert legacy.exists()


class TestDefaultPathsUseAppDir:
    def test_oauth_tokens_default_path(self, app_dir, monkeypatch):
        monkeypatch.setenv("CLAUDE_OAUTH_TOKENS_PATH", "")
        from providers import claude_oauth

        assert claude_oauth.tokens_path() == app_dir / "claude_oauth_tokens.json"

    def test_oauth_tokens_env_override_still_wins(self, tmp_path, monkeypatch):
        override = tmp_path / "custom.json"
        monkeypatch.setenv("CLAUDE_OAUTH_TOKENS_PATH", str(override))
        from providers import claude_oauth

        assert claude_oauth.tokens_path() == override
