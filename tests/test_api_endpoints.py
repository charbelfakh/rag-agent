"""API endpoint tests for Sprint J/K additions and shared routes."""
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import app


@pytest.fixture(autouse=True)
def disable_api_key(monkeypatch):
    """api.main reads API_KEY at import time from .env."""
    monkeypatch.setattr("api.main.API_KEY", None)


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_deps_all_up(self, client, monkeypatch):
        monkeypatch.setattr("api.main._check_qdrant", lambda: (True, "http://localhost:6333"))
        monkeypatch.setattr("api.main._check_redis", lambda: (True, "redis://localhost:6379"))
        monkeypatch.setattr("api.main._ollama_probe", lambda: (True, "Ollama 0.9", ["mistral"]))
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        response = client.get("/health/deps")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["qdrant"]["ok"] is True
        assert body["llm"]["provider"] == "ollama"

    def test_health_deps_qdrant_down_is_503(self, client, monkeypatch):
        monkeypatch.setattr("api.main._check_qdrant", lambda: (False, "unreachable"))
        monkeypatch.setattr("api.main._check_redis", lambda: (True, "redis://localhost:6379"))
        monkeypatch.setattr("api.main._ollama_probe", lambda: (True, "Ollama 0.9", []))
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        response = client.get("/health/deps")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_health_deps_redis_down_degrades_but_200(self, client, monkeypatch):
        monkeypatch.setattr("api.main._check_qdrant", lambda: (True, "http://localhost:6333"))
        monkeypatch.setattr("api.main._check_redis", lambda: (False, "unreachable"))
        monkeypatch.setattr("api.main._ollama_probe", lambda: (True, "Ollama 0.9", []))
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        response = client.get("/health/deps")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["redis"]["ok"] is False

    def test_health_deps_hosted_llm_not_probed(self, client, monkeypatch):
        monkeypatch.setattr("api.main._check_qdrant", lambda: (True, "http://localhost:6333"))
        monkeypatch.setattr("api.main._check_redis", lambda: (True, "redis://localhost:6379"))
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")

        response = client.get("/health/deps")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["llm"]["provider"] == "anthropic"
        assert "not probed" in body["llm"]["detail"]


class TestUiEndpoints:
    def test_index_served_with_no_cache_headers(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "renderCitationPlayers" in response.text
        assert response.headers.get("cache-control") == "no-cache, no-store, must-revalidate"

    def test_ui_version_endpoint(self, client):
        response = client.get("/ui-version")
        assert response.status_code == 200
        assert response.json()["ui_build"] == "2026-06-17-youtube"

    def test_health_embed(self, client, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        monkeypatch.setenv("EMBED_PROVIDER", "tei")
        monkeypatch.setattr("api.main.get_embedder", lambda: embedder)

        response = client.get("/health/embed")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["dimensions"] == 3
        assert body["provider"] == "tei"

    def test_health_embed_unavailable(self, client, monkeypatch):
        def fail():
            raise ConnectionError("embed down")

        monkeypatch.setattr("api.main.get_embedder", fail)
        response = client.get("/health/embed")
        assert response.status_code == 503

    def test_health_queue(self, client, monkeypatch):
        monkeypatch.setenv("INGEST_QUEUE_ENABLED", "true")
        queue = MagicMock()
        queue.available = True
        queue.queue_depth.return_value = 2
        monkeypatch.setattr("api.main.get_ingest_queue", lambda: queue)
        monkeypatch.setattr("api.main.is_ingest_queue_enabled", lambda: True)

        response = client.get("/health/queue")
        assert response.status_code == 200
        body = response.json()
        assert body["ingest_queue_enabled"] is True
        assert body["available"] is True
        assert body["depth"] == 2


class TestFeedbackEndpoint:
    def test_feedback_accepts_rating(self, client, monkeypatch):
        captured: list[dict] = []

        def fake_log(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr("api.main.log_feedback", fake_log)

        response = client.post(
            "/feedback",
            json={
                "question": "How to calibrate?",
                "answer": "Use the wizard.",
                "rating": 1,
                "trace_id": "trace-abc",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert captured[0]["rating"] == 1
        assert captured[0]["trace_id"] == "trace-abc"

    def test_feedback_rejects_invalid_rating(self, client):
        response = client.post(
            "/feedback",
            json={"question": "q", "answer": "a", "rating": 0},
        )
        assert response.status_code == 400


class TestUploadQueueMode:
    def test_upload_enqueues_when_queue_enabled(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("api.main.ROOT", tmp_path)
        monkeypatch.setattr("api.main.DATA_DIR", tmp_path / "data")

        queue = MagicMock()
        queue.available = True
        queue.enqueue.return_value = True
        job_store = MagicMock()
        job_store.get.return_value = None

        monkeypatch.setattr("api.main.is_ingest_queue_enabled", lambda: True)
        monkeypatch.setattr("api.main.get_ingest_queue", lambda: queue)
        monkeypatch.setattr("api.main._job_store", lambda: job_store)

        response = client.post(
            "/upload",
            files={"file": ("note.txt", BytesIO(b"hello world"), "text/plain")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["queued"] is True
        queue.enqueue.assert_called_once()
        job_store.create.assert_called_once()

    def test_upload_runs_inline_when_queue_disabled(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("api.main.ROOT", tmp_path)
        monkeypatch.setattr("api.main.DATA_DIR", tmp_path / "data")
        monkeypatch.setattr("api.main.is_ingest_queue_enabled", lambda: False)

        queue = MagicMock()
        queue.available = False
        job_store = MagicMock()
        monkeypatch.setattr("api.main.get_ingest_queue", lambda: queue)
        monkeypatch.setattr("api.main._job_store", lambda: job_store)

        created_tasks: list = []

        def fake_create_task(coro):
            created_tasks.append(coro)
            coro.close()
            return MagicMock()

        monkeypatch.setattr("api.main.asyncio.create_task", fake_create_task)

        response = client.post(
            "/upload",
            files={"file": ("note.txt", BytesIO(b"hello world"), "text/plain")},
        )
        assert response.status_code == 200
        assert response.json()["queued"] is False
        assert len(created_tasks) == 1


class TestDocumentsEndpoint:
    def test_documents_uses_registry_when_populated(self, client, monkeypatch):
        registry = MagicMock()
        registry.list_documents.return_value = [
            {"source": "data/a.pdf", "vendor": "pekat", "chunks": 3}
        ]
        store = MagicMock()
        monkeypatch.setattr("api.main.get_doc_registry", lambda: registry)
        monkeypatch.setattr("api.main.get_vector_store", lambda: store)

        response = client.get("/documents")
        assert response.status_code == 200
        docs = response.json()["documents"]
        assert len(docs) == 1
        store.list_sources.assert_not_called()

    def test_documents_falls_back_to_qdrant(self, client, monkeypatch):
        registry = MagicMock()
        registry.list_documents.return_value = []
        store = MagicMock()
        store.list_sources.return_value = [{"source": "data/b.pdf", "chunks": 1}]
        monkeypatch.setattr("api.main.get_doc_registry", lambda: registry)
        monkeypatch.setattr("api.main.get_vector_store", lambda: store)

        response = client.get("/documents")
        assert response.status_code == 200
        assert len(response.json()["documents"]) == 1
        store.list_sources.assert_called_once()


class TestCaptionImageEndpoint:
    _PNG_BYTES = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDAT\x08\xd79c\xf8\xcf\xc0"
        b"\x00\x00\x00\x02\x00\x01\xe5'\xde\xfc"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    @pytest.fixture
    def caption_image_root(self, tmp_path, monkeypatch):
        img_dir = tmp_path / "data" / "pekat" / "images" / "_caption_cache"
        img_dir.mkdir(parents=True)
        image_path = img_dir / "sample.img"
        image_path.write_bytes(self._PNG_BYTES)
        monkeypatch.setattr("api.main.ROOT", tmp_path)
        return image_path

    def test_serves_valid_image_with_content_type(self, client, caption_image_root):
        rel = "data/pekat/images/_caption_cache/sample.img"
        response = client.get("/image", params={"path": rel})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/png")
        assert response.content == self._PNG_BYTES

    def test_rejects_path_traversal(self, client, caption_image_root):
        response = client.get(
            "/image",
            params={"path": "data/pekat/images/_caption_cache/../../secrets.img"},
        )
        assert response.status_code == 400

    def test_rejects_path_outside_allowed_base(self, client, caption_image_root):
        response = client.get("/image", params={"path": "data/pekat/docs/secret.png"})
        assert response.status_code == 400

    def test_missing_file_returns_404(self, client, caption_image_root):
        response = client.get(
            "/image",
            params={"path": "data/pekat/images/_caption_cache/missing.img"},
        )
        assert response.status_code == 404


class TestVideoEndpoint:
    _VIDEO_BYTES = bytes(range(256)) * 2  # 512 bytes

    @pytest.fixture
    def video_root(self, tmp_path, monkeypatch):
        video_dir = tmp_path / "data" / "pekat" / "videos" / "be4a2d03a9cc29b0e37e"
        video_dir.mkdir(parents=True)
        video_path = video_dir / "source.mp4"
        video_path.write_bytes(self._VIDEO_BYTES)
        monkeypatch.setattr("api.main.ROOT", tmp_path)
        return video_path

    @property
    def rel_path(self):
        return "data/pekat/videos/be4a2d03a9cc29b0e37e/source.mp4"

    def test_full_get_returns_200_with_accept_ranges(self, client, video_root):
        response = client.get("/video", params={"path": self.rel_path})
        assert response.status_code == 200
        assert response.headers.get("accept-ranges") == "bytes"
        assert len(response.content) == len(self._VIDEO_BYTES)
        assert int(response.headers["content-length"]) == len(self._VIDEO_BYTES)
        assert response.headers["content-type"].startswith("video/mp4")

    def test_range_get_bytes_0_99(self, client, video_root):
        response = client.get(
            "/video",
            params={"path": self.rel_path},
            headers={"Range": "bytes=0-99"},
        )
        assert response.status_code == 206
        assert response.headers.get("content-range") == "bytes 0-99/512"
        assert response.content == self._VIDEO_BYTES[0:100]
        assert response.headers["content-length"] == "100"

    def test_range_get_open_ended(self, client, video_root):
        response = client.get(
            "/video",
            params={"path": self.rel_path},
            headers={"Range": "bytes=100-"},
        )
        assert response.status_code == 206
        assert response.headers.get("content-range") == "bytes 100-511/512"
        assert response.content == self._VIDEO_BYTES[100:]

    def test_range_get_suffix(self, client, video_root):
        response = client.get(
            "/video",
            params={"path": self.rel_path},
            headers={"Range": "bytes=-50"},
        )
        assert response.status_code == 206
        assert response.headers.get("content-range") == "bytes 462-511/512"
        assert response.content == self._VIDEO_BYTES[-50:]

    def test_unsatisfiable_range_returns_416(self, client, video_root):
        response = client.get(
            "/video",
            params={"path": self.rel_path},
            headers={"Range": "bytes=1000-"},
        )
        assert response.status_code == 416
        assert response.headers.get("content-range") == "bytes */512"

    def test_rejects_path_traversal(self, client, video_root):
        response = client.get(
            "/video",
            params={"path": "data/pekat/videos/be4a2d03a9cc29b0e37e/../../.env"},
        )
        assert response.status_code == 400
        assert b"API_KEY" not in response.content

    def test_rejects_path_outside_videos_subtree(self, client, video_root):
        response = client.get(
            "/video",
            params={"path": "data/pekat/images/secret.mp4"},
        )
        assert response.status_code == 400

    def test_missing_file_returns_404(self, client, video_root):
        response = client.get(
            "/video",
            params={"path": "data/pekat/videos/be4a2d03a9cc29b0e37e/missing.mp4"},
        )
        assert response.status_code == 404


class TestTranscribeEndpoint:
    def test_transcribe_returns_glossary_corrected_text(self, client, monkeypatch):
        captured: list[str] = []

        def fake_transcribe_plain_text(path: str) -> str:
            captured.append(path)
            return "What is Peacot Vision?"

        monkeypatch.setattr("api.main.transcribe_plain_text", fake_transcribe_plain_text)

        response = client.post(
            "/transcribe",
            files={"file": ("clip.webm", BytesIO(b"fake-audio-bytes"), "audio/webm")},
        )
        assert response.status_code == 200
        assert response.json() == {"text": "What is PEKAT Vision?"}
        assert captured
        assert captured[0].endswith(".webm")

    def test_transcribe_empty_upload_returns_400(self, client):
        response = client.post(
            "/transcribe",
            files={"file": ("clip.webm", BytesIO(b""), "audio/webm")},
        )
        assert response.status_code == 400
        assert "Empty" in response.json()["detail"]

    def test_transcribe_no_speech_returns_400(self, client, monkeypatch):
        monkeypatch.setattr("api.main.transcribe_plain_text", lambda _path: "  ")

        response = client.post(
            "/transcribe",
            files={"file": ("clip.webm", BytesIO(b"audio"), "audio/webm")},
        )
        assert response.status_code == 400
        assert "No speech" in response.json()["detail"]

    def test_transcribe_unsupported_format_returns_400(self, client):
        response = client.post(
            "/transcribe",
            files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_transcribe_whisper_error_returns_400(self, client, monkeypatch):
        def boom(_path: str) -> str:
            raise RuntimeError("decode failed")

        monkeypatch.setattr("api.main.transcribe_plain_text", boom)

        response = client.post(
            "/transcribe",
            files={"file": ("clip.ogg", BytesIO(b"audio"), "audio/ogg")},
        )
        assert response.status_code == 400
        assert "Could not transcribe" in response.json()["detail"]


class TestLlmProviderEndpoints:
    @pytest.fixture(autouse=True)
    def _pin_provider_env(self, monkeypatch, tmp_path):
        """Register LLM env keys with monkeypatch so endpoint mutations are undone,
        stub the Ollama probe, and isolate the OAuth token store (no live state)."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_LLM_MODEL", "mistral:7b-instruct-q4_K_M")
        monkeypatch.setenv("CLAUDE_SUBSCRIPTION_MODEL", "")
        monkeypatch.setenv("CLAUDE_OAUTH_TOKENS_PATH", str(tmp_path / "tokens.json"))
        monkeypatch.setenv("LLM_PROVIDER_STATE_PATH", str(tmp_path / "llm_provider.json"))
        monkeypatch.setenv("LLM_API_KEYS_PATH", str(tmp_path / "llm_api_keys.json"))
        monkeypatch.setattr(
            "api.main._ollama_probe",
            lambda: (True, "Ollama test at http://localhost:11434", ["mistral:7b"]),
        )
        yield
        from providers.factory import reset_providers

        reset_providers()

    def test_llm_status_reports_provider_connection_and_models(self, client, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        response = client.get("/llm/status")
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "ollama"
        assert data["model"] == "mistral:7b-instruct-q4_K_M"
        assert data["connection"] == {
            "connected": True,
            "detail": "Ollama test at http://localhost:11434",
        }
        assert data["providers"]["ollama"]["models"] == ["mistral:7b"]
        anthropic = data["providers"]["anthropic"]
        assert anthropic["api_key_set"] is False
        assert anthropic["connected"] is False
        assert anthropic["models"]
        claude = data["providers"]["claude_subscription"]
        assert claude["signed_in"] is False
        assert claude["connected"] is False
        assert "Not signed in" in claude["detail"]
        assert claude["models"]

    def test_llm_status_disconnected_when_ollama_down(self, client, monkeypatch):
        monkeypatch.setattr(
            "api.main._ollama_probe", lambda: (False, "Ollama not reachable", [])
        )
        data = client.get("/llm/status").json()
        assert data["connection"]["connected"] is False
        assert "not reachable" in data["connection"]["detail"]

    def test_disconnect_claude_subscription_keeps_tokens(self, client):
        from providers import claude_oauth

        claude_oauth._store_tokens({"access_token": "tok", "expires_in": 3600})
        assert claude_oauth.is_signed_in() is True

        response = client.post(
            "/llm/disconnect", json={"provider": "claude_subscription"}
        )

        assert response.status_code == 200
        # ForgeStation-style: token retained so reconnect needs no re-sign-in.
        assert claude_oauth.is_signed_in() is True
        assert "reconnect" in response.json()["disconnect_detail"].lower()

    def test_disconnect_active_provider_reverts_to_ollama(self, client, monkeypatch):
        import os
        from providers import claude_oauth

        # Subscription is signed in AND the active provider.
        claude_oauth._store_tokens({"access_token": "tok", "expires_in": 3600})
        monkeypatch.setenv("LLM_PROVIDER", "claude_subscription")

        response = client.post(
            "/llm/disconnect", json={"provider": "claude_subscription"}
        )

        assert response.status_code == 200
        # Model is deactivated (falls back to local) but the token is retained.
        assert os.environ["LLM_PROVIDER"] == "ollama"
        assert response.json()["provider"] == "ollama"
        assert claude_oauth.is_signed_in() is True

    def test_disconnect_inactive_provider_keeps_active(self, client, monkeypatch):
        import os

        # Active provider is ollama; disconnecting a different one must not move it.
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        response = client.post("/llm/disconnect", json={"provider": "openai"})

        assert response.status_code == 200
        assert os.environ["LLM_PROVIDER"] == "ollama"

    def test_disconnect_claude_cli_moves_credentials(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        creds = tmp_path / ".credentials.json"
        creds.write_text('{"claudeAiOauth": {"accessToken": "tok"}}', encoding="utf-8")

        response = client.post("/llm/disconnect", json={"provider": "claude_cli"})

        assert response.status_code == 200
        assert not creds.exists()
        assert (tmp_path / ".credentials.json.bak").exists()
        assert "Signed out" in response.json()["disconnect_detail"]

    def test_disconnect_api_provider_keeps_saved_key(self, client, monkeypatch):
        import os

        # ForgeStation-style: the saved key is retained across a disconnect.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        response = client.post("/llm/disconnect", json={"provider": "openai"})
        assert response.status_code == 200
        assert os.environ["OPENAI_API_KEY"] == "sk-test"
        assert "reconnect" in response.json()["disconnect_detail"].lower()

    def test_disconnect_unknown_provider_rejected(self, client):
        response = client.post("/llm/disconnect", json={"provider": "skynet"})
        assert response.status_code == 400

    def test_switch_provider_updates_env_and_resets_factory(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        reset = MagicMock()
        fake_llm = MagicMock()
        monkeypatch.setattr("api.main.provider_factory.reset_providers", reset)
        monkeypatch.setattr("api.main.provider_factory.get_llm", MagicMock(return_value=fake_llm))

        response = client.post(
            "/llm/provider", json={"provider": "openai", "model": "gpt-4o-mini"}
        )

        assert response.status_code == 200
        assert response.json()["provider"] == "openai"
        import os

        assert os.environ["LLM_PROVIDER"] == "openai"
        assert os.environ["OPENAI_MODEL"] == "gpt-4o-mini"
        reset.assert_called_once()

    def test_switch_unknown_provider_rejected(self, client):
        response = client.post("/llm/provider", json={"provider": "skynet"})
        assert response.status_code == 400
        assert "Unknown provider" in response.json()["detail"]

    def test_failed_switch_reverts_previous_provider(self, client, monkeypatch):
        import os

        monkeypatch.setattr("api.main.provider_factory.reset_providers", MagicMock())
        monkeypatch.setattr(
            "api.main.provider_factory.get_llm",
            MagicMock(side_effect=RuntimeError("claude CLI not found")),
        )

        response = client.post("/llm/provider", json={"provider": "claude_cli"})

        assert response.status_code == 400
        assert "claude CLI not found" in response.json()["detail"]
        assert os.environ["LLM_PROVIDER"] == "ollama"

    def test_switch_provider_persists_preference(self, client, monkeypatch):
        import json
        import api.main as api_main

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("api.main.provider_factory.reset_providers", MagicMock())
        monkeypatch.setattr(
            "api.main.provider_factory.get_llm", MagicMock(return_value=MagicMock())
        )

        response = client.post(
            "/llm/provider", json={"provider": "openai", "model": "gpt-4o"}
        )

        assert response.status_code == 200
        saved = json.loads(api_main._llm_pref_path().read_text(encoding="utf-8"))
        assert saved == {"provider": "openai", "model": "gpt-4o"}

    def test_disconnect_clears_persisted_preference(self, client):
        import api.main as api_main

        api_main._save_llm_pref("openai", "gpt-4o")
        assert api_main._llm_pref_path().exists()

        response = client.post("/llm/disconnect", json={"provider": "openai"})

        assert response.status_code == 200
        assert not api_main._llm_pref_path().exists()

    def test_persisted_subscription_restored_on_startup(self, monkeypatch):
        import os
        import api.main as api_main
        from providers import claude_oauth

        claude_oauth._store_tokens({"access_token": "tok", "expires_in": 3600})
        api_main._save_llm_pref("claude_subscription", "claude-opus-4-8")
        monkeypatch.setattr("api.main.provider_factory.reset_providers", MagicMock())

        api_main._apply_persisted_llm_provider()

        assert os.environ["LLM_PROVIDER"] == "claude_subscription"
        assert os.environ["CLAUDE_SUBSCRIPTION_MODEL"] == "claude-opus-4-8"

    def test_persisted_subscription_skipped_when_not_signed_in(self, monkeypatch):
        import os
        import api.main as api_main

        # Isolated token store is empty -> not signed in, so it must not restore.
        api_main._save_llm_pref("claude_subscription", "claude-opus-4-8")
        monkeypatch.setattr("api.main.provider_factory.reset_providers", MagicMock())

        api_main._apply_persisted_llm_provider()

        assert os.environ["LLM_PROVIDER"] == "ollama"

    def test_persisted_api_provider_skipped_without_key(self, monkeypatch):
        import os
        import api.main as api_main

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        api_main._save_llm_pref("openai", "gpt-4o")
        monkeypatch.setattr("api.main.provider_factory.reset_providers", MagicMock())

        api_main._apply_persisted_llm_provider()

        assert os.environ["LLM_PROVIDER"] == "ollama"

    def test_switch_provider_persists_api_key(self, client, monkeypatch):
        import os
        import api.main as api_main

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("api.main.provider_factory.reset_providers", MagicMock())
        monkeypatch.setattr(
            "api.main.provider_factory.get_llm", MagicMock(return_value=MagicMock())
        )

        response = client.post(
            "/llm/provider",
            json={"provider": "openai", "model": "gpt-4o", "api_key": "sk-live"},
        )

        assert response.status_code == 200
        # Applied to the environment now, and persisted for next run.
        assert os.environ["OPENAI_API_KEY"] == "sk-live"
        assert api_main._load_api_keys()["openai"] == "sk-live"
        # The key itself is never echoed back to the client.
        assert "sk-live" not in response.text

    def test_switch_provider_reverts_api_key_on_failure(self, client, monkeypatch):
        import os
        import api.main as api_main

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("api.main.provider_factory.reset_providers", MagicMock())
        monkeypatch.setattr(
            "api.main.provider_factory.get_llm",
            MagicMock(side_effect=RuntimeError("bad key")),
        )

        response = client.post(
            "/llm/provider", json={"provider": "openai", "api_key": "sk-bad"}
        )

        assert response.status_code == 400
        assert "OPENAI_API_KEY" not in os.environ
        assert "openai" not in api_main._load_api_keys()

    def test_switch_api_provider_without_key_rejected(self, client, monkeypatch):
        import os

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("api.main.provider_factory.reset_providers", MagicMock())

        response = client.post("/llm/provider", json={"provider": "openai"})

        assert response.status_code == 400
        assert "API key is required" in response.json()["detail"]
        # The active provider is left untouched.
        assert os.environ["LLM_PROVIDER"] == "ollama"

    def test_persisted_api_keys_loaded_on_startup(self, monkeypatch):
        import os
        import api.main as api_main

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        api_main._save_api_key("openai", "sk-saved")

        api_main._apply_persisted_api_keys()

        assert os.environ["OPENAI_API_KEY"] == "sk-saved"

    def test_status_reports_saved_key_as_connected(self, client, monkeypatch):
        import api.main as api_main

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        api_main._save_api_key("openai", "sk-saved")
        api_main._apply_persisted_api_keys()

        data = client.get("/llm/status").json()
        assert data["providers"]["openai"]["api_key_set"] is True

    def test_oauth_start_short_circuits_when_signed_in(self, client, monkeypatch):
        monkeypatch.setattr("api.main.claude_oauth.is_signed_in", lambda: True)
        response = client.post("/llm/claude/oauth/start")
        assert response.status_code == 200
        assert response.json() == {"signed_in": True}

    def test_oauth_start_returns_authorize_url(self, client):
        from providers import claude_oauth as oauth_module

        try:
            response = client.post("/llm/claude/oauth/start")
        finally:
            oauth_module.reset_pending()

        assert response.status_code == 200
        data = response.json()
        assert data["signed_in"] is False
        assert data["url"].startswith("https://claude.ai/oauth/authorize?")
        assert "code_challenge=" in data["url"]

    def test_oauth_finish_returns_result(self, client, monkeypatch):
        monkeypatch.setattr(
            "api.main.claude_oauth.finish_login",
            lambda code: {"signed_in": True, "subscription_type": "max"},
        )
        response = client.post("/llm/claude/oauth/finish", json={"code": "abc#state"})
        assert response.status_code == 200
        assert response.json() == {"signed_in": True, "subscription_type": "max"}

    def test_oauth_finish_maps_flow_errors_to_400(self, client, monkeypatch):
        from providers.claude_oauth import ClaudeOAuthError

        def _raise(code):
            raise ClaudeOAuthError("State mismatch — restart the sign-in and try again.")

        monkeypatch.setattr("api.main.claude_oauth.finish_login", _raise)
        response = client.post("/llm/claude/oauth/finish", json={"code": "bad"})
        assert response.status_code == 400
        assert "State mismatch" in response.json()["detail"]

    def test_ui_contains_llm_provider_controls(self, client):
        response = client.get("/")
        assert 'id="llmProviderSelect"' in response.text
        assert 'id="llmConnectBtn"' in response.text
        # Model selection lives in the chat window, not Settings.
        assert 'id="chatModelSelect"' in response.text
        assert 'id="llmOauthCodeInput"' in response.text
        assert 'id="llmOauthFinishBtn"' in response.text
