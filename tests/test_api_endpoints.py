"""API endpoint tests for Sprint J/K additions and shared routes."""
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

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
