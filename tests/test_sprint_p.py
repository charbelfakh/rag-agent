"""Sprint P tests: production sampling, sessions API, markdown citations."""
import json
from unittest.mock import MagicMock

import pytest

from eval.production_sampler import record_sample, should_sample
from providers.session_store import SessionStore, reset_session_store
from providers.rag_pipeline import collect_citations


@pytest.fixture(autouse=True)
def reset_sessions():
    reset_session_store()
    yield
    reset_session_store()


class TestProductionSampler:
    def test_record_sample_writes_jsonl(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PRODUCTION_EVAL_SAMPLING_ENABLED", "true")
        monkeypatch.setenv("PRODUCTION_EVAL_SAMPLE_RATE", "1.0")
        path = tmp_path / "samples.jsonl"
        monkeypatch.setenv("PRODUCTION_EVAL_SAMPLE_PATH", str(path))
        record_sample(question="How to wire?", answer_preview="Use terminal 3.")
        assert path.exists()
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert "question_hash" in row

    def test_should_sample_disabled(self, monkeypatch):
        monkeypatch.setenv("PRODUCTION_EVAL_SAMPLING_ENABLED", "false")
        assert should_sample("anything") is False


class TestSessionStore:
    def test_create_and_append(self, tmp_path):
        store = SessionStore(db_path=str(tmp_path / "sessions.db"))
        session = store.create_session(title="Test")
        store.append_message(session["session_id"], role="user", content="Hi")
        messages = store.get_messages(session["session_id"])
        assert len(messages) == 1
        assert messages[0]["content"] == "Hi"


class TestMultimodalCitations:
    def test_collect_citations_includes_media(self):
        chunks = [
            {
                "source": "docs/a.pdf",
                "page": 1,
                "section": "Wiring",
                "content_type": "image",
                "media_uri": "/media/abc/1_0.png",
                "ocr_text": "PWR +24V",
                "image_class": "wiring",
            }
        ]
        cites = collect_citations(chunks)
        assert cites[0]["media_uri"] == "/media/abc/1_0.png"
        assert cites[0]["ocr_preview"] == "PWR +24V"


class TestSessionsAPI:
    def test_sessions_crud(self, api_client, monkeypatch, tmp_path):
        monkeypatch.setattr("api.main.API_KEY", None)
        monkeypatch.setenv("SESSION_STORAGE_ENABLED", "true")
        monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
        reset_session_store()

        created = api_client.post("/sessions", json={"title": "Chat 1"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        api_client.post(
            f"/sessions/{session_id}/messages",
            json={"role": "user", "content": "Hello"},
        )
        loaded = api_client.get(f"/sessions/{session_id}")
        assert loaded.status_code == 200
        assert len(loaded.json()["messages"]) == 1
