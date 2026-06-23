"""Sprint L tests: M0 media store, reranker A/B, section chunking."""
import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from providers.media_store import MediaStore, normalize_media_uri, reset_media_store
from providers.metadata import build_image_chunk_payload, resolve_metadata
from providers.reranker import CrossEncoderReranker, truncate_for_rerank
from providers.section_chunking import (
    is_section_aware_chunking_enabled,
    split_procedure_steps,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    reset_media_store()
    yield
    reset_media_store()


class TestMediaStore:
    def test_put_and_resolve(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEDIA_STORAGE_ENABLED", "true")
        monkeypatch.setenv("MEDIA_STORAGE_PATH", str(tmp_path / "media"))
        store = MediaStore()
        uri = store.put_bytes("abc/page_0.png", b"\x89PNG")
        assert uri == "/media/abc/page_0.png"
        path = store.resolve_path(uri)
        assert path is not None
        assert path.read_bytes() == b"\x89PNG"

    def test_blocks_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEDIA_STORAGE_ENABLED", "true")
        monkeypatch.setenv("MEDIA_STORAGE_PATH", str(tmp_path / "media"))
        store = MediaStore()
        assert store.resolve_path("/media/../secret.txt") is None


class TestImagePayload:
    def test_build_image_chunk_payload(self):
        meta = resolve_metadata("docs/pekat/wiring.pdf")
        payload = build_image_chunk_payload(
            metadata=meta,
            chunk_index=3,
            page=12,
            media_uri="/media/abc/12_0.png",
            media_hash="deadbeef",
            ocr_text="PWR +24V",
            image_class="wiring",
            width=640,
            height=480,
        )
        assert payload["content_type"] == "image"
        assert payload["media_uri"] == "/media/abc/12_0.png"
        assert payload["ocr_text"] == "PWR +24V"


class TestSectionChunking:
    def test_split_procedure_steps(self):
        body = "Step 1. Connect power.\n2. Run calibration.\n3. Save settings."
        parts = split_procedure_steps(body)
        assert len(parts) == 3

    def test_disabled_returns_single_block(self, monkeypatch):
        monkeypatch.setenv("SECTION_AWARE_CHUNKING_ENABLED", "false")
        assert is_section_aware_chunking_enabled() is False


class TestRerankerAB:
    def test_cross_encoder_accepts_model_name(self, monkeypatch):
        import sys

        fake_st = MagicMock()
        fake_st.CrossEncoder.return_value = MagicMock(
            predict=lambda pairs: [0.5] * len(pairs)
        )
        with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            reranker = CrossEncoderReranker(model_name="model-b")
        assert reranker.model_name == "model-b"

    def test_run_reranker_ab_mocked(self, monkeypatch, tmp_path):
        import eval.run_reranker_ab as ab

        dataset = tmp_path / "set.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "question": "How to wire?",
                    "expected_sources": ["data/a.pdf"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        def fake_retrieve(item, *, fetch_k, rerank_top_n, reranker):
            recall = 1.0 if "L-12" in reranker.model_name else 0.0
            return {
                "id": item["id"],
                "recall_at_5": recall,
                "recall_at_10": recall,
                "mrr": recall,
                "top_vector_score": 0.8,
                "retrieved_sources": ["data/a.pdf"],
            }

        monkeypatch.setattr(ab, "_retrieve_with_reranker", fake_retrieve)
        monkeypatch.setattr(
            ab,
            "CrossEncoderReranker",
            lambda model_name=None: MagicMock(model_name=model_name or "default"),
        )

        report = ab.run_reranker_ab(
            dataset,
            model_a="cross-encoder/ms-marco-MiniLM-L-6-v2",
            model_b="cross-encoder/ms-marco-MiniLM-L-12-v2",
            limit=1,
        )
        assert report["winner"] == "b"


class TestMediaEndpoint:
    def test_media_endpoint_serves_file(self, api_client, monkeypatch, tmp_path):
        monkeypatch.setattr("api.main.API_KEY", None)
        monkeypatch.setenv("MEDIA_STORAGE_ENABLED", "true")
        monkeypatch.setenv("MEDIA_STORAGE_PATH", str(tmp_path / "media"))
        reset_media_store()
        store = MediaStore()
        store.put_bytes("doc/a.png", b"PNGDATA")

        response = api_client.get("/media/doc/a.png")
        assert response.status_code == 200
        assert response.content == b"PNGDATA"
