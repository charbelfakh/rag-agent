"""Sprint I tests: RedisVL cache, v2 migration, Langfuse experiments."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from providers.semantic_cache import (
    LegacySemanticCacheBackend,
    SemanticCache,
    cosine_similarity,
    reset_semantic_cache,
)


@pytest.fixture(autouse=True)
def reset_cache():
    reset_semantic_cache()
    yield
    reset_semantic_cache()


class TestLegacySemanticCache:
    def test_lookup_hit(self):
        redis = MagicMock()
        vector = [1.0, 0.0]
        redis.lrange.return_value = [
            json.dumps(
                {
                    "question": "q",
                    "vector": vector,
                    "answer": "cached answer",
                }
            )
        ]
        backend = LegacySemanticCacheBackend(redis, threshold=0.99, max_size=10)
        hit = backend.lookup(vector)
        assert hit == ("cached answer", pytest.approx(1.0))

    def test_semantic_cache_selects_redisvl_backend(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")
        monkeypatch.setenv("SEMANTIC_CACHE_BACKEND", "redisvl")
        mock_backend = MagicMock()
        mock_backend.lookup.return_value = ("answer", 0.95)

        with patch("redis.from_url") as from_url:
            client = MagicMock()
            client.ping.return_value = True
            from_url.return_value = client
            with patch(
                "providers.semantic_cache_redisvl.RedisVLSemanticCacheBackend",
                return_value=mock_backend,
            ) as constructor:
                cache = SemanticCache()
                result = cache.lookup([0.1, 0.2])

        assert result == ("answer", 0.95)
        constructor.assert_called_once()

    def test_redisvl_import_error_falls_back_to_legacy(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")
        monkeypatch.setenv("SEMANTIC_CACHE_BACKEND", "redisvl")

        with patch("redis.from_url") as from_url:
            client = MagicMock()
            client.ping.return_value = True
            client.lrange.return_value = []
            from_url.return_value = client
            with patch.dict(sys.modules, {"redisvl.index": None}):
                with patch(
                    "providers.semantic_cache_redisvl.RedisVLSemanticCacheBackend",
                    side_effect=ImportError("no redisvl"),
                ):
                    cache = SemanticCache()
                    assert isinstance(cache._backend, LegacySemanticCacheBackend)


class TestMigrateToV2:
    def test_point_id_prefers_chunk_id(self):
        from scripts.ops.migrate_to_v2 import _point_id

        payload = {"chunk_id": "abc123", "source": "data/a.pdf", "chunk_index": 1}
        assert _point_id(payload, "legacy-id") == "abc123"

    def test_point_id_builds_from_source_index(self):
        from scripts.ops.migrate_to_v2 import _point_id
        from providers.metadata import make_chunk_id

        payload = {"source": "data/a.pdf", "chunk_index": 2}
        assert _point_id(payload, "legacy-id") == make_chunk_id("data/a.pdf", 2)


class TestLangfuseExperiment:
    def test_sync_dataset_creates_items(self, tmp_path, monkeypatch):
        dataset = tmp_path / "set.jsonl"
        dataset.write_text(
            '{"id":"q1","question":"How to calibrate?","expected_sources":["data/a.pdf"]}\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

        lf = MagicMock()
        with patch("eval.langfuse_experiment.get_langfuse_client", return_value=lf):
            from eval.langfuse_experiment import sync_dataset

            count = sync_dataset(dataset, "rag-retrieval-golden")

        assert count == 1
        lf.create_dataset_item.assert_called_once()

    def test_run_retrieval_experiment_invokes_langfuse(self, tmp_path, monkeypatch):
        dataset = tmp_path / "set.jsonl"
        dataset.write_text(
            '{"id":"q1","question":"How to calibrate?","expected_sources":["data/a.pdf"]}\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

        lf = MagicMock()
        lf.run_experiment.return_value = MagicMock(run_evaluations=[])
        fake_row = {
            "id": "q1",
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
            "retrieved_sources": ["data/a.pdf"],
        }

        with patch("eval.langfuse_experiment.get_langfuse_client", return_value=lf):
            with patch(
                "eval.run_retrieval_eval.retrieve_for_item",
                return_value=fake_row,
            ):
                from eval.langfuse_experiment import run_retrieval_experiment

                result = run_retrieval_experiment(
                    dataset,
                    fetch_k=10,
                    rerank_top_n=5,
                    sync_items=False,
                )

        assert result["item_count"] == 1
        lf.run_experiment.assert_called_once()
