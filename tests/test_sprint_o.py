"""Sprint O tests: speculative generation, map-reduce, dynamic top-N."""
from concurrent.futures import Future
from unittest.mock import MagicMock


from providers.query_enhancements import (
    dynamic_rerank_top_n,
    is_map_reduce_enabled,
    map_reduce_search,
    speculative_prompt_chunks,
    split_sub_questions,
)


class TestMapReduce:
    def test_split_sub_questions(self):
        parts = split_sub_questions("How to wire sensor; and calibrate output")
        assert len(parts) == 2

    def test_map_reduce_search_merges(self, monkeypatch):
        monkeypatch.setenv("MAP_REDUCE_RETRIEVAL_ENABLED", "true")
        assert is_map_reduce_enabled() is True
        embedder = MagicMock()
        embedder.embed.side_effect = lambda texts: [[0.1] for _ in texts]
        store = MagicMock()
        store.search.side_effect = [
            [{"chunk_id": "a", "score": 0.9, "source": "a.pdf"}],
            [{"chunk_id": "b", "score": 0.8, "source": "b.pdf"}],
        ]
        merged = map_reduce_search(
            "part one and part two",
            embedder=embedder,
            store=store,
            top_k=5,
            filter_payload=None,
        )
        assert len(merged) == 2


class TestSpeculative:
    def test_uses_top_one_when_future_pending(self, monkeypatch):
        monkeypatch.setenv("SPECULATIVE_GENERATION_ENABLED", "true")
        chunks = [
            {"chunk_id": "1", "score": 0.9},
            {"chunk_id": "2", "score": 0.5},
        ]
        future = Future()
        selected = speculative_prompt_chunks(chunks, future, default_top_n=3)
        assert len(selected) == 1

    def test_dynamic_rerank_top_n(self, monkeypatch):
        monkeypatch.setenv("DYNAMIC_RERANK_TOP_N_ENABLED", "true")
        monkeypatch.setenv("RERANKER_TOP_N", "5")
        chunks = [{"text": "a" * 100}, {"text": "b" * 100}, {"text": "c" * 100}]
        top_n = dynamic_rerank_top_n(
            chunks,
            prompt_token_budget=60,
            estimate_tokens=lambda chunk: len(chunk["text"]) // 4,
        )
        assert 1 <= top_n <= 5
