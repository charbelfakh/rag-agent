"""Cost-optimization increment 1: fast-tier LLM routing + batched dense search.

Covers:
- ``factory.get_fast_llm`` model/token selection (anthropic Haiku vs. main llm).
- ``AnthropicLLM`` per-instance model/max_tokens overrides in the payload.
- HyDE and sufficiency routing through the fast tier.
- ``QdrantLocalStore.search_batch`` + the orchestrator's ``_batched_dense_search``
  round-trip consolidation (with sequential fallback).
"""
import json
from unittest.mock import MagicMock, patch

import pytest

import providers.factory as factory
import providers.rag_pipeline as rag_pipeline
from conftest import patch_retrieval_pipeline
from providers.anthropic_llm import AnthropicLLM
from providers.query_orchestrator import (
    QueryOrchestrator,
    RetrievalContext,
    _batched_dense_search,
)
from providers.rag_pipeline import _build_generation_plan


@pytest.fixture
def reset_factory():
    factory.reset_providers()
    yield
    factory.reset_providers()


class TestFastTier:
    def test_anthropic_provider_uses_haiku_with_small_cap(self, monkeypatch, reset_factory):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("LLM_FAST_MODEL", raising=False)
        monkeypatch.delenv("LLM_FAST_MAX_TOKENS", raising=False)

        fast = factory.get_fast_llm()

        assert isinstance(fast, AnthropicLLM)
        assert fast.model == "claude-haiku-4-5"
        assert fast.max_tokens == 512

    def test_fast_tier_env_overrides(self, monkeypatch, reset_factory):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_FAST_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("LLM_FAST_MAX_TOKENS", "128")

        fast = factory.get_fast_llm()

        assert fast.model == "claude-sonnet-5"
        assert fast.max_tokens == 128

    def test_non_anthropic_reuses_main_llm(self, monkeypatch, reset_factory):
        # Ollama (and every non-anthropic provider) keeps a single model — the
        # fast tier is the same object as the main llm, so those paths never change.
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        assert factory.get_fast_llm() is factory.get_llm()


class TestAnthropicPayloadOverrides:
    def test_constructor_overrides_reach_payload(self):
        llm = AnthropicLLM(model="claude-haiku-4-5", max_tokens=333)
        payload = llm._payload("hi", stream=False)

        assert payload["model"] == "claude-haiku-4-5"
        assert payload["max_tokens"] == 333

    def test_defaults_from_env_when_unset(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "2048")

        llm = AnthropicLLM()

        assert llm.model == "claude-sonnet-5"
        assert llm.max_tokens == 2048


class TestHydeSufficiencyTiering:
    def test_hyde_runs_on_fast_tier(self, monkeypatch):
        monkeypatch.setenv("HYDE_ENABLED", "true")
        monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
        monkeypatch.setenv("QUERY_ORCHESTRATOR_ENABLED", "false")
        monkeypatch.setenv("RERANKER_ENABLED", "false")

        fast_llm = MagicMock()
        fast_llm.generate.return_value = "hypothetical document"
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2]]
        store = MagicMock()
        store.search.return_value = [{"source": "m.pdf", "content_type": "text", "score": 0.9}]
        cache = MagicMock()
        cache.enabled = False
        analytics = MagicMock()

        with patch("providers.query_orchestrator.get_fast_llm", lambda: fast_llm), \
             patch("providers.query_orchestrator.get_embedder", lambda: embedder), \
             patch("providers.query_orchestrator.get_vector_store", lambda: store), \
             patch("providers.query_orchestrator.get_semantic_cache", lambda: cache):
            result = QueryOrchestrator().run_retrieval(
                RetrievalContext(
                    question="How do I configure exposure on the Zivid camera?",
                    top_k=5,
                    history_turns=0,
                    vendor_filter=None,
                    document_type_filter=None,
                    analytics=analytics,
                )
            )

        fast_llm.generate.assert_called_once()  # HyDE used the fast tier
        assert result.hyde_used is True
        assert result.search_text == "hypothetical document"

    def test_sufficiency_runs_on_fast_tier(self, monkeypatch):
        monkeypatch.setenv("SUFFICIENCY_CHECK_ENABLED", "true")
        monkeypatch.setenv("HYDE_ENABLED", "false")

        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = [
            {"text": "excerpt", "source": "m.pdf", "content_type": "text", "score": 0.9}
        ]
        main_llm = MagicMock()
        main_llm.generate.return_value = "answer"
        fast_llm = MagicMock()
        fast_llm.generate.return_value = "YES"
        cache = MagicMock()
        cache.enabled = False

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=main_llm,
            cache=cache,
            reranker=MagicMock(),
        )
        # Override the fast tier to a distinct object so we can prove the
        # sufficiency call targets it (not the main synthesis llm).
        monkeypatch.setattr(rag_pipeline, "get_fast_llm", lambda: fast_llm)
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        _build_generation_plan("What is Pekat?", top_k=5)

        fast_llm.generate.assert_called_once()  # sufficiency check ran on the fast tier
        assert not main_llm.generate.called  # main model untouched (definitional → no HyDE/condense)


class TestBatchedDenseSearch:
    def test_uses_search_batch_when_store_supports_it(self):
        class BatchStore:
            def __init__(self):
                self.batch_calls = 0
                self.search_calls = 0

            def search(self, vector, top_k=5, filter_payload=None):
                self.search_calls += 1
                return [{"seq": True}]

            def search_batch(self, requests):
                self.batch_calls += 1
                return [[{"kind_filter": r.get("filter_payload")}] for r in requests]

        store = BatchStore()
        specs = [("main", None, 5), ("video", {"content_type": "video_transcript"}, 30)]

        out = _batched_dense_search(store, [0.1], specs)

        assert store.batch_calls == 1
        assert store.search_calls == 0
        assert set(out) == {"main", "video"}

    def test_falls_back_to_sequential_without_batch(self):
        class SeqStore:  # no search_batch on the class → sequential path
            def __init__(self):
                self.search_calls = 0

            def search(self, vector, top_k=5, filter_payload=None):
                self.search_calls += 1
                return [{"top_k": top_k}]

        store = SeqStore()
        specs = [("main", None, 5), ("video", {"x": 1}, 7)]

        out = _batched_dense_search(store, [0.1], specs)

        assert store.search_calls == 2
        assert out["video"][0]["top_k"] == 7

    def test_single_spec_skips_batch(self):
        class BatchStore:
            def __init__(self):
                self.batch_calls = 0
                self.search_calls = 0

            def search(self, vector, top_k=5, filter_payload=None):
                self.search_calls += 1
                return [{"seq": True}]

            def search_batch(self, requests):
                self.batch_calls += 1
                return [[] for _ in requests]

        store = BatchStore()

        out = _batched_dense_search(store, [0.1], [("main", None, 5)])

        assert store.batch_calls == 0
        assert store.search_calls == 1
        assert "main" in out


class TestQdrantSearchBatch:
    def test_search_batch_single_round_trip_and_hydration(self, monkeypatch):
        import sys as _sys

        from providers.qdrant_store import QdrantLocalStore

        # Other test modules swap in their own fake ``qdrant_client.models`` at
        # collection time; guarantee ``QueryRequest`` exists on whichever is active.
        monkeypatch.setattr(
            _sys.modules["qdrant_client.models"], "QueryRequest", MagicMock(), raising=False
        )

        store = object.__new__(QdrantLocalStore)  # skip __init__ (no collection setup)
        store.collection = "rag_docs_v3"

        def _point(text, score):
            pt = MagicMock()
            pt.payload = {"text": text, "source": "s"}
            pt.score = score
            return pt

        resp_a = MagicMock()
        resp_a.points = [_point("a", 0.9)]
        resp_b = MagicMock()
        resp_b.points = [_point("b", 0.8)]
        client = MagicMock()
        client.query_batch_points.return_value = [resp_a, resp_b]
        store.client = client

        out = store.search_batch(
            [
                {"vector": [0.1], "top_k": 5, "filter_payload": None},
                {"vector": [0.1], "top_k": 3, "filter_payload": {"content_type": "image"}},
            ]
        )

        assert client.query_batch_points.call_count == 1  # one round-trip for both
        assert [h["text"] for h in out[0]] == ["a"]
        assert [h["text"] for h in out[1]] == ["b"]
        assert out[0][0]["score"] == 0.9

    def test_search_batch_empty_requests(self):
        from providers.qdrant_store import QdrantLocalStore

        store = object.__new__(QdrantLocalStore)
        store.client = MagicMock()
        assert store.search_batch([]) == []
        store.client.query_batch_points.assert_not_called()


class TestRunRetrievalBatchesSupplements:
    def test_video_supplement_shares_one_batched_request(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
        monkeypatch.setenv("HYDE_ENABLED", "false")
        monkeypatch.setenv("RERANKER_ENABLED", "false")

        class BatchStore:
            def __init__(self):
                self.batch_calls = 0

            def search(self, vector, top_k=5, filter_payload=None, **kwargs):
                return []

            def search_batch(self, requests):
                self.batch_calls += 1
                out = []
                for req in requests:
                    ct = (req.get("filter_payload") or {}).get("content_type")
                    if ct == "video_transcript":
                        out.append(
                            [{"source": "vid", "content_type": "video_transcript", "score": 0.8}]
                        )
                    else:
                        out.append([{"source": "m.pdf", "content_type": "text", "score": 0.9}])
                return out

        store = BatchStore()
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2]]
        cache = MagicMock()
        cache.enabled = False
        analytics = MagicMock()

        with patch("providers.query_orchestrator.get_embedder", lambda: embedder), \
             patch("providers.query_orchestrator.get_vector_store", lambda: store), \
             patch("providers.query_orchestrator.get_semantic_cache", lambda: cache):
            result = QueryOrchestrator().run_retrieval(
                RetrievalContext(
                    question="Mech-Mind bin picking demo with 3D camera",
                    top_k=5,
                    history_turns=0,
                    vendor_filter=None,
                    document_type_filter=None,
                    analytics=analytics,
                )
            )

        assert store.batch_calls == 1  # main + video collapsed into one round-trip
        assert analytics.video_transcript_supplement is True
        assert any(c.get("content_type") == "video_transcript" for c in result.chunks)


# --- Increment 2: prompt caching (cached system block) + offline Batch API ---


class TestAnthropicSystemCaching:
    def test_system_block_is_cached_when_provided(self):
        llm = AnthropicLLM(model="claude-sonnet-5")
        payload = llm._payload("question body", stream=False, system="STABLE RULES")

        assert payload["system"] == [
            {"type": "text", "text": "STABLE RULES", "cache_control": {"type": "ephemeral"}}
        ]
        assert payload["messages"] == [{"role": "user", "content": "question body"}]

    def test_no_system_key_when_omitted(self):
        payload = AnthropicLLM()._payload("hi", stream=False)
        assert "system" not in payload

    def test_subscription_keeps_identity_first_then_cached_rules(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_SUBSCRIPTION_MODEL", "claude-sonnet-5")
        from providers.claude_subscription_llm import CLAUDE_CODE_IDENTITY, ClaudeSubscriptionLLM

        llm = ClaudeSubscriptionLLM()
        payload = llm._payload("body", stream=False, system="RULES")

        assert payload["system"][0] == {"type": "text", "text": CLAUDE_CODE_IDENTITY}
        assert payload["system"][1]["text"] == "RULES"
        assert payload["system"][1]["cache_control"] == {"type": "ephemeral"}

    def test_subscription_identity_only_without_system(self, monkeypatch):
        from providers.claude_subscription_llm import ClaudeSubscriptionLLM

        payload = ClaudeSubscriptionLLM()._payload("body", stream=False)
        assert len(payload["system"]) == 1


def _http_response(*, status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.text = text
    return resp


class TestAnthropicBatch:
    def _client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from providers.anthropic_batch import AnthropicBatchClient

        client = AnthropicBatchClient(model="claude-haiku-4-5", max_tokens=64)
        client._client = MagicMock()
        return client

    def test_create_posts_requests_and_returns_id(self, monkeypatch):
        client = self._client(monkeypatch)
        client._client.post.return_value = _http_response(json_body={"id": "batch_123"})

        batch_id = client.create({"0": "prompt-a", "1": "prompt-b"})

        assert batch_id == "batch_123"
        posted = client._client.post.call_args.kwargs["json"]["requests"]
        assert {r["custom_id"] for r in posted} == {"0", "1"}
        assert posted[0]["params"]["model"] == "claude-haiku-4-5"
        assert posted[0]["params"]["messages"][0]["content"] == "prompt-a"
        assert "system" not in posted[0]["params"]

    def test_create_adds_cached_system_prefix(self, monkeypatch):
        client = self._client(monkeypatch)
        client._client.post.return_value = _http_response(json_body={"id": "b"})

        client.create({"0": "p"}, system="SHARED RUBRIC")

        params = client._client.post.call_args.kwargs["json"]["requests"][0]["params"]
        assert params["system"][0]["text"] == "SHARED RUBRIC"
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_results_parse_jsonl_keyed_by_custom_id(self, monkeypatch):
        client = self._client(monkeypatch)
        lines = [
            {
                "custom_id": "0",
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"type": "text", "text": "answer-0"}]},
                },
            },
            {"custom_id": "1", "result": {"type": "errored"}},
        ]
        client._client.get.return_value = _http_response(
            text="\n".join(json.dumps(row) for row in lines)
        )

        out = client.results({"results_url": "https://x/results"})

        assert out == {"0": "answer-0", "1": ""}

    def test_run_end_to_end(self, monkeypatch):
        client = self._client(monkeypatch)
        client._client.post.return_value = _http_response(json_body={"id": "batch_9"})
        client._client.get.side_effect = [
            _http_response(json_body={"processing_status": "ended", "results_url": "u"}),
            _http_response(
                text=json.dumps(
                    {
                        "custom_id": "0",
                        "result": {
                            "type": "succeeded",
                            "message": {"content": [{"type": "text", "text": "done"}]},
                        },
                    }
                )
            ),
        ]

        assert client.run({"0": "p"}) == {"0": "done"}


class TestAnswerEvalBatch:
    def test_batch_mode_grades_via_batch_api(self, monkeypatch, tmp_path):
        import eval.run_answer_eval as rae
        import providers.anthropic_batch as anthropic_batch

        dataset = tmp_path / "d.jsonl"
        dataset.write_text(
            json.dumps({"id": "q1", "question": "What is X?", "expected_sources": ["a.pdf"]})
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(rae, "get_llm", lambda: MagicMock())
        monkeypatch.setattr(rae, "query", lambda q, top_k=5: {"answer": "X is a thing."})

        class FakeBatchClient:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, prompts, system=None):
                return {cid: '{"pass": true, "score": 1.0, "reason": "grounded"}' for cid in prompts}

        monkeypatch.setattr(anthropic_batch, "AnthropicBatchClient", FakeBatchClient)

        report = rae.run_answer_eval(dataset, batch=True)

        assert report["count"] == 1
        assert report["pass_rate"] == 1.0
        assert report["items"][0]["pass"] is True
