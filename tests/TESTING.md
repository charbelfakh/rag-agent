# Testing policy

Every new feature, provider, script, or API route **must ship with unit tests** in the same change.

## Requirements

1. **New Python module** under `providers/`, `eval/`, `scripts/`, or `api/` → add or extend tests under `tests/`.
2. **New env knob** → at least one test asserting enabled/disabled behavior.
3. **New HTTP route** → add a case in `tests/test_api_endpoints.py` (or a focused API test file).
4. **Bug fix** → regression test when reproduction is practical.

## Conventions

- Mock external services (Ollama, Qdrant, Redis, Langfuse, vLLM, TEI); do not require live infra for CI.
- Use `tests/conftest.py` → `patch_retrieval_pipeline()` when testing `providers.rag_pipeline._build_generation_plan()`.
- Prefer small, named test classes grouped by module (see `tests/test_coverage_complete.py`).
- Run the full suite before merging:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## Coverage map (Sprints G–Q)

| Module / area | Test file(s) |
|---------------|----------------|
| Incremental ingest, early web | `test_sprint_g.py` |
| TEI embed, doc registry, ingest jobs, web compression | `test_sprint_h.py` |
| RedisVL cache, v2 migration, Langfuse experiments | `test_sprint_i.py` |
| Feedback, orchestrator, two-stage, judge, OTel | `test_sprint_j.py` |
| Ingest queue, blob store, sharding, vLLM | `test_sprint_k.py` |
| M0 media store, reranker A/B, section chunking | `test_sprint_l.py` |
| PDF images, SigLIP, hybrid RRF | `test_sprint_m.py` |
| VLM LLM, ColPali stub | `test_sprint_n.py` |
| Speculative gen, map-reduce, dynamic top-N | `test_sprint_o.py` |
| Production sampling, sessions API, multimodal citations | `test_sprint_p.py` |
| Qdrant Cloud store | `test_sprint_q.py` |
| API routes (health, feedback, upload queue, media, sessions) | `test_api_endpoints.py` |
| Cross-module gap audit | `test_coverage_complete.py` |

Run before merge: `python -m pytest tests/ -q` (currently **169** tests).
