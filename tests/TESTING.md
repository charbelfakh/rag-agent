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
| Fast-tier LLM, cached system block, Batch API, Qdrant `search_batch` | `test_cost_optimization.py` |
| API routes (health, feedback, upload queue, media, sessions) | `test_api_endpoints.py` |
| Cross-module gap audit | `test_coverage_complete.py` |
| Eval harness gate logic (offline) | `test_eval_harness.py` |

Run before merge: `python -m pytest tests/ -q` (currently **528** tests; 1 skips when jsdom is not installed).

## Pre-push recall gate (requires live services)

The CI workflow runs only offline unit tests — it cannot reach Qdrant or Ollama.
Before merging **any retrieval-path change** (embedder, reranker, HyDE prompt, filter
logic, score floors), run the live gate locally:

```bash
# Start Qdrant (Docker) and Ollama, then:
python eval/run_retrieval_eval.py \
  --dataset eval/dataset.jsonl \
  --fetch-k 20 \
  --rerank-top-n 5 \
  --check-baseline
```

Exit 0 = within tolerance. Exit 1 = regression printed to stderr.

### Updating the baseline

After intentionally improving retrieval (new model, corpus expansion, etc.):

```bash
python eval/run_retrieval_eval.py \
  --dataset eval/dataset.jsonl \
  --fetch-k 20 \
  --rerank-top-n 5 \
  --write-baseline
```

Commit the updated `eval/baseline.json` alongside the change.

### Dataset composition (`eval/dataset.jsonl`)

| Content type | Count | Vendors |
|---|---|---|
| `text` | 25 | lmi (8), mechmind (8), zivid (5), pekat (4) |
| `image_caption` | 20 | lmi (8), pekat (12) |
| **Total** | **45** | all four |

Source files: `eval/dataset_golden.jsonl` (text) + `eval/dataset_caption.jsonl` (caption).
`eval/dataset.jsonl` is the merged file used by the gate; do not edit it directly.

### Gated metrics (as of last baseline)

| Metric | Baseline | Tolerance |
|---|---|---|
| `recall_at_5` | 0.933 | ±0.05 |
| `recall_at_10` | 0.933 | ±0.05 |
| `mrr` | 0.800 | ±0.05 |
| `caption_recall_at_5` | 0.400 | ±0.05 |
| `strict_caption_recall_at_5` | 0.333 | ±0.05 |
