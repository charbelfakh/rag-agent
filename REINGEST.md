# Full Re-ingest Runbook

> Plan for rebuilding the vector index with the Markdown ingest path, hybrid
> sparse+dense retrieval, cleaned video transcripts, and (optionally) a new
> embedding model. **The old collection is never deleted** — the new index is
> built side-by-side and cut over only after the eval gate passes.

## What changed since the current index was built

| Change | Flag / mechanism | Why |
|---|---|---|
| Markdown extraction for PDF + HTML | `MARKDOWN_INGEST_ENABLED=true` | Tables/lists survive into chunk text (pymupdf4llm per-page MD, markdownify for HTML); page/section metadata preserved |
| Hybrid dense+sparse retrieval | `QDRANT_SPARSE_ENABLED=true` | Lexical matching for exact terms (model numbers like "2120A", "XL250"); sparse vectors **must exist at collection creation** |
| VTT rolling-caption dedupe | automatic | YouTube transcripts were stored with every line duplicated 2–3× |
| VTT glossary pass | automatic | ASR fixes (homoids→humanoids, cuttingedge→cutting-edge, Mech-Mind variants) now apply to YouTube captions too |
| PyMuPDF 1.27.2.3 → 1.28.0 | requirements.txt | pymupdf4llm requires matching versions |

Pre-migration snapshot (2026-07-04, old `rag_docs` index — now superseded by v3): recall@5 **0.930**,
recall@10 0.930, MRR **0.806**, caption recall@5 0.395 —
`eval/reports/pre_migration_2026-07-04.json`.

## Phase 0 — prerequisites

- Qdrant, Redis, Ollama running; API servers can stay up (they read the old collection until env changes + restart).
- Working tree committed (your call — nothing here commits for you).
- Pre-migration eval report saved (done, see above).

## Phase 1 — embedder bake-off (~1–2 h, decide the model)

Pull candidates, then compare on a corpus sample (golden-set sources always included):

```powershell
ollama pull bge-m3
ollama pull mxbai-embed-large
.venv\Scripts\python.exe eval/run_embedder_bakeoff.py `
  --models nomic-embed-text,bge-m3,mxbai-embed-large `
  --sample-size 2000 --output eval/reports/bakeoff_2026-07.json
```

Pick on recall@5/MRR **and** `mean_query_embed_ms` (query embed latency is the
single worst latency contributor). Put the winner in `.env` as
`OLLAMA_EMBED_MODEL=<model>`. Skip this phase → keep `nomic-embed-text`.

## Phase 2 — Markdown pilot sanity check (~15 min)

Spot-check MD extraction on 2–3 representative manuals before committing hours:

```powershell
$env:MARKDOWN_INGEST_ENABLED='true'
.venv\Scripts\python.exe -c "from providers.markdown_ingest import pdf_markdown_sections; s = pdf_markdown_sections('data/zivid/Zivid 2+ L110 Datasheet.pdf'); print(len(s), 'sections'); print(s[5])"
```

Verify: headings sensible, page numbers present, spec tables render as `|`-pipe rows.

## Phase 3 — full re-ingest into a NEW collection

Set in `.env` (then restart any running API processes):

```ini
QDRANT_COLLECTION=rag_docs_v3      # new index; old rag_docs untouched
MARKDOWN_INGEST_ENABLED=true
QDRANT_SPARSE_ENABLED=true         # must be set BEFORE first ingest creates the collection
OLLAMA_EMBED_MODEL=<bake-off winner>
# optional: QDRANT_QUANTIZATION=int8   (≈4× RAM reduction, slight recall cost)
```

Then run (resumable; `--fresh` ignores previous state):

```powershell
# 1. Documents (PDF + HTML + captions) — the long part (hours on Ollama; use gpu/tei embed profile if available)
.venv\Scripts\python.exe -m scripts.ingest.reingest_all --fresh

# 2. YouTube transcripts — picks up the stutter fix + glossary automatically
.venv\Scripts\python.exe -m scripts.ingest.ingest --video-dir data/videos/transcripts --force

# 3. Registry backfill for the documents panel
.venv\Scripts\python.exe -m scripts.ops.backfill_doc_registry
```

Notes:
- The embed model at query time **must match** the model used to build the index.
- `--force` purges per-source chunks in the *target* collection only.
- For bulk speed, `EMBED_PROVIDER=gpu` or `tei` embeds far faster than Ollama —
  only if the same model is then served for queries.

## Phase 4 — eval gate and cutover

```powershell
# Fresh index numbers
.venv\Scripts\python.exe eval/run_retrieval_eval.py --output eval/reports/post_migration.json

# Compare against eval/reports/pre_migration_2026-07-04.json.
# Gate: recall@5 and MRR within −0.05 of pre-migration (ideally better).
```

- **Pass** → keep `QDRANT_COLLECTION=rag_docs_v3`, run
  `eval/run_retrieval_eval.py --write-baseline`, restart API + GraphQL servers,
  update the README metrics table with before/after numbers.
- **Fail** → investigate (chunk sizes? sparse noise? MD extraction of a specific
  vendor?); the old index is still live — nothing user-facing changed.

## Rollback

Set `QDRANT_COLLECTION=rag_docs` in `.env`, restart servers. Delete
`rag_docs_v3` only when confident:
`curl -X DELETE http://localhost:6333/collections/rag_docs_v3`

## Cleanup after successful cutover

- Delete the old collection when comfortable (or keep it for the portfolio A/B story).
- Delete `bakeoff_*` collections if `--keep-collections` was used.
- Update `tests/TESTING.md` / `CLAUDE.md` if baselines or counts changed.

## Cutover result (2026-07-05)

Cut over to **`rag_docs_v3`** built with Markdown ingest, full caption coverage
(10,756 migrated from the old index + 883 newly VLM-captioned), and 361 video
transcripts (dedupe + glossary). Shipping config: **dense + HyDE, sparse OFF,
`RERANKER_FETCH_K=60`**.

Retrieval eval (45 golden Qs), v3 vs old `rag_docs` baseline:

| Metric | old | v3 @fetch20 | v3 @fetch60 (shipped) | Δ vs old |
|---|---|---|---|---|
| recall@5 | 0.933 | 0.911 | **0.933** | ties |
| MRR | 0.761 | 0.783 | **0.800** | **+0.039** |
| caption recall@5 | 0.356 | 0.400 | **0.400** | **+0.044** |

`RERANKER_FETCH_K` raised 20→60: the reranker often needs candidates ranked
21–60, which recovers the last recall point (0.911→0.933) and lifts MRR. Cost:
mean rerank latency ~500→1100 ms/query. Lower to 20 if latency outweighs the gain.

**Sparse hybrid deferred.** A/B (see `eval/reports/v3_*`) showed sparse RRF keeps
recall@5 but *hurts* MRR (0.78→0.65) and caption recall — it pushes the best
chunk below rank 1. HyDE already closes the lexical gap sparse targets, so the
two are redundant on this (mostly NL) golden set. The collection retains sparse
vectors; re-enabling is a `QDRANT_SPARSE_ENABLED` flip once the sparse text
function is improved (stopwords / BM25 length-norm / query-adaptive) **and** the
golden set gains exact model-number lookups to prove the benefit.

New baseline written from `eval/reports/post_migration_v3_dense.json`. Rollback:
set `QDRANT_COLLECTION=rag_docs` in `.env` and restart.
