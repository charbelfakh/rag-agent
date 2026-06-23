# Vision Systems Assistant (rag-agent)

A local-first **Retrieval-Augmented Generation (RAG)** assistant for industrial vision-systems documentation (Pekat, Mechmind, Zivid, LMI, and others). Ingest PDFs and HTML help articles into Qdrant, query them through a streaming chat UI, and optionally fall back to web search when local context is insufficient.

## Features

- **Chat UI** — Dark-theme SPA with SSE streaming, conversation history, stop control, thumbs up/down feedback, optional dev analytics, and a documents panel (upload, ingest progress, delete).
- **Vendor/product scoping** — Toolbar dropdowns populated from `GET /vendors`; keyword inference from the question when filters are unset; explicit filters always win.
- **RAG pipeline** — Query condensation for follow-ups, HyDE, parallel `QueryOrchestrator`, cross-encoder reranking, semantic cache, sufficiency check, and SearXNG web fallback.
- **Ingest v2** — PDF + HTML (URL list or single URL), schema v2 Qdrant payloads, ingest manifest, collision-proof filenames, pending caption/video sidecars.
- **Provider abstraction** — Ollama (default LLM + embed), local GPU or TEI embeddings, OpenAI-compatible chat endpoints (`vllm` / `tgi` / `openai_compatible`), optional VLM; Qdrant local, sharded, or cloud.
- **Ops tooling** — `python -m scripts.ingest.reingest_all` (full-corpus re-ingest), `python -m scripts.ingest.extract_pdf_images` + `python -m scripts.ingest.caption_worker` (PDF/HTML image caption pipeline), `python -m scripts.ops.audit`, `python -m scripts.ops.cleanup`, eval suite with CI baseline checks.
- **Observability** — Langfuse v3 per-query analytics; optional OpenTelemetry spans.

## Architecture

### System overview

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────────────────┐
│  Chat UI     │────▶│  FastAPI     │────▶│  providers/rag_pipeline.py  │
│  (SSE)       │     │  api/main.py │     │  plan → stream → analytics  │
└──────────────┘     └──────────────┘     └──────────────┬──────────────┘
                                                         │
                         ┌───────────────────────────────┼───────────────────────────────┐
                         ▼                               ▼                               ▼
              QueryOrchestrator                   GenerationPlan                    Langfuse
              (embed ∥ HyDE ∥ cache)              (prompt + citations)              + OTel
                         │
         ┌───────────────┼───────────────┬───────────────┬───────────────┐
         ▼               ▼               ▼               ▼               ▼
      Qdrant         Reranker        Redis cache      SearXNG         Ollama / vLLM
      vectors        (CrossEncoder)  (RedisVL opt.)   web search      LLM + embed
```

### Query path

1. **Optional condensation** — When chat history is non-empty, the LLM rewrites the follow-up into a standalone retrieval query (original question still goes to the generation prompt).
2. **Retrieval scope** — Explicit `vendor` / `product` from the UI or API; else single-vendor keyword detection on the (condensed) question; multiple vendor keywords → no filter (comparison queries). Zero filtered hits → unfiltered fallback (logged).
3. **Retrieval** — `QueryOrchestrator`: embed question (parallel with HyDE when enabled), semantic cache lookup (skipped when history or filters are active), Qdrant search with payload filters, optional hybrid image retrieval, rerank.
4. **Generation** — Sufficiency check and/or early web fallback; assemble prompt with history + chunk headers; stream LLM tokens; retry with web if answer is insufficient.

### Ingest path

```
PDF / HTML / TXT  →  read/split (producer thread)  →  chunk queue
                              ↓
                    embed batches (worker thread)
                              ↓
                    Qdrant upsert + doc_registry (main thread)
```

- **HTML** — BeautifulSoup extraction; Confluence pages prefer `.ak-renderer-document` when present (avoids SPA chrome). Heading-aware sections, image/video sidecars (`pending_captions.json`, `pending_videos.json`).
- **Incremental** — `data/ingest_manifest.json` tracks SHA-256 per source; unchanged files skip re-embed unless `--force`.
- **Metadata** — Upload modal and CLI flags set vendor, product, doc type, version, URL.

### Key modules

| Module | Role |
|--------|------|
| `providers/rag_pipeline.py` | RAG orchestration, condensation, filters, prompt assembly, SSE events |
| `providers/query_orchestrator.py` | Parallel embed / HyDE / cache / search / rerank |
| `scripts/ingest/ingest.py` | v2 ingest pipeline, manifest, HTML loader |
| `providers/factory.py` | Provider singletons (LLM, embed, store, cache, reranker) |
| `providers/metadata.py` | Vendor inference, chunk IDs, upload metadata |
| `scripts/ops/audit.py` | Read-only Qdrant coverage vs manifest (`python -m scripts.ops.audit`) |
| `scripts/ops/cleanup.py` | Delete or retag sources by `payload.source` (`python -m scripts.ops.cleanup`) |
| `scripts/ingest/reingest_all.py` | Resumable full-corpus re-ingest (`python -m scripts.ingest.reingest_all`) |
| `scripts/ingest/extract_pdf_images.py` | Extract embedded PDF images → caption queue (`python -m scripts.ingest.extract_pdf_images`) |
| `scripts/ingest/caption_worker.py` | Vision-caption queued images → `image_caption` points (`python -m scripts.ingest.caption_worker`) |
| `scripts/ingest/ingest_worker.py` | Redis ingest queue consumer (`python -m scripts.ingest.ingest_worker`) |
| `scripts/ingest/video_frame_worker.py` | Scene-change video frame captions (`python -m scripts.ingest.video_frame_worker`) |
| `scripts/mechmind_batch_ingest.py` | One-off Mech-Mind PDF batch (`python -m scripts.mechmind_batch_ingest`) |
| `scripts/zivid_batch_ingest.py` | One-off Zivid PDF batch (`python -m scripts.zivid_batch_ingest`) |
| `scripts/backfill_manifest.py` | Manifest backfill for legacy Qdrant sources (`python -m scripts.backfill_manifest`) |
| `eval/` | Offline retrieval and answer evaluation |

> **Note:** Deep design notes and agent context live in local-only `ARCHITECTURE.md` / `CLAUDE.md` (gitignored).

## Document metadata (schema v2)

Each Qdrant point payload includes:

| Field | Description |
|-------|-------------|
| `text` | Chunk content (or preview + `text_uri` when blob storage enabled) |
| `source` | Filename only (e.g. `manual-abc12345.html`) |
| `vendor` | Lowercase vendor slug |
| `product` | Product line (e.g. `gocator`, `pekat vision`, `hexsight`) |
| `product_version` | Optional software/firmware version |
| `doc_type` | `manual`, `article`, `datasheet`, `tutorial`, etc. |
| `page` | PDF page (0-based); `null` for HTML |
| `section` | Heading label when detected |
| `url` | Original article URL for HTML ingests |
| `content_type` | `text` or `image_caption` (vision-captioned figures) |
| `schema_version` | `2` |
| `ingested_at` | ISO-8601 UTC timestamp |

Citations returned to the UI include `source`, `vendor`, `page`, `section`, and scores when available.

### Retrieval scoping

| Mechanism | When | Cache |
|-----------|------|-------|
| **Explicit** | UI vendor/product dropdowns or API `vendor` / `product` | Skipped |
| **Keyword** | Question mentions one known vendor (no explicit filter) | Skipped |
| **None** | No filter, or multiple vendors in question | Normal |

Filter precedence is logged as `filter_mechanism`: `explicit`, `keyword`, or `none` (Langfuse + dev analytics).

## Project structure

```
rag-agent/
├── api/main.py                 # FastAPI: /query, /upload, /vendors, /documents, sessions
├── ui/index.html               # Chat UI (vanilla JS, no build step)
├── providers/rag_pipeline.py   # RAG query pipeline
├── scripts/ingest/ingest.py    # Ingest v2 (PDF, HTML, URL, VTT modes)
├── config/
│   ├── vendors.json            # Vendor folders, YouTube channel aliases
│   └── searxng/settings.yml    # SearXNG instance config
├── providers/                  # LLM, embed, Qdrant, cache, reranker, orchestrator, …
├── scripts/
│   ├── ingest/                 # ingest, reingest, caption_worker, ingest_video, …
│   ├── ops/                    # audit, cleanup, migrate_*, backfill_doc_registry
│   ├── data/                   # download_docs, fetch_lmi_urls
│   └── backfill_manifest.py    # Manifest backfill for legacy Qdrant sources
├── eval/                       # Retrieval/answer eval, CI baseline
├── tests/                      # Unit tests (see tests/TESTING.md)
├── data/                       # Runtime: uploads, manifest, registries (gitignored)
├── logs/                       # Operational logs (gitignored)
├── docs/                       # Vendor manuals + internal notes (content gitignored)
├── docker-compose.yml          # qdrant, redis, searxng, api, langfuse stack
└── Dockerfile                  # API container image
```

## Prerequisites

- **Python 3.11+** with a virtual environment
- **Ollama** on the host (`ollama serve`) for default LLM and embeddings
- **Docker** for Qdrant, Redis, SearXNG, and optional Langfuse

```powershell
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
ollama pull llava:7b
```

Default stack: **LLM** `qwen3.5:9b`, **embed** `qwen3-embedding:0.6b` (1024-dim), **vision** `llava:7b` for image captions. Set `OLLAMA_THINK_ENABLED=false` for qwen3.x RAG (see `.env.example`).

## Installation

```powershell
git clone https://github.com/charbelfakh/rag-agent.git
cd rag-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Configuration

Copy `.env.example` to `.env`. Common variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `ollama`, `vllm`, `openai_compatible`, `tgi`, `vlm` | `ollama` |
| `EMBED_PROVIDER` | `ollama`, `gpu`, `tei` | `ollama` |
| `VECTOR_STORE` | `qdrant_local`, `qdrant_cloud` | `qdrant_local` |
| `QDRANT_COLLECTION` | Collection name | `rag_docs` |
| `HYDE_ENABLED` | Hypothetical document embeddings | `true` |
| `RERANKER_ENABLED` | Cross-encoder reranking | `true` |
| `SEMANTIC_CACHE_ENABLED` | Redis-backed answer cache | `true` |
| `ANALYTICS_ENABLED` | Langfuse query logging | `true` |
| `API_KEY` | Optional `X-API-Key` auth (unset = disabled) | — |
| `CHAT_HISTORY_TURNS` | Messages in LLM prompt | `6` |

See `.env.example` for the full list (orchestrator flags, blob storage, sharding, multimodal, etc.).

## Running locally

```powershell
docker compose up -d qdrant redis searxng
ollama serve
python api/main.py
```

Open **http://localhost:8000**. Hard-refresh after UI changes.

### Ingest

```powershell
# PDF
python -m scripts.ingest.ingest data/pekat/manual.pdf --vendor pekat --doc-type manual

# HTML from URL list
python -m scripts.ingest.ingest --url-list data/lmi/urls_kb.txt --vendor lmi --doc-type article --product gocator

# UI: Documents panel → upload with metadata modal
```

### Full corpus re-ingest (LMI + Pekat)

Resumable pipeline: text ingest → PDF image extraction → vision caption drain → report.

```powershell
# Dry-run first
python -m scripts.ingest.reingest_all --vendors lmi pekat --dry-run

# Full run (GPU-heavy during caption phase — avoid concurrent chat queries)
python -m scripts.ingest.reingest_all --vendors lmi pekat

# Or run stages individually:
python -m scripts.ingest.extract_pdf_images --all
python -m scripts.ingest.caption_worker --vendor lmi --reconcile-missing -v
python -m scripts.ingest.caption_worker --vendor pekat -v
```

State is saved to `data/reingest_state.json`; reports go to `reingest_report_*.md` (both gitignored).

### Vendor doc downloads

```powershell
python -m scripts.data.download_docs --list
python -m scripts.data.download_docs --vendor pekat
```

### Audit and cleanup

```powershell
python -m scripts.ops.audit --detail --vendor lmi
python -m scripts.ops.audit --json logs/audits/report.json

python -m scripts.ops.cleanup --dry-run --delete-sources data/lmi/sources_to_delete.txt
python -m scripts.ops.cleanup --retag data/lmi/sources_to_retag.txt --product hexsight --yes
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `GET` | `/health` | Health check |
| `GET` | `/health/embed` | Embedder health |
| `GET` | `/vendors` | Distinct vendor/product pairs (5 min cache; `?refresh=true`) |
| `POST` | `/query` | SSE stream (`token`, `citations`, `done`, `meta`) |
| `GET` | `/documents` | List ingested sources |
| `POST` | `/upload` | Upload file → ingest job |
| `GET` | `/ingest/status/{job_id}` | Ingest progress |
| `POST` | `/ingest` | Ingest by server path |
| `DELETE` | `/ingest` | Delete chunks by source |
| `POST` | `/feedback` | Thumbs up/down → Langfuse |
| `GET` | `/sessions` | List sessions (when `SESSION_STORAGE_ENABLED`) |

When `API_KEY` is set, send `X-API-Key` on protected routes.

### `POST /query` body

```json
{
  "question": "How does surface flatness work?",
  "top_k": 10,
  "history": [],
  "vendor": "pekat",
  "product": null
}
```

Omit `vendor` and `product` when using “All vendors” / “All products” in the UI.

## Observability

Enable Langfuse:

```env
ANALYTICS_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

```powershell
docker compose up -d langfuse-web langfuse-worker langfuse-postgres langfuse-clickhouse langfuse-minio langfuse-redis
```

Dev analytics in the UI: **Settings → Show query analytics**.

### Offline eval

```powershell
python eval/run_retrieval_eval.py
python eval/run_retrieval_eval.py --write-baseline
python eval/run_retrieval_eval.py --check-baseline
python eval/run_answer_eval.py

# Caption ablation (text-only vs full corpus)
python eval/run_retrieval_eval.py --dataset eval/dataset_caption.jsonl --content-type-filter text
python eval/run_retrieval_eval.py --dataset eval/dataset_caption.jsonl --content-type-filter none
```

CI runs unit tests and baseline checks via `.github/workflows/eval.yml`.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

**339 tests** across sprints A–Q, API endpoints, ingest v2, caption worker, and coverage audit. Requires `pytest-asyncio` (included in `requirements-dev.txt`). Policy: new modules, env knobs, and API routes need tests in the same change ([tests/TESTING.md](tests/TESTING.md)).

## Roadmap

### Shipped (current)

| Area | Capability |
|------|------------|
| **Core RAG** | HyDE, reranker, semantic cache, web fallback, sufficiency check, early web fallback |
| **Query UX** | Multi-turn history, query condensation, vendor/product filters + keyword inference |
| **Ingest** | Schema v2, HTML/URL ingest, Confluence `.ak-renderer-document` extraction, manifest, upload metadata, incremental skip |
| **Scale** | Parallel orchestrator, ingest queue + workers, blob externalization, vendor sharding |
| **Quality** | Section-aware chunking, map-reduce retrieval, speculative generation, dynamic rerank top-N |
| **Multimodal** | PDF image extract, vision captions (`image_caption` points), hybrid retrieval stubs, ColPali/VLM stubs |
| **Ops** | `scripts/ops/*`, `scripts/ingest/*`, `scripts/data/*`, Langfuse + OTel |
| **UI** | Documents panel, upload modal, vendor/product toolbar, feedback buttons, markdown rendering |

### Next (prioritized)

| Priority | Item | Notes |
|----------|------|-------|
| 1 | **Remaining vendor ingest** | Photoneo, Basler, etc. — deferred by choice |
| 2 | **KB3150 Mask + ML Training visuals** | Shell print-to-PDFs have no extractable images; needs proper Confluence PDF export (cf. KB3150-Detector) |
| 3 | **Typo-tolerant vendor keywords** | e.g. `pekt` → `pekat` for inference |
| 4 | **Server-side sessions by default** | `SESSION_STORAGE_ENABLED` exists; UI still uses `localStorage` for conversations |
| 5 | **Scheduled doc refresh** | Cron/worker around `python -m scripts.data.download_docs` + URL lists per vendor |
| 6 | **ColPali / diagram PDF index** | Stub embedder present; needs end-to-end ingest + retrieval |
| 7 | **GoPxL caption disambiguation** | Same-family manuals share figure captions; enrich with `product`/`device_family` metadata |

### Later / experimental

- Cross-encoder cache for repeated retrieval patterns
- Automated golden-set expansion from production sampling (`eval/production_sampler.py`)
- Multi-tenant API keys and per-tenant collections
- Hosted deployment guide (Qdrant Cloud + GPU embed sidecar)

## Optional feature flags (quick reference)

Enable via `.env` — each has tests in `tests/test_sprint_*.py`:

```env
# Retrieval
QUERY_ORCHESTRATOR_ENABLED=true
TWO_STAGE_RETRIEVAL_ENABLED=true
HYBRID_RETRIEVAL_ENABLED=true
MAP_REDUCE_RETRIEVAL_ENABLED=true
SPECULATIVE_GENERATION_ENABLED=true

# Ingest / storage
INGEST_QUEUE_ENABLED=true
BLOB_STORAGE_ENABLED=true
MULTIMODAL_IMAGE_INGEST_ENABLED=true
QDRANT_VENDOR_SHARDING=true

# Cache
SEMANTIC_CACHE_BACKEND=redisvl

# Infra
OTEL_ENABLED=true
SESSION_STORAGE_ENABLED=true
```

See `.env.example` for defaults and Docker Compose profiles (`workers`, `gpu-embed`, `vllm-pool`, `multimodal`).

## Known limitations

- **Ollama on host** — Not in Docker; `[WinError 10061]` means Ollama is not running on port 11434.
- **Client-side chat history** — Conversations default to browser `localStorage` unless session storage is enabled.
- **Semantic cache** — Skipped when chat history or vendor/product filters are active.
- **Large ingests** — Full vendor corpora (e.g. 270 LMI articles) take significant time on local Ollama embed.
- **Web fallback** — Quality depends on SearXNG instance and upstream engines; not vendor-scoped.
- **Corpus gaps** — Strong retrieval cannot answer topics absent from ingested docs. Some Pekat KB3150 PDFs are print-to-PDF shells (text-only, no figures); captioning cannot recover missing screenshots.
- **GPU contention** — On 8 GB cards, do not run `python -m scripts.ingest.caption_worker` concurrently with live chat queries (vision + LLM + embed compete for VRAM).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `[WinError 10061]` | Ollama not running | `ollama serve` |
| `401 Unauthorized` | `API_KEY` set, key missing in UI | Settings → API key |
| Mixed vendor citations | No filter; comparison or broad topic | Set vendor dropdown or mention one vendor |
| “Insufficient” with on-topic chunks | Topic not in vendor corpus | Ingest missing docs or query correct vendor |
| Stale answers | Semantic cache | `redis-cli DEL semantic_cache:entries` |
| UI changes invisible | Browser cache | Hard refresh (Ctrl+Shift+R) |

## License

See [LICENSE](LICENSE).
