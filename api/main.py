"""FastAPI application: chat UI, RAG query streaming, document upload/ingest, and API-key auth."""
import sys



from pathlib import Path







ROOT = Path(__file__).resolve().parent.parent



if str(ROOT) not in sys.path:



    sys.path.insert(0, str(ROOT))







import asyncio
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from collections import defaultdict

from dotenv import load_dotenv

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile



from fastapi.middleware.cors import CORSMiddleware



from fastapi.responses import FileResponse, Response, StreamingResponse



from pydantic import BaseModel, Field, field_validator



from providers.rag_pipeline import stream_query



from scripts.ingest.ingest import ingest



from providers import claude_oauth
from providers import factory as provider_factory
from providers.doc_registry import get_doc_registry
from providers.factory import get_embedder, get_vector_store
from providers.ingest_jobs import get_ingest_job_store
from providers.ingest_queue import get_ingest_queue, is_ingest_queue_enabled
from providers.langfuse_logger import log_feedback
from providers.caption_image_serve import (
    detect_image_content_type,
    resolve_caption_image_candidate,
)
from providers.video_serve import (
    detect_video_content_type,
    iter_file_bytes,
    parse_range_header,
    resolve_video_candidate,
)
from providers.media_store import get_media_store
from providers.otel_tracing import setup_otel
from providers.session_store import get_session_store, is_session_storage_enabled
from providers.transcript_glossary import normalize_transcript_text
from providers.video_transcribe import transcribe_plain_text







load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)
API_KEY = os.getenv("API_KEY", "").strip() or None
VENDORS_CACHE_TTL_SECONDS = 300
_vendors_cache: dict | None = None
_vendors_cache_at: float = 0.0

app = FastAPI(title="RAG Agent API")


async def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    """Require ``X-API-Key`` when ``API_KEY`` is set in the environment."""
    if API_KEY is None:
        return
    if (
        not x_api_key
        or len(x_api_key) != len(API_KEY)
        or not secrets.compare_digest(x_api_key, API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


@app.on_event("startup")
async def _startup() -> None:
    if API_KEY is None:
        logger.warning("API_KEY is not set — API authentication is disabled")
    _apply_persisted_api_keys()
    _apply_persisted_llm_provider()
    setup_otel()
    interrupted = get_ingest_job_store().mark_stale_ingesting_as_interrupted()
    if interrupted:
        logger.info("Marked %d stale ingest jobs as interrupted", interrupted)


DATA_DIR = ROOT / "data"







app.add_middleware(



    CORSMiddleware,



    allow_origins=["*"],



    allow_credentials=True,



    allow_methods=["*"],



    allow_headers=["*"],



)







UI_DIR = ROOT / "ui"







ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".txt"}
ALLOWED_TRANSCRIBE_SUFFIXES = {
    ".webm",
    ".ogg",
    ".oga",
    ".opus",
    ".wav",
    ".mp3",
    ".m4a",
}
_TRANSCRIBE_CONTENT_TYPE_SUFFIX = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/opus": ".opus",
}


def _transcribe_upload_suffix(filename: str, content_type: str | None) -> str | None:
    suffix = Path(filename).suffix.lower()
    if suffix in ALLOWED_TRANSCRIBE_SUFFIXES:
        return suffix
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        return _TRANSCRIBE_CONTENT_TYPE_SUFFIX.get(mime)
    return None

def _job_store():
    return get_ingest_job_store()


def _job_snapshot(job: dict) -> dict:
    now = time.time()
    snap = dict(job)
    snap["elapsed_seconds"] = int(now - job["started_at"])
    if job["status"] == "ingesting":
        done = job.get("chunks_done", 0)
        total = job.get("chunks_total", 0)
        if done > 0 and total > done:
            elapsed = now - job["started_at"]
            snap["estimated_remaining_seconds"] = int(
                (elapsed / done) * (total - done)
            )
        else:
            snap["estimated_remaining_seconds"] = job.get(
                "estimated_remaining_seconds", 0
            )
    return snap


def _update_ingest_job(
    job_id: str, stage: str, chunks_done: int, chunks_total: int
) -> None:
    job = _job_store().get(job_id)
    if not job or job["status"] != "ingesting":
        return
    now = time.time()
    elapsed = now - job["started_at"]
    eta = 0
    if chunks_done > 0 and chunks_total > chunks_done:
        eta = int((elapsed / chunks_done) * (chunks_total - chunks_done))
    _job_store().update(
        job_id,
        stage=stage,
        chunks_done=chunks_done,
        chunks_total=chunks_total,
        elapsed_seconds=int(elapsed),
        estimated_remaining_seconds=eta,
    )


def _run_ingest_job(
    job_id: str,
    ingest_path: str,
    *,
    vendor: str | None = None,
    document_type: str | None = None,
    product_line: str | None = None,
    software_version: str | None = None,
) -> None:
    def callback(stage: str, chunks_done: int, chunks_total: int) -> None:
        _update_ingest_job(job_id, stage, chunks_done, chunks_total)

    try:
        total = ingest(
            ingest_path,
            progress_callback=callback,
            vendor=vendor,
            document_type=document_type,
            product_line=product_line,
            software_version=software_version,
        )
        job = _job_store().get(job_id)
        if not job:
            return
        now = time.time()
        stage = "skipped" if total == 0 else "storing"
        _job_store().update(
            job_id,
            status="done",
            stage=stage,
            chunks_done=total,
            chunks_total=total,
            elapsed_seconds=int(now - job["started_at"]),
            estimated_remaining_seconds=0,
            finished_at=now,
        )
    except Exception as exc:
        job = _job_store().get(job_id)
        if not job:
            return
        now = time.time()
        _job_store().update(
            job_id,
            status="error",
            error=str(exc),
            elapsed_seconds=int(now - job["started_at"]),
            estimated_remaining_seconds=0,
            finished_at=now,
        )


def _source_path_candidates(file_path: str) -> list[str]:
    raw = file_path.strip()
    p = Path(raw)
    candidates = [raw]

    if p.is_absolute():
        candidates.append(str(p.resolve()))
        try:
            rel = p.resolve().relative_to(ROOT.resolve())
            candidates.append(str(rel))
            candidates.append(str(rel).replace("\\", "/"))
        except ValueError:
            pass
    else:
        candidates.append(str((ROOT / p).resolve()))
        candidates.append(raw.replace("\\", "/"))

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique











class HistoryMessage(BaseModel):



    role: str



    content: str











class QueryRequest(BaseModel):



    question: str



    top_k: int = 5



    history: list[HistoryMessage] = Field(default_factory=list)

    vendor_filter: str | None = None

    document_type_filter: str | None = None

    vendor: str | None = None

    product: str | None = None

    @field_validator("vendor", "product", mode="before")
    @classmethod
    def _normalize_filter_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None











class IngestRequest(BaseModel):



    file_path: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(ge=-1, le=1)
    trace_id: str | None = None
    comment: str | None = None











@app.get("/health")



def health():



    return {"status": "ok"}


@app.get("/health/queue")
def queue_health():
    queue = get_ingest_queue()
    return {
        "ingest_queue_enabled": is_ingest_queue_enabled(),
        "available": queue.available,
        "depth": queue.queue_depth(),
    }


@app.get("/health/embed")
def embed_health():
    try:
        embedder = get_embedder()
        vector = embedder.embed(["health check"])[0]
        return {
            "status": "ok",
            "provider": os.getenv("EMBED_PROVIDER", "ollama"),
            "dimensions": len(vector),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _check_qdrant() -> tuple[bool, str]:
    import httpx

    base = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
    try:
        httpx.get(f"{base}/readyz", timeout=2.0).raise_for_status()
        return True, base
    except Exception as exc:
        return False, f"{base} ({exc.__class__.__name__})"


def _check_redis() -> tuple[bool, str]:
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        import redis

        redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2).ping()
        return True, url
    except Exception as exc:
        return False, f"{url} ({exc.__class__.__name__})"


@app.get("/health/deps")
def deps_health(response: Response):
    """Per-dependency reachability so orchestrators can restart or alert.

    503 when Qdrant is down (no retrieval without it); Redis and the LLM
    only degrade the status — queries still work without cache, and hosted
    LLM providers are not probed (cost/latency), only reported.
    """
    qdrant_ok, qdrant_detail = _check_qdrant()
    redis_ok, redis_detail = _check_redis()
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        llm_ok, llm_detail, _ = _ollama_probe()
    else:
        llm_ok, llm_detail = True, f"{provider} (configured; not probed)"

    status = "ok" if (qdrant_ok and redis_ok and llm_ok) else "degraded"
    if not qdrant_ok:
        response.status_code = 503
    return {
        "status": status,
        "qdrant": {"ok": qdrant_ok, "detail": qdrant_detail},
        "redis": {"ok": redis_ok, "detail": redis_detail},
        "llm": {"ok": llm_ok, "provider": provider, "detail": llm_detail},
    }


class SessionCreateRequest(BaseModel):
    title: str = "New chat"


class SessionMessageRequest(BaseModel):
    role: str
    content: str
    meta: dict | None = None


@app.get("/sessions", dependencies=[Depends(verify_api_key)])
def list_sessions_endpoint():
    if not is_session_storage_enabled():
        raise HTTPException(status_code=404, detail="Session storage disabled")
    return {"sessions": get_session_store().list_sessions()}


@app.post("/sessions", dependencies=[Depends(verify_api_key)])
def create_session_endpoint(req: SessionCreateRequest):
    if not is_session_storage_enabled():
        raise HTTPException(status_code=404, detail="Session storage disabled")
    return get_session_store().create_session(title=req.title)


@app.get("/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
def get_session_endpoint(session_id: str):
    if not is_session_storage_enabled():
        raise HTTPException(status_code=404, detail="Session storage disabled")
    return {
        "session_id": session_id,
        "messages": get_session_store().get_messages(session_id),
    }


@app.post("/sessions/{session_id}/messages", dependencies=[Depends(verify_api_key)])
def append_session_message_endpoint(session_id: str, req: SessionMessageRequest):
    if not is_session_storage_enabled():
        raise HTTPException(status_code=404, detail="Session storage disabled")
    get_session_store().append_message(
        session_id,
        role=req.role,
        content=req.content,
        meta=req.meta,
    )
    return {"status": "ok"}


@app.get("/media/{media_path:path}")
def media_endpoint(media_path: str):
    """Serve multimodal blobs (images, thumbnails) from the local media store."""
    store = get_media_store()
    uri = f"/media/{media_path}"
    path = store.resolve_path(uri)
    if path is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(path)


@app.get("/image")
def caption_image_endpoint(path: str):
    """Serve caption-source images from data/<vendor>/images/ (path-restricted)."""
    candidate = resolve_caption_image_candidate(path, ROOT)
    if candidate is None:
        raise HTTPException(status_code=400, detail="Invalid image path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    content_type = detect_image_content_type(candidate.read_bytes()[:16])
    return FileResponse(candidate, media_type=content_type)


@app.get("/video")
def video_endpoint(request: Request, path: str):
    """Serve staged videos from data/<vendor>/videos/ with HTTP Range support."""
    candidate = resolve_video_candidate(path, ROOT)
    if candidate is None:
        raise HTTPException(status_code=400, detail="Invalid video path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = candidate.stat().st_size
    content_type = detect_video_content_type(candidate)
    parsed = parse_range_header(request.headers.get("range"), file_size)

    if parsed == "unsatisfiable":
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    if parsed is None:
        return FileResponse(
            candidate,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes"},
        )

    start, end = parsed
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": content_type,
    }
    return StreamingResponse(
        iter_file_bytes(candidate, start, end),
        status_code=206,
        media_type=content_type,
        headers=headers,
    )


@app.post("/feedback", dependencies=[Depends(verify_api_key)])
def feedback_endpoint(req: FeedbackRequest):
    if req.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be -1 or 1")
    log_feedback(
        question=req.question,
        answer=req.answer,
        rating=req.rating,
        trace_id=req.trace_id,
        comment=req.comment,
    )
    return {"status": "ok"}





def _fetch_vendors_from_qdrant() -> dict:
    store = get_vector_store()
    client = store.client
    collection = store.collection
    vendor_products: dict[str, set[str]] = defaultdict(set)
    offset = None
    scroll_limit = int(os.getenv("QDRANT_SCROLL_PAGE_SIZE", "500"))

    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=scroll_limit,
            offset=offset,
            with_payload=["vendor", "product"],
            with_vectors=False,
        )
        if not records:
            break
        for record in records:
            payload = record.payload or {}
            vendor = str(payload.get("vendor") or "").strip().lower()
            if not vendor:
                continue
            product = str(payload.get("product") or "").strip().lower()
            if product:
                vendor_products[vendor].add(product)
            else:
                vendor_products[vendor]
        if offset is None:
            break

    vendors = [
        {
            "name": name,
            "products": sorted(vendor_products[name]),
        }
        for name in sorted(vendor_products)
    ]
    return {"vendors": vendors}


@app.get("/vendors", dependencies=[Depends(verify_api_key)])
async def vendors_endpoint(refresh: bool = False):
    """Distinct vendor/product pairs from Qdrant payloads (cached 5 minutes)."""
    global _vendors_cache, _vendors_cache_at

    now = time.time()
    if (
        not refresh
        and _vendors_cache is not None
        and now - _vendors_cache_at < VENDORS_CACHE_TTL_SECONDS
    ):
        return _vendors_cache

    try:
        data = await asyncio.to_thread(_fetch_vendors_from_qdrant)
    except Exception as exc:
        logger.exception("Failed to fetch vendors from Qdrant")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _vendors_cache = data
    _vendors_cache_at = now
    return data


@app.get("/documents", dependencies=[Depends(verify_api_key)])
def list_documents_endpoint():



    try:



        registry = get_doc_registry()
        documents = registry.list_documents()
        if not documents:
            documents = get_vector_store().list_sources()

        return {"documents": documents}



    except ValueError as e:



        raise HTTPException(status_code=404, detail=str(e))



    except Exception as e:



        raise HTTPException(status_code=500, detail=str(e))











@app.post("/query", dependencies=[Depends(verify_api_key)])
async def query_endpoint(req: QueryRequest, request: Request):
    """Stream a RAG answer as Server-Sent Events (tokens, citations, meta)."""
    history = [msg.model_dump() for msg in req.history]

    _STREAM_END = object()

    def _pull_next_event(iterator):
        try:
            return next(iterator)
        except StopIteration:
            return _STREAM_END

    cancel_event = threading.Event()

    async def event_stream():
        try:
            iterator = stream_query(
                req.question,
                top_k=req.top_k,
                history=history,
                vendor=req.vendor,
                product=req.product,
                vendor_filter=req.vendor_filter,
                document_type_filter=req.document_type_filter,
                cancel_event=cancel_event,
            )
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected; stopping query stream")
                    cancel_event.set()
                    break
                event = await asyncio.to_thread(_pull_next_event, iterator)
                if event is _STREAM_END:
                    break
                public_event = {
                    k: v for k, v in event.items() if not k.startswith("_")
                }
                yield f"data: {json.dumps(public_event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )











@app.post("/ingest", dependencies=[Depends(verify_api_key)])
def ingest_endpoint(req: IngestRequest):



    try:



        ingest(req.file_path)



        return {"status": "ingested", "file": req.file_path}



    except Exception as e:



        raise HTTPException(status_code=500, detail=str(e))











@app.delete("/ingest", dependencies=[Depends(verify_api_key)])
def delete_ingest_endpoint(req: IngestRequest):



    try:



        store = get_vector_store()
        deleted = 0
        matched_source = req.file_path

        for candidate in _source_path_candidates(req.file_path):
            deleted = store.delete_by_source(candidate)
            if deleted > 0:
                matched_source = candidate
                break

        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No ingested chunks found for source: {req.file_path}",
            )

        get_doc_registry().delete(matched_source)

        return {"deleted": deleted, "file_path": matched_source}



    except HTTPException:



        raise



    except ValueError as e:



        raise HTTPException(status_code=404, detail=str(e))



    except Exception as e:



        raise HTTPException(status_code=500, detail=str(e))





@app.get("/ingest/status/{job_id}", dependencies=[Depends(verify_api_key)])
def ingest_status_endpoint(job_id: str):
    store = _job_store()
    store.prune()
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_snapshot(job)


@app.post("/transcribe", dependencies=[Depends(verify_api_key)])
async def transcribe_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    suffix = _transcribe_upload_suffix(file.filename, file.content_type)
    if not suffix:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Use webm, ogg, or wav.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        text = await asyncio.to_thread(transcribe_plain_text, tmp_path)
        text = (text or "").strip()
        if not text:
            raise HTTPException(
                status_code=400,
                detail="No speech detected in audio",
            )
        return {"text": normalize_transcript_text(text)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed for %s", file.filename)
        raise HTTPException(
            status_code=400,
            detail=f"Could not transcribe audio: {exc}",
        ) from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Failed to remove temp transcription file %s", tmp_path)


# --- LLM provider selection (runtime, per-process) -------------------------

LLM_PROVIDER_MODEL_ENV: dict[str, str] = {
    "ollama": "OLLAMA_LLM_MODEL",
    "claude_subscription": "CLAUDE_SUBSCRIPTION_MODEL",
    "claude_cli": "CLAUDE_CLI_MODEL",  # legacy CLI-based path, not shown in UI
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
    "gemini": "GEMINI_MODEL",
}


LLM_API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

_CLAUDE_MODEL_SUGGESTIONS = ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]

_STATIC_MODEL_SUGGESTIONS: dict[str, list[str]] = {
    "claude_subscription": _CLAUDE_MODEL_SUGGESTIONS,
    "claude_cli": _CLAUDE_MODEL_SUGGESTIONS,
    "anthropic": _CLAUDE_MODEL_SUGGESTIONS,
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
}


def _ollama_probe() -> tuple[bool, str, list[str]]:
    """Reachability + installed model tags for the local Ollama server."""
    import httpx

    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        version = httpx.get(f"{base}/api/version", timeout=1.5).json().get("version", "?")
        tags = httpx.get(f"{base}/api/tags", timeout=1.5).json().get("models") or []
        models = [m.get("name", "") for m in tags if m.get("name")]
        return True, f"Ollama {version} at {base}", models
    except Exception as exc:
        return False, f"Ollama not reachable at {base} ({exc.__class__.__name__})", []


def _api_key_set(provider: str) -> bool:
    return any(os.getenv(env) for env in LLM_API_KEY_ENVS.get(provider, ()))


# --- Persisted provider selection ------------------------------------------
# The runtime switch below sets LLM_PROVIDER in-process; on its own that resets
# to the .env default on every restart, forcing the user to re-pick their
# provider (e.g. Claude subscription) each run even though the credential is
# already stored. We persist the last successful selection to a small JSON file
# under data/ and restore it on startup, so a connection survives restarts.


def _llm_pref_path() -> Path:
    configured = os.getenv("LLM_PROVIDER_STATE_PATH", "").strip()
    if configured:
        return Path(configured)
    return DATA_DIR / "llm_provider.json"


def _load_llm_pref() -> dict | None:
    try:
        record = json.loads(_llm_pref_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _save_llm_pref(provider: str, model: str | None) -> None:
    record: dict[str, str] = {"provider": provider}
    if model:
        record["model"] = model
    path = _llm_pref_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        logger.warning("Could not persist LLM provider preference to %s", path)


def _clear_llm_pref() -> None:
    try:
        _llm_pref_path().unlink()
    except OSError:
        pass


# --- Persisted API keys ----------------------------------------------------
# LLM API keys (Claude/OpenAI/Gemini) can be entered in Settings and saved
# locally so they survive restarts without living in .env. Stored provider ->
# key in the per-user app-data dir (%APPDATA%\rag-agent or ~/.rag-agent, see
# providers/app_paths.py); loaded into the environment on startup.


def _llm_keys_path() -> Path:
    configured = os.getenv("LLM_API_KEYS_PATH", "").strip()
    if configured:
        return Path(configured)
    from providers.app_paths import secret_file

    return secret_file("llm_api_keys.json", legacy_path=DATA_DIR / "llm_api_keys.json")


def _load_api_keys() -> dict:
    try:
        record = json.loads(_llm_keys_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _save_api_key(provider: str, key: str) -> None:
    keys = _load_api_keys()
    keys[provider] = key
    path = _llm_keys_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(keys), encoding="utf-8")
        try:  # best-effort owner-only perms on POSIX (no-op on Windows)
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        logger.warning("Could not persist LLM API key to %s", path)


def _apply_persisted_api_keys() -> None:
    """Load locally-saved API keys into the environment at startup so the
    matching providers work without the key being present in .env. A key already
    set in the environment (e.g. from .env) wins and is left untouched."""
    keys = _load_api_keys()
    for provider, key in keys.items():
        if not key:
            continue
        envs = LLM_API_KEY_ENVS.get(str(provider).strip().lower())
        if envs and not any(os.getenv(env) for env in envs):
            os.environ[envs[0]] = str(key)


def _apply_persisted_llm_provider() -> None:
    """Restore the last-selected provider at startup so a stored subscription /
    API-key connection is used automatically without reconnecting each run.

    A credential-backed provider is only restored when its credential is still
    present, so a stale preference can never wedge the app into a provider that
    would fail on every query.
    """
    pref = _load_llm_pref()
    if not pref:
        return
    provider = str(pref.get("provider", "")).strip().lower()
    if provider not in LLM_PROVIDER_MODEL_ENV:
        return
    if provider == "claude_subscription" and not claude_oauth.is_signed_in():
        logger.info("Persisted provider %s not restored: not signed in", provider)
        return
    if provider in LLM_API_KEY_ENVS and not _api_key_set(provider):
        logger.info("Persisted provider %s not restored: no API key set", provider)
        return

    os.environ["LLM_PROVIDER"] = provider
    model = str(pref.get("model", "")).strip()
    if model:
        os.environ[LLM_PROVIDER_MODEL_ENV[provider]] = model
    provider_factory.reset_providers()
    logger.info("Restored persisted LLM provider: %s", provider)


def _llm_status_payload() -> dict:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model_env = LLM_PROVIDER_MODEL_ENV.get(provider)

    ollama_ok, ollama_detail, ollama_models = _ollama_probe()
    signed_in = claude_oauth.is_signed_in()
    claude_detail = (
        "Signed in — answers use your Claude subscription"
        if signed_in
        else "Not signed in — click Connect to sign in with your subscription"
    )

    providers: dict[str, dict] = {
        "ollama": {
            "connected": ollama_ok,
            "detail": ollama_detail,
            "models": ollama_models,
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
        "claude_subscription": {
            "connected": signed_in,
            "detail": claude_detail,
            "signed_in": signed_in,
            "models": _STATIC_MODEL_SUGGESTIONS["claude_subscription"],
        },
    }
    for api_provider in ("anthropic", "openai", "gemini"):
        key_set = _api_key_set(api_provider)
        providers[api_provider] = {
            "connected": key_set,
            "detail": (
                "API key configured (verified on first query)"
                if key_set
                else "No API key set — add it to .env"
            ),
            "api_key_set": key_set,
            "models": _STATIC_MODEL_SUGGESTIONS[api_provider],
        }

    active = providers.get(provider, {})
    return {
        "provider": provider,
        "model": os.getenv(model_env, "") if model_env else "",
        "providers": providers,
        "connection": {
            "connected": bool(active.get("connected")),
            "detail": active.get("detail", ""),
        },
        "note": "Selection is remembered across restarts; .env LLM_PROVIDER is the initial default.",
    }


@app.get("/llm/status", dependencies=[Depends(verify_api_key)])
def llm_status_endpoint():
    return _llm_status_payload()


class LLMProviderRequest(BaseModel):
    provider: str
    model: str | None = None
    api_key: str | None = None


@app.post("/llm/provider", dependencies=[Depends(verify_api_key)])
def set_llm_provider_endpoint(req: LLMProviderRequest):
    provider = req.provider.strip().lower()
    if provider not in LLM_PROVIDER_MODEL_ENV:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider {provider!r}. "
            f"Choose one of: {', '.join(sorted(LLM_PROVIDER_MODEL_ENV))}",
        )

    previous_provider = os.getenv("LLM_PROVIDER", "ollama")
    model_env = LLM_PROVIDER_MODEL_ENV[provider]
    previous_model = os.getenv(model_env)

    # A freshly-entered API key is applied to the environment so eager
    # construction below validates against it; the previous value is captured
    # for rollback on failure.
    key_env = LLM_API_KEY_ENVS.get(provider, ())[:1]
    new_key = (req.api_key or "").strip()
    previous_key = os.getenv(key_env[0]) if key_env else None
    if key_env and new_key:
        os.environ[key_env[0]] = new_key

    # API-key providers need a key (the constructor only checks it on the first
    # request), so reject up front rather than "connecting" into a broken state.
    if provider in LLM_API_KEY_ENVS and not _api_key_set(provider):
        raise HTTPException(
            status_code=400,
            detail=f"An API key is required for {provider}. Enter it in Settings.",
        )

    os.environ["LLM_PROVIDER"] = provider
    if req.model and req.model.strip():
        os.environ[model_env] = req.model.strip()
    provider_factory.reset_providers()
    try:
        # Construct eagerly so misconfiguration (e.g. claude CLI missing)
        # fails here instead of on the next user query.
        provider_factory.get_llm()
    except Exception as exc:
        os.environ["LLM_PROVIDER"] = previous_provider
        if previous_model is None:
            os.environ.pop(model_env, None)
        else:
            os.environ[model_env] = previous_model
        if key_env and new_key:
            if previous_key is None:
                os.environ.pop(key_env[0], None)
            else:
                os.environ[key_env[0]] = previous_key
        provider_factory.reset_providers()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Persist the key locally so it survives restarts (no re-entry needed).
    if key_env and new_key:
        _save_api_key(provider, new_key)

    # Persist so the selection is restored on the next restart (no reconnect).
    _save_llm_pref(provider, os.getenv(model_env))
    logger.info("LLM provider switched: %s -> %s", previous_provider, provider)
    return _llm_status_payload()


class LLMDisconnectRequest(BaseModel):
    provider: str


@app.post("/llm/disconnect", dependencies=[Depends(verify_api_key)])
def llm_disconnect_endpoint(req: LLMDisconnectRequest):
    """Forget the credential for a provider (Forgestation-style disconnect)."""
    provider = req.provider.strip().lower()
    if provider not in LLM_PROVIDER_MODEL_ENV:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider!r}")

    was_active = os.getenv("LLM_PROVIDER", "ollama") == provider

    if provider == "ollama":
        detail = "Local provider — nothing to disconnect."
    elif provider == "claude_subscription":
        # ForgeStation-style: disconnect only deactivates the model. The stored
        # subscription token is kept so reconnecting needs no re-sign-in.
        if claude_oauth.is_signed_in():
            detail = "Disconnected — your Claude subscription stays saved; reconnect anytime without signing in again."
        else:
            detail = "Already disconnected."
    elif provider == "claude_cli":
        path = claude_oauth.credentials_path()
        if path.is_file():
            backup = path.with_name(path.name + ".bak")
            backup.unlink(missing_ok=True)
            path.replace(backup)
            detail = (
                "Signed out — this also signs the claude CLI out everywhere "
                "(credentials kept in .credentials.json.bak)."
            )
        else:
            detail = "Already signed out."
    else:
        # ForgeStation-style: keep the saved API key so reconnect is one click.
        detail = "Disconnected — your saved API key is kept; reconnect anytime."

    # Deactivate the model too: if the disconnected provider was the active one,
    # fall back to the local default so the app isn't left pointing at a provider
    # whose credential we just removed (queries would otherwise error).
    if was_active and provider != "ollama":
        os.environ["LLM_PROVIDER"] = "ollama"

    # Drop the saved selection so a disconnected provider is not auto-restored
    # on the next restart.
    pref = _load_llm_pref()
    if pref and str(pref.get("provider", "")).strip().lower() == provider:
        _clear_llm_pref()

    provider_factory.reset_providers()
    return {**_llm_status_payload(), "disconnect_detail": detail}


@app.post("/llm/claude/oauth/start", dependencies=[Depends(verify_api_key)])
def claude_oauth_start_endpoint():
    """Begin the in-browser Claude subscription sign-in; returns the authorize URL."""
    if claude_oauth.is_signed_in():
        return {"signed_in": True}
    return {"signed_in": False, **claude_oauth.start_login()}


class ClaudeOAuthFinishRequest(BaseModel):
    code: str


@app.post("/llm/claude/oauth/finish", dependencies=[Depends(verify_api_key)])
def claude_oauth_finish_endpoint(req: ClaudeOAuthFinishRequest):
    """Exchange the pasted authorization code and store CLI credentials."""
    try:
        result = claude_oauth.finish_login(req.code)
    except claude_oauth.ClaudeOAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_endpoint(
    file: UploadFile = File(...),
    vendor: str | None = Form(None),
    document_type: str | None = Form(None),
    product_line: str | None = Form(None),
    software_version: str | None = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()

    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Only .pdf and .txt files are accepted",
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / safe_name

    try:
        content = await file.read()
        dest.write_bytes(content)

        ingest_path = str(dest.relative_to(ROOT)).replace("\\", "/")
        job_id = str(uuid.uuid4())
        now = time.time()
        queue = get_ingest_queue()
        use_queue = is_ingest_queue_enabled() and queue.available
        job = {
            "job_id": job_id,
            "filename": safe_name,
            "status": "queued" if use_queue else "ingesting",
            "stage": "queued" if use_queue else "reading",
            "chunks_total": 0,
            "chunks_done": 0,
            "started_at": now,
            "elapsed_seconds": 0,
            "estimated_remaining_seconds": 0,
        }
        _job_store().prune()
        _job_store().create(job)

        if use_queue:
            queue.enqueue(
                {
                    "job_id": job_id,
                    "ingest_path": ingest_path,
                    "vendor": vendor,
                    "document_type": document_type,
                    "product_line": product_line,
                    "software_version": software_version,
                }
            )
        else:
            asyncio.create_task(
                asyncio.to_thread(
                    _run_ingest_job,
                    job_id,
                    ingest_path,
                    vendor=vendor,
                    document_type=document_type,
                    product_line=product_line,
                    software_version=software_version,
                )
            )

        return {
            "job_id": job_id,
            "filename": safe_name,
            "queued": use_queue,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_UI_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/")
@app.get("/index.html")
def serve_chat_ui():
    """Serve chat UI with no-cache headers so citation/video UI updates apply immediately."""
    index_path = UI_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(
        index_path,
        media_type="text/html",
        headers=_UI_CACHE_HEADERS,
    )


@app.get("/ui-version")
def ui_version():
    """Lightweight build marker for verifying the browser loaded the current UI."""
    index_path = UI_DIR / "index.html"
    text = index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
    marker = 'RAG_UI_BUILD = "'
    start = text.find(marker)
    if start != -1:
        start += len(marker)
        end = text.find('"', start)
        if end != -1:
            return {"ui_build": text[start:end]}
    return {"ui_build": "unknown"}















if __name__ == "__main__":



    import uvicorn







    uvicorn.run(app, host="0.0.0.0", port=8000)




