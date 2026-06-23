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




