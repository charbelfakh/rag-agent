"""RAG query pipeline: retrieval orchestration, prompt build, and LLM generation.

Exposes ``stream_query()`` for SSE and ``query()`` for synchronous answers.
"""
import contextvars
import json
import logging
import os
import re
import threading
import time
import uuid
from urllib.parse import quote
from collections.abc import Iterator
from queue import Empty, Queue
from dataclasses import dataclass, field
from pathlib import Path

import providers.query_orchestrator as _query_orchestrator  # patched below; avoids import cycle

from providers.factory import get_fast_llm, get_llm
from providers.semantic_cache import get_semantic_cache
from providers.searxng_search import web_search

logger = logging.getLogger(__name__)

KNOWN_VENDORS = (
    "photoneo",
    "mechmind",
    "basler",
    "pekat",
    "zivid",
    "lmi",
)

RAG_SCORE_THRESHOLD = 0.25
TEMPORAL_KEYWORDS = ("latest version", "newest version", "recent update", "released in")
DEFINITIONAL_PATTERN = re.compile(
    r"\b(what is|what are|what's|define|explain|describe|tell me about)\b",
    re.IGNORECASE,
)


def is_early_web_fallback_enabled() -> bool:
    return os.getenv("EARLY_WEB_FALLBACK_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def _early_web_score_threshold() -> float:
    return float(os.getenv("EARLY_WEB_SCORE_THRESHOLD", str(RAG_SCORE_THRESHOLD)))


def _retrieval_confidence_score(
    chunks: list[dict],
    top_score: float,
    analytics: "QueryAnalytics | None" = None,
) -> float:
    """Score for web-fallback gating; prefers rerank over fused/RRF ranking scores."""
    if chunks:
        rerank = chunks[0].get("rerank_score")
        if rerank is not None:
            return float(rerank)
    if analytics is not None and analytics.top_rerank_score is not None:
        return analytics.top_rerank_score
    if chunks:
        vector = chunks[0].get("vector_score")
        if vector is not None:
            return float(vector)
    return float(top_score)


def score_indicates_early_web(
    chunks: list[dict],
    top_score: float,
    analytics: "QueryAnalytics",
) -> bool:
    """Score-only signal to skip local-only generation and use web upfront."""
    if not is_early_web_fallback_enabled() or not chunks:
        return False
    threshold = _early_web_score_threshold()
    confidence = _retrieval_confidence_score(chunks, top_score, analytics)
    if confidence < threshold:
        return True
    if chunks[0].get("rerank_score") is not None:
        return False
    if (
        analytics.top_vector_score is not None
        and analytics.top_vector_score < threshold
    ):
        return True
    mean_vector = analytics.mean_vector_score
    if mean_vector is not None and mean_vector < threshold * 0.75:
        return True
    return False


def needs_web_fallback(
    question: str,
    chunks: list[dict],
    top_score: float,
    analytics: "QueryAnalytics | None" = None,
) -> bool:
    """Return whether retrieval scores or temporal cues warrant a SearXNG lookup."""
    if _retrieval_confidence_score(chunks, top_score, analytics) < RAG_SCORE_THRESHOLD:
        return True

    context = " ".join(c.get("text", "") for c in chunks).lower()
    q = question.lower()

    years_in_question = re.findall(r"\b(20\d{2})\b", question)
    if years_in_question and not any(year in context for year in years_in_question):
        return True

    if any(keyword in q for keyword in TEMPORAL_KEYWORDS):
        recent_years = re.findall(r"\b(202[4-9]|203\d)\b", context)
        if not recent_years:
            return True

    return False


def is_insufficient_answer(answer: str) -> bool:
    normalized = answer.strip().lower()
    if "don't have enough information" in normalized:
        return True
    if "do not have enough information" in normalized:
        return True
    # Model restated the topic without explaining (e.g. a one-line label from industrial vision systems vendor documentation).
    words = normalized.rstrip(".").split()
    if len(words) <= 6 and len(normalized) < 80:
        return True
    return False


def _classify_answer_outcome(
    answer: str,
    *,
    generation_error: str | None,
    generation_timed_out: bool,
    chunks_in_prompt: int,
) -> str:
    if generation_timed_out:
        return "generation_timeout"
    if generation_error:
        return "generation_error"
    if not answer.strip():
        if chunks_in_prompt > 0:
            return "empty_generation"
        return "empty_retrieval"
    if is_insufficient_answer(answer):
        return "insufficient_content"
    return "ok"


def _mark_generation_exception(
    analytics: "QueryAnalytics | None",
    exc: BaseException,
) -> None:
    if analytics is None:
        return
    err = str(exc)
    analytics.generation_error = err
    lowered = err.lower()
    if "timed out" in lowered or "timeout" in lowered:
        analytics.generation_timed_out = True


def _mark_generation_timeout(analytics: "QueryAnalytics | None") -> None:
    if analytics is None:
        return
    analytics.generation_timed_out = True
    analytics.generation_error = "answer generation timed out"


def _log_generation_outcome(analytics: "QueryAnalytics", citations: list) -> None:
    outcome = analytics.answer_outcome
    if outcome == "empty_generation":
        logger.error(
            "Answer generation returned no text despite %d chunks in prompt "
            "(citations=%d, embed_ms=%s, pre_stream_ms=%s, llm_generation_ms=%s)",
            analytics.chunks_in_prompt,
            len(citations),
            analytics.embed_ms,
            analytics.pre_stream_ms,
            analytics.llm_generation_ms,
        )
    elif outcome in ("generation_timeout", "generation_error"):
        logger.error(
            "Answer generation failed: outcome=%s error=%r",
            outcome,
            analytics.generation_error,
        )


GENERATION_FAILURE_USER_MESSAGE = (
    "Answer generation timed out or failed while relevant sources were found. "
    "Please try again in a moment."
)


def _resolve_generation_answer(
    plan: "GenerationPlan",
    raw_answer: str,
    citations: list[dict],
) -> str:
    analytics = plan.analytics
    if analytics is None:
        return raw_answer

    analytics.answer_outcome = _classify_answer_outcome(
        raw_answer,
        generation_error=analytics.generation_error,
        generation_timed_out=analytics.generation_timed_out,
        chunks_in_prompt=analytics.chunks_in_prompt,
    )
    _log_generation_outcome(analytics, citations)

    if analytics.answer_outcome in (
        "empty_generation",
        "generation_timeout",
        "generation_error",
    ):
        if citations and not raw_answer.strip():
            return GENERATION_FAILURE_USER_MESSAGE
    return raw_answer


def _should_retry_insufficient(plan: "GenerationPlan", answer: str) -> bool:
    if plan.analytics and plan.analytics.answer_outcome in (
        "generation_timeout",
        "generation_error",
        "empty_generation",
    ):
        return False
    return is_insufficient_answer(answer)


SUFFICIENCY_PROMPT = """Do these document excerpts contain ANY information relevant to answering the question, even partially?
Question: {question}
Excerpts:
{excerpts}
Reply with only YES or NO. When in doubt, reply YES."""

HYDE_PROMPT = """You are an industrial automation expert. Write a short hypothetical passage (2-3 sentences maximum) that would answer the question below, as if it appeared in a technical manual for the relevant product or vendor. Reply with only the passage — no preamble, labels, or explanation.

Question: {question}"""

CONDENSATION_PROMPT = """Given the conversation below, rewrite the user's last question as a single, fully self-contained question. Replace pronouns and vague references (it, they, that, this, them) with the specific subject from the conversation. Keep it short — one sentence. Do not answer the question. Output ONLY the rewritten question, nothing else.

Conversation:
{conversation}

Last question: {question}

Rewritten question:"""

CONDENSATION_MAX_CHARS = 300


def _condensation_timeout_seconds() -> float:
    return float(os.getenv("CONDENSATION_TIMEOUT_SECONDS", "10"))


def _answer_generation_timeout_seconds() -> float:
    return float(os.getenv("ANSWER_GENERATION_TIMEOUT_SECONDS", "120"))


def is_hyde_enabled() -> bool:
    return os.getenv("HYDE_ENABLED", "false").lower() in ("true", "1", "yes")


def is_sufficiency_check_enabled() -> bool:
    return os.getenv("SUFFICIENCY_CHECK_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _chunk_file_name(chunk: dict) -> str:
    return chunk.get("file_name") or Path(chunk.get("source", "")).name


def _chunk_page_index(chunk: dict) -> int | None:
    """0-based page index, or None for non-paginated sources (HTML, etc.)."""
    page = chunk.get("page")
    if page is None:
        return None
    return int(page)


def format_chunk_header(chunk: dict) -> str:
    """Human-readable source line for prompt grounding."""
    file_name = _chunk_file_name(chunk)
    content_type = chunk.get("content_type", "text") or "text"
    start_seconds = chunk.get("start_seconds")
    if content_type in ("video_transcript", "video_frame") and start_seconds is not None:
        return f"[{file_name} t={float(start_seconds):.1f}s]"
    page = _chunk_page_index(chunk)
    section = (chunk.get("section") or "").strip()
    if page is None:
        if section:
            return f"[{file_name} §{section}]"
        return f"[{file_name}]"
    page_display = page + 1
    if section:
        return f"[{file_name} p.{page_display} §{section}]"
    return f"[{file_name} p.{page_display}]"


def _max_prompt_tokens() -> int:
    return int(os.getenv("MAX_PROMPT_TOKENS", "12000"))


def _chunk_prompt_max_chars() -> int:
    return int(os.getenv("CHUNK_PROMPT_MAX_CHARS", "1000"))


def format_chunk_for_prompt(chunk: dict) -> str:
    text = chunk.get("text", "")
    max_chars = _chunk_prompt_max_chars()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "…"
    header = format_chunk_header(chunk)
    return f"{header}\n{text}" if header else text


def _prompt_overhead_tokens(
    question: str,
    history: list[dict] | None,
    web_results: list[dict] | None,
) -> int:
    overhead = estimate_prompt_tokens(question) + 500
    if history:
        overhead += estimate_prompt_tokens(format_history_block(history))
    if web_results:
        overhead += estimate_prompt_tokens(format_web_results(web_results)) + 200
    return overhead


def select_chunks_for_prompt(
    chunks: list[dict],
    question: str,
    history: list[dict] | None = None,
    web_results: list[dict] | None = None,
) -> list[dict]:
    """Greedy pack chunks by rerank order until ``MAX_PROMPT_TOKENS`` budget."""
    if not chunks:
        return []

    max_tokens = _max_prompt_tokens()
    if max_tokens <= 0:
        return chunks

    budget = max_tokens - _prompt_overhead_tokens(question, history, web_results)
    if budget <= 0:
        return chunks[:1]

    selected: list[dict] = []
    used = 0
    for chunk in chunks:
        cost = estimate_prompt_tokens(format_chunk_for_prompt(chunk)) + 2
        if selected and used + cost > budget:
            break
        selected.append(chunk)
        used += cost
    return selected or chunks[:1]


def _is_youtube_url(url: str) -> bool:
    lowered = url.lower()
    return (
        "youtube.com/watch" in lowered
        or "youtu.be/" in lowered
        or "youtube.com/embed/" in lowered
        or "youtube.com/shorts/" in lowered
    )


def _youtube_url_with_timestamp(url: str, start_seconds: float | None) -> str:
    if start_seconds is None:
        return url
    t = max(0, int(round(float(start_seconds))))
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={t}s"


def _parse_youtube_video_id(url: str) -> str | None:
    if not url:
        return None
    text = url.strip()
    match = re.search(
        r"(?:youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{6,})",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(r"[?&]v=([a-zA-Z0-9_-]{6,})", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _attach_video_citation_fields(
    citation: dict,
    chunk: dict,
    content_type: str,
) -> None:
    """Attach local/YouTube playback fields for transcript and YouTube-backed chunks."""
    is_video = content_type in ("video_transcript", "video_frame")
    video_path = chunk.get("video_path")
    chunk_url = (chunk.get("url") or "").strip()
    video_id = chunk.get("video_id")
    if not chunk_url and video_id:
        chunk_url = f"https://www.youtube.com/watch?v={video_id}"
    has_youtube = bool(chunk_url and _is_youtube_url(chunk_url))

    if not is_video and not video_path and not has_youtube:
        return

    if has_youtube or is_video:
        citation["content_type"] = content_type if is_video else "video_transcript"

    if video_path:
        citation["video_path"] = video_path
        citation["video_url"] = f"/video?path={quote(video_path, safe='/')}"
    elif has_youtube:
        start_seconds = chunk.get("start_seconds")
        citation["youtube_url"] = _youtube_url_with_timestamp(
            chunk_url,
            float(start_seconds) if start_seconds is not None else None,
        )
        resolved_id = video_id or _parse_youtube_video_id(chunk_url)
        if resolved_id:
            citation["video_id"] = str(resolved_id)
        title = (chunk.get("source") or "").strip()
        if title:
            citation["source"] = title

    if chunk.get("start_seconds") is not None:
        citation["start_seconds"] = float(chunk["start_seconds"])
    transcript = chunk.get("text")
    if transcript and is_video:
        citation["text"] = transcript


def collect_citations(chunks: list[dict]) -> list[dict]:
    """Build UI citation dicts from prompt chunks, including video playback fields."""
    seen: set[tuple] = set()
    citations: list[dict] = []
    for chunk in chunks:
        source = chunk.get("source", "")
        content_type = chunk.get("content_type", "text") or "text"
        if content_type in ("video_transcript", "video_frame"):
            start_seconds = chunk.get("start_seconds")
            key = (
                source,
                round(float(start_seconds), 3) if start_seconds is not None else -1.0,
                content_type,
            )
        else:
            page = _chunk_page_index(chunk)
            section = chunk.get("section", "") or ""
            key = (source, page, section)
        if key in seen:
            continue
        seen.add(key)
        page = _chunk_page_index(chunk)
        section = chunk.get("section", "") or ""
        citation = {
            "source": source,
            "file_name": _chunk_file_name(chunk),
            "vendor": chunk.get("vendor", "") or "",
            "page": page,
            "section": section,
            "chunk_index": chunk.get("chunk_index"),
            "total_chunks": chunk.get("total_chunks"),
            "document_type": chunk.get("document_type") or chunk.get("doc_type", "") or "",
            "content_type": chunk.get("content_type", "text") or "text",
        }
        if chunk.get("media_uri"):
            citation["media_uri"] = chunk["media_uri"]
            thumb = chunk.get("thumbnail_uri")
            if thumb:
                citation["thumbnail_uri"] = thumb
            elif citation["media_uri"]:
                citation["thumbnail_uri"] = citation["media_uri"]
        if chunk.get("ocr_text"):
            citation["ocr_preview"] = chunk["ocr_text"][:120]
        if chunk.get("image_class"):
            citation["image_class"] = chunk["image_class"]
        image_path = chunk.get("image_path")
        if citation.get("content_type") == "image_caption" and image_path:
            citation["image_path"] = image_path
            citation["image_url"] = f"/image?path={quote(image_path, safe='/')}"
        _attach_video_citation_fields(citation, chunk, content_type)
        citations.append(citation)
    citations.sort(
        key=lambda c: (
            0
            if c.get("youtube_url")
            and c.get("content_type") in ("video_transcript", "video_frame")
            else 1
        )
    )
    return citations


def _rerank_score_floor() -> float:
    return float(os.getenv("RERANK_SCORE_FLOOR", "0"))


def filter_chunks_by_score_floor(chunks: list[dict]) -> list[dict]:
    floor = _rerank_score_floor()
    if floor <= 0 or not chunks:
        return chunks
    kept = [c for c in chunks if float(c.get("score", 0.0)) >= floor]
    return kept if kept else chunks[:1]


def check_context_sufficient(
    question: str,
    chunks: list[dict],
    llm,
) -> bool | None:
    """Ask the LLM whether top chunks can answer the question; ``None`` on failure."""
    excerpts = "\n\n---\n\n".join(
        chunk.get("text", "") for chunk in chunks[:3]
    )
    prompt = SUFFICIENCY_PROMPT.format(question=question, excerpts=excerpts)
    try:
        response = llm.generate(prompt).strip().upper()
        if response.startswith("NO") or response == "NO":
            return False
        if response.startswith("YES") or response == "YES":
            return True
        if "NO" in response and "YES" not in response:
            return False
        if "YES" in response:
            return True
        return True
    except Exception as exc:
        logger.warning("Sufficiency check failed: %s", exc)
        return None


def generate_hypothetical_document(question: str, llm) -> str:
    prompt = HYDE_PROMPT.format(question=question)
    return llm.generate(prompt).strip()


def _is_reranker_enabled() -> bool:
    return os.getenv("RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")


def _retrieval_top_k(request_top_k: int) -> int:
    from providers.query_orchestrator import is_two_stage_retrieval_enabled

    if is_two_stage_retrieval_enabled():
        return int(os.getenv("BI_ENCODER_TOP_K", "50"))
    if _is_reranker_enabled():
        return int(os.getenv("RERANKER_FETCH_K", "50"))
    return request_top_k


def get_search_text(question: str, llm) -> tuple[str, bool]:
    # HyDE hurts definitional queries — embed the question directly instead.
    if is_hyde_enabled() and not DEFINITIONAL_PATTERN.search(question):
        return generate_hypothetical_document(question, llm), True
    return question, False


def estimate_prompt_tokens(prompt: str) -> int:
    """Rough token estimate (~4 chars/token) for English technical prose."""
    if not prompt:
        return 0
    return max(1, len(prompt) // 4)


def score_stats(chunks: list[dict], field: str = "score") -> tuple[float | None, float | None]:
    if not chunks:
        return None, None
    scores = [float(chunk.get(field, 0.0)) for chunk in chunks]
    return scores[0], sum(scores) / len(scores)


@dataclass
class QueryAnalytics:
    """Per-query timing, retrieval, and outcome fields for logging and SSE meta."""

    question: str
    condensed_question: str | None = None
    history_turns: int = 0
    hyde_used: bool = False
    cache_hit: bool = False
    web_fallback_used: bool = False
    web_fallback_reason: str | None = None
    sufficiency_check_result: str = "skipped"
    chunks_retrieved: int | None = None
    chunks_after_rerank: int | None = None
    chunks_in_prompt: int = 0
    top_chunk_score: float | None = None
    top_chunk_source: str | None = None
    top_chunk_page: int | None = None
    top_chunk_vendor: str | None = None
    top_chunk_document_type: str | None = None
    top_vector_score: float | None = None
    mean_vector_score: float | None = None
    top_rerank_score: float | None = None
    mean_rerank_score: float | None = None
    embed_ms: int | None = None
    hyde_ms: int | None = None
    qdrant_search_ms: int | None = None
    rerank_ms: int | None = None
    prompt_token_count: int | None = None
    prompt_eval_count: int | None = None
    answer_token_count: int | None = None
    citations_count: int = 0
    insufficient_answer: bool = False
    answer_outcome: str = "ok"
    generation_error: str | None = None
    generation_timed_out: bool = False
    empty_retrieval: bool = False
    pre_stream_ms: int | None = None
    llm_ttft_ms: int | None = None
    llm_generation_ms: int | None = None
    cache_skipped_reason: str | None = None
    filter_vendor: str | None = None
    filter_product: str | None = None
    filter_document_type: str | None = None
    filter_fallback: bool = False
    inferred_vendor_filter: bool = False
    filter_mechanism: str = "none"
    video_transcript_supplement: bool = False
    answer_length: int = 0
    latency_ms: int = 0
    trace_id: str = ""

    def to_meta(self) -> dict:
        meta = {
            "trace_id": self.trace_id or None,
            "cache_hit": self.cache_hit,
            "web_fallback_used": self.web_fallback_used,
            "latency_ms": self.latency_ms,
            "insufficient_answer": self.insufficient_answer,
            "answer_outcome": self.answer_outcome,
            "empty_retrieval": self.empty_retrieval,
            "citations_count": self.citations_count,
        }
        if self.generation_error:
            meta["generation_error"] = self.generation_error
        if not self.cache_hit:
            meta["chunks_retrieved"] = self.chunks_retrieved
            meta["chunks_after_rerank"] = self.chunks_after_rerank
            meta["chunks_in_prompt"] = self.chunks_in_prompt
            meta["top_vector_score"] = self.top_vector_score
            meta["mean_vector_score"] = self.mean_vector_score
            if self.top_rerank_score is not None:
                meta["top_rerank_score"] = self.top_rerank_score
                meta["mean_rerank_score"] = self.mean_rerank_score
            if self.embed_ms is not None:
                meta["embed_ms"] = self.embed_ms
            if self.hyde_ms is not None:
                meta["hyde_ms"] = self.hyde_ms
            if self.qdrant_search_ms is not None:
                meta["qdrant_search_ms"] = self.qdrant_search_ms
            if self.rerank_ms is not None:
                meta["rerank_ms"] = self.rerank_ms
            if self.prompt_token_count is not None:
                meta["prompt_token_count"] = self.prompt_token_count
            if self.pre_stream_ms is not None:
                meta["pre_stream_ms"] = self.pre_stream_ms
            if self.llm_ttft_ms is not None:
                meta["llm_ttft_ms"] = self.llm_ttft_ms
            if self.llm_generation_ms is not None:
                meta["llm_generation_ms"] = self.llm_generation_ms
            if self.prompt_eval_count is not None:
                meta["prompt_eval_count"] = self.prompt_eval_count
            if self.answer_token_count is not None:
                meta["answer_token_count"] = self.answer_token_count
            if self.cache_skipped_reason:
                meta["cache_skipped_reason"] = self.cache_skipped_reason
            if self.condensed_question:
                meta["condensed_question"] = self.condensed_question
            if self.filter_vendor:
                meta["filter_vendor"] = self.filter_vendor
            if self.filter_product:
                meta["filter_product"] = self.filter_product
            if self.filter_document_type:
                meta["filter_document_type"] = self.filter_document_type
            if self.filter_fallback:
                meta["filter_fallback"] = True
            if self.inferred_vendor_filter:
                meta["inferred_vendor_filter"] = True
            if self.filter_mechanism and self.filter_mechanism != "none":
                meta["filter_mechanism"] = self.filter_mechanism
            if self.video_transcript_supplement:
                meta["video_transcript_supplement"] = True
        return meta

    def finalize(
        self,
        answer: str,
        start_time: float,
        *,
        citations: list | None = None,
        prompt: str | None = None,
        raw_answer: str | None = None,
    ) -> dict:
        self.answer_length = len(answer)
        self.latency_ms = int((time.time() - start_time) * 1000)
        classify_from = raw_answer if raw_answer is not None else answer
        if self.answer_outcome == "ok":
            self.answer_outcome = _classify_answer_outcome(
                classify_from,
                generation_error=self.generation_error,
                generation_timed_out=self.generation_timed_out,
                chunks_in_prompt=self.chunks_in_prompt,
            )
        self.insufficient_answer = self.answer_outcome != "ok"
        if citations is not None:
            self.citations_count = len(citations)
        if prompt:
            self.prompt_token_count = estimate_prompt_tokens(prompt)
        try:
            from providers.otel_tracing import record_query_span

            record_query_span(
                question=self.question,
                latency_ms=self.latency_ms,
                cache_hit=self.cache_hit,
                chunks_retrieved=self.chunks_retrieved,
                trace_id=self.trace_id or None,
            )
        except Exception:
            pass
        log_query(
            question=self.question,
            condensed_question=self.condensed_question,
            trace_id=self.trace_id or None,
            hyde_used=self.hyde_used,
            cache_hit=self.cache_hit,
            web_fallback_used=self.web_fallback_used,
            web_fallback_reason=self.web_fallback_reason,
            sufficiency_check_result=self.sufficiency_check_result,
            chunks_retrieved=self.chunks_retrieved,
            chunks_after_rerank=self.chunks_after_rerank,
            chunks_in_prompt=self.chunks_in_prompt,
            top_chunk_score=self.top_chunk_score,
            top_chunk_source=self.top_chunk_source,
            top_chunk_page=self.top_chunk_page,
            top_chunk_vendor=self.top_chunk_vendor,
            top_chunk_document_type=self.top_chunk_document_type,
            top_vector_score=self.top_vector_score,
            mean_vector_score=self.mean_vector_score,
            top_rerank_score=self.top_rerank_score,
            mean_rerank_score=self.mean_rerank_score,
            embed_ms=self.embed_ms,
            hyde_ms=self.hyde_ms,
            qdrant_search_ms=self.qdrant_search_ms,
            rerank_ms=self.rerank_ms,
            prompt_token_count=self.prompt_token_count,
            pre_stream_ms=self.pre_stream_ms,
            llm_ttft_ms=self.llm_ttft_ms,
            llm_generation_ms=self.llm_generation_ms,
            prompt_eval_count=self.prompt_eval_count,
            answer_token_count=self.answer_token_count,
            cache_skipped_reason=self.cache_skipped_reason,
            citations_count=self.citations_count,
            insufficient_answer=self.insufficient_answer,
            answer_outcome=self.answer_outcome,
            generation_error=self.generation_error,
            empty_retrieval=self.empty_retrieval,
            answer_length=self.answer_length,
            latency_ms=self.latency_ms,
            history_turns=self.history_turns,
            filter_vendor=self.filter_vendor,
            filter_product=self.filter_product,
            filter_document_type=self.filter_document_type,
            filter_fallback=self.filter_fallback,
            inferred_vendor_filter=self.inferred_vendor_filter,
            filter_mechanism=self.filter_mechanism,
        )
        meta = self.to_meta()
        if citations:
            youtube_count = sum(
                1
                for c in citations
                if c.get("youtube_url") or c.get("video_id")
            )
            if youtube_count:
                meta["youtube_citations"] = youtube_count
        return meta


def log_query(*, condensed_question: str | None = None, **kwargs) -> None:
    """Schedule Langfuse analytics; includes optional condensed question metadata."""
    from providers.langfuse_logger import _is_analytics_enabled, _log_query_sync

    if not _is_analytics_enabled():
        return
    payload = dict(kwargs)
    if condensed_question:
        payload["condensed_question"] = condensed_question
    threading.Thread(target=_log_query_sync, args=(payload,), daemon=True).start()


def _populate_top_chunk(analytics: QueryAnalytics, chunks: list[dict]) -> None:
    if not chunks:
        return
    top = chunks[0]
    analytics.top_chunk_score = float(top.get("score", 0.0))
    analytics.top_chunk_source = top.get("source", "")
    page = top.get("page")
    analytics.top_chunk_page = int(page) if page is not None else 0
    analytics.top_chunk_vendor = top.get("vendor", "") or None
    analytics.top_chunk_document_type = top.get("document_type", "") or None
    if "rerank_score" in top:
        analytics.top_rerank_score, analytics.mean_rerank_score = score_stats(
            chunks, "rerank_score"
        )


def _truncate_history(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    max_messages = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
    valid = [
        msg
        for msg in history
        if msg.get("role") in ("user", "assistant") and msg.get("content")
    ]
    return _cap_history_chars(valid[-max_messages:])


def _cap_history_chars(turns: list[dict]) -> list[dict]:
    max_chars = int(os.getenv("HISTORY_MAX_CHARS", "4000"))
    if max_chars <= 0 or not turns:
        return turns

    selected: list[dict] = []
    total = 0
    for msg in reversed(turns):
        content = msg["content"]
        if total + len(content) <= max_chars:
            selected.append(msg)
            total += len(content)
            continue
        remaining = max_chars - total
        if remaining > 0:
            selected.append({**msg, "content": content[:remaining] + "…"})
        break
    return list(reversed(selected))


def format_history_block(history: list[dict] | None) -> str:
    turns = _truncate_history(history)
    if not turns:
        return ""
    lines = []
    for msg in turns:
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{label}: {msg['content']}")
    return "Conversation history:\n" + "\n".join(lines) + "\n\n"


def _format_condensation_conversation(history: list[dict] | None) -> str:
    turns = _truncate_history(history)
    lines = []
    for msg in turns:
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{label}: {msg['content']}")
    return "\n".join(lines)


def _llm_generate_with_timeout(llm, prompt: str, timeout: float) -> str | None:
    result: list[str] = []
    errors: list[Exception] = []

    def _target() -> None:
        try:
            result.append(llm.generate(prompt))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return None
    if errors:
        raise errors[0]
    return result[0] if result else None


def _generate_answer_sync(
    llm,
    prompt: str,
    analytics: "QueryAnalytics | None",
) -> str:
    try:
        raw = _llm_generate_with_timeout(
            llm, prompt, _answer_generation_timeout_seconds()
        )
        if raw is None:
            _mark_generation_timeout(analytics)
            logger.error("Answer generation timed out")
            return ""
        return raw
    except Exception as exc:
        _mark_generation_exception(analytics, exc)
        logger.error("Answer generation failed: %s", exc)
        return ""


def _sanitize_condensed_question(raw: str) -> str:
    text = raw.strip().splitlines()[0].strip()
    for prefix in ("Rewritten question:", "rewritten question:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    for prefix in ("Rewritten question:", "rewritten question:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return text


def _accept_condensed_question(original: str, condensed: str) -> bool:
    if not condensed:
        return False
    if len(condensed) > CONDENSATION_MAX_CHARS:
        return False
    if original and len(condensed) > 3 * len(original):
        return False
    if original.rstrip().endswith("?") and not condensed.rstrip().endswith("?"):
        return False
    return True


def condense_question_for_retrieval(
    question: str,
    history: list[dict] | None,
    llm,
) -> str:
    """Rewrite a follow-up into a standalone retrieval query; no-op without history."""
    if not _truncate_history(history):
        return question

    conversation = _format_condensation_conversation(history)
    prompt = CONDENSATION_PROMPT.format(
        conversation=conversation,
        question=question,
    )
    try:
        raw = _llm_generate_with_timeout(
            llm,
            prompt,
            _condensation_timeout_seconds(),
        )
        if raw is None:
            logger.warning(
                "Query condensation timed out; using original %r",
                question,
            )
            return question
        condensed = _sanitize_condensed_question(raw)
        if not _accept_condensed_question(question, condensed):
            logger.warning(
                "Query condensation rejected; using original %r (got %r)",
                question,
                condensed,
            )
            return question
        logger.info(
            "Query condensation: original=%r condensed=%r",
            question,
            condensed,
        )
        return condensed
    except Exception as exc:
        logger.warning(
            "Query condensation failed (%s); using original %r",
            exc,
            question,
        )
        return question


def _web_snippet_max_chars() -> int:
    return int(os.getenv("WEB_SNIPPET_MAX_CHARS", "1200"))


def compress_web_snippet(text: str, max_chars: int | None = None) -> str:
    """Extractive compression (~300 tokens) for web fallback prompts."""
    limit = max_chars if max_chars is not None else _web_snippet_max_chars()
    if limit <= 0 or len(text) <= limit:
        return text
    window = text[:limit]
    for sep in (". ", ".\n", "\n"):
        idx = window.rfind(sep)
        if idx > limit // 2:
            return window[: idx + 1].strip() + "…"
    return window.rstrip() + "…"


def compress_web_results(web_results: list[dict]) -> list[dict]:
    max_chars = _web_snippet_max_chars()
    if max_chars <= 0 or not web_results:
        return web_results
    return [
        {
            **result,
            "content": compress_web_snippet(result.get("content", ""), max_chars),
        }
        for result in web_results
    ]


def fetch_web_results(question: str) -> list[dict]:
    """Run SearXNG and compress snippets for the web-fallback prompt."""
    return compress_web_results(web_search(question))


def format_web_results(web_results: list[dict]) -> str:
    parts = []
    for result in web_results:
        parts.append(
            f"{result['title']}\n{result['url']}\n{result['content']}"
        )
    return "\n\n".join(parts)


def build_prompt(
    question: str,
    chunks: list[dict],
    web_results: list[dict] | None = None,
    history: list[dict] | None = None,
) -> str:
    """Assemble the grounded LLM prompt from chunks, optional web results, and history."""
    local_context = "\n\n---\n\n".join(
        format_chunk_for_prompt(chunk) for chunk in chunks
    )
    history_block = format_history_block(history)
    if web_results:
        web_context = format_web_results(web_results)
        return f"""You are an expert assistant for industrial vision systems including Pekat, Mechmind, Zivid, LMI, and similar vendors. Answer the question using the sources below.

Use all sources below to answer the question as completely as possible.

Web Search Results:
{web_context}

---

Local Documentation (reference only):
{local_context}

{history_block}Question: {question}

Answer using the sources above. If none contain enough information, say "I don't have enough information to answer that."

Answer:"""

    # Only mention video transcripts when the packed context actually contains
    # them — otherwise the model volunteers "no video transcripts were provided"
    # disclaimers in ordinary answers.
    has_video_context = any(
        (chunk.get("content_type") or "") in ("video_transcript", "video_frame")
        for chunk in chunks
    )
    if has_video_context:
        return f"""You are an industrial vision systems expert assistant. Answer the question using the context below from vendor documentation and video transcripts.
Excerpts marked with t=<seconds> are from video tutorials or demonstrations—summarize what those videos show and cite the video title and timestamp.
When the question asks for a demo, tutorial, or video, answer from matching transcript excerpts when present; do not refuse merely because the user asked for a demonstration.
If the answer is not in the context at all, say "I don't have enough information to answer that."

Context:
{local_context}

{history_block}Question: {question}

Answer:"""

    return f"""You are an industrial vision systems expert assistant. Answer the question using the context below from vendor documentation.
If the answer is not in the context at all, say "I don't have enough information to answer that."

Context:
{local_context}

{history_block}Question: {question}

Answer:"""


@dataclass
class GenerationPlan:
    """Mutable plan produced by ``_build_generation_plan`` for stream/sync paths."""

    prompt: str | None = None
    citations: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    immediate_answer: str | None = None
    allow_insufficient_retry: bool = False
    question_vector: list[float] | None = None
    use_cache: bool = False
    cached_answer: str | None = None
    retrieval_question: str | None = None
    analytics: QueryAnalytics | None = None


def _mark_web_fallback(analytics: QueryAnalytics, reason: str) -> None:
    analytics.web_fallback_used = True
    analytics.web_fallback_reason = reason


_filter_product: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "retrieval_product_filter",
    default=None,
)
_filter_vendors: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "retrieval_vendor_filters",
    default=None,
)


def _normalize_filter(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _search_filter(
    vendor_filter: str | None,
    document_type_filter: str | None,
    product_filter: str | None = None,
    *,
    vendors: list[str] | None = None,
) -> dict | None:
    payload: dict = {}
    if vendors:
        payload["vendors"] = vendors
    else:
        vendor = _normalize_filter(vendor_filter)
        if vendor:
            payload["vendor"] = vendor
    doc_type = _normalize_filter(document_type_filter)
    if doc_type:
        payload["document_type"] = doc_type
    product = _normalize_filter(product_filter)
    if product:
        payload["product"] = product
    return payload or None


def _orchestrator_search_filter(
    vendor_filter: str | None,
    document_type_filter: str | None,
) -> dict | None:
    vendors = _filter_vendors.get()
    if vendors:
        return _search_filter(
            None,
            document_type_filter,
            _filter_product.get(),
            vendors=vendors,
        )
    return _search_filter(
        vendor_filter,
        document_type_filter,
        _filter_product.get(),
    )


_query_orchestrator._search_filter = _orchestrator_search_filter  # reads contextvars set in _run_retrieval


def _effective_vendor(vendor: str | None, vendor_filter: str | None) -> str | None:
    return _normalize_filter(vendor) or _normalize_filter(vendor_filter)


def _normalize_for_vendor_match(text: str) -> str:
    """Lowercase alphanumeric-only form for hyphen/space tolerant brand matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _vendor_match_aliases() -> dict[str, list[str]]:
    """Vendor slug -> alias strings (from KNOWN_VENDORS + vendors.json channel names)."""
    aliases: dict[str, list[str]] = {vendor: [vendor] for vendor in KNOWN_VENDORS}
    from providers.config_paths import VENDORS_JSON

    vendors_path = VENDORS_JSON
    try:
        data = json.loads(vendors_path.read_text(encoding="utf-8"))
        channel_names = data.get("youtube_channel_names") or {}
        for vendor, names in channel_names.items():
            if vendor not in aliases or not isinstance(names, list):
                continue
            seen = {alias.lower() for alias in aliases[vendor]}
            for name in names:
                if isinstance(name, str) and name.strip():
                    seen.add(name.strip().lower())
            aliases[vendor] = sorted(seen)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return aliases


def _infer_vendors_from_text(*texts: str) -> list[str]:
    combined = " ".join(t for t in texts if t).lower()
    if not combined:
        return []
    normalized = _normalize_for_vendor_match(combined)
    found: list[str] = []
    seen: set[str] = set()
    for vendor in KNOWN_VENDORS:
        for alias in _vendor_match_aliases().get(vendor, [vendor]):
            norm_alias = _normalize_for_vendor_match(alias)
            if len(norm_alias) < 2:
                continue
            if norm_alias in normalized:
                if vendor not in seen:
                    seen.add(vendor)
                    found.append(vendor)
                break
    return found


def _explicit_filters_set(
    vendor: str | None,
    product: str | None,
    vendor_filter: str | None,
) -> bool:
    return bool(
        _normalize_filter(vendor)
        or _normalize_filter(product)
        or _normalize_filter(vendor_filter)
    )


def _resolve_retrieval_vendors(
    *,
    question: str,
    retrieval_question: str,
    history: list[dict] | None,
    vendor: str | None,
    product: str | None,
    vendor_filter: str | None,
) -> tuple[list[str] | None, bool, str]:
    if _explicit_filters_set(vendor, product, vendor_filter):
        explicit_vendor = _effective_vendor(vendor, vendor_filter)
        if explicit_vendor:
            return [explicit_vendor], False, "explicit"
        return None, False, "explicit"

    if _truncate_history(history):
        keyword_texts = [retrieval_question]
    else:
        keyword_texts = [retrieval_question or question]

    inferred = _infer_vendors_from_text(*keyword_texts)
    if len(inferred) > 1:
        return None, False, "none"
    if len(inferred) == 1:
        return inferred, True, "keyword"
    return None, False, "none"


def _skip_cache_for_filters(
    vendor: str | None,
    product: str | None,
    *,
    vendor_scope: list[str] | None = None,
) -> bool:
    return bool(
        _normalize_filter(vendor)
        or _normalize_filter(product)
        or vendor_scope
    )


def _run_retrieval(
    *,
    retrieval_question: str,
    top_k: int,
    history_turns: int,
    analytics: QueryAnalytics,
    vendor: str | None,
    product: str | None,
    vendor_filter: str | None,
    document_type_filter: str | None,
    question: str,
    history: list[dict] | None,
):
    # Lazy import: query_orchestrator imports rag_pipeline for dynamic top-N.
    from providers.query_orchestrator import QueryOrchestrator, RetrievalContext

    vendor_scope, inferred, filter_mechanism = _resolve_retrieval_vendors(
        question=question,
        retrieval_question=retrieval_question,
        history=history,
        vendor=vendor,
        product=product,
        vendor_filter=vendor_filter,
    )
    analytics.filter_mechanism = filter_mechanism
    effective_vendor = (
        vendor_scope[0] if vendor_scope and len(vendor_scope) == 1 else None
    )
    multi_vendors = vendor_scope if vendor_scope and len(vendor_scope) > 1 else None
    effective_product = _normalize_filter(product)
    effective_doc_type = _normalize_filter(document_type_filter)
    has_qdrant_filter = bool(
        vendor_scope or effective_product or effective_doc_type
    )
    skip_cache = _skip_cache_for_filters(
        vendor,
        product,
        vendor_scope=vendor_scope,
    )

    if has_qdrant_filter:
        analytics.filter_vendor = (
            ",".join(vendor_scope) if vendor_scope else None
        )
        analytics.filter_product = effective_product
        analytics.filter_document_type = effective_doc_type
        analytics.inferred_vendor_filter = inferred
        if inferred:
            logger.info(
                "Inferred retrieval vendors from question: %s",
                vendor_scope,
            )
        logger.info(
            "Retrieval filters active (mechanism=%s): vendor=%r product=%r document_type=%r",
            filter_mechanism,
            analytics.filter_vendor,
            effective_product,
            effective_doc_type,
        )

    cache = get_semantic_cache()
    cache_was_enabled = cache.enabled
    if skip_cache and cache_was_enabled:
        # Filtered queries must not return answers cached from unfiltered runs.
        cache.enabled = False
        analytics.cache_skipped_reason = "filters"
        cache.record_skip("filters")

    product_token = _filter_product.set(effective_product)
    vendors_token = _filter_vendors.set(multi_vendors)
    try:
        ctx = RetrievalContext(
            question=retrieval_question,
            top_k=top_k,
            history_turns=history_turns,
            vendor_filter=effective_vendor,
            document_type_filter=effective_doc_type,
            analytics=analytics,
        )
        retrieval = QueryOrchestrator().run_retrieval(ctx)

        if has_qdrant_filter and not retrieval.cached_answer and not retrieval.chunks:
            logger.info(
                "Filtered retrieval returned zero chunks for %r; "
                "falling back to unfiltered search",
                retrieval_question,
            )
            analytics.filter_fallback = True
            _filter_product.set(None)
            _filter_vendors.set(None)
            fallback_ctx = RetrievalContext(
                question=retrieval_question,
                top_k=top_k,
                history_turns=history_turns,
                vendor_filter=None,
                document_type_filter=None,
                analytics=analytics,
            )
            retrieval = QueryOrchestrator().run_retrieval(fallback_ctx)

        if skip_cache:
            retrieval.use_cache = False
        return retrieval
    finally:
        _filter_product.reset(product_token)
        _filter_vendors.reset(vendors_token)
        if skip_cache:
            cache.enabled = cache_was_enabled


def _build_generation_plan(
    question: str,
    top_k: int,
    history: list[dict] | None = None,
    *,
    vendor: str | None = None,
    product: str | None = None,
    vendor_filter: str | None = None,
    document_type_filter: str | None = None,
) -> GenerationPlan:
    llm = get_llm()
    analytics = QueryAnalytics(
        question=question,
        history_turns=len(_truncate_history(history)),
        trace_id=uuid.uuid4().hex,
    )
    retrieval_question = condense_question_for_retrieval(question, history, llm)
    if retrieval_question != question:
        analytics.condensed_question = retrieval_question

    retrieval = _run_retrieval(
        retrieval_question=retrieval_question,
        top_k=top_k,
        history_turns=analytics.history_turns,
        analytics=analytics,
        vendor=vendor,
        product=product,
        vendor_filter=vendor_filter,
        document_type_filter=document_type_filter,
        question=question,
        history=history,
    )

    if retrieval.cached_answer is not None:
        return GenerationPlan(
            cached_answer=retrieval.cached_answer,
            citations=[],
            question_vector=retrieval.question_vector,
            use_cache=True,
            retrieval_question=retrieval_question,
            analytics=analytics,
        )

    chunks = retrieval.chunks
    question_vector = retrieval.question_vector
    use_cache = retrieval.use_cache

    _populate_top_chunk(analytics, chunks)
    top_score = chunks[0].get("score", 0.0) if chunks else 0.0
    logger.debug(
        "Retrieval top_score=%.3f needs_web=%s",
        top_score,
        needs_web_fallback(retrieval_question, chunks, top_score, analytics),
    )

    if not chunks:
        analytics.empty_retrieval = True
        return GenerationPlan(
            immediate_answer=(
                "No relevant documents found. Please ingest some documents first."
            ),
            citations=[],
            question_vector=question_vector,
            use_cache=use_cache,
            retrieval_question=retrieval_question,
            analytics=analytics,
        )

    citations = collect_citations(chunks)
    top_score = chunks[0].get("score", 1.0)

    def _make_plan(
        plan_citations: list[dict],
        *,
        web_results: list[dict] | None = None,
        allow_retry: bool = False,
    ) -> GenerationPlan:
        prompt_chunks = select_chunks_for_prompt(
            chunks, question, history, web_results
        )
        analytics.chunks_in_prompt = len(prompt_chunks)
        return GenerationPlan(
            prompt=build_prompt(question, prompt_chunks, web_results, history),
            citations=plan_citations,
            chunks=chunks,
            allow_insufficient_retry=allow_retry,
            question_vector=question_vector,
            use_cache=use_cache,
            retrieval_question=retrieval_question,
            analytics=analytics,
        )

    if is_sufficiency_check_enabled():
        # YES/NO context check — use the cheaper fast tier, not the synthesis model.
        sufficient = check_context_sufficient(retrieval_question, chunks, get_fast_llm())
        if sufficient is True:
            analytics.sufficiency_check_result = "YES"
            return _make_plan(citations)
        elif sufficient is False:
            analytics.sufficiency_check_result = "NO"
            _mark_web_fallback(analytics, "upfront")
            web_results = fetch_web_results(retrieval_question)
            return _make_plan([], web_results=web_results)
        else:
            analytics.sufficiency_check_result = "skipped"

    if needs_web_fallback(retrieval_question, chunks, top_score, analytics):
        _mark_web_fallback(analytics, "upfront")
        web_results = fetch_web_results(retrieval_question)
        return _make_plan(citations, web_results=web_results)

    if score_indicates_early_web(chunks, top_score, analytics):
        _mark_web_fallback(analytics, "score_early")
        web_results = fetch_web_results(retrieval_question)
        return _make_plan(citations, web_results=web_results)

    allow_retry = (
        not is_sufficiency_check_enabled()
        and not is_early_web_fallback_enabled()
    )
    return _make_plan(citations, allow_retry=allow_retry)


def _apply_llm_stream_stats(analytics: QueryAnalytics | None, llm) -> None:
    if analytics is None:
        return
    stats = getattr(llm, "last_stream_stats", None) or {}
    if stats.get("prompt_eval_count") is not None:
        analytics.prompt_eval_count = int(stats["prompt_eval_count"])
    if stats.get("eval_count") is not None:
        analytics.answer_token_count = int(stats["eval_count"])


def _stream_llm_tokens(
    llm,
    prompt: str,
    cancel_event: threading.Event | None = None,
) -> Iterator[str]:
    timeout = _answer_generation_timeout_seconds()
    if not hasattr(llm, "generate_stream"):
        result = _llm_generate_with_timeout(llm, prompt, timeout)
        if result is None:
            raise TimeoutError("answer generation timed out")
        yield result
        return

    token_queue: Queue = Queue()

    def _worker() -> None:
        try:
            for token in llm.generate_stream(prompt, cancel_event=cancel_event):
                token_queue.put(("token", token))
            token_queue.put(("done", None))
        except Exception as exc:
            token_queue.put(("error", exc))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    deadline = time.perf_counter() + timeout
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if cancel_event:
                cancel_event.set()
            raise TimeoutError("answer generation timed out")
        try:
            kind, payload = token_queue.get(timeout=remaining)
        except Empty:
            if cancel_event:
                cancel_event.set()
            raise TimeoutError("answer generation timed out")
        if kind == "token":
            yield payload
        elif kind == "done":
            break
        elif kind == "error":
            raise payload


def _yield_text_as_tokens(text: str) -> Iterator[dict]:
    """Yield cached/immediate answers in one or few SSE events (not per-character)."""
    if not text:
        yield {"token": ""}
        return

    word_chunk = int(os.getenv("CACHE_SSE_WORD_CHUNK", "0"))
    if word_chunk <= 0:
        yield {"token": text}
        return

    words = text.split()
    for index in range(0, len(words), word_chunk):
        chunk = " ".join(words[index : index + word_chunk])
        if index + word_chunk < len(words):
            chunk += " "
        yield {"token": chunk}


def _retrieval_stage_event(plan: GenerationPlan, start_time: float) -> dict:
    if plan.analytics:
        plan.analytics.pre_stream_ms = int((time.time() - start_time) * 1000)
        return {
            "stage": "retrieval_complete",
            "meta": plan.analytics.to_meta(),
        }
    return {"stage": "retrieval_complete"}


def _stream_prompt(
    llm,
    prompt: str,
    citations: list[dict] | None = None,
    cancel_event: threading.Event | None = None,
    analytics: QueryAnalytics | None = None,
) -> tuple[Iterator[dict], list[str]]:
    tokens: list[str] = []
    done_citations = citations if citations is not None else []

    def generator() -> Iterator[dict]:
        try:
            for token in _stream_llm_tokens(llm, prompt, cancel_event=cancel_event):
                if cancel_event and cancel_event.is_set():
                    break
                tokens.append(token)
                yield {"token": token}
        except Exception as exc:
            _mark_generation_exception(analytics, exc)
            logger.error("Answer generation streaming failed: %s", exc)
            if tokens:
                yield {"token": f"\n\n[Stream interrupted: {exc}]"}
            yield {
                "citations": done_citations,
                "done": True,
                "error": str(exc),
            }

    return generator(), tokens


def _done_event(
    plan: GenerationPlan,
    answer: str,
    start_time: float,
    citations: list[dict],
    *,
    raw_answer: str | None = None,
) -> dict:
    meta = (
        plan.analytics.finalize(
            answer,
            start_time,
            citations=citations,
            prompt=plan.prompt,
            raw_answer=raw_answer,
        )
        if plan.analytics
        else {}
    )
    try:
        from eval.production_sampler import record_sample

        record_sample(
            question=plan.analytics.question if plan.analytics else "",
            answer_preview=answer,
            meta=meta,
        )
    except Exception:
        pass
    return {"citations": citations, "done": True, "meta": meta}


def stream_query(
    question: str,
    top_k: int = 5,
    history: list[dict] | None = None,
    *,
    vendor: str | None = None,
    product: str | None = None,
    vendor_filter: str | None = None,
    document_type_filter: str | None = None,
    cancel_event: threading.Event | None = None,
) -> Iterator[dict]:
    """Run the RAG pipeline and yield SSE-friendly events (tokens, citations, done/meta)."""
    start_time = time.time()
    plan = _build_generation_plan(
        question,
        top_k,
        history,
        vendor=vendor,
        product=product,
        vendor_filter=vendor_filter,
        document_type_filter=document_type_filter,
    )

    if plan.cached_answer is not None:
        yield from _yield_text_as_tokens(plan.cached_answer)
        yield _done_event(plan, plan.cached_answer, start_time, plan.citations)
        return

    if plan.immediate_answer is not None:
        yield from _yield_text_as_tokens(plan.immediate_answer)
        yield _done_event(plan, plan.immediate_answer, start_time, plan.citations)
        if plan.use_cache and not is_insufficient_answer(plan.immediate_answer):
            get_semantic_cache().store(
                question, plan.question_vector, plan.immediate_answer
            )
        return

    yield _retrieval_stage_event(plan, start_time)

    llm = get_llm()
    stream, tokens = _stream_prompt(
        llm,
        plan.prompt,
        plan.citations,
        cancel_event=cancel_event,
        analytics=plan.analytics,
    )
    llm_start = time.perf_counter()
    first_token = True
    for event in stream:
        if event.get("done") and event.get("error"):
            full_answer = "".join(tokens)
            _apply_llm_stream_stats(plan.analytics, llm)
            if plan.analytics and not plan.analytics.generation_error:
                plan.analytics.generation_error = event["error"]
                lowered = event["error"].lower()
                if "timed out" in lowered or "timeout" in lowered:
                    plan.analytics.generation_timed_out = True
            error_citations = event.get("citations", plan.citations)
            display_answer = _resolve_generation_answer(
                plan, full_answer, error_citations
            )
            if not tokens and display_answer.strip():
                yield from _yield_text_as_tokens(display_answer)
            done = _done_event(
                plan,
                display_answer,
                start_time,
                error_citations,
                raw_answer=full_answer,
            )
            done["error"] = event["error"]
            yield done
            return
        if event.get("token") is not None and first_token:
            if plan.analytics:
                plan.analytics.llm_ttft_ms = int(
                    (time.perf_counter() - llm_start) * 1000
                )
            first_token = False
        yield event

    full_answer = "".join(tokens)
    if (
        cancel_event
        and cancel_event.is_set()
        and not full_answer.strip()
        and plan.analytics
        and not plan.analytics.generation_error
    ):
        plan.analytics.generation_error = "generation cancelled"
    if plan.analytics:
        total_llm_ms = int((time.perf_counter() - llm_start) * 1000)
        plan.analytics.llm_generation_ms = total_llm_ms - (
            plan.analytics.llm_ttft_ms or 0
        )
    _apply_llm_stream_stats(plan.analytics, llm)

    display_answer = _resolve_generation_answer(
        plan, full_answer, plan.citations
    )

    if not tokens and display_answer.strip():
        yield from _yield_text_as_tokens(display_answer)

    if plan.allow_insufficient_retry and _should_retry_insufficient(
        plan, full_answer
    ):
        search_question = plan.retrieval_question or question
        web_results = fetch_web_results(search_question)
        if web_results:
            if plan.analytics:
                _mark_web_fallback(plan.analytics, "retry")
            retry_prompt = build_prompt(
                question, plan.chunks, web_results, history
            )
            retry_stream, retry_tokens = _stream_prompt(
                llm,
                retry_prompt,
                analytics=plan.analytics,
            )
            for event in retry_stream:
                if event.get("done") and event.get("error"):
                    retry_raw = "".join(retry_tokens)
                    retry_display = _resolve_generation_answer(
                        plan, retry_raw, []
                    )
                    done = _done_event(
                        plan,
                        retry_display,
                        start_time,
                        event.get("citations", []),
                        raw_answer=retry_raw,
                    )
                    done["error"] = event["error"]
                    yield done
                    return
                yield event
            retry_raw = "".join(retry_tokens)
            retry_display = _resolve_generation_answer(plan, retry_raw, [])
            if plan.use_cache and plan.analytics.answer_outcome == "ok":
                get_semantic_cache().store(
                    question, plan.question_vector, retry_display
                )
            yield _done_event(
                plan, retry_display, start_time, [], raw_answer=retry_raw
            )
            return

    if plan.use_cache and plan.analytics and plan.analytics.answer_outcome == "ok":
        get_semantic_cache().store(question, plan.question_vector, display_answer)

    yield _done_event(
        plan,
        display_answer,
        start_time,
        plan.citations,
        raw_answer=full_answer,
    )


def _run_rag_pipeline(
    question: str,
    top_k: int,
    history: list[dict] | None = None,
    *,
    vendor: str | None = None,
    product: str | None = None,
    vendor_filter: str | None = None,
    document_type_filter: str | None = None,
) -> dict:
    start_time = time.time()
    llm = get_llm()
    plan = _build_generation_plan(
        question,
        top_k,
        history,
        vendor=vendor,
        product=product,
        vendor_filter=vendor_filter,
        document_type_filter=document_type_filter,
    )

    if plan.cached_answer is not None:
        meta = plan.analytics.finalize(
            plan.cached_answer, start_time, citations=plan.citations
        )
        return {
            "answer": plan.cached_answer,
            "citations": plan.citations,
            "meta": meta,
        }

    if plan.immediate_answer is not None:
        if plan.use_cache and not is_insufficient_answer(plan.immediate_answer):
            get_semantic_cache().store(
                question, plan.question_vector, plan.immediate_answer
            )
        meta = plan.analytics.finalize(
            plan.immediate_answer,
            start_time,
            citations=plan.citations,
        )
        return {
            "answer": plan.immediate_answer,
            "citations": plan.citations,
            "meta": meta,
        }

    answer = _generate_answer_sync(llm, plan.prompt, plan.analytics)

    display_answer = _resolve_generation_answer(plan, answer, plan.citations)

    if plan.allow_insufficient_retry and _should_retry_insufficient(plan, answer):
        search_question = plan.retrieval_question or question
        web_results = fetch_web_results(search_question)
        if web_results:
            if plan.analytics:
                _mark_web_fallback(plan.analytics, "retry")
            answer = _generate_answer_sync(
                llm,
                build_prompt(question, plan.chunks, web_results, history),
                plan.analytics,
            )
            display_answer = _resolve_generation_answer(plan, answer, [])
            if plan.use_cache and plan.analytics.answer_outcome == "ok":
                get_semantic_cache().store(
                    question, plan.question_vector, display_answer
                )
            meta = plan.analytics.finalize(
                display_answer,
                start_time,
                citations=[],
                prompt=plan.prompt,
                raw_answer=answer,
            )
            return {"answer": display_answer, "citations": [], "meta": meta}

    if plan.use_cache and plan.analytics and plan.analytics.answer_outcome == "ok":
        get_semantic_cache().store(question, plan.question_vector, display_answer)

    meta = plan.analytics.finalize(
        display_answer,
        start_time,
        citations=plan.citations,
        prompt=plan.prompt,
        raw_answer=answer,
    )
    return {"answer": display_answer, "citations": plan.citations, "meta": meta}


def query(
    question: str,
    top_k: int = 5,
    history: list[dict] | None = None,
    *,
    vendor: str | None = None,
    product: str | None = None,
    vendor_filter: str | None = None,
    document_type_filter: str | None = None,
) -> dict:
    """Run the full RAG pipeline synchronously; return answer, citations, and meta."""
    return _run_rag_pipeline(
        question,
        top_k,
        history,
        vendor=vendor,
        product=product,
        vendor_filter=vendor_filter,
        document_type_filter=document_type_filter,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m providers.rag_pipeline \"your question here\"")
        sys.exit(1)
    result = query(sys.argv[1])
    print("\nAnswer:", result["answer"])
