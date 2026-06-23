"""Offline speech-to-text for video ingest via faster-whisper (segment timestamps)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_whisper_model: Any | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def whisper_model_name() -> str:
    return os.getenv("VIDEO_WHISPER_MODEL", "small")


def whisper_device() -> str:
    return os.getenv("VIDEO_WHISPER_DEVICE", "cuda")


def whisper_compute_type() -> str:
    return os.getenv("VIDEO_WHISPER_COMPUTE_TYPE", "float16")


def whisper_initial_prompt() -> str:
    return os.getenv(
        "VIDEO_WHISPER_INITIAL_PROMPT",
        "This video discusses PEKAT Vision, the Detector, Classifier, and Unifier modules.",
    ).strip()


def transcript_chunk_seconds() -> float:
    return float(os.getenv("VIDEO_TRANSCRIPT_CHUNK_SECONDS", "45"))


def reset_whisper_model() -> None:
    """Clear cached model (tests)."""
    global _whisper_model
    _whisper_model = None


def _load_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    from faster_whisper import WhisperModel

    _whisper_model = WhisperModel(
        whisper_model_name(),
        device=whisper_device(),
        compute_type=whisper_compute_type(),
    )
    logger.info(
        "Loaded faster-whisper model=%s device=%s compute_type=%s",
        whisper_model_name(),
        whisper_device(),
        whisper_compute_type(),
    )
    return _whisper_model


def transcribe_plain_text(audio_path: str) -> str:
    """Transcribe an audio or video file; return the full plain-text transcript."""
    segments = transcribe(audio_path)
    return " ".join(segment.text for segment in segments).strip()


def transcribe(audio_path: str) -> list[TranscriptSegment]:
    """Transcribe audio/video file; return segment-level timestamps."""
    model = _load_whisper_model()
    segments: list[TranscriptSegment] = []
    transcribe_kwargs: dict = {"beam_size": 5}
    prompt = whisper_initial_prompt()
    if prompt:
        transcribe_kwargs["initial_prompt"] = prompt
    for segment in model.transcribe(audio_path, **transcribe_kwargs)[0]:
        text = (segment.text or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=text,
            )
        )
    return segments


def group_segments_into_windows(
    segments: list[TranscriptSegment],
    max_seconds: float,
) -> list[TranscriptSegment]:
    """Merge whisper segments into paragraph windows capped by max_seconds."""
    if not segments:
        return []
    if max_seconds <= 0:
        return list(segments)

    windows: list[TranscriptSegment] = []
    window_start = segments[0].start
    window_end = segments[0].end
    texts = [segments[0].text]

    for segment in segments[1:]:
        if segment.end - window_start > max_seconds and texts:
            windows.append(
                TranscriptSegment(
                    start=window_start,
                    end=window_end,
                    text=" ".join(texts),
                )
            )
            window_start = segment.start
            texts = []
        texts.append(segment.text)
        window_end = segment.end

    if texts:
        windows.append(
            TranscriptSegment(
                start=window_start,
                end=window_end,
                text=" ".join(texts),
            )
        )
    return windows


def build_transcript_index_text(source: str, start_seconds: float, text: str) -> str:
    return f"[Video transcript, {source}, t={start_seconds:.1f}s] {text}"
