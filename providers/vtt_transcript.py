"""Parse yt-dlp WebVTT transcripts and map .info.json metadata for ingest."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VTT_TIMESTAMP = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3}).*$"
)
_VTT_LANG_SUFFIX = re.compile(r"^(.+)\.([a-z]{2}(?:-[a-zA-Z]+)?)\.vtt$", re.IGNORECASE)
_BRACKET_ID = re.compile(r"\[([^\]]+)\]\s*$")
_INLINE_TAG = re.compile(r"<[^>]+>")


def _normalize_cue_line(line: str) -> str:
    cleaned = _INLINE_TAG.sub("", line)
    cleaned = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _lines_overlap(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    prev = previous.casefold()
    cur = current.casefold()
    if prev == cur:
        return True
    return prev.startswith(cur) or cur.startswith(prev)


def collapse_rolling_duplicates(lines: list[str]) -> list[str]:
    """Collapse consecutive identical or prefix-overlapping caption lines."""
    collapsed: list[str] = []
    for line in lines:
        if not line:
            continue
        if not collapsed:
            collapsed.append(line)
            continue
        if _lines_overlap(collapsed[-1], line):
            if len(line) > len(collapsed[-1]):
                collapsed[-1] = line
            continue
        collapsed.append(line)
    return collapsed


@dataclass(frozen=True)
class VttCue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class VttTranscriptChunk:
    start_seconds: float
    end_seconds: float
    text: str


def _vtt_clock_to_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def parse_vtt_cues(path: str) -> list[VttCue]:
    """Return timestamped cues from a WebVTT file."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    cues: list[VttCue] = []
    in_note = False
    current_start: float | None = None
    current_end: float | None = None
    cue_lines: list[str] = []

    def flush_cue() -> None:
        nonlocal current_start, current_end, cue_lines
        if current_start is None or current_end is None:
            cue_lines = []
            return
        merged = collapse_rolling_duplicates(
            [_normalize_cue_line(line) for line in cue_lines if line]
        )
        text = re.sub(r"\s+", " ", " ".join(merged)).strip()
        if text:
            cues.append(VttCue(start=current_start, end=current_end, text=text))
        current_start = None
        current_end = None
        cue_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_cue()
            in_note = False
            continue
        if stripped.upper() == "WEBVTT":
            continue
        if stripped.upper().startswith("NOTE"):
            flush_cue()
            in_note = True
            continue
        if in_note:
            continue
        if stripped.upper() == "STYLE" or stripped.startswith("::"):
            flush_cue()
            in_note = True
            continue
        match = _VTT_TIMESTAMP.match(stripped)
        if match:
            flush_cue()
            current_start = _vtt_clock_to_seconds(*match.groups()[0:4])
            current_end = _vtt_clock_to_seconds(*match.groups()[4:8])
            continue
        if re.match(r"^(align|position|line|size):", stripped, re.IGNORECASE):
            continue
        if current_start is not None:
            cue_lines.append(stripped)

    flush_cue()
    return cues


def group_vtt_cues_into_chunks(
    cues: list[VttCue],
    *,
    max_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[VttTranscriptChunk]:
    """Merge VTT cues into ingest windows with start/end timestamps."""
    if not cues:
        return []
    if max_chars <= 0:
        return [
            VttTranscriptChunk(
                start_seconds=cue.start,
                end_seconds=cue.end,
                text=cue.text,
            )
            for cue in cues
            if cue.text
        ]

    windows: list[VttTranscriptChunk] = []
    current_texts: list[str] = []
    window_start: float | None = None
    window_end: float | None = None
    char_count = 0

    def flush_window() -> None:
        nonlocal current_texts, window_start, window_end, char_count
        if not current_texts or window_start is None or window_end is None:
            current_texts = []
            window_start = None
            window_end = None
            char_count = 0
            return
        text = re.sub(r"\s+", " ", " ".join(current_texts)).strip()
        if text:
            windows.append(
                VttTranscriptChunk(
                    start_seconds=window_start,
                    end_seconds=window_end,
                    text=text,
                )
            )
        current_texts = []
        window_start = None
        window_end = None
        char_count = 0

    for cue in cues:
        cue_text = cue.text.strip()
        if not cue_text:
            continue

        extra = len(cue_text) + (1 if current_texts else 0)
        if current_texts and char_count + extra > max_chars:
            flush_window()
            if overlap_chars > 0 and windows:
                tail = windows[-1].text
                overlap = tail[-overlap_chars:].strip() if len(tail) > overlap_chars else tail
                if overlap:
                    current_texts = [overlap]
                    char_count = len(overlap)
                    window_start = cue.start
                    window_end = cue.end

        if window_start is None:
            window_start = cue.start
        window_end = cue.end
        current_texts.append(cue_text)
        char_count += extra

    flush_window()
    return windows


def parse_vtt(path: str) -> str:
    """Return clean continuous transcript text from a WebVTT file."""
    cue_texts = [cue.text for cue in parse_vtt_cues(path) if cue.text]
    merged = collapse_rolling_duplicates(cue_texts)
    return re.sub(r"\s+", " ", " ".join(merged)).strip()


def load_info_json(path: str) -> dict:
    """Load yt-dlp ``.info.json`` sidecar; return ``{}`` on missing/invalid input."""
    info_path = Path(path)
    if not info_path.is_file():
        return {}
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read info json %s: %s", info_path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def load_vendors_config(path: str | Path | None = None) -> dict:
    """Load ``config/vendors.json`` with empty defaults when missing."""
    if path is None:
        from providers.config_paths import VENDORS_JSON

        config_path = VENDORS_JSON
    else:
        config_path = Path(path)
    if not config_path.is_file():
        return {
            "vendors": {},
            "custom_folders": [],
            "keywords": {},
            "youtube_channels": {},
            "youtube_channel_names": {},
        }
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "vendors": {},
            "custom_folders": [],
            "keywords": {},
            "youtube_channels": {},
            "youtube_channel_names": {},
        }
    data.setdefault("vendors", {})
    data.setdefault("custom_folders", [])
    data.setdefault("keywords", {})
    data.setdefault("youtube_channels", {})
    data.setdefault("youtube_channel_names", {})
    return data


def vtt_language_from_filename(vtt_path: str | Path) -> str | None:
    match = _VTT_LANG_SUFFIX.match(Path(vtt_path).name)
    if not match:
        return None
    return match.group(2).lower()


def infer_transcript_source(vtt_path: str | Path, info: dict) -> str:
    name = Path(vtt_path).name.lower()
    if ".auto." in name:
        return "auto"
    lang = vtt_language_from_filename(vtt_path)
    if lang:
        auto_caps = info.get("automatic_captions") or {}
        manual_subs = info.get("subtitles") or {}
        if isinstance(auto_caps, dict) and lang in auto_caps:
            return "auto"
        if isinstance(manual_subs, dict) and lang in manual_subs:
            return "manual"
    return "auto"


def video_id_from_vtt_path(vtt_path: Path) -> str | None:
    match = _VTT_LANG_SUFFIX.match(vtt_path.name)
    if not match:
        return None
    base = match.group(1)
    bracket = _BRACKET_ID.search(base)
    if bracket:
        return bracket.group(1).strip()
    return base.strip() or None


def find_info_json_for_vtt(vtt_path: Path) -> Path | None:
    """Locate sibling ``.info.json`` for a VTT file."""
    video_id = video_id_from_vtt_path(vtt_path)
    if not video_id:
        return None

    direct = vtt_path.with_name(f"{video_id}.info.json")
    if direct.is_file():
        return direct

    for candidate in sorted(vtt_path.parent.glob("*.info.json")):
        info = load_info_json(str(candidate))
        if str(info.get("id") or "") == video_id:
            return candidate
    return None


def resolve_vendor_from_info(
    info: dict,
    config: dict | None = None,
    *,
    cli_vendor: str | None = None,
) -> str | None:
    """Map yt-dlp channel metadata to a vendor slug via ``vendors.json``."""
    if cli_vendor and cli_vendor.strip():
        return cli_vendor.strip().lower()

    cfg = config or load_vendors_config()
    channel_map = cfg.get("youtube_channels") or {}
    channel_id = str(info.get("channel_id") or info.get("uploader_id") or "").strip()
    if channel_id and channel_id in channel_map:
        return str(channel_map[channel_id]).strip().lower()

    channel_text = " ".join(
        str(info.get(key) or "")
        for key in ("channel", "uploader", "channel_url", "uploader_url")
    ).casefold()
    if channel_text:
        for vendor, aliases in (cfg.get("youtube_channel_names") or {}).items():
            for alias in aliases or []:
                alias_cf = str(alias).casefold()
                if alias_cf and alias_cf in channel_text:
                    return str(vendor).strip().lower()
        for vendor in (cfg.get("custom_folders") or []):
            slug = str(vendor).casefold()
            if slug and slug in channel_text:
                return str(vendor).strip().lower()
    return None


def resolve_language(info: dict, vtt_path: str | Path) -> str:
    lang = info.get("language")
    if isinstance(lang, str) and lang.strip():
        return lang.strip().lower()[:2]
    from_vtt = vtt_language_from_filename(vtt_path)
    if from_vtt:
        return from_vtt.split("-", 1)[0].lower()
    return "en"


def resolve_product_hint(
    info: dict,
    *,
    cli_product: str | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(product, device_family)`` best-effort hints."""
    if cli_product and cli_product.strip():
        product = cli_product.strip().lower()
        return product, cli_product.strip()

    playlist = str(info.get("playlist_title") or "").strip()
    if playlist:
        return playlist.lower(), playlist
    title = str(info.get("title") or "").strip()
    if title:
        return title.lower(), title
    return None, None


def build_video_transcript_metadata(
    info: dict,
    vtt_path: str | Path,
    *,
    cli_vendor: str | None = None,
    cli_product: str | None = None,
    cli_product_version: str | None = None,
    vendors_config: dict | None = None,
) -> dict[str, Any] | None:
    """Map yt-dlp info json fields to ingest metadata; ``None`` if video id/title missing."""
    video_id = str(info.get("id") or video_id_from_vtt_path(Path(vtt_path)) or "").strip()
    title = str(info.get("title") or "").strip()
    if not video_id or not title:
        return None

    vendor = resolve_vendor_from_info(info, vendors_config, cli_vendor=cli_vendor)
    if not vendor:
        return None

    product, device_family = resolve_product_hint(info, cli_product=cli_product)
    upload_date = str(info.get("upload_date") or "").strip() or None
    doc_version = (cli_product_version or upload_date or "").strip() or None

    return {
        "video_id": video_id,
        "source": title,
        "url": str(info.get("webpage_url") or info.get("original_url") or "").strip() or None,
        "vendor": vendor,
        "product": product,
        "device_family": device_family,
        "product_version": doc_version,
        "doc_version": doc_version,
        "language": resolve_language(info, vtt_path),
        "doc_type": "tutorial",
        "content_type": "video_transcript",
        "source_type": "video",
        "transcript_source": infer_transcript_source(vtt_path, info),
        "page": None,
    }


_ACCEPTED_ENGLISH_VTT_LANGS = frozenset({"en", "en-orig"})


def _vtt_lang_tag(path: Path) -> str | None:
    match = _VTT_LANG_SUFFIX.match(path.name)
    if not match:
        return None
    return match.group(2).lower()


def _is_accepted_english_vtt_lang(lang: str) -> bool:
    return lang.lower() in _ACCEPTED_ENGLISH_VTT_LANGS


def _accepted_english_preference_rank(lang: str) -> int:
    lowered = lang.lower()
    if lowered == "en":
        return 0
    if lowered == "en-orig":
        return 1
    raise ValueError(f"unexpected accepted lang tag: {lang}")


def _pick_preferred_vtt(paths: list[Path]) -> Path | None:
    accepted = [
        path
        for path in paths
        if (lang := _vtt_lang_tag(path)) and _is_accepted_english_vtt_lang(lang)
    ]
    if not accepted:
        return None

    def sort_key(path: Path) -> tuple[int, str]:
        lang = _vtt_lang_tag(path) or ""
        return (_accepted_english_preference_rank(lang), path.name)

    # Prefer plain ``en`` over ``en-orig`` when both exist for the same video.
    return min(accepted, key=sort_key)


def discover_vtt_files(video_dir: str | Path) -> list[Path]:
    """Pick one English VTT per video id under a transcript directory."""
    root = Path(video_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Video transcript directory not found: {video_dir}")

    grouped: dict[str, list[Path]] = {}
    for path in sorted(root.glob("*.vtt")):
        video_id = video_id_from_vtt_path(path)
        if not video_id:
            continue
        grouped.setdefault(video_id, []).append(path)

    selected: list[Path] = []
    for video_id in sorted(grouped):
        picked = _pick_preferred_vtt(grouped[video_id])
        if picked is not None:
            selected.append(picked)
    return selected
