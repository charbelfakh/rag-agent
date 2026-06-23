"""Tests for ui/index.html stripTextForTts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "ui" / "index.html"


def strip_text_for_tts_python(text: str) -> str:
    """Mirror of ui/index.html stripTextForTts — keep regex order in sync."""
    s = str(text or "")
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"`[^`]+`", " ", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*\n]+)\*", r"\1", s)
    s = re.sub(r"(^|\s)[*\-]\s+", r"\1", s)
    s = re.sub(r"(^|\n)\s*#{1,6}\s*", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"/video\?[^\s]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"/media/[^\s]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[*_#`]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_js_function(source: str, name: str) -> str:
    marker = f"function {name}("
    start = source.index(marker)
    brace_start = source.index("{", start)
    depth = 0
    for i in range(brace_start, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise ValueError(f"unclosed function body for {name}")


def _run_strip_text_for_tts_js(text: str) -> str:
    if not shutil.which("node"):
        pytest.skip("node not installed")
    fn_source = _extract_js_function(INDEX_HTML.read_text(encoding="utf-8"), "stripTextForTts")
    payload = json.dumps(text)
    script = f"{fn_source}\nconsole.log(JSON.stringify(stripTextForTts({payload})));"
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return json.loads(result.stdout.strip())


def _assert_strip_sample(out: str) -> None:
    assert "*" not in out
    assert "#" not in out
    assert "`" not in out
    for word in ("Step", "one", "PEKAT", "bold", "Heading"):
        assert word in out


def test_strip_text_for_tts_removes_markdown_noise():
    sample = "* Step one\n*PEKAT* and **bold** and # Heading"
    out = strip_text_for_tts_python(sample)
    _assert_strip_sample(out)
    if shutil.which("node"):
        js_out = _run_strip_text_for_tts_js(sample)
        assert js_out == out


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_strip_text_for_tts_js_matches_python_mirror():
    sample = "* Step one\n*PEKAT* and **bold** and # Heading"
    assert _run_strip_text_for_tts_js(sample) == strip_text_for_tts_python(sample)
