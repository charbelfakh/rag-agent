"""UI helpers for YouTube citations (index.html; browser or Node with jsdom)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "ui" / "index.html"

_CITATION_UI_FUNCTIONS = (
    "formatCitationLabel",
    "isVideoCitation",
    "formatVideoTimestamp",
    "citationVideoSrc",
    "parseYoutubeVideoIdFromUrl",
    "parseYoutubeVideoId",
    "citationStartSeconds",
    "citationYoutubeWatchUrl",
    "citationYoutubeEmbedSrc",
    "isYoutubeVideoCitation",
    "isLocalVideoCitation",
    "youtubeCitationTitle",
    "renderCitationsList",
    "renderCitationPlayers",
    "setCitationsContent",
    "createVideoCitationElement",
    "createYoutubeCitationElement",
    "runCitationYoutubeUnitTest",
)


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


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_youtube_citation_ui_unit_in_browser_dom():
    html = INDEX_HTML.read_text(encoding="utf-8")
    bodies = "\n".join(_extract_js_function(html, name) for name in _CITATION_UI_FUNCTIONS)
    script = f"""
const {{ JSDOM }} = require('jsdom');
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {{
  url: 'http://localhost/',
}});
global.window = dom.window;
global.document = dom.window.document;
{bodies}
const result = runCitationYoutubeUnitTest();
console.log(JSON.stringify(result));
"""
    try:
        result = subprocess.run(
            ["node", "-e", script],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as exc:
        if "Cannot find module 'jsdom'" in (exc.stderr or ""):
            pytest.skip("jsdom not installed")
        raise
    payload = json.loads(result.stdout.strip())
    assert payload.get("ok"), payload.get("errors")
