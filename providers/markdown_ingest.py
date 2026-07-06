"""Markdown-based extraction for PDF/HTML ingest (``MARKDOWN_INGEST_ENABLED``).

Converts documents to Markdown before sectioning so tables, lists, and code
blocks survive into chunk text, while preserving the page/section metadata
contract the citations, UI, and eval set depend on.

- PDF: ``pymupdf4llm`` per-page Markdown (``page_chunks=True`` keeps page numbers)
- HTML: ``markdownify`` over a cleaned DOM
- Both feed :func:`split_markdown_sections` → ``(heading, body)`` tuples that
  slot into the existing section chunker unchanged.
"""
from __future__ import annotations

import os
import re

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_MD_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")
_EMPHASIS_RE = re.compile(r"[*_`]+")

# Tags whose text content should never reach chunks.
_HTML_STRIP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "iframe")


def is_markdown_ingest_enabled() -> bool:
    return os.getenv("MARKDOWN_INGEST_ENABLED", "false").lower() in ("true", "1", "yes")


def _clean_heading(raw: str) -> str:
    """Strip markdown emphasis/trailing anchors from a heading line's text."""
    text = _EMPHASIS_RE.sub("", raw).strip()
    return re.sub(r"\s+", " ", text)


def split_markdown_sections(md_text: str) -> list[tuple[str | None, str]]:
    """Split Markdown into ``(heading, body)`` sections on ATX headings.

    Content before the first heading becomes a section with ``heading=None``.
    Table/list/code lines are preserved verbatim in the body. Image-only lines
    are dropped (images travel through the caption pipeline, not chunk text).
    """
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    body_lines: list[str] = []
    in_code_block = False

    def flush() -> None:
        nonlocal body_lines
        body = "\n".join(body_lines).strip()
        if body or heading:
            sections.append((heading, body))
        body_lines = []

    for line in md_text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            body_lines.append(line)
            continue
        if not in_code_block:
            match = _HEADING_RE.match(line)
            if match:
                flush()
                heading = _clean_heading(match.group(2))
                continue
            if _MD_IMAGE_RE.match(line.strip()):
                continue
        body_lines.append(line)

    flush()
    return sections


def pdf_markdown_sections(pdf_path: str) -> list[tuple[str | None, str, int]]:
    """Return ``(heading, body, page)`` sections from per-page pymupdf4llm Markdown."""
    import pymupdf4llm  # heavy; imported only when the flag is on

    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, show_progress=False)
    sections: list[tuple[str | None, str, int]] = []
    last_heading: str | None = None
    for page_entry in pages:
        page_number = _page_number(page_entry)
        md_text = page_entry.get("text") or ""
        for heading, body in split_markdown_sections(md_text):
            if heading is None:
                # Section content continuing across a page break keeps the
                # previous page's heading, matching the legacy extractor.
                heading = last_heading
            else:
                last_heading = heading
            if body or heading:
                sections.append((heading, body, page_number))
    return sections


def _page_number(page_entry: dict) -> int:
    """0-based page index from a pymupdf4llm page chunk (its metadata is 1-based)."""
    metadata = page_entry.get("metadata") or {}
    page = metadata.get("page")
    if isinstance(page, int) and page >= 1:
        return page - 1
    return 0


def html_to_markdown(raw_html: str) -> str:
    """Convert an HTML document body to Markdown with chrome/script removed."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(raw_html, "lxml")
    root = soup.body or soup
    for tag_name in _HTML_STRIP_TAGS:
        for tag in root.find_all(tag_name):
            tag.decompose()
    return markdownify(str(root), heading_style="ATX", strip=["img"]).strip()


def html_markdown_sections(raw_html: str) -> list[tuple[str | None, str]]:
    """Return ``(heading, body)`` sections from a Markdown-converted HTML doc."""
    return split_markdown_sections(html_to_markdown(raw_html))
