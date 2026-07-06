"""Tests for the Markdown ingest path (MARKDOWN_INGEST_ENABLED)."""
import sys
import types
from unittest.mock import patch


from providers.markdown_ingest import (
    html_to_markdown,
    is_markdown_ingest_enabled,
    pdf_markdown_sections,
    split_markdown_sections,
)


class TestFlag:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MARKDOWN_INGEST_ENABLED", raising=False)
        assert is_markdown_ingest_enabled() is False

    def test_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("MARKDOWN_INGEST_ENABLED", "true")
        assert is_markdown_ingest_enabled() is True


class TestSplitMarkdownSections:
    def test_headings_create_sections(self):
        md = "# Intro\nwelcome\n\n## Setup\nstep one\nstep two"
        sections = split_markdown_sections(md)
        assert sections == [("Intro", "welcome"), ("Setup", "step one\nstep two")]

    def test_content_before_first_heading_has_none_heading(self):
        sections = split_markdown_sections("preamble text\n# Later\nbody")
        assert sections[0] == (None, "preamble text")
        assert sections[1] == ("Later", "body")

    def test_heading_emphasis_stripped(self):
        sections = split_markdown_sections("## **Laser Safety**\nbody")
        assert sections[0][0] == "Laser Safety"

    def test_tables_preserved_in_body(self):
        md = "# Specs\n| Model | NOHD |\n| --- | --- |\n| 2120A | 10 cm |"
        sections = split_markdown_sections(md)
        assert "| 2120A | 10 cm |" in sections[0][1]

    def test_image_only_lines_dropped(self):
        sections = split_markdown_sections("# S\n![fig](img.png)\ntext")
        assert sections[0][1] == "text"

    def test_hash_inside_code_block_not_treated_as_heading(self):
        md = "# Real\n```\n# not a heading\n```\nafter"
        sections = split_markdown_sections(md)
        assert len(sections) == 1
        assert "# not a heading" in sections[0][1]


class TestPdfMarkdownSections:
    def _fake_pymupdf4llm(self, pages):
        module = types.ModuleType("pymupdf4llm")
        module.to_markdown = lambda path, page_chunks=True, show_progress=False: pages
        return module

    def test_sections_carry_zero_based_pages(self):
        pages = [
            {"metadata": {"page": 1}, "text": "# Overview\nintro text"},
            {"metadata": {"page": 2}, "text": "continuation text\n## Details\nmore"},
        ]
        with patch.dict(sys.modules, {"pymupdf4llm": self._fake_pymupdf4llm(pages)}):
            sections = pdf_markdown_sections("doc.pdf")

        assert sections[0] == ("Overview", "intro text", 0)
        # Cross-page continuation inherits the previous heading.
        assert sections[1] == ("Overview", "continuation text", 1)
        assert sections[2] == ("Details", "more", 1)

    def test_empty_pages_skipped(self):
        pages = [{"metadata": {"page": 1}, "text": ""}]
        with patch.dict(sys.modules, {"pymupdf4llm": self._fake_pymupdf4llm(pages)}):
            assert pdf_markdown_sections("doc.pdf") == []


class TestHtmlToMarkdown:
    def test_headings_paragraphs_and_tables_convert(self):
        html = (
            "<html><body><h2>Fast Training</h2><p>Use it for quick tests.</p>"
            "<table><tr><th>Mode</th></tr><tr><td>Fast</td></tr></table>"
            "</body></html>"
        )
        md = html_to_markdown(html)
        assert "## Fast Training" in md
        assert "Use it for quick tests." in md
        assert "Fast" in md

    def test_script_nav_and_images_stripped(self):
        html = (
            "<html><body><nav>menu</nav><script>evil()</script>"
            "<h1>T</h1><img src='x.png'><p>body</p></body></html>"
        )
        md = html_to_markdown(html)
        assert "menu" not in md
        assert "evil" not in md
        assert "x.png" not in md
        assert "body" in md

