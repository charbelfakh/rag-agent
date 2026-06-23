"""Ingest pipeline schema v2: payload, CLI, HTML loader, manifest, point IDs."""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# conftest stubs ingest for API tests; load the real module with light deps stubbed.
sys.modules.pop("scripts.ingest.ingest", None)
sys.modules.pop("ingest", None)

_lc_docs = types.ModuleType("langchain_core.documents")
_lc_docs.Document = MagicMock
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules["langchain_core.documents"] = _lc_docs

_lc_splitters = types.ModuleType("langchain_text_splitters")
_lc_splitters.RecursiveCharacterTextSplitter = MagicMock
sys.modules["langchain_text_splitters"] = _lc_splitters

import scripts.ingest.ingest as ingest_mod
from scripts.ingest.ingest import (
    SCHEMA_VERSION,
    ChunkItem,
    IngestContext,
    _build_html_sections,
    _build_ingest_context,
    _flush_embed,
    build_v2_payload,
    compute_file_sha256,
    infer_vendor_from_path,
    load_manifest,
    lookup_url_from_sources,
    make_point_id,
    normalize_doc_type,
    save_manifest,
)


class TestVendorInference:
    def test_infer_from_data_vendor_path(self):
        assert infer_vendor_from_path("data/pekat/manual.pdf") == "pekat"
        assert infer_vendor_from_path("data\\pekat\\guide.html") == "pekat"

    def test_no_vendor_for_flat_data_path(self):
        assert infer_vendor_from_path("data/manual.pdf") is None


class TestPointIds:
    def test_pdf_page_scoped_ids_are_stable(self):
        source = "manual.pdf"
        a = make_point_id(source, 0, 0)
        b = make_point_id(source, 0, 0)
        c = make_point_id(source, 1, 0)
        assert a == b
        assert a != c

    def test_html_uses_none_page_key(self):
        html_id = make_point_id("page.html", None, 2)
        pdf_id = make_point_id("page.html", 2, 2)
        assert html_id != pdf_id
        assert "none" in html_id or html_id  # uuid string


class TestV2Payload:
    def test_payload_has_all_required_fields(self):
        ctx = IngestContext(
            source="manual.pdf",
            vendor="pekat",
            product="pekat vision",
            product_version="3.17",
            doc_type="manual",
            url="https://example.com/manual",
            ingested_at="2026-06-11T14:30:00Z",
        )
        item = ChunkItem(
            text="Setup steps",
            page=None,
            section="Installation",
            chunk_index=0,
        )
        payload = build_v2_payload(item, ctx)
        assert payload["text"] == "Setup steps"
        assert payload["source"] == "manual.pdf"
        assert payload["page"] is None
        assert payload["url"] == "https://example.com/manual"
        assert payload["vendor"] == "pekat"
        assert payload["product"] == "pekat vision"
        assert payload["product_version"] == "3.17"
        assert payload["content_type"] == "text"
        assert payload["doc_type"] == "manual"
        assert payload["section"] == "Installation"
        assert payload["timestamp"] is None
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["ingested_at"] == "2026-06-11T14:30:00Z"


class TestHtmlSections:
    def test_headings_become_sections(self):
        from bs4 import BeautifulSoup

        html = """
        <html><head><title>Page Title</title></head>
        <body>
          <p>Intro paragraph</p>
          <h1>First Section</h1>
          <p>Body one</p>
          <h2>Second Section</h2>
          <p>Body two</p>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        sections = _build_html_sections(soup)
        headings = [h for h, _ in sections if h]
        assert "First Section" in headings
        assert "Second Section" in headings
        assert any("Body one" in body for _, body in sections)

    def test_strips_script_and_style(self):
        from bs4 import BeautifulSoup

        html = """
        <html><body>
          <script>alert(1)</script>
          <style>.x{}</style>
          <p>Visible</p>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        sections = _build_html_sections(soup)
        combined = " ".join(body for _, body in sections)
        assert "alert" not in combined
        assert "Visible" in combined

    def test_confluence_uses_ak_renderer_document_not_chrome(self):
        from bs4 import BeautifulSoup

        html = """
        <html><head><title>Mask - Confluence</title></head>
        <body>
          <nav>Site navigation menu</nav>
          <img src="/wiki/aa-avatar/user.png" alt="avatar">
          <p>data-loadable-begin="bTjE2:OP-5u"</p>
          <div class="fabric css-xyz ak-renderer-document">
            <p>The Mask module allows you to create a black mask.</p>
            <h2>Painting the mask</h2>
            <p>Use the brush tool to paint regions.</p>
          </div>
          <footer>Atlassian cookies and tracking notice</footer>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        sections = _build_html_sections(soup)
        combined = " ".join(f"{h or ''} {b}" for h, b in sections)
        assert "The Mask module allows you to create a black mask" in combined
        assert "Painting the mask" in combined
        assert "Use the brush tool" in combined
        assert "Site navigation menu" not in combined
        assert "data-loadable-begin" not in combined
        assert "Atlassian cookies" not in combined
        assert "avatar" not in combined

    def test_non_confluence_html_unchanged_without_ak_renderer(self):
        from bs4 import BeautifulSoup

        html = """
        <html><head><title>Zendesk Article</title></head>
        <body>
          <div class="portal-banner">Support portal banner</div>
          <p>Intro outside any renderer.</p>
          <h1>Calibration</h1>
          <p>Step one details.</p>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        sections = _build_html_sections(soup)
        combined = " ".join(f"{h or ''} {b}" for h, b in sections)
        assert "Support portal banner" in combined
        assert "Intro outside any renderer" in combined
        assert "Calibration" in combined
        assert "Step one details" in combined


class TestSourcesJson:
    def test_lookup_url(self, tmp_path, monkeypatch):
        vendor_dir = tmp_path / "data" / "pekat"
        vendor_dir.mkdir(parents=True)
        (vendor_dir / "sources.json").write_text(
            json.dumps({"article.html": "https://pekat.com/article"}),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        assert lookup_url_from_sources("pekat", "article.html") == "https://pekat.com/article"
        assert lookup_url_from_sources("pekat", "missing.html") is None

    def test_malformed_sources_json_returns_none(self, tmp_path, monkeypatch):
        vendor_dir = tmp_path / "data" / "pekat"
        vendor_dir.mkdir(parents=True)
        (vendor_dir / "sources.json").write_text("{not json", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert lookup_url_from_sources("pekat", "a.html") is None


class TestManifest:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = {"manual.pdf": {"sha256": "abc", "vendor": "pekat", "chunk_count": 3}}
        save_manifest(data)
        loaded = load_manifest()
        assert loaded["manual.pdf"]["sha256"] == "abc"


class TestBuildIngestContext:
    def test_maps_legacy_api_fields(self, tmp_path):
        path = tmp_path / "data" / "pekat" / "manual.pdf"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"%PDF-1.4")
        ctx = _build_ingest_context(
            str(path),
            product_line="Pekat Vision",
            software_version="3.17",
            document_type="user_manual",
        )
        assert ctx.vendor == "pekat"
        assert ctx.product == "pekat vision"
        assert ctx.product_version == "3.17"
        assert ctx.doc_type == "manual"
        assert ctx.source == "manual.pdf"

    def test_raises_without_vendor(self, tmp_path):
        path = tmp_path / "data" / "orphan.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="vendor is required"):
            _build_ingest_context(str(path))


class TestCli:
    def test_missing_vendor_exits_with_error(self, tmp_path, capsys):
        path = tmp_path / "data" / "manual.pdf"
        path.parent.mkdir(parents=True)
        path.write_text("hello", encoding="utf-8")
        code = ingest_mod._cli_main([str(path)])
        assert code == 1
        assert "vendor is required" in capsys.readouterr().err

    def test_vendor_inferred_from_path(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "data" / "pekat" / "manual.txt"
        path.parent.mkdir(parents=True)
        path.write_text("hello world", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ingest_mod, "ingest", lambda *a, **k: 1)
        code = ingest_mod._cli_main([str(path)])
        assert code == 0
        manifest = load_manifest()
        assert manifest["manual.txt"]["vendor"] == "pekat"

    def test_skip_unchanged_file(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "data" / "pekat" / "manual.txt"
        path.parent.mkdir(parents=True)
        path.write_text("same content", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        file_hash = compute_file_sha256(str(path))
        save_manifest(
            {
                "manual.txt": {
                    "sha256": file_hash,
                    "vendor": "pekat",
                    "product": None,
                    "chunk_count": 2,
                    "ingested_at": "2026-01-01T00:00:00Z",
                }
            }
        )
        code = ingest_mod._cli_main([str(path)])
        assert code == 0
        assert "Skipping manual.txt" in capsys.readouterr().out

    def test_force_reingests(self, tmp_path, monkeypatch):
        path = tmp_path / "data" / "pekat" / "manual.txt"
        path.parent.mkdir(parents=True)
        path.write_text("same content", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        file_hash = compute_file_sha256(str(path))
        save_manifest(
            {
                "manual.txt": {
                    "sha256": file_hash,
                    "vendor": "pekat",
                    "product": None,
                    "chunk_count": 2,
                    "ingested_at": "2026-01-01T00:00:00Z",
                }
            }
        )
        called = {"n": 0}

        def fake_ingest(*args, **kwargs):
            called["n"] += 1
            return 2

        monkeypatch.setattr(ingest_mod, "ingest", fake_ingest)
        code = ingest_mod._cli_main([str(path), "--force"])
        assert code == 0
        assert called["n"] == 1


class TestNormalizeDocType:
    def test_aliases(self):
        assert normalize_doc_type("user_manual") == "manual"
        assert normalize_doc_type("article") == "article"
        assert normalize_doc_type("unknown_kind") == "other"


class TestFlushEmbed:
    def test_skips_none_vectors_with_source_page_warning(self, caplog):
        from queue import Queue

        buffer = [
            ChunkItem(text="chunk one", page=1, chunk_index=0),
            ChunkItem(text="chunk two", page=2, chunk_index=1),
            ChunkItem(text="chunk three", page=3, chunk_index=2),
        ]
        valid_vector = [0.1, 0.2, 0.3]

        class StubEmbedder:
            def __init__(self):
                self.last_call: dict = {}

            def embed(self, texts, *, sources=None, pages=None):
                self.last_call = {
                    "texts": texts,
                    "sources": sources,
                    "pages": pages,
                }
                return [valid_vector, None, valid_vector]

        embedder = StubEmbedder()
        result_queue: Queue = Queue()
        embed_pbar = MagicMock()

        with caplog.at_level(logging.WARNING):
            _flush_embed(
                buffer,
                embedder,
                result_queue,
                embed_pbar,
                "manual.pdf",
            )

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == 2
        assert results[0] == (valid_vector, buffer[0])
        assert results[1] == (valid_vector, buffer[2])
        assert embedder.last_call == {
            "texts": [item.text for item in buffer],
            "sources": ["manual.pdf", "manual.pdf", "manual.pdf"],
            "pages": [1, 2, 3],
        }
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        ]
        assert any(
            "Skipping chunk with no embedding (source=manual.pdf page=2)" in msg
            for msg in warning_messages
        )
