"""Tests for ingest_manifest tracking on URL ingest paths."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

_lc_docs = types.ModuleType("langchain_core.documents")
_lc_docs.Document = MagicMock
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules["langchain_core.documents"] = _lc_docs

_lc_splitters = types.ModuleType("langchain_text_splitters")
_lc_splitters.RecursiveCharacterTextSplitter = lambda **kwargs: MagicMock(
    split_text=lambda text: [text]
)
sys.modules["langchain_text_splitters"] = _lc_splitters

_INGEST_MODULE = "scripts.ingest.ingest"

sys.modules.pop(_INGEST_MODULE, None)
sys.modules.pop("ingest", None)
import scripts.ingest.ingest  # noqa: F401


@pytest.fixture(autouse=True)
def _reset_provider_singletons():
    from providers.factory import reset_providers

    reset_providers()
    yield
    reset_providers()


@pytest.fixture
def isolated_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "ingest_manifest.json"
    monkeypatch.setattr(sys.modules[_INGEST_MODULE], "manifest_path", lambda: manifest)
    return manifest


def test_process_url_ingest_writes_manifest_keyed_by_qdrant_source(
    tmp_path, isolated_manifest, monkeypatch
):
    url = "https://example.com/docs/calibration-guide"
    vendor = "lmi"
    html = "<html><body><h1>Calibration</h1><p>" + ("word " * 200) + "</p></body></html>"
    filename = "calibration-guide-deadbeef.html"

    ingest_mod = sys.modules[_INGEST_MODULE]
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(ingest_mod, "_filename_from_url", return_value=filename),
        patch.object(ingest_mod, "_fetch_url_html", return_value=html),
        patch.object(ingest_mod, "ingest", return_value=7) as ingest_mock,
    ):
        status = ingest_mod._process_url_ingest(
            url,
            vendor=vendor,
            product="gocator",
            product_version=None,
            doc_type="article",
            force=True,
        )

    assert status == "ingested"
    ingest_mock.assert_called_once()
    data = json.loads(isolated_manifest.read_text(encoding="utf-8"))
    assert filename in data
    assert url not in data
    row = data[filename]
    assert row["vendor"] == vendor
    assert row["product"] == "gocator"
    assert row["chunk_count"] == 7
    assert row["content_type"] == "text"
    assert row["source_type"] == "html"
    assert row["sha256"]
    assert row["ingested_at"]


def test_reingest_html_url_writes_manifest_keyed_by_url(tmp_path, isolated_manifest):
    import scripts.ingest.reingest_all as reingest_all

    url = "https://support.example.com/article/setup-12345678.html"
    vendor = "pekat"
    html = "<html><body><h1>Setup</h1><p>" + ("setup " * 200) + "</p></body></html>"

    fake_store = MagicMock()
    config = {"keywords": {vendor: {"category": vendor, "device_families": []}}}

    ingest_mod = sys.modules[_INGEST_MODULE]
    with (
        patch.object(reingest_all, "fetch_html_with_retry", return_value=html),
        patch("providers.factory.get_vector_store", return_value=fake_store),
        patch.object(ingest_mod, "manifest_path", lambda: isolated_manifest),
        patch.object(ingest_mod, "ingest", return_value=5),
        patch.object(reingest_all, "derive_html_metadata") as derive_meta,
    ):
        meta = MagicMock()
        meta.ingest_kwargs.return_value = {
            "vendor": vendor,
            "product": None,
            "product_version": None,
            "doc_type": "article",
        }
        meta.payload_extra.return_value = {"source_type": "html"}
        derive_meta.return_value = meta

        ok = reingest_all.ingest_html_url(
            url,
            vendor=vendor,
            config=config,
            extra_by_source={},
            stats=reingest_all.RunStats(),
        )

    assert ok is True
    data = json.loads(isolated_manifest.read_text(encoding="utf-8"))
    assert url in data
    row = data[url]
    assert row["vendor"] == vendor
    assert row["chunk_count"] == 5
    assert row["content_type"] == "text"
    assert row["source_type"] == "html"
