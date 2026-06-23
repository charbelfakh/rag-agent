"""Sprint C unit tests: metadata resolution and chunk payloads."""
from providers.metadata import (
    build_chunk_payload,
    infer_vendor_from_source,
    make_chunk_id,
    normalize_source,
    resolve_metadata,
)


class _FakeChunk:
    def __init__(self, text: str, page: int = 0, section: str = ""):
        self.text = text
        self.page = page
        self.section = section


class TestMetadataResolution:
    def test_normalize_source_posix(self):
        assert normalize_source("data\\manual.pdf") == "data/manual.pdf"

    def test_infer_vendor_from_docs_path(self):
        assert infer_vendor_from_source("docs/pekat/guide.pdf") == "pekat"

    def test_infer_vendor_from_data_upload(self):
        assert infer_vendor_from_source("data/upload.pdf") == "uploads"

    def test_resolve_metadata_with_override(self):
        meta = resolve_metadata(
            "docs/zivid/a.pdf",
            vendor="Custom",
            document_type="datasheet",
        )
        assert meta.vendor == "custom"
        assert meta.document_type == "datasheet"
        assert meta.file_name == "a.pdf"
        assert meta.ingestion_timestamp

    def test_stable_chunk_id(self):
        a = make_chunk_id("data/a.pdf", 3)
        b = make_chunk_id("data/a.pdf", 3)
        c = make_chunk_id("data/a.pdf", 4)
        assert a == b
        assert a != c

    def test_build_chunk_payload_includes_extended_fields(self):
        meta = resolve_metadata("docs/pekat/manual.pdf")
        payload = build_chunk_payload(_FakeChunk("hello", page=2, section="Setup"), meta, 5)
        assert payload["text"] == "hello"
        assert payload["vendor"] == "pekat"
        assert payload["chunk_index"] == 5
        assert payload["chunk_id"] == make_chunk_id(meta.source, 5)
        assert payload["file_name"] == "manual.pdf"
        assert "total_chunks" not in payload

    def test_build_chunk_payload_total_chunks(self):
        meta = resolve_metadata("data/x.pdf")
        payload = build_chunk_payload(_FakeChunk("x"), meta, 0, total_chunks=12)
        assert payload["total_chunks"] == 12

    def test_ingest_batch_pattern_stable_ids(self):
        """Mirrors ingest._flush_store ID assignment without importing ingest."""
        meta = resolve_metadata("docs/zivid/guide.pdf")
        items = [
            _FakeChunk("one", page=0, section="Intro"),
            _FakeChunk("two", page=1, section="Setup"),
        ]
        payloads = []
        ids = []
        for index, item in enumerate(items):
            payload = build_chunk_payload(item, meta, index)
            ids.append(payload["chunk_id"])
            payloads.append(payload)

        assert ids[0] == make_chunk_id(meta.source, 0)
        assert ids[1] == make_chunk_id(meta.source, 1)
        assert payloads[0]["vendor"] == "zivid"
        assert payloads[1]["chunk_index"] == 1
