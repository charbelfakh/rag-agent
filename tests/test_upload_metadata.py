"""Upload metadata inference, chunk flags, and API form fields."""
from io import BytesIO
from unittest.mock import MagicMock

from providers.metadata import (
    build_chunk_payload,
    detect_has_code,
    detect_is_procedure,
    infer_product_line_from_filename,
    infer_software_version_from_filename,
    infer_vendor_from_filename,
    resolve_metadata,
)


class _FakeChunk:
    def __init__(self, text: str, page: int = 0, section: str = ""):
        self.text = text
        self.page = page
        self.section = section


class TestVendorFilenameInference:
    def test_infer_vendor_from_filename(self):
        assert infer_vendor_from_filename("Pekat_Vision_Manual.pdf") == "pekat"
        assert infer_vendor_from_filename("pekat_vision_guide.txt") == "pekat"
        assert infer_vendor_from_filename("random_manual.pdf") is None


class TestSoftwareVersionInference:
    def test_v_dotted_pattern(self):
        assert infer_software_version_from_filename("manual_v3.15.0_release.pdf") == "3.15.0"
        assert infer_software_version_from_filename("guide_v3.15_notes.pdf") == "3.15"

    def test_compact_pattern(self):
        assert infer_software_version_from_filename("firmware_3150_build.pdf") == "3.15.0"

    def test_v_prefix_patterns(self):
        assert infer_software_version_from_filename("InSight_V3.15_Manual.pdf") == "3.15"
        assert infer_software_version_from_filename("release_V18.pdf") == "18"


class TestProductLineInference:
    def test_lmi_product_lines(self):
        assert infer_product_line_from_filename("lmi", "gocator_2500.pdf") == "Gocator"
        assert infer_product_line_from_filename("lmi", "hdrstudio_manual.pdf") == "HDR Studio"

    def test_unknown_vendor_returns_empty(self):
        assert infer_product_line_from_filename("pekat", "vision_manual.pdf") == ""


class TestChunkFlags:
    def test_detect_is_procedure(self):
        text = "1. Power on\n2. Connect camera\n3. Run calibration\nDone."
        assert detect_is_procedure(text) is True
        assert detect_is_procedure("1. Only one step") is False

    def test_detect_has_code(self):
        assert detect_has_code("def setup():\n    return 1") is True
        assert detect_has_code("#include <stdio.h>") is True
        assert detect_has_code("  indented line\n  another") is True
        assert detect_has_code("Plain prose only.") is False


class TestResolveMetadataOverrides:
    def test_explicit_product_line_and_version(self):
        meta = resolve_metadata(
            "data/upload.pdf",
            vendor="pekat",
            document_type="user_manual",
            product_line="Vision",
            software_version="3.15.0",
        )
        assert meta.vendor == "pekat"
        assert meta.product_line == "Vision"
        assert meta.software_version == "3.15.0"

    def test_filename_auto_detect_when_not_overridden(self):
        meta = resolve_metadata(
            "data/pekat_vision_v3.15_manual.pdf",
            vendor="pekat",
            document_type="user_manual",
        )
        assert meta.product_line == ""
        assert meta.software_version == "3.15"

    def test_build_chunk_payload_includes_metadata_and_flags(self):
        meta = resolve_metadata(
            "data/test.pdf",
            vendor="lmi",
            product_line="Gocator",
            software_version="5.2",
        )
        procedure = _FakeChunk("1. Step\n2. Step\n3. Step")
        payload = build_chunk_payload(procedure, meta, 0)
        assert payload["product_line"] == "Gocator"
        assert payload["software_version"] == "5.2"
        assert payload["is_procedure"] is True
        assert payload["has_code"] is False


class TestUploadApiMetadataFields:
    def test_upload_passes_metadata_to_queue(self, api_client, monkeypatch, tmp_path):
        monkeypatch.setattr("api.main.API_KEY", None)
        monkeypatch.setattr("api.main.ROOT", tmp_path)
        monkeypatch.setattr("api.main.DATA_DIR", tmp_path / "data")

        queue = MagicMock()
        queue.available = True
        queue.enqueue.return_value = True
        job_store = MagicMock()
        job_store.get.return_value = None

        monkeypatch.setattr("api.main.is_ingest_queue_enabled", lambda: True)
        monkeypatch.setattr("api.main.get_ingest_queue", lambda: queue)
        monkeypatch.setattr("api.main._job_store", lambda: job_store)

        response = api_client.post(
            "/upload",
            files={"file": ("pekat_manual.pdf", BytesIO(b"pdf"), "application/pdf")},
            data={
                "vendor": "pekat",
                "document_type": "user_manual",
                "product_line": "Vision",
                "software_version": "3.15.0",
            },
        )
        assert response.status_code == 200
        enqueued = queue.enqueue.call_args[0][0]
        assert enqueued["vendor"] == "pekat"
        assert enqueued["document_type"] == "user_manual"
        assert enqueued["product_line"] == "Vision"
        assert enqueued["software_version"] == "3.15.0"
