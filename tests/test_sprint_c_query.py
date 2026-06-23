"""Sprint C unit tests: prompt headers, token packing, vendor filter."""
import os
from unittest.mock import MagicMock

import pytest

import providers.rag_pipeline as rag_pipeline
from conftest import patch_retrieval_pipeline
from providers.rag_pipeline import (
    _accept_condensed_question,
    _build_generation_plan,
    _condensation_timeout_seconds,
    _infer_vendors_from_text,
    _resolve_retrieval_vendors,
    _sanitize_condensed_question,
    _search_filter,
    build_prompt,
    collect_citations,
    condense_question_for_retrieval,
    format_chunk_for_prompt,
    format_chunk_header,
    select_chunks_for_prompt,
)


def _chunk(**kwargs) -> dict:
    base = {
        "text": "chunk body text",
        "source": "docs/pekat/manual.pdf",
        "file_name": "manual.pdf",
        "vendor": "pekat",
        "page": 1,
        "section": "Triggers",
        "score": 0.9,
    }
    base.update(kwargs)
    return base


@pytest.fixture(autouse=True)
def env_defaults(monkeypatch):
    monkeypatch.setenv("HYDE_ENABLED", "false")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("MAX_PROMPT_TOKENS", "12000")
    monkeypatch.setenv("CHUNK_PROMPT_MAX_CHARS", "1000")
    monkeypatch.delenv("ANALYTICS_ENABLED", raising=False)


class TestChunkFormatting:
    def test_format_chunk_header_with_section(self):
        header = format_chunk_header(_chunk())
        assert header == "[manual.pdf p.2 §Triggers]"

    def test_format_chunk_header_null_page_html(self):
        header = format_chunk_header(
            _chunk(
                source="what-is-pekat-vision-abc12345.html",
                file_name="what-is-pekat-vision-abc12345.html",
                page=None,
                section="Overview",
            )
        )
        assert header == "[what-is-pekat-vision-abc12345.html §Overview]"

    def test_format_chunk_header_video_timestamp(self):
        header = format_chunk_header(
            {
                "source": "Bin Picking Demo",
                "content_type": "video_transcript",
                "start_seconds": 125.4,
            }
        )
        assert header == "[Bin Picking Demo t=125.4s]"

    def test_collect_citations_handles_null_page(self):
        cites = collect_citations([_chunk(page=None, source="page.html")])
        assert cites[0]["page"] is None

    def test_format_chunk_truncates_long_text(self, monkeypatch):
        monkeypatch.setenv("CHUNK_PROMPT_MAX_CHARS", "20")
        formatted = format_chunk_for_prompt(_chunk(text="x" * 50))
        assert "…" in formatted
        assert len(formatted.split("\n", 1)[1]) == 21

    def test_build_prompt_includes_headers(self):
        prompt = build_prompt("What is Pekat?", [_chunk()])
        assert "[manual.pdf p.2 §Triggers]" in prompt
        assert "chunk body text" in prompt

    def test_build_prompt_mentions_video_transcript_demonstrations(self):
        prompt = build_prompt(
            "Mech-Mind bin picking demo",
            [
                _chunk(
                    source="Random Bin Picking Tutorial Introduction",
                    content_type="video_transcript",
                    start_seconds=0.12,
                    page=None,
                )
            ],
        )
        assert "video tutorials or demonstrations" in prompt
        assert "do not refuse merely because the user asked for a demonstration" in prompt


class TestTokenPacking:
    def test_select_all_chunks_when_budget_large(self):
        chunks = [_chunk(text="short"), _chunk(text="also short", page=2)]
        selected = select_chunks_for_prompt(chunks, "question?")
        assert len(selected) == 2

    def test_select_fewer_chunks_when_budget_tight(self, monkeypatch):
        monkeypatch.setenv("MAX_PROMPT_TOKENS", "50")
        chunks = [
            _chunk(text="a" * 400),
            _chunk(text="b" * 400, page=2),
        ]
        selected = select_chunks_for_prompt(chunks, "question?")
        assert len(selected) == 1


class TestCitations:
    def test_collect_citations_includes_metadata(self):
        cites = collect_citations([_chunk()])
        assert cites[0]["file_name"] == "manual.pdf"
        assert cites[0]["vendor"] == "pekat"
        assert cites[0]["section"] == "Triggers"

    def test_collect_citations_includes_image_url_for_image_caption(self):
        cites = collect_citations(
            [
                {
                    "source": "data/pekat/KB3150.pdf",
                    "page": 2,
                    "section": "Mask",
                    "content_type": "image_caption",
                    "image_path": "data/pekat/images/_caption_cache/abc123.img",
                    "vendor": "pekat",
                }
            ]
        )
        assert cites[0]["image_path"] == "data/pekat/images/_caption_cache/abc123.img"
        assert cites[0]["image_url"] == "/image?path=data/pekat/images/_caption_cache/abc123.img"

    def test_collect_citations_includes_video_url_and_start_seconds(self):
        cites = collect_citations(
            [
                {
                    "source": "detector-and-classifier-cba3f5c6.html",
                    "page": None,
                    "section": "Tutorial",
                    "content_type": "video_transcript",
                    "video_path": "data/pekat/videos/be4a2d03a9cc29b0e37e/source.mp4",
                    "start_seconds": 330.08,
                    "text": "Click start training to open the training options window.",
                    "vendor": "pekat",
                }
            ]
        )
        assert cites[0]["video_url"] == (
            "/video?path=data/pekat/videos/be4a2d03a9cc29b0e37e/source.mp4"
        )
        assert cites[0]["start_seconds"] == 330.08
        assert cites[0]["text"] == "Click start training to open the training options window."
        assert "youtube_url" not in cites[0]

    def test_collect_citations_youtube_url_without_video_path(self):
        cites = collect_citations(
            [
                {
                    "source": "Bin Picking Demo",
                    "content_type": "video_transcript",
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "video_id": "abc123",
                    "text": "Robot picks parts from the bin.",
                    "vendor": "mechmind",
                }
            ]
        )
        assert cites[0]["youtube_url"] == "https://www.youtube.com/watch?v=abc123"
        assert cites[0]["video_id"] == "abc123"
        assert cites[0]["source"] == "Bin Picking Demo"
        assert "video_url" not in cites[0]

    def test_collect_citations_youtube_url_appends_timestamp(self):
        cites = collect_citations(
            [
                {
                    "source": "Calibration Tutorial",
                    "content_type": "video_transcript",
                    "url": "https://www.youtube.com/watch?v=xyz789",
                    "start_seconds": 42.5,
                    "text": "Align the calibration board.",
                }
            ]
        )
        assert cites[0]["youtube_url"] == "https://www.youtube.com/watch?v=xyz789&t=42s"
        assert cites[0]["video_id"] == "xyz789"
        assert cites[0]["start_seconds"] == 42.5
        assert cites[0]["text"] == "Align the calibration board."

    def test_collect_citations_video_id_parsed_from_url(self):
        cites = collect_citations(
            [
                {
                    "source": "Bin Picking Demo",
                    "content_type": "video_transcript",
                    "url": "https://www.youtube.com/watch?v=WMcn5O9GFKM",
                    "start_seconds": 93.4,
                    "text": "Overview of random bin picking.",
                }
            ]
        )
        assert cites[0]["video_id"] == "WMcn5O9GFKM"
        assert cites[0]["youtube_url"] == (
            "https://www.youtube.com/watch?v=WMcn5O9GFKM&t=93s"
        )

    def test_collect_citations_youtube_from_video_id_without_url(self):
        cites = collect_citations(
            [
                {
                    "source": "Random Bin Picking Tutorial Introduction",
                    "content_type": "text",
                    "video_id": "WMcn5O9GFKM",
                    "start_seconds": 12.5,
                    "text": "Overview of random bin picking.",
                }
            ]
        )
        assert cites[0]["content_type"] == "video_transcript"
        assert cites[0]["video_id"] == "WMcn5O9GFKM"
        assert cites[0]["youtube_url"] == (
            "https://www.youtube.com/watch?v=WMcn5O9GFKM&t=12s"
        )

    def test_collect_citations_video_transcripts_sorted_first(self):
        cites = collect_citations(
            [
                _chunk(source="manual.pdf", content_type="text"),
                {
                    "source": "Tutorial",
                    "content_type": "video_transcript",
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "video_id": "abc123",
                    "text": "demo",
                },
            ]
        )
        assert cites[0]["content_type"] == "video_transcript"

    def test_local_video_path_preferred_over_youtube_url(self):
        cites = collect_citations(
            [
                {
                    "source": "Tutorial",
                    "content_type": "video_transcript",
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "video_path": "data/pekat/videos/be4a2d03a9cc29b0e37e/source.mp4",
                }
            ]
        )
        assert "video_url" in cites[0]
        assert "youtube_url" not in cites[0]

    def test_text_citation_payload_unchanged_without_video_fields(self):
        cites = collect_citations([_chunk()])
        assert "video_url" not in cites[0]
        assert "start_seconds" not in cites[0]
        assert cites[0]["content_type"] == "text"
        assert cites[0]["source"] == "docs/pekat/manual.pdf"
        assert cites[0]["vendor"] == "pekat"


class TestVendorInference:
    def test_infer_single_vendor_from_question(self):
        assert _infer_vendors_from_text(
            "how does surface flatness work in pekat?"
        ) == ["pekat"]

    def test_infer_multiple_vendors(self):
        assert _infer_vendors_from_text(
            "compare pekat and lmi surface inspection"
        ) == ["pekat", "lmi"]

    def test_multiple_keywords_apply_no_filter(self):
        vendors, inferred, mechanism = _resolve_retrieval_vendors(
            question="compare pekat and lmi surface inspection",
            retrieval_question="compare pekat and lmi surface inspection",
            history=None,
            vendor=None,
            product=None,
            vendor_filter=None,
        )
        assert vendors is None
        assert inferred is False
        assert mechanism == "none"

    def test_infer_no_vendor(self):
        assert _infer_vendors_from_text("how does surface flatness work?") == []

    @pytest.mark.parametrize(
        "question",
        [
            "Mech-Mind bin picking",
            "MechMind bin picking demo",
            "mech mind robotic camera",
        ],
    )
    def test_infer_mechmind_hyphen_and_spacing_variants(self, question):
        assert _infer_vendors_from_text(question) == ["mechmind"]

    def test_infer_zivid_unchanged(self):
        assert _infer_vendors_from_text("Zivid hand eye calibration tutorial") == ["zivid"]

    def test_infer_lmi_technologies_alias(self):
        assert _infer_vendors_from_text("LMI Technologies Gocator setup") == ["lmi"]

    def test_explicit_filter_overrides_inference(self):
        vendors, inferred, mechanism = _resolve_retrieval_vendors(
            question="how does flatness work in pekat?",
            retrieval_question="how does flatness work in pekat?",
            history=None,
            vendor="lmi",
            product=None,
            vendor_filter=None,
        )
        assert vendors == ["lmi"]
        assert inferred is False
        assert mechanism == "explicit"

    def test_build_plan_infers_vendor_for_store(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = [_chunk(vendor="pekat")]
        cache = MagicMock()
        cache.enabled = False

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=MagicMock(),
            cache=cache,
            reranker=MagicMock(),
        )
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        plan = _build_generation_plan(
            "how does surface flatness work in pekat?",
            top_k=5,
        )

        store.search.assert_called()
        assert store.search.call_args.kwargs["filter_payload"] == {
            "vendor": "pekat",
        }
        assert plan.analytics.inferred_vendor_filter is True
        assert plan.analytics.filter_mechanism == "keyword"
        assert plan.prompt is not None


class TestVendorFilter:
    def test_search_filter_builds_payload(self):
        assert _search_filter("Pekat", None) == {"vendor": "pekat"}
        assert _search_filter(None, "Manual") == {"document_type": "manual"}
        assert _search_filter("pekat", "datasheet") == {
            "vendor": "pekat",
            "document_type": "datasheet",
        }
        assert _search_filter(None, None) is None

    def test_build_plan_passes_vendor_filter_to_store(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = [_chunk()]
        cache = MagicMock()
        cache.enabled = False

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=MagicMock(),
            cache=cache,
            reranker=MagicMock(),
        )
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        plan = _build_generation_plan(
            "question",
            top_k=5,
            vendor_filter="pekat",
        )

        store.search.assert_called_once()
        assert store.search.call_args.kwargs["filter_payload"] == {"vendor": "pekat"}
        assert plan.prompt is not None
        assert "[manual.pdf" in plan.prompt


class TestQueryCondensation:
    def test_sanitize_strips_prefix_and_quotes(self):
        assert (
            _sanitize_condensed_question('Rewritten question: "What does LMI provide?"')
            == "What does LMI provide?"
        )

    def test_accept_rejects_answer_without_question_mark(self):
        assert not _accept_condensed_question(
            "what do they provide?",
            "LMI Technologies provides 3D sensors.",
        )

    def test_accept_rejects_overlong_output(self):
        original = "tell me more?"
        condensed = "x" * 301
        assert not _accept_condensed_question(original, condensed)

    def test_empty_history_skips_llm(self):
        llm = MagicMock()
        result = condense_question_for_retrieval("what do they provide?", [], llm)
        assert result == "what do they provide?"
        llm.generate.assert_not_called()

    def test_history_condenses_for_retrieval(self):
        llm = MagicMock()
        llm.generate.return_value = "What products does LMI Technologies provide?"
        history = [
            {"role": "user", "content": "what is LMI technologies?"},
            {"role": "assistant", "content": "LMI makes 3D sensors."},
        ]
        result = condense_question_for_retrieval("what do they provide?", history, llm)
        assert result == "What products does LMI Technologies provide?"
        llm.generate.assert_called_once()
        assert "what is LMI technologies?" in llm.generate.call_args[0][0]

    def test_condensation_failure_falls_back(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("ollama down")
        history = [{"role": "user", "content": "what is Pekat?"}]
        result = condense_question_for_retrieval("tell me more", history, llm)
        assert result == "tell me more"

    def test_condensation_timeout_seconds_reads_env(self, monkeypatch):
        monkeypatch.delenv("CONDENSATION_TIMEOUT_SECONDS", raising=False)
        assert _condensation_timeout_seconds() == 10.0
        monkeypatch.setenv("CONDENSATION_TIMEOUT_SECONDS", "25")
        assert _condensation_timeout_seconds() == 25.0

    def test_build_plan_uses_condensed_for_retrieval(self, monkeypatch):
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1]]
        store = MagicMock()
        store.search.return_value = [_chunk(vendor="lmi")]
        cache = MagicMock()
        cache.enabled = False
        llm = MagicMock()
        llm.generate.return_value = "What does LMI Technologies provide?"

        patch_retrieval_pipeline(
            monkeypatch,
            embedder=embedder,
            store=store,
            llm=llm,
            cache=cache,
            reranker=MagicMock(),
        )
        monkeypatch.setattr(rag_pipeline, "log_query", lambda **kwargs: None)

        history = [
            {"role": "user", "content": "what is LMI technologies?"},
            {"role": "assistant", "content": "LMI makes 3D sensors."},
        ]
        plan = _build_generation_plan(
            "what do they provide?",
            top_k=5,
            history=history,
        )

        embedder.embed.assert_called()
        assert embedder.embed.call_args[0][0] == [
            "What does LMI Technologies provide?"
        ]
        assert plan.retrieval_question == "What does LMI Technologies provide?"
        assert "what do they provide?" in plan.prompt
        assert plan.analytics.condensed_question == "What does LMI Technologies provide?"
