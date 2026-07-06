"""Tests for the public rag_interface.answer() wrapper."""
import threading
from unittest.mock import patch

import pytest

from rag_interface import answer, answer_stream


@patch("rag_interface.query")
def test_answer_returns_answer_and_sources(mock_query):
    mock_query.return_value = {
        "answer": "Surface flatness is measured via ...",
        "citations": [
            {
                "source": "manual.pdf",
                "vendor": "pekat",
                "page": 3,
                "content_type": "text",
            }
        ],
        "meta": {"latency_ms": 42},
    }

    result = answer("How is surface flatness measured?")

    assert set(result.keys()) == {"answer", "sources"}
    assert result["answer"] == "Surface flatness is measured via ..."
    assert result["sources"] == mock_query.return_value["citations"]
    mock_query.assert_called_once_with("How is surface flatness measured?")


@patch("rag_interface.query")
def test_answer_sources_defaults_to_empty_list(mock_query):
    mock_query.return_value = {
        "answer": "Cached reply",
        "citations": [],
        "meta": {"cache_hit": True},
    }

    result = answer("cached question")

    assert result["sources"] == []
    assert isinstance(result["sources"], list)


@patch("rag_interface.query")
def test_answer_sources_empty_when_citations_key_missing(mock_query):
    mock_query.return_value = {
        "answer": "Reply without citations key",
        "meta": {},
    }

    result = answer("some question")

    assert result["sources"] == []


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_answer_rejects_empty_question(question):
    with pytest.raises(ValueError, match="question must be non-empty"):
        answer(question)


@patch("rag_interface.query")
def test_answer_forwards_filters_when_set(mock_query):
    mock_query.return_value = {"answer": "ok", "citations": []}

    answer(
        "scoped question",
        top_k=8,
        vendor_filter="pekat",
        document_type_filter="manual",
    )

    mock_query.assert_called_once_with(
        "scoped question",
        top_k=8,
        vendor_filter="pekat",
        document_type_filter="manual",
    )


@patch("rag_interface.query")
def test_answer_omits_unset_filters(mock_query):
    mock_query.return_value = {"answer": "ok", "citations": []}

    answer("plain question")

    mock_query.assert_called_once_with("plain question")


@patch("rag_interface.stream_query")
def test_answer_stream_yields_tokens_then_done_with_sources(mock_stream):
    mock_stream.return_value = iter(
        [
            {"token": "Surface "},
            {"stage": "retrieval_complete", "meta": {"chunks": 3}},
            {"token": "flatness."},
            {
                "done": True,
                "citations": [{"source": "manual.pdf", "vendor": "pekat"}],
                "meta": {"latency_ms": 12},
            },
        ]
    )

    events = list(answer_stream("How flat?"))

    assert events == [
        {"token": "Surface "},
        {"token": "flatness."},
        {"done": True, "sources": [{"source": "manual.pdf", "vendor": "pekat"}]},
    ]
    mock_stream.assert_called_once_with("How flat?")


@patch("rag_interface.stream_query")
def test_answer_stream_surfaces_error_event(mock_stream):
    mock_stream.return_value = iter(
        [{"token": "partial"}, {"done": True, "error": "llm exploded"}]
    )

    events = list(answer_stream("boom"))

    assert events == [
        {"token": "partial"},
        {"done": True, "error": "llm exploded"},
    ]


@patch("rag_interface.stream_query")
def test_answer_stream_forwards_filters_when_set(mock_stream):
    mock_stream.return_value = iter([{"done": True, "citations": []}])

    list(answer_stream("q", top_k=3, vendor_filter="pekat"))

    mock_stream.assert_called_once_with("q", top_k=3, vendor_filter="pekat")


@patch("rag_interface.stream_query")
def test_answer_stream_forwards_cancel_event_when_set(mock_stream):
    mock_stream.return_value = iter([{"done": True, "citations": []}])
    cancel_event = threading.Event()

    list(answer_stream("q", cancel_event=cancel_event))

    mock_stream.assert_called_once_with("q", cancel_event=cancel_event)


@pytest.mark.parametrize("question", ["", "   "])
def test_answer_stream_rejects_empty_question(question):
    with pytest.raises(ValueError, match="question must be non-empty"):
        list(answer_stream(question))


def test_answer_stream_validates_question_at_call_time():
    """The ValueError must fire on the call itself, before iteration —
    callers (e.g. the GraphQL subscription) rely on eager validation."""
    with pytest.raises(ValueError, match="question must be non-empty"):
        answer_stream("   ")
