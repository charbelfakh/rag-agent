"""Tests for the public rag_interface.answer() wrapper."""
from unittest.mock import patch

import pytest

from rag_interface import answer


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
