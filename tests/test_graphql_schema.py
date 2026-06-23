"""Tests for the standalone Strawberry GraphQL schema."""
from unittest.mock import patch

from graphql_app.schema import schema

ASK_QUERY = """
query Ask($question: String!) {
    ask(question: $question) {
        answer
        sources {
            source
            vendor
            page
            file_name
            content_type
            section
            text
        }
    }
}
"""


@patch("graphql_app.schema.answer")
def test_ask_returns_answer_and_sources(mock_answer):
    mock_answer.return_value = {
        "answer": "Surface flatness is measured via laser triangulation.",
        "sources": [
            {
                "source": "manual.pdf",
                "vendor": "pekat",
                "page": 3,
                "content_type": "text",
            }
        ],
    }

    result = schema.execute_sync(
        ASK_QUERY,
        variable_values={"question": "How is surface flatness measured?"},
    )

    assert result.errors is None
    ask = result.data["ask"]
    assert ask["answer"] == "Surface flatness is measured via laser triangulation."
    assert len(ask["sources"]) == 1
    assert ask["sources"][0]["source"] == "manual.pdf"
    assert ask["sources"][0]["vendor"] == "pekat"
    mock_answer.assert_called_once_with("How is surface flatness measured?")


@patch("graphql_app.schema.answer")
def test_ask_maps_missing_optional_citation_fields(mock_answer):
    mock_answer.return_value = {
        "answer": "Reply",
        "sources": [{"source": "doc.pdf"}],
    }

    result = schema.execute_sync(
        ASK_QUERY,
        variable_values={"question": "test"},
    )

    assert result.errors is None
    source = result.data["ask"]["sources"][0]
    assert source["source"] == "doc.pdf"
    assert source["vendor"] is None
    assert source["page"] is None


def test_ask_empty_question_returns_graphql_error():
    result = schema.execute_sync(
        ASK_QUERY,
        variable_values={"question": "   "},
    )

    assert result.errors is not None
    assert any("question must be non-empty" in str(err) for err in result.errors)
