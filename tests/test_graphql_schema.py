"""Tests for the standalone Strawberry GraphQL schema."""
import asyncio
import threading
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


def _execute(query, variables):
    """Run a query through the async executor (``ask`` offloads to a thread)."""
    return asyncio.run(schema.execute(query, variable_values=variables))


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

    result = _execute(
        ASK_QUERY,
        {"question": "How is surface flatness measured?"},
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

    result = _execute(ASK_QUERY, {"question": "test"})

    assert result.errors is None
    source = result.data["ask"]["sources"][0]
    assert source["source"] == "doc.pdf"
    assert source["vendor"] is None
    assert source["page"] is None


@patch("graphql_app.schema.answer")
def test_ask_forwards_filters_to_answer(mock_answer):
    mock_answer.return_value = {"answer": "ok", "sources": []}

    scoped_query = """
    query Ask($question: String!, $vendor_filter: String) {
        ask(question: $question, vendor_filter: $vendor_filter) {
            answer
        }
    }
    """
    result = _execute(
        scoped_query,
        {"question": "scoped", "vendor_filter": "pekat"},
    )

    assert result.errors is None
    mock_answer.assert_called_once_with("scoped", vendor_filter="pekat")


def test_ask_empty_question_returns_graphql_error():
    result = _execute(ASK_QUERY, {"question": "   "})

    assert result.errors is not None
    assert any("question must be non-empty" in str(err) for err in result.errors)


ASK_STREAM_SUBSCRIPTION = """
subscription AskStream($question: String!, $vendor_filter: String) {
    askStream(question: $question, vendor_filter: $vendor_filter) {
        token
        done
        error
        sources {
            source
            vendor
        }
    }
}
"""


def _collect_subscription(query, variables):
    async def run():
        result = await schema.subscribe(query, variable_values=variables)
        # On validation/argument errors subscribe returns an ExecutionResult.
        if hasattr(result, "errors"):
            return result
        return [item async for item in result]

    return asyncio.run(run())


def _assert_stream_call(mock_stream, question, **expected_kwargs):
    """Assert answer_stream was called with question, expected kwargs, and a cancel event."""
    mock_stream.assert_called_once()
    args, kwargs = mock_stream.call_args
    assert args == (question,)
    cancel_event = kwargs.pop("cancel_event")
    assert isinstance(cancel_event, threading.Event)
    assert kwargs == expected_kwargs


@patch("graphql_app.schema.answer_stream")
def test_ask_stream_emits_tokens_then_done(mock_stream):
    mock_stream.return_value = iter(
        [
            {"token": "Surface "},
            {"token": "flatness."},
            {"done": True, "sources": [{"source": "manual.pdf", "vendor": "pekat"}]},
        ]
    )

    results = _collect_subscription(
        ASK_STREAM_SUBSCRIPTION, {"question": "How flat?"}
    )

    payloads = [r.data["askStream"] for r in results]
    assert payloads[0]["token"] == "Surface "
    assert payloads[1]["token"] == "flatness."
    done = payloads[2]
    assert done["done"] is True
    assert done["sources"] == [{"source": "manual.pdf", "vendor": "pekat"}]
    _assert_stream_call(mock_stream, "How flat?")


@patch("graphql_app.schema.answer_stream")
def test_ask_stream_forwards_vendor_filter(mock_stream):
    mock_stream.return_value = iter([{"done": True, "sources": []}])

    _collect_subscription(
        ASK_STREAM_SUBSCRIPTION,
        {"question": "scoped", "vendor_filter": "pekat"},
    )

    _assert_stream_call(mock_stream, "scoped", vendor_filter="pekat")


@patch("graphql_app.schema.answer_stream")
def test_ask_stream_surfaces_error_event(mock_stream):
    mock_stream.return_value = iter(
        [{"token": "partial"}, {"done": True, "error": "llm exploded"}]
    )

    results = _collect_subscription(ASK_STREAM_SUBSCRIPTION, {"question": "boom"})

    last = results[-1].data["askStream"]
    assert last["done"] is True
    assert last["error"] == "llm exploded"


@patch("graphql_app.schema.answer_stream")
def test_ask_stream_cancels_pipeline_on_client_disconnect(mock_stream):
    """Closing the subscription mid-stream must set the cancel event and
    close the underlying generator (so pipeline cleanup runs)."""
    generator_closed = threading.Event()
    captured: dict = {}

    def fake_stream(question, *, cancel_event=None, **kwargs):
        captured["cancel_event"] = cancel_event

        def gen():
            try:
                while True:
                    yield {"token": "x"}
            finally:
                generator_closed.set()

        return gen()

    mock_stream.side_effect = fake_stream

    async def run():
        stream = await schema.subscribe(
            ASK_STREAM_SUBSCRIPTION, variable_values={"question": "q"}
        )
        first = await stream.__anext__()
        assert first.data["askStream"]["token"] == "x"
        # Simulate the websocket client going away.
        await stream.aclose()

    asyncio.run(run())

    assert captured["cancel_event"] is not None
    assert captured["cancel_event"].is_set()
    assert generator_closed.is_set()


class TestCorsOrigins:
    """graphql_app.main.cors_origins() env-driven allowlist."""

    def test_defaults_to_localhost_dev_hosts(self, monkeypatch):
        monkeypatch.delenv("GRAPHQL_CORS_ORIGINS", raising=False)
        from graphql_app.main import cors_origins

        origins = cors_origins()
        assert "http://localhost:5173" in origins
        assert all(o.startswith(("http://localhost", "http://127.0.0.1")) for o in origins)

    def test_env_override_replaces_defaults(self, monkeypatch):
        monkeypatch.setenv("GRAPHQL_CORS_ORIGINS", "https://rag.example.com, https://ops.example.com")
        from graphql_app.main import cors_origins

        assert cors_origins() == ["https://rag.example.com", "https://ops.example.com"]
