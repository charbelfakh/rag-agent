"""Strawberry GraphQL schema for the standalone RAG query API."""
from __future__ import annotations

import asyncio
import threading
from typing import Annotated, AsyncGenerator

import strawberry

from rag_interface import answer, answer_stream


@strawberry.type
class Source:
    source: str
    file_name: str | None = strawberry.field(default=None, name="file_name")
    vendor: str | None = None
    content_type: str | None = strawberry.field(default=None, name="content_type")
    page: int | None = None
    section: str | None = None
    text: str | None = None


@strawberry.type
class AskResult:
    answer: str
    sources: list[Source]


@strawberry.type
class AskStreamEvent:
    """One streamed event: a token, or the terminal done event with sources/error."""

    token: str | None = None
    done: bool = False
    sources: list[Source] | None = None
    error: str | None = None


def _citation_to_source(citation: dict) -> Source:
    return Source(
        source=citation.get("source") or "",
        file_name=citation.get("file_name"),
        vendor=citation.get("vendor"),
        content_type=citation.get("content_type"),
        page=citation.get("page"),
        section=citation.get("section"),
        text=citation.get("text"),
    )


def _retrieval_kwargs(
    top_k: int | None,
    vendor_filter: str | None,
    document_type_filter: str | None,
) -> dict:
    kwargs: dict = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if vendor_filter is not None:
        kwargs["vendor_filter"] = vendor_filter
    if document_type_filter is not None:
        kwargs["document_type_filter"] = document_type_filter
    return kwargs


@strawberry.type
class Query:
    @strawberry.field
    async def ask(
        self,
        question: str,
        top_k: Annotated[int | None, strawberry.argument(name="top_k")] = None,
        vendor_filter: Annotated[
            str | None, strawberry.argument(name="vendor_filter")
        ] = None,
        document_type_filter: Annotated[
            str | None, strawberry.argument(name="document_type_filter")
        ] = None,
    ) -> AskResult:
        kwargs = _retrieval_kwargs(top_k, vendor_filter, document_type_filter)
        # ``answer`` runs the full blocking RAG pipeline (seconds); offload it
        # so concurrent requests and websocket subscriptions stay responsive.
        result = await asyncio.to_thread(answer, question, **kwargs)
        sources = [_citation_to_source(c) for c in result.get("sources") or []]
        return AskResult(answer=result["answer"], sources=sources)


def _event_to_stream_event(event: dict) -> AskStreamEvent:
    if event.get("token") is not None:
        return AskStreamEvent(token=event["token"])
    if event.get("error"):
        return AskStreamEvent(done=True, error=event["error"])
    sources = [_citation_to_source(c) for c in event.get("sources") or []]
    return AskStreamEvent(done=True, sources=sources)


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def ask_stream(
        self,
        question: str,
        top_k: Annotated[int | None, strawberry.argument(name="top_k")] = None,
        vendor_filter: Annotated[
            str | None, strawberry.argument(name="vendor_filter")
        ] = None,
        document_type_filter: Annotated[
            str | None, strawberry.argument(name="document_type_filter")
        ] = None,
    ) -> AsyncGenerator[AskStreamEvent, None]:
        # ``answer_stream`` is a blocking sync generator (Ollama HTTP stream);
        # pump it on a worker thread one event at a time so the event loop and
        # other websocket clients stay responsive.
        kwargs = _retrieval_kwargs(top_k, vendor_filter, document_type_filter)
        cancel_event = threading.Event()
        gen = answer_stream(question, cancel_event=cancel_event, **kwargs)
        sentinel = object()

        def _next():
            return next(gen, sentinel)

        def _close():
            close = getattr(gen, "close", None)
            if close is None:
                return
            try:
                close()
            except ValueError:
                # A _next call is still executing on its worker thread;
                # cancel_event already tells the pipeline to stop.
                pass

        try:
            while True:
                event = await asyncio.to_thread(_next)
                if event is sentinel:
                    break
                yield _event_to_stream_event(event)
        finally:
            # Runs when the client disconnects mid-stream (strawberry closes
            # this async generator): stop LLM generation and release the
            # pipeline instead of letting it run to completion detached.
            cancel_event.set()
            await asyncio.to_thread(_close)


schema = strawberry.Schema(query=Query, subscription=Subscription)
