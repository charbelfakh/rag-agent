"""Strawberry GraphQL schema for the standalone RAG query API."""
from __future__ import annotations

import strawberry

from rag_interface import answer


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


@strawberry.type
class Query:
    @strawberry.field
    def ask(self, question: str) -> AskResult:
        result = answer(question)
        sources = [_citation_to_source(c) for c in result.get("sources") or []]
        return AskResult(answer=result["answer"], sources=sources)


schema = strawberry.Schema(query=Query)
