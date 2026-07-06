"""Standalone FastAPI app serving the Strawberry GraphQL RAG API."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from graphql_app.schema import schema

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
)


def cors_origins() -> list[str]:
    """Allowed browser origins; localhost dev hosts unless GRAPHQL_CORS_ORIGINS
    (comma-separated) narrows or extends the list for a real deployment."""
    configured = os.getenv("GRAPHQL_CORS_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return list(DEFAULT_CORS_ORIGINS)


app = FastAPI(title="RAG GraphQL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graphql_router = GraphQLRouter(schema, graphql_ide="graphiql")
app.include_router(graphql_router, prefix="/graphql")
