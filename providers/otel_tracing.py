"""Optional OpenTelemetry tracing alongside Langfuse (Sprint J)."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_tracer = None
_provider = None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def is_otel_enabled() -> bool:
    return _env_bool("OTEL_ENABLED")


def setup_otel(service_name: str | None = None) -> None:
    """Initialize OpenTelemetry tracing when ``OTEL_ENABLED`` is set."""
    global _tracer, _provider
    if not is_otel_enabled():
        return
    if _tracer is not None:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        resource = Resource.create(
            {
                "service.name": service_name
                or os.getenv("OTEL_SERVICE_NAME", "rag-agent"),
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = os.getenv("OTEL_EXPORTER", "console").lower()
        if exporter == "console":
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        elif exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
            )
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
        trace.set_tracer_provider(provider)
        _provider = provider
        _tracer = trace.get_tracer("rag-agent")
        logger.info("OpenTelemetry tracing enabled (%s exporter)", exporter)
    except Exception as exc:
        logger.warning("OpenTelemetry setup failed: %s", exc)


def get_tracer():
    return _tracer


def record_query_span(
    *,
    question: str,
    latency_ms: int,
    cache_hit: bool,
    chunks_retrieved: int | None,
    trace_id: str | None = None,
) -> None:
    """Emit a single ``rag_query`` span with core latency and retrieval attrs."""
    if _tracer is None:
        return
    with _tracer.start_as_current_span("rag_query") as span:
        if trace_id:
            span.set_attribute("rag.trace_id", trace_id)
        span.set_attribute("rag.question_length", len(question))
        span.set_attribute("rag.latency_ms", latency_ms)
        span.set_attribute("rag.cache_hit", cache_hit)
        if chunks_retrieved is not None:
            span.set_attribute("rag.chunks_retrieved", chunks_retrieved)


def shutdown_otel() -> None:
    global _tracer, _provider
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:
            pass
    _tracer = None
    _provider = None
