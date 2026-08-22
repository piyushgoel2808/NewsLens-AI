"""Prometheus metrics instrumentation for NewsLens-AI.

Provides non-blocking, thread-safe metrics tracking for HTTP traffic,
LLM token consumption, dollar costs, cache efficiency, provider fallbacks,
and broadsheet ingestion performance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# HTTP Traffic & Latency Metrics
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "newslens_http_requests_total",
    "Total count of HTTP requests processed by endpoint and status code.",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "newslens_http_request_duration_seconds",
    "Histogram of HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# Agentic RAG Query & Token Metrics
# ---------------------------------------------------------------------------

AGENT_QUERIES_TOTAL = Counter(
    "newslens_agent_queries_total",
    "Total count of agentic research queries executed.",
    ["archetype", "status", "model"],
)

AGENT_QUERY_DURATION_SECONDS = Histogram(
    "newslens_agent_query_duration_seconds",
    "End-to-end agentic query processing duration in seconds.",
    ["archetype"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

LLM_TOKENS_TOTAL = Counter(
    "newslens_llm_tokens_total",
    "Cumulative count of tokens processed by LLM/VLM providers.",
    ["provider", "model", "direction"],  # direction: input | output
)

LLM_COST_USD_TOTAL = Counter(
    "newslens_llm_cost_usd_total",
    "Cumulative dollar expenditure on model provider completions.",
    ["provider", "model"],
)

# ---------------------------------------------------------------------------
# Caching, Resilience & Fallback Metrics
# ---------------------------------------------------------------------------

CACHE_EVENTS_TOTAL = Counter(
    "newslens_cache_events_total",
    "Cache lookup events (hit, miss, set, error).",
    ["cache_type", "event"],  # cache_type: query | embedding; event: hit | miss | set | error
)

CASCADE_FALLBACKS_TOTAL = Counter(
    "newslens_cascade_fallbacks_total",
    "Count of provider fallback cascade activations.",
    ["primary_provider", "fallback_provider", "reason"],
)

# ---------------------------------------------------------------------------
# Ingestion Pipeline Metrics
# ---------------------------------------------------------------------------

INGESTION_PAGES_TOTAL = Counter(
    "newslens_ingestion_pages_total",
    "Total newspaper broadsheet pages ingested into archive.",
    ["newspaper", "extraction_mode"],  # extraction_mode: digital | ocr | advertisement
)

INGESTION_STAGE_DURATION_SECONDS = Histogram(
    "newslens_ingestion_stage_duration_seconds",
    "Duration of individual broadsheet ingestion pipeline stages in seconds.",
    ["stage"],  # stage: rasterize | docling_parse | ocr | segment | embed
    buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

CELERY_ACTIVE_TASKS = Gauge(
    "newslens_celery_active_tasks",
    "Number of active ingestion tasks in Celery queue.",
)


# ---------------------------------------------------------------------------
# Helpers & Middleware
# ---------------------------------------------------------------------------


def generate_prometheus_metrics() -> tuple[bytes, str]:
    """Export all registered Prometheus metrics in the official exposition format."""
    return generate_latest(), CONTENT_TYPE_LATEST


def record_agent_query(
    archetype: str,
    status: str,
    model: str,
    duration_seconds: float,
) -> None:
    """Record an agent query execution metrics event."""
    AGENT_QUERIES_TOTAL.labels(archetype=archetype, status=status, model=model).inc()
    AGENT_QUERY_DURATION_SECONDS.labels(archetype=archetype).observe(duration_seconds)


def record_llm_tokens_and_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """Record token counts and cost in Prometheus."""
    if input_tokens > 0:
        LLM_TOKENS_TOTAL.labels(
            provider=provider, model=model, direction="input"
        ).inc(input_tokens)
    if output_tokens > 0:
        LLM_TOKENS_TOTAL.labels(
            provider=provider, model=model, direction="output"
        ).inc(output_tokens)
    if cost_usd > 0.0:
        LLM_COST_USD_TOTAL.labels(provider=provider, model=model).inc(cost_usd)


def record_cache_event(cache_type: str, event: str) -> None:
    """Record a cache event (hit, miss, set, error)."""
    CACHE_EVENTS_TOTAL.labels(cache_type=cache_type, event=event).inc()


def record_cascade_fallback(
    primary_provider: str,
    fallback_provider: str,
    reason: str,
) -> None:
    """Record a fallback event when primary provider fails."""
    CASCADE_FALLBACKS_TOTAL.labels(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        reason=reason[:64],
    ).inc()


class PrometheusMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for non-blocking HTTP request latency and status metrics."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        # Ignore metrics and health endpoint calls from cluttering request metrics
        path = request.url.path
        if path in ("/metrics", "/health", "/favicon.ico"):
            return await call_next(request)  # type: ignore[no-any-return]

        method = request.method
        start_time = time.monotonic()

        try:
            response: Response = await call_next(request)
            duration = time.monotonic() - start_time
            status_code = str(response.status_code)

            HTTP_REQUESTS_TOTAL.labels(
                method=method, endpoint=path, status_code=status_code
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method, endpoint=path
            ).observe(duration)
            return response
        except Exception:
            duration = time.monotonic() - start_time
            HTTP_REQUESTS_TOTAL.labels(
                method=method, endpoint=path, status_code="500"
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method, endpoint=path
            ).observe(duration)
            raise
