"""Tests for Prometheus metrics and telemetry middleware."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import app
from app.core.metrics import (
    generate_prometheus_metrics,
    record_agent_query,
    record_cache_event,
    record_cascade_fallback,
    record_llm_tokens_and_cost,
)


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint() -> None:
    """Verify /metrics endpoint returns valid Prometheus formatted data."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        text = response.text
        assert "newslens_http_requests_total" in text
        assert "newslens_agent_queries_total" in text
        assert "newslens_llm_tokens_total" in text
        assert "newslens_llm_cost_usd_total" in text


def test_record_agent_query_metric() -> None:
    """Verify recording an agent query updates Prometheus counters."""
    record_agent_query(
        archetype="thematic_timeline",
        status="success",
        model="llama3.3:70b",
        duration_seconds=1.25,
    )
    raw_data, _ = generate_prometheus_metrics()
    assert b"newslens_agent_queries_total" in raw_data
    assert b'archetype="thematic_timeline"' in raw_data


def test_record_llm_tokens_and_cost() -> None:
    """Verify token count and cost recording."""
    record_llm_tokens_and_cost(
        provider="anthropic",
        model="claude-3-5-sonnet",
        input_tokens=1500,
        output_tokens=300,
        cost_usd=0.009,
    )
    raw_data, _ = generate_prometheus_metrics()
    assert b'provider="anthropic"' in raw_data
    assert b"newslens_llm_cost_usd_total" in raw_data


def test_record_cache_events_and_cascade() -> None:
    """Verify cache and cascade fallback tracking."""
    record_cache_event("query", "hit")
    record_cache_event("query", "miss")
    record_cascade_fallback("groq", "anthropic", "Rate limit HTTP 429")

    raw_data, _ = generate_prometheus_metrics()
    assert b"newslens_cache_events_total" in raw_data
    assert b"newslens_cascade_fallbacks_total" in raw_data
