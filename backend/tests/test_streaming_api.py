"""Unit tests for Server-Sent Events (SSE) Query Streaming API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.planner import PlannedToolCall, PlanResult
from app.api.main import create_app
from app.retrieval.hybrid_search import HybridSearchResult


@pytest.fixture(autouse=True)
def mock_cache_store():
    with patch("app.storage.cache_store.CacheStore.get_query", new_callable=AsyncMock, return_value=None):
        yield


@pytest.mark.asyncio
async def test_stream_query_endpoint() -> None:
    app = create_app()

    mock_session_factory = MagicMock()
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    async def mock_stream_gen(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield "Stocks surged in heavy trading."

    mock_plan = PlanResult(
        archetype="fact_lookup",
        reasoning="Analyzing market rally query.",
        tool_calls=[
            PlannedToolCall(
                tool_name="hybrid_search",
                arguments={"query": "What happened to the markets?", "top_k": 6},
                purpose="Retrieve factual details",
            )
        ],
    )

    with (
        patch("app.api.routers.query.get_session_factory", return_value=mock_session_factory),
        patch("app.agent.planner.QueryPlanner.plan_query_async", new_callable=AsyncMock, return_value=mock_plan),
        patch("app.agent.graph.HybridSearchEngine.search", new_callable=AsyncMock) as mock_search,
        patch(
            "app.agent.synthesizer.AnswerSynthesizer.synthesize_stream",
            side_effect=mock_stream_gen,
        ),
    ):
        mock_search.return_value = [
            HybridSearchResult(
                article_id=1,
                headline="MARKET RALLIES",
                subheadline=None,
                byline_author="Reporter",
                section="Business",
                article_type="news",
                prominence_score=0.9,
                rrf_score=0.03,
                vector_rank=1,
                keyword_rank=1,
                snippet="Stocks surged in heavy trading.",
                newspaper_name="The Daily Record",
                issue_date="2026-08-21",
                pages=[1],
            )
        ]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query/stream",
                json={"query": "What happened to the markets?"},
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            content = response.text
            assert "event: stage" in content
            assert "event: plan" in content
            assert "event: token" in content
            assert "event: citations" in content
            assert "event: done" in content


@pytest.mark.asyncio
async def test_stream_query_with_think_tags() -> None:
    """Verify that <think>...</think> reasoning tags are parsed into event: thought."""
    app = create_app()

    mock_session_factory = MagicMock()
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    async def mock_stream_gen(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield "<think>\nAnalyzing user request...\nEvaluating evidence...\n</think>\n\n"
        yield "This is the final "
        yield "synthesized answer."

    mock_plan = PlanResult(
        archetype="fact_lookup",
        reasoning="Analyzing 3-nation trip query.",
        tool_calls=[
            PlannedToolCall(
                tool_name="hybrid_search",
                arguments={"query": "PM 3-nation trip to boost economic engagement", "top_k": 6},
                purpose="Retrieve trip details",
            )
        ],
    )

    with (
        patch("app.api.routers.query.get_session_factory", return_value=mock_session_factory),
        patch("app.agent.planner.QueryPlanner.plan_query_async", new_callable=AsyncMock, return_value=mock_plan),
        patch("app.agent.graph.HybridSearchEngine.search", new_callable=AsyncMock) as mock_search,
        patch(
            "app.agent.synthesizer.AnswerSynthesizer.synthesize_stream",
            side_effect=mock_stream_gen,
        ),
    ):
        mock_search.return_value = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query/stream",
                json={"query": "PM 3-nation trip to boost economic engagement"},
            )
            assert response.status_code == 200
            content = response.text
            assert "event: thought" in content
            assert "event: thought_done" in content
            assert "event: token" in content
            assert "event: done" in content
            assert "Analyzing user request" in content
            assert "This is the final" in content


@pytest.mark.asyncio
async def test_stream_query_with_exclusion_phrase_in_history() -> None:
    """Verify that chat history containing exclusion phrases like 'not in' with 1 newspaper does not raise IndexError."""
    app = create_app()

    mock_session_factory = MagicMock()
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    async def mock_stream_gen(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield "No visual photos were attached to this article."

    mock_plan = PlanResult(
        archetype="fact_lookup",
        reasoning="Looking up visual photos.",
        tool_calls=[],
    )

    with (
        patch("app.api.routers.query.get_session_factory", return_value=mock_session_factory),
        patch("app.agent.planner.QueryPlanner.plan_query_async", new_callable=AsyncMock, return_value=mock_plan),
        patch(
            "app.agent.synthesizer.AnswerSynthesizer.synthesize_stream",
            side_effect=mock_stream_gen,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query/stream",
                json={
                    "query": "any visual photo attached with it?",
                    "chat_history": [
                        {
                            "role": "user",
                            "content": "Why was the article in The Goan but not in the front page?",
                        },
                        {
                            "role": "assistant",
                            "content": "### ⚡ Executive Summary\nThe Goan published on 2026-08-05, Page 5.",
                        },
                    ],
                },
            )
            assert response.status_code == 200
            assert "event: done" in response.text


@pytest.mark.asyncio
async def test_stream_query_empty_answer_preserves_archetype_in_fallback() -> None:
    app = create_app()

    mock_session_factory = MagicMock()
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    # Model produces reasoning but 0 answer chunks
    async def mock_stream_gen(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        yield "<think>\nDeep comparative analysis across broadsheet archives...\n</think>\n\n"

    mock_plan = PlanResult(
        archetype="cross_newspaper_comparison",
        reasoning="Comparative query.",
        tool_calls=[
            PlannedToolCall(
                tool_name="hybrid_search",
                arguments={"query": "health news", "top_k": 6},
                purpose="Compare health",
            )
        ],
    )

    with (
        patch("app.api.routers.query.get_session_factory", return_value=mock_session_factory),
        patch("app.agent.planner.QueryPlanner.plan_query_async", new_callable=AsyncMock, return_value=mock_plan),
        patch("app.agent.graph.HybridSearchEngine.search", new_callable=AsyncMock) as mock_search,
        patch(
            "app.agent.synthesizer.AnswerSynthesizer.synthesize_stream",
            side_effect=mock_stream_gen,
        ),
    ):
        mock_search.return_value = [
            HybridSearchResult(
                article_id=1,
                headline="Hospital Wing Opened",
                subheadline=None,
                byline_author="Staff",
                section="Health",
                article_type="news",
                prominence_score=0.9,
                rrf_score=0.03,
                vector_rank=1,
                keyword_rank=1,
                snippet="Modern hospital opened in state.",
                newspaper_name="The Goan",
                issue_date="2026-08-01",
                pages=[1],
            ),
            HybridSearchResult(
                article_id=2,
                headline="Clinic Subsidies Given",
                subheadline=None,
                byline_author="Staff",
                section="Health",
                article_type="news",
                prominence_score=0.9,
                rrf_score=0.03,
                vector_rank=1,
                keyword_rank=1,
                snippet="Clinics received new medical funds.",
                newspaper_name="The Morning Standard",
                issue_date="2026-08-01",
                pages=[5],
            ),
        ]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/query/stream",
                json={"query": "COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on health related news"},
            )
            assert response.status_code == 200
            assert "event: done" in response.text
            # The fallback must preserve archetype and render the comparison matrix, NOT single paper
            assert "Comparison" in response.text and "Matrix" in response.text
            assert "The Goan" in response.text
            assert "The Morning Standard" in response.text


