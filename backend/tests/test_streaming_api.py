"""Unit tests for Server-Sent Events (SSE) Query Streaming API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.retrieval.hybrid_search import HybridSearchResult


@pytest.mark.asyncio
async def test_stream_query_endpoint() -> None:
    app = create_app()

    mock_session_factory = MagicMock()
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    with (
        patch("app.api.routers.query.get_session_factory", return_value=mock_session_factory),
        patch("app.agent.graph.HybridSearchEngine.search", new_callable=AsyncMock) as mock_search,
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

    with (
        patch("app.api.routers.query.get_session_factory", return_value=mock_session_factory),
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
