"""Unit tests for LangGraph Agentic RAG state machine."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.graph import AgentWorkflow
from app.retrieval.hybrid_search import HybridSearchResult


class TestAgentWorkflow:
    """Test suite for AgentWorkflow state machine."""

    @pytest.mark.asyncio
    async def test_agent_workflow_execution_cycle(self) -> None:
        mock_session_factory = MagicMock()
        mock_db = MagicMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_db

        workflow = AgentWorkflow(session_factory=mock_session_factory)

        # Mock hybrid search return
        workflow._hybrid_search.search = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                HybridSearchResult(
                    article_id=1,
                    headline="MARKET SURGE REPORTED",
                    subheadline=None,
                    byline_author="Staff",
                    section="Finance",
                    article_type="news",
                    prominence_score=0.95,
                    rrf_score=0.033,
                    vector_rank=1,
                    keyword_rank=1,
                    snippet="Heavy buying drove stocks upward today.",
                    newspaper_name="The Daily Record",
                    issue_date="2026-08-21",
                    pages=[1],
                )
            ]
        )

        state = await workflow.run(query="What happened in the markets?", user_id="test_user")

        assert state["query"] == "What happened in the markets?"
        assert state["archetype"] in ("factual_lookup", "quantitative_trend")
        assert len(state["plan"]) >= 1
        assert len(state["evidence_items"]) >= 1
        assert len(state["citations"]) >= 1
        assert "MARKET SURGE" in state["synthesized_answer"]
        assert state["latency_ms"] >= 0
        assert mock_db.commit.called
