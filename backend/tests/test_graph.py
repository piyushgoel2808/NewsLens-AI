"""Unit tests for LangGraph Agentic RAG state machine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.graph import AgentWorkflow
from app.providers.base import ModelResponse
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
        workflow._cache.get_query = AsyncMock(return_value=None)  # type: ignore[method-assign]

        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        mock_provider.complete = AsyncMock(
            return_value=ModelResponse(
                text="MARKET SURGE REPORTED: Heavy buying drove stocks upward.",
                input_tokens=50,
                output_tokens=20,
            )
        )
        workflow._synthesizer._provider = mock_provider
        workflow._planner._provider = mock_provider

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

    @pytest.mark.asyncio
    async def test_execute_tools_node_parallel_gather(self) -> None:
        """Verify that _execute_tools_node executes multiple tools concurrently."""
        import asyncio
        mock_session_factory = MagicMock()
        workflow = AgentWorkflow(session_factory=mock_session_factory)

        execution_order = []

        async def slow_tool_1(*args: Any, **kwargs: Any) -> list[Any]:
            await asyncio.sleep(0.05)
            execution_order.append("slow_1")
            return []

        async def slow_tool_2(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(0.05)
            execution_order.append("slow_2")
            return {"total_articles": 5, "articles": []}

        workflow._hybrid_search.search = AsyncMock(side_effect=slow_tool_1)  # type: ignore[method-assign]
        workflow._sql_analytics.get_issue_summary = AsyncMock(side_effect=slow_tool_2)  # type: ignore[method-assign]

        state = {
            "query": "compare coverage",
            "archetype": "cross_newspaper_comparison",
            "plan": [
                {"tool_name": "hybrid_search", "arguments": {"query": "test"}},
                {"tool_name": "sql_analytics", "arguments": {"analysis_type": "issue_summary", "newspaper_name": "Paper A"}},
            ],
            "active_issue_id": None,
            "active_newspaper_name": None,
            "active_issue_date": None,
        }

        t_start = asyncio.get_event_loop().time()
        res = await workflow._execute_tools_node(state)  # type: ignore[arg-type]
        elapsed = asyncio.get_event_loop().time() - t_start

        # If executed sequentially, elapsed >= 0.10s. If parallel, elapsed ~0.05s (< 0.09s)
        assert len(res["tool_executions"]) == 2
        assert len(execution_order) == 2
        assert elapsed < 0.09

    @pytest.mark.asyncio
    async def test_execute_single_tool_category_fallback(self) -> None:
        """Verify that hybrid_search falls back to unconstrained search if category returns 0 hits."""
        mock_session_factory = MagicMock()
        workflow = AgentWorkflow(session_factory=mock_session_factory)

        call_counts = []

        async def mock_search(query: str, top_k: int = 6, filters: Any = None) -> list[Any]:
            call_counts.append(filters.category_name if filters else None)
            if filters and filters.category_name == "SpecificNonExistentCat":
                return []
            return [
                HybridSearchResult(
                    article_id=99,
                    headline="Unconstrained Result Found",
                    subheadline=None,
                    byline_author="Staff",
                    section="General",
                    article_type="news",
                    prominence_score=0.8,
                    rrf_score=0.02,
                    vector_rank=1,
                    keyword_rank=1,
                    snippet="Found without category constraint.",
                    newspaper_name="The Daily",
                    issue_date="2026-08-01",
                    pages=[1],
                )
            ]

        workflow._hybrid_search.search = AsyncMock(side_effect=mock_search)  # type: ignore[method-assign]

        call = {
            "tool_name": "hybrid_search",
            "arguments": {
                "query": "market news",
                "category_filter": "SpecificNonExistentCat",
            },
        }
        state = {"query": "market news"}

        items, record, _ = await workflow._execute_single_tool(call, state)  # type: ignore[arg-type]

        # First call with SpecificNonExistentCat returned 0, second fallback call with None returned 1
        assert len(call_counts) == 2
        assert call_counts[0] == "SpecificNonExistentCat"
        assert call_counts[1] is None
        assert record["results_count"] == 1
        assert len(items) == 1
        assert items[0]["headline"] == "Unconstrained Result Found"
