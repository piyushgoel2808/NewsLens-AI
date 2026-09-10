"""Unit tests for LangGraph Agentic RAG state machine."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.graph import AgentWorkflow
from app.agent.state import AgentState
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

    @pytest.mark.asyncio
    async def test_crag_semantic_hit_protection(self) -> None:
        """Verify CRAG evaluator preserves vector search hits (prominence >= 0.65) without lexical overlap."""
        mock_session_factory = MagicMock()
        workflow = AgentWorkflow(session_factory=mock_session_factory)

        # Query asks about 'pharmaceuticals', evidence contains 'vaccines' and 'drugs' (no stem match)
        evidence = [
            {
                "article_id": 42,
                "headline": "New Vaccine Distributed Across Clinics",
                "snippet": "Health authorities released clinical supplies of essential drugs.",
                "prominence_score": 0.85,
                "source_tool": "hybrid_search",
            },
            {
                "article_id": 99,
                "headline": "Sports Match Rescheduled",
                "snippet": "The tournament has been moved to next weekend due to rain.",
                "prominence_score": 0.20,
                "source_tool": "hybrid_search",
            },
        ]
        state = {
            "query": "Show me reports about pharmaceuticals",
            "archetype": "factual_lookup",
            "evidence_items": evidence,
            "tool_executions": [],
        }

        res = await workflow._evaluate_and_fallback_node(state)  # type: ignore[arg-type]
        filtered = res["evidence_items"]

        # High-confidence vector hit (0.85) should be preserved despite 0 lexical token overlap
        assert any(item["article_id"] == 42 for item in filtered)
        # Irrelevant low-prominence hit (0.20) with no lexical overlap should be pruned
        assert not any(item["article_id"] == 99 for item in filtered)

    def test_conditional_edge_short_circuit_routing(self) -> None:
        """Verify _route_after_planning correctly determines next node."""
        # 1. Clarification needed skips to log_query
        assert AgentWorkflow._route_after_planning(cast(AgentState, {"archetype": "clarification_needed"})) == "log_query"

        # 2. Conversational meta query skips to synthesize_answer
        assert AgentWorkflow._route_after_planning(cast(AgentState, {"archetype": "conversational_meta_query"})) == "synthesize_answer"

        # 3. Empty plan skips to synthesize_answer
        assert AgentWorkflow._route_after_planning(cast(AgentState, {"archetype": "factual_lookup", "plan": []})) == "synthesize_answer"

        # 4. Standard plan routes to execute_tools
        assert AgentWorkflow._route_after_planning(cast(AgentState, {
            "archetype": "factual_lookup",
            "plan": [{"tool_name": "hybrid_search", "arguments": {}}],
        })) == "execute_tools"

    def test_date_normalization_multi_issue(self) -> None:
        """Verify normalize_date_to_iso handles slash dates and returns clean ISO format."""
        from app.retrieval.sql_analytics import normalize_date_to_iso

        assert normalize_date_to_iso("1/8/2026") == "2026-08-01"
        assert normalize_date_to_iso("28/08/2026") == "2026-08-28"
        assert normalize_date_to_iso("2026-08-01") == "2026-08-01"
        assert normalize_date_to_iso(None) is None

    def test_format_issue_manifest_deduplication(self) -> None:
        """Verify format_issue_manifest generates structured broadsheet text."""
        from app.agent.executor import format_issue_manifest

        summary = {
            "newspaper": "The Daily Star",
            "issue_date": "2026-08-01",
            "total_articles": 2,
            "total_pages": 4,
            "section_breakdown": {"Finance": 1, "Sports": 1},
            "articles": [
                {
                    "headline": "MARKET HIGHS REACHED",
                    "byline_author": "John Doe",
                    "section": "Finance",
                    "page_number": 1,
                    "printed_page": "1",
                    "word_count": 250,
                },
                {
                    "headline": "LOCAL TEAM WINS",
                    "byline_author": None,
                    "section": "Sports",
                    "page_number": 3,
                    "printed_page": "3",
                    "word_count": 180,
                },
            ],
        }

        manifest = format_issue_manifest(summary, category_filter="Finance")
        assert "=== RELATIONAL ARCHIVE MANIFEST FOR The Daily Star (2026-08-01) - CATEGORY: Finance ===" in manifest
        assert "Total Articles Ingested: 2" in manifest
        assert "1. [Finance] \"MARKET HIGHS REACHED\" (Page 1 by John Doe, 250 words)" in manifest
        assert "2. [Sports] \"LOCAL TEAM WINS\" (Page 3, 180 words)" in manifest

