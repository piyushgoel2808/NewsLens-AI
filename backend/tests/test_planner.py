"""Unit tests for QueryPlanner: Archetype classification and tool execution sequences."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.planner import QueryPlanner
from app.providers.base import ModelResponse


class TestQueryPlanner:
    """Test suite for QueryPlanner."""

    def test_classify_factual_lookup(self) -> None:
        planner = QueryPlanner()
        archetype, _ = planner.classify_archetype("What was the outcome of the tax vote yesterday?")
        assert archetype == "factual_lookup"

        plan = planner.plan_query("What was the outcome of the tax vote yesterday?")
        assert plan.archetype == "factual_lookup"
        assert len(plan.tool_calls) == 1
        assert plan.tool_calls[0].tool_name == "hybrid_search"

    def test_classify_thematic_timeline(self) -> None:
        planner = QueryPlanner()
        queries = [
            "Provide a timeline of the stock market crash in October",
            "Trace the evolution of the transit strike over time",
            "What is the chronological progression of the naval treaty talks?",
        ]
        for q in queries:
            archetype, _ = planner.classify_archetype(q)
            assert archetype == "thematic_timeline"

        plan = planner.plan_query("Provide a timeline of the stock market crash")
        assert plan.archetype == "thematic_timeline"
        tool_names = [t.tool_name for t in plan.tool_calls]
        assert "timeline_builder" in tool_names
        assert "hybrid_search" in tool_names

    def test_classify_quantitative_trend(self) -> None:
        planner = QueryPlanner()
        queries = [
            "How many articles covered the transit strike?",
            "What is the frequency trend of mentions for Reserve Bank?",
            "Show the distribution of articles across topics",
        ]
        for q in queries:
            archetype, _ = planner.classify_archetype(q)
            assert archetype == "quantitative_trend"

        plan = planner.plan_query("How many articles covered the strike?")
        assert plan.archetype == "quantitative_trend"
        tool_names = [t.tool_name for t in plan.tool_calls]
        assert "sql_analytics" in tool_names

    def test_classify_cross_newspaper_comparison(self) -> None:
        planner = QueryPlanner()
        queries = [
            "Compare the editorial perspectives on the new tariff law",
            "How did different papers cover the election results?",
            "Contrast the coverage across newspapers for the Mayor's speech",
        ]
        for q in queries:
            archetype, _ = planner.classify_archetype(q)
            assert archetype == "cross_newspaper_comparison"

        plan = planner.plan_query("Compare the coverage of the tax law across different papers")
        assert plan.archetype == "cross_newspaper_comparison"
        assert plan.tool_calls[0].tool_name == "hybrid_search"
        assert plan.tool_calls[0].arguments.get("top_k", 0) >= 10

    def test_classify_entity_deep_dive(self) -> None:
        planner = QueryPlanner()
        queries = [
            "Show me everything about Winston Churchill",
            "All mentions of Reserve Bank in 1930",
            "Profile the coverage of Prime Minister John Smith",
        ]
        for q in queries:
            archetype, _ = planner.classify_archetype(q)
            assert archetype == "entity_deep_dive"

        plan = planner.plan_query("Show me everything about Winston Churchill")
        assert plan.archetype == "entity_deep_dive"
        tool_names = [t.tool_name for t in plan.tool_calls]
        assert "entity_search" in tool_names

    def test_plan_issue_manifest_aggregate_query(self) -> None:
        planner = QueryPlanner()
        queries = [
            "How many articles are in this newspaper?",
            "List all the articles in today's paper",
            "Summarize issue overview",
            "What articles are in the July 7 edition?",
            "Summarize the whole newspaper issue 81 of Mint 2026-8-28",
        ]
        for q in queries:
            plan = planner.plan_query(q)
            assert plan.archetype == "quantitative_trend"
            assert len(plan.tool_calls) >= 1
            assert plan.tool_calls[0].tool_name == "sql_analytics"
            assert plan.tool_calls[0].arguments.get("analysis_type") in ("issue_summary", "count_articles")

    def test_plan_page_specific_article_queries(self) -> None:
        planner = QueryPlanner()
        queries = [
            ("list no of articles on pg 7", "7"),
            ("how many articles on page 3", "3"),
            ("articles on page 10", "10"),
            ("what articles are on pg 4", "4"),
            ("List articles on page 5", "5"),
        ]
        for q, expected_page in queries:
            plan = planner.plan_query(q)
            assert plan.archetype == "quantitative_trend"
            sql_tool = next((t for t in plan.tool_calls if t.tool_name == "sql_analytics"), None)
            assert sql_tool is not None
            assert sql_tool.arguments.get("analysis_type") == "issue_summary"
            assert sql_tool.arguments.get("page_filter") == expected_page

    def test_plan_page_specific_factual_lookup(self) -> None:
        planner = QueryPlanner()
        plan = planner.plan_query("What did the minister announce on page 4?")
        assert plan.archetype == "factual_lookup"
        assert len(plan.tool_calls) == 1
        assert plan.tool_calls[0].tool_name == "hybrid_search"
        assert plan.tool_calls[0].arguments.get("page_filter") == "4"

    @pytest.mark.asyncio
    async def test_plan_query_async_with_llm_structured_cot(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = ModelResponse(
            text="",
            parsed={
                "thought_process": "User wants a complete macro-issue summary of Mint issue 81. Relational sql_analytics is required.",
                "archetype": "quantitative_trend",
                "primary_tool": "sql_analytics",
                "arguments": {
                    "newspaper_name": "Mint",
                    "issue_date": "2026-08-28",
                    "issue_id": 81,
                    "analysis_type": "issue_summary",
                },
                "include_secondary_hybrid_search": False,
            },
        )

        planner = QueryPlanner(provider=mock_provider)
        plan = await planner.plan_query_async("Summarize the whole newspaper issue 81 of Mint 2026-8-28")

        assert plan.archetype == "quantitative_trend"
        assert "macro-issue summary" in plan.reasoning
        assert len(plan.tool_calls) == 1
        assert plan.tool_calls[0].tool_name == "sql_analytics"
        assert plan.tool_calls[0].arguments["newspaper_name"] == "Mint"
        assert plan.tool_calls[0].arguments["issue_id"] == 81
        assert plan.tool_calls[0].arguments["analysis_type"] == "issue_summary"

    @pytest.mark.asyncio
    async def test_plan_query_async_factual_page_question(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = ModelResponse(
            text="",
            parsed={
                "thought_process": "User is asking about a specific entity event (Tata Power) on page 3. Requires hybrid search filtered to page 3.",
                "archetype": "factual_lookup",
                "primary_tool": "hybrid_search",
                "arguments": {
                    "query": "Tata Power",
                    "page_filter": "3",
                    "top_k": 6,
                },
                "include_secondary_hybrid_search": False,
            },
        )

        planner = QueryPlanner(provider=mock_provider)
        plan = await planner.plan_query_async("What happened to Tata Power on page 3?")

        assert plan.archetype == "factual_lookup"
        assert len(plan.tool_calls) == 1
        assert plan.tool_calls[0].tool_name == "hybrid_search"
        assert plan.tool_calls[0].arguments["page_filter"] == "3"
        assert plan.tool_calls[0].arguments["query"] == "Tata Power"


class TestAgentWorkflowToolExecution:
    """Integration test suite verifying full LangGraph state machine tool execution nodes."""

    @pytest.mark.asyncio
    async def test_execute_sql_analytics_count_articles(self) -> None:
        from app.agent.graph import AgentWorkflow

        mock_session_factory = MagicMock()
        workflow = AgentWorkflow(session_factory=mock_session_factory)
        workflow._sql_analytics.count_articles = AsyncMock(
            return_value={"count": 42, "filters": {"newspaper_name": "Mint"}}
        )

        state = {
            "query": "How many articles are in Mint?",
            "plan": [
                {
                    "tool_name": "sql_analytics",
                    "arguments": {"analysis_type": "count_articles", "newspaper_name": "Mint"},
                    "purpose": "Count total articles",
                }
            ],
            "archetype": "quantitative_trend",
        }

        res = await workflow._execute_tools_node(state)  # type: ignore[arg-type]
        assert len(res["evidence_items"]) == 1
        assert res["evidence_items"][0]["source_tool"] == "sql_analytics"
        assert "42" in res["evidence_items"][0]["snippet"]
        assert len(res["tool_executions"]) == 1
        assert res["tool_executions"][0]["results_count"] == 42

    @pytest.mark.asyncio
    async def test_execute_coverage_analysis_matrix(self) -> None:
        from app.agent.graph import AgentWorkflow
        from app.retrieval.coverage_analyzer import CoverageMatrix, CoverageStatus, PublicationCoverageReport

        mock_session_factory = MagicMock()
        workflow = AgentWorkflow(session_factory=mock_session_factory)

        matrix = CoverageMatrix(
            target_query_or_event="defense budget",
            target_date="2026-08-28",
            total_publications=2,
            covered_count=1,
            not_found_count=1,
            reports={
                "Mint": PublicationCoverageReport(
                    newspaper_id=1,
                    newspaper_name="Mint",
                    status=CoverageStatus.COVERED,
                    confidence=0.92,
                    matched_headlines=["Defense Outlay Boosted"],
                ),
                "Business Standard": PublicationCoverageReport(
                    newspaper_id=2,
                    newspaper_name="Business Standard",
                    status=CoverageStatus.NOT_FOUND,
                    confidence=0.95,
                    audit_notes="0 articles found",
                ),
            },
        )
        workflow._coverage_analyzer.generate_coverage_matrix = AsyncMock(return_value=matrix)

        state = {
            "query": "Compare coverage on defense budget",
            "plan": [
                {
                    "tool_name": "coverage_analysis",
                    "arguments": {"query": "defense budget"},
                    "purpose": "Coverage audit",
                }
            ],
            "archetype": "cross_newspaper_comparison",
        }

        res = await workflow._execute_tools_node(state)  # type: ignore[arg-type]
        assert len(res["evidence_items"]) == 1
        assert res["evidence_items"][0]["source_tool"] == "coverage_analysis"
        assert "COVERED" in res["evidence_items"][0]["snippet"]
        assert "NOT_FOUND" in res["evidence_items"][0]["snippet"]


