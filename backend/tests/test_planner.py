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
        from app.retrieval.coverage_analyzer import (
            CoverageMatrix,
            CoverageStatus,
            PublicationCoverageReport,
        )

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

    def test_plan_section_listing_sports_manifest(self) -> None:
        planner = QueryPlanner()
        plan = planner.plan_query("list all its sports related news")
        assert plan.archetype in ("quantitative_trend", "article_catalog")
        assert len(plan.tool_calls) >= 1
        sql_call = plan.tool_calls[0]
        assert sql_call.tool_name == "sql_analytics"
        assert sql_call.arguments.get("category_filter") == "Sports"
        assert sql_call.arguments.get("analysis_type") == "issue_summary"

    def test_plan_newspaper_issue_and_date_extraction(self) -> None:
        planner = QueryPlanner()
        plan = planner.plan_query("summrizze the whole newspaper of THE ECONOMICS times issue 84 dated 27/8/2026")
        assert plan.archetype == "quantitative_trend"
        assert len(plan.tool_calls) >= 1
        sql_call = plan.tool_calls[0]
        assert sql_call.tool_name == "sql_analytics"
        assert sql_call.arguments.get("newspaper_name") == "The Economic Times"
        assert sql_call.arguments.get("issue_id") == 84
        assert sql_call.arguments.get("issue_date") == "2026-08-27"
        assert sql_call.arguments.get("analysis_type") == "issue_summary"

    def test_single_newspaper_multi_date_comparison(self) -> None:
        planner = QueryPlanner()
        plan = planner.plan_query("COMPARE NEWSPAPER THE GOAN DATE 1/8/2026 AND THE THE GOAN NEWSPAPER DATED 2/8/2026")
        assert plan.archetype == "quantitative_trend"
        # Verify tools: targeted SQL issue summaries and scoped hybrid search, NOT all-newspaper coverage analysis
        tool_names = [t.tool_name for t in plan.tool_calls]
        assert "coverage_analysis" not in tool_names
        assert "sql_analytics" in tool_names
        assert "hybrid_search" in tool_names

        sql_calls = [t for t in plan.tool_calls if t.tool_name == "sql_analytics"]
        assert len(sql_calls) == 2
        dates = {c.arguments.get("issue_date") for c in sql_calls}
        assert "2026-08-01" in dates
        assert "2026-08-02" in dates

        hybrid_calls = [t for t in plan.tool_calls if t.tool_name == "hybrid_search"]
        assert len(hybrid_calls) == 1
        assert hybrid_calls[0].arguments.get("newspaper_name") == "The Goan"
        assert hybrid_calls[0].arguments.get("date_from") == "2026-08-01"
        assert hybrid_calls[0].arguments.get("date_to") == "2026-08-02"

    def test_extract_parameters_multi_date(self) -> None:
        from app.agent.planner import extract_parameters_from_query
        res = extract_parameters_from_query("COMPARE NEWSPAPER THE GOAN DATE 1/8/2026 AND THE THE GOAN NEWSPAPER DATED 2/8/2026")
        assert res.get("newspaper_name") == "The Goan"
        assert res.get("date_from") == "2026-08-01"
        assert res.get("date_to") == "2026-08-02"
        assert res.get("target_dates") == ["2026-08-01", "2026-08-02"]

    def test_compare_all_available_newspapers_dated(self) -> None:
        planner = QueryPlanner()
        plan = planner.plan_query("comapare all the available newspaper dated 1/8/2026")
        assert plan.archetype == "cross_newspaper_comparison"
        
        tool_names = [t.tool_name for t in plan.tool_calls]
        assert "sql_analytics" in tool_names
        assert "hybrid_search" in tool_names
        assert "coverage_analysis" in tool_names

        # Verify sql_analytics issue_summary has target date
        sql_summary = next(t for t in plan.tool_calls if t.tool_name == "sql_analytics" and t.arguments.get("analysis_type") == "issue_summary")
        assert sql_summary.arguments.get("issue_date") == "2026-08-01"

        # Verify hybrid_search has promoted date_from and date_to
        hybrid = next(t for t in plan.tool_calls if t.tool_name == "hybrid_search")
        assert hybrid.arguments.get("date_from") == "2026-08-01"
        assert hybrid.arguments.get("date_to") == "2026-08-01"

        # Verify coverage_analysis has target_date
        cov = next(t for t in plan.tool_calls if t.tool_name == "coverage_analysis")
        assert cov.arguments.get("target_date") == "2026-08-01"

        # Verify sql_analytics coverage_comparison has target_date
        sql_cov = next(t for t in plan.tool_calls if t.tool_name == "sql_analytics" and t.arguments.get("analysis_type") == "coverage_comparison")
        assert sql_cov.arguments.get("target_date") == "2026-08-01"

    @pytest.mark.asyncio
    async def test_llm_plan_cross_newspaper_date_promotion(self) -> None:
        from unittest.mock import AsyncMock, MagicMock
        from app.providers.base import ModelResponse

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(
            return_value=ModelResponse(
                text="",
                parsed={
                    "thought_process": "Compare all newspapers for date 1/8/2026.",
                    "archetype": "cross_newspaper_comparison",
                    "primary_tool": "coverage_analysis",
                    "arguments": {
                        "issue_date": "2026-08-01",
                        "query": "newspaper coverage comparison",
                    },
                    "include_secondary_hybrid_search": False,
                },
            )
        )
        planner = QueryPlanner(provider=mock_provider)
        plan = await planner.plan_query_async("comapare all the available newspaper dated 1/8/2026")

        assert plan.archetype == "cross_newspaper_comparison"
        
        # Verify sql_analytics issue_summary was injected for all newspapers on date
        sql_summary = next((t for t in plan.tool_calls if t.tool_name == "sql_analytics" and t.arguments.get("analysis_type") == "issue_summary"), None)
        assert sql_summary is not None
        assert sql_summary.arguments.get("issue_date") == "2026-08-01"

        # Verify hybrid_search received promoted dates
        hybrid = next(t for t in plan.tool_calls if t.tool_name == "hybrid_search")
        assert hybrid.arguments.get("date_from") == "2026-08-01"
        assert hybrid.arguments.get("date_to") == "2026-08-01"

        # Verify coverage_analysis received target_date
        cov = next(t for t in plan.tool_calls if t.tool_name == "coverage_analysis")
        assert cov.arguments.get("target_date") == "2026-08-01"

    def test_plan_differential_coverage_in_x_but_not_in_y(self) -> None:
        planner = QueryPlanner()
        query = "List the news that are in the GOAN dated 1/8/2026 but not in he Morning Standard dated 1/8/2026"
        plan = planner.plan_query(query)

        assert plan.archetype == "cross_newspaper_comparison"
        
        # Verify coverage_difference tool was planned
        diff_tool = next((t for t in plan.tool_calls if t.tool_name == "sql_analytics" and t.arguments.get("analysis_type") == "coverage_difference"), None)
        assert diff_tool is not None, "Expected sql_analytics(coverage_difference) in planned tools!"
        assert diff_tool.arguments.get("newspaper_name") == "The Goan"
        assert diff_tool.arguments.get("comparison_newspaper") == "The Morning Standard"
        assert diff_tool.arguments.get("issue_date") == "2026-08-01"

        # Verify hybrid_search was planned for The Goan
        hybrid_tool = next((t for t in plan.tool_calls if t.tool_name == "hybrid_search"), None)
        assert hybrid_tool is not None
        assert hybrid_tool.arguments.get("newspaper_name") == "The Goan"
        assert hybrid_tool.arguments.get("date_from") == "2026-08-01"

    def test_extract_parameters_differential_single_brand_does_not_index_error(self) -> None:
        """Verify queries with exclusion phrases like 'not in' and only 1 newspaper don't raise IndexError."""
        from app.agent.planner import extract_parameters_from_query

        # Query has "not in" but only 1 newspaper brand ("The Goan")
        params = extract_parameters_from_query("Were there any articles not in The Goan?")
        assert params["is_differential"] is True
        assert params["source_newspaper"] == "The Goan"
        assert "comparison_newspaper" not in params

        # Query has "exclusive to" and 1 brand
        params2 = extract_parameters_from_query("Stories exclusive to The Goan on 2026-08-05")
        assert params2["is_differential"] is True
        assert params2["source_newspaper"] == "The Goan"
        assert "comparison_newspaper" not in params2

    def test_cross_newspaper_domain_comparison_preserves_topic(self) -> None:
        """Verify cross-newspaper domain comparisons preserve query topic and avoid heavy coverage analysis."""
        from app.agent.planner import ExtractedToolArguments, QueryPlan, QueryPlanner

        planner = QueryPlanner()
        q = "COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on health related news"
        raw_plan = QueryPlan(
            thought_process="Cross newspaper comparison on health",
            archetype="cross_newspaper_comparison",
            primary_tool="sql_analytics",
            arguments=ExtractedToolArguments(
                issue_date="2026-08-01",
                query="health related news",
                category_filter="Health",
                analysis_type="issue_summary",
            ),
            include_secondary_hybrid_search=False,
        )

        plan = planner._build_plan_from_structured_model(q, raw_plan)
        assert plan.archetype == "cross_newspaper_comparison"

        # Verify SQL analytics is scheduled for all newspapers with Health category
        sql_tool = next((t for t in plan.tool_calls if t.tool_name == "sql_analytics"), None)
        assert sql_tool is not None
        assert sql_tool.arguments.get("issue_date") == "2026-08-01"
        assert sql_tool.arguments.get("category_filter") == "Health"
        assert sql_tool.arguments.get("newspaper_name") is None  # all newspapers

        # Verify hybrid search is scheduled with Health category filter
        hs_tool = next((t for t in plan.tool_calls if t.tool_name == "hybrid_search"), None)
        assert hs_tool is not None
        assert hs_tool.arguments.get("category_filter") == "Health"
        assert hs_tool.arguments.get("query") == "health related news"

        # Verify coverage_analysis is NOT scheduled for domain-filtered comparison without explicit audit
        cov_tool = next((t for t in plan.tool_calls if t.tool_name == "coverage_analysis"), None)
        assert cov_tool is None, "Coverage analysis should NOT be scheduled when domain filter is present!"

    def test_generic_filler_query_sanitized_to_domain_topic(self) -> None:
        """Verify filler queries like 'newspaper coverage comparison' are cleaned to domain topics."""
        from app.agent.planner import ExtractedToolArguments, QueryPlan, QueryPlanner

        planner = QueryPlanner()
        q = "COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on health related news"
        # Simulate LLM copying Example 9 filler query
        raw_plan = QueryPlan(
            thought_process="Copied example 9 filler query",
            archetype="cross_newspaper_comparison",
            primary_tool="coverage_analysis",
            arguments=ExtractedToolArguments(
                issue_date="2026-08-01",
                query="newspaper coverage comparison",
                category_filter="Health",
            ),
            include_secondary_hybrid_search=False,
        )

        plan = planner._build_plan_from_structured_model(q, raw_plan)

        # args.query should be automatically cleaned to health related news
        hs_tool = next((t for t in plan.tool_calls if t.tool_name == "hybrid_search"), None)
        assert hs_tool is not None
        assert "health" in hs_tool.arguments.get("query", "").lower()
        assert hs_tool.arguments.get("query") != "newspaper coverage comparison"

    def test_extract_parameters_brand_does_not_bleed_into_category(self) -> None:
        """Verify brand name tokens like 'Economic' or 'Financial' do not bleed into category filters."""
        from app.agent.planner import extract_parameters_from_query

        # Mentioning The Economic Times should not trigger Economy & Policy filter
        p1 = extract_parameters_from_query("Summarize the front page of The Economic Times dated 2026-08-01")
        assert p1.get("newspaper_name") == "The Economic Times"
        assert p1.get("category_filter") is None

        # Explicitly mentioning economy and The Economic Times SHOULD trigger Economy & Policy
        p2 = extract_parameters_from_query("list all economy news from The Economic Times dated 2026-08-01")
        assert p2.get("newspaper_name") == "The Economic Times"
        assert p2.get("category_filter") == "Economy & Policy"

        # Mentioning Financial Times should not trigger Business & Markets category filter
        p3 = extract_parameters_from_query("Summarize the front page of Financial Times dated 2026-08-01")
        assert p3.get("newspaper_name") == "Financial Times"
        assert p3.get("category_filter") is None

        # Mentioning Business Standard should not trigger Business & Markets category filter
        p4 = extract_parameters_from_query("Summarize the front page of Business Standard dated 2026-08-01")
        assert p4.get("newspaper_name") == "Business Standard"
        assert p4.get("category_filter") is None

    def test_legacy_macro_summary_and_negative_audit_archetypes(self) -> None:
        """Verify macro_summary and negative_coverage_audit validate and route cleanly."""
        from app.agent.planner import ExtractedToolArguments, QueryPlan, QueryPlanner

        planner = QueryPlanner()
        q = "Summarize the entire newspaper of The Goan dated 2026-08-01"
        plan1 = QueryPlan(
            thought_process="Macro issue overview",
            archetype="macro_summary",
            primary_tool="sql_analytics",
            arguments=ExtractedToolArguments(newspaper_name="The Goan", issue_date="2026-08-01"),
        )
        res1 = planner._build_plan_from_structured_model(q, plan1)
        assert res1.archetype == "quantitative_trend"
        assert any(t.tool_name == "sql_analytics" for t in res1.tool_calls)

        q2 = "Audit what The Goan omitted compared to The Morning Standard on 2026-08-01"
        plan2 = QueryPlan(
            thought_process="Negative audit",
            archetype="negative_coverage_audit",
            primary_tool="sql_analytics",
            arguments=ExtractedToolArguments(
                newspaper_name="The Goan",
                comparison_newspaper="The Morning Standard",
                issue_date="2026-08-01",
            ),
        )
        res2 = planner._build_plan_from_structured_model(q2, plan2)
        assert res2.archetype == "cross_newspaper_comparison"
        assert any(t.tool_name == "sql_analytics" and t.arguments.get("analysis_type") == "coverage_difference" for t in res2.tool_calls)

        q3 = "Audit negative coverage omissions across all newspapers on 2026-08-01"
        plan3 = QueryPlan(
            thought_process="Archive-wide negative audit",
            archetype="negative_coverage_audit",
            primary_tool="coverage_analysis",
            arguments=ExtractedToolArguments(
                issue_date="2026-08-01",
            ),
        )
        res3 = planner._build_plan_from_structured_model(q3, plan3)
        assert res3.archetype == "cross_newspaper_comparison"
        assert any(t.tool_name == "coverage_analysis" for t in res3.tool_calls)

    def test_edge_case_a_article_catalog_heuristic_classification(self) -> None:
        """Verify heuristic fallback classifies topic listings as article_catalog instead of quantitative_trend."""
        planner = QueryPlanner()
        plan_health = planner.plan_query("list all their health news")
        assert plan_health.archetype == "article_catalog"
        assert len(plan_health.tool_calls) >= 1
        assert plan_health.tool_calls[0].tool_name == "sql_analytics"
        assert plan_health.tool_calls[0].arguments.get("category_filter") == "Health"
        assert plan_health.tool_calls[0].arguments.get("analysis_type") == "issue_summary"

        plan_catalog = planner.plan_query("catalog of sports articles")
        assert plan_catalog.archetype == "article_catalog"
        assert plan_catalog.tool_calls[0].arguments.get("category_filter") == "Sports"

    def test_edge_case_b_undated_cross_newspaper_coverage_suppression(self) -> None:
        """Verify general undated cross-newspaper comparison omits heavy coverage_analysis unless requested."""
        planner = QueryPlanner()
        # General comparison -> hybrid_search only, no coverage_analysis
        plan_general = planner.plan_query("Compare how different newspapers cover climate change")
        assert plan_general.archetype == "cross_newspaper_comparison"
        tool_names = [t.tool_name for t in plan_general.tool_calls]
        assert "hybrid_search" in tool_names
        assert "coverage_analysis" not in tool_names

        # Comparison with omission keyword -> includes coverage_analysis
        plan_audit = planner.plan_query("What did different newspapers omit regarding the budget speech?")
        assert plan_audit.archetype == "cross_newspaper_comparison"
        tool_names_audit = [t.tool_name for t in plan_audit.tool_calls]
        assert "hybrid_search" in tool_names_audit
        assert "coverage_analysis" in tool_names_audit

    def test_edge_case_c_context_retention_reconcile_and_sanitize(self) -> None:
        """Verify active context brand and date are preserved during multi-turn follow-ups."""
        from app.agent.models import AgentPlan, ToolCallSpec

        planner = QueryPlanner()
        follow_up_query = "What was reported on page 4?"
        plan_obj = AgentPlan(
            thought_process="Follow-up turn checking page 4 for the active newspaper and date",
            archetype="factual_lookup",
            tool_calls=[
                ToolCallSpec(
                    tool_name="hybrid_search",
                    arguments={
                        "query": "page 4 news",
                        "newspaper_name": "The Goan",
                        "date_from": "2026-08-01",
                        "date_to": "2026-08-01",
                        "page_filter": "4",
                    },
                    purpose="Search page 4 in active newspaper",
                )
            ],
        )

        # Passing active context preserves The Goan even though not in prompt
        res = planner._build_plan_from_structured_model(
            query=follow_up_query,
            plan_obj=plan_obj,
            active_issue_date="2026-08-01",
            active_newspapers=["The Goan"],
        )
        assert len(res.tool_calls) == 1
        args = res.tool_calls[0].arguments
        assert args.get("newspaper_name") == "The Goan"
        assert args.get("page_filter") == "4"

        # Without active context, hallucinated brand (not in prompt) is pruned
        res_no_ctx = planner._build_plan_from_structured_model(
            query=follow_up_query,
            plan_obj=plan_obj,
        )
        assert res_no_ctx.tool_calls[0].arguments.get("newspaper_name") is None









