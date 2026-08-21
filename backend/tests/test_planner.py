"""Unit tests for QueryPlanner: Archetype classification and tool execution sequences."""
from __future__ import annotations

from app.agent.planner import QueryPlanner


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
