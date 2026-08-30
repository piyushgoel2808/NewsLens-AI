"""Unit tests for query condensation, coreference resolution, and anti-hallucination guardrails."""

import pytest
from app.agent.condenser import (
    condense_conversational_query,
    extract_active_issue_from_history,
    needs_condensation,
)
from app.agent.planner import (
    QueryPlanner,
    QueryPlan,
    extract_parameters_from_query,
)


def test_extract_active_issue_from_summary_header():
    """Verify that assistant executive summaries of The Goan cleanly yield active context."""
    history = [
        {"role": "user", "content": "Summrizze the whole newspaper issue 94 the Goan 2/8/2026"},
        {
            "role": "assistant",
            "content": (
                "⚡ EXECUTIVE SUMMARY\n"
                "The Goan newspaper issue 94 (2026-08-02) is a comprehensive and diverse edition "
                "with a focus on governance, environment, crime, and socio-economic issues. "
                "The issue features 180 articles across 16 pages, distributed across sections."
            ),
        },
    ]
    active_ctx = extract_active_issue_from_history(history)
    assert active_ctx.get("newspaper_name") == "The Goan"
    assert active_ctx.get("issue_id") == 94
    assert active_ctx.get("issue_date") == "2026-08-02"


def test_needs_condensation_detects_pronouns():
    """Verify needs_condensation flags relative pronouns like 'its'."""
    history = [{"role": "user", "content": "Summarize issue 94"}]
    assert needs_condensation("list all its sports related news", history) is True
    assert needs_condensation("what about its main headline", history) is True
    assert needs_condensation("who won the elections in India?", history) is False


@pytest.mark.asyncio
async def test_condense_query_resolves_pronouns_with_active_context():
    """Verify conversational pronoun 'its' is rewritten to the active publication."""
    history = [
        {"role": "user", "content": "Summrizze the whole newspaper issue 94 the Goan 2/8/2026"},
        {
            "role": "assistant",
            "content": "⚡ EXECUTIVE SUMMARY\nThe Goan newspaper issue 94 (2026-08-02) is a comprehensive edition...",
        },
    ]
    resolved = await condense_conversational_query(
        query="list all its sports related news",
        chat_history=history,
        provider=None,
    )
    assert "The Goan" in resolved
    assert "issue 94" in resolved
    assert "2026-08-02" in resolved
    assert "sports related news" in resolved
    assert "its" not in resolved.lower().split()


def test_planner_prunes_hallucinated_page_filter_and_brand():
    """Verify planner drops page_filter and newspaper_name if they were hallucinated by LLM."""
    planner = QueryPlanner()
    query = "list all sports related news from The Goan issue 94 dated 2026-08-02"

    hallucinated_plan = QueryPlan.model_validate(
        {
            "thought_process": "Hallucinating random page and wrong newspaper",
            "archetype": "quantitative_trend",
            "primary_tool": "sql_analytics",
            "arguments": {
                "newspaper_name": "The Economic Times",
                "issue_date": "2026-08-28",
                "page_filter": "5",
                "category_filter": "Sports",
            },
        }
    )

    plan_res = planner._build_plan_from_structured_model(query, hallucinated_plan)
    assert len(plan_res.tool_calls) == 1
    call = plan_res.tool_calls[0]
    args = call.arguments

    assert "page_filter" not in args
    assert args.get("newspaper_name") == "The Goan"
    assert args.get("issue_id") == 94
    assert args.get("category_filter") == "Sports"


def test_extract_parameters_the_goan():
    """Verify extract_parameters_from_query recognizes The Goan brand and date format."""
    q = "Summarize the whole newspaper issue 94 the Goan 2/8/2026"
    params = extract_parameters_from_query(q)
    assert params.get("newspaper_name") == "The Goan"
    assert params.get("issue_id") == 94
    assert params.get("issue_date") == "2026-08-02"
