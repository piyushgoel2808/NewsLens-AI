"""Unit tests for query condensation, coreference resolution, and anti-hallucination guardrails."""

import pytest

from app.agent.condenser import (
    condense_conversational_query,
    extract_active_issue_from_history,
    needs_condensation,
)
from app.agent.planner import (
    QueryPlan,
    QueryPlanner,
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


def test_parse_inline_citation_formats():
    """Verify inline citation parsing across broadsheet and bracketed formats."""
    from app.agent.condenser import parse_inline_citation

    q1 = '[4] Hindustan Times, 2026-09-10, Page 4, Headline: "The growing bipolarity in the world complicates the ability of Brics-like groupings to push for a radical Global South agenda",, TWLL ME ABOUT THE INFOGRPHICS ATTACHED WITH THIS'
    c1 = parse_inline_citation(q1)
    assert c1.get("newspaper_name") == "Hindustan Times"
    assert c1.get("issue_date") == "2026-09-10"
    assert c1.get("page_number") == 4
    assert "bipolarity" in c1.get("headline", "")

    q2 = '[{The Hindu}, 2026-09-08, Page 1, "ISRO Launches Satellite"]'
    c2 = parse_inline_citation(q2)
    assert c2.get("newspaper_name") == "The Hindu"
    assert c2.get("issue_date") == "2026-09-08"
    assert c2.get("page_number") == 1
    assert c2.get("headline") == "ISRO Launches Satellite"


def test_cross_turn_article_eviction():
    """Verify that stale article_id and photo_id from prior turn are strictly evicted when query specifies a new citation."""
    history = [
        {
            "role": "assistant",
            "content": "No infographics found for Brics... but Mint has photos: #8671...",
            "citations": [
                {
                    "photo_id": 8671,
                    "article_id": 42426,
                    "headline": "SEBI APPROVES NSE IPO, CLEARS WAY FOR LISTING",
                    "newspaper_name": "Mint",
                    "issue_date": "2026-09-05",
                }
            ],
        }
    ]
    turn3_query = '[4] Hindustan Times, 2026-09-10, Page 4, Headline: "The growing bipolarity in the world complicates the ability of Brics-like groupings to push for a radical Global South agenda",, TWLL ME ABOUT THE INFOGRPHICS ATTACHED WITH THIS'

    ctx = extract_active_issue_from_history(history, turn3_query)
    assert ctx.get("article_id") is None
    assert ctx.get("photo_id") is None
    assert ctx.get("newspaper_name") == "Hindustan Times"
    assert ctx.get("issue_date") == "2026-09-10"
    assert ctx.get("page_number") == 4
    assert "bipolarity" in ctx.get("headline", "")


def test_attached_asset_supersedes_stale_history_date():
    """Verify that attaching an asset with a new issue date purges stale date from dialogue history."""
    history = [
        {
            "role": "user",
            "content": "Tell me about education policy in Hindustan Times on 2026-08-01",
        },
        {
            "role": "assistant",
            "content": "In the Hindustan Times issue of 2026-08-01, education reforms were announced...",
            "citations": [
                {
                    "photo_id": 9100,
                    "article_id": 41500,
                    "headline": "Education Policy Overhaul",
                    "newspaper_name": "Hindustan Times",
                    "issue_date": "2026-08-01",
                }
            ],
        },
    ]

    # New turn: user asks about an attached photo without typing a date in text
    query = "what is the name of the person in the photo"
    ctx = extract_active_issue_from_history(
        history,
        current_query=query,
        attached_photo_id=9607,
        attached_article_id=42869,
        attached_issue_date="2026-08-03",
        attached_newspaper_name="Hindustan Times",
    )

    assert ctx.get("issue_date") == "2026-08-03"
    assert ctx.get("newspaper_name") == "Hindustan Times"
    assert ctx.get("photo_id") == 9607
    assert ctx.get("article_id") == 42869


@pytest.mark.asyncio
async def test_resolve_attached_asset_context_mock():
    """Verify resolve_attached_asset_context queries session for photo and extracts attributes."""
    from unittest.mock import AsyncMock, MagicMock
    from app.agent.condenser import resolve_attached_asset_context

    mock_newspaper = MagicMock()
    mock_newspaper.name = "Hindustan Times"

    mock_issue = MagicMock()
    mock_issue.id = 107
    mock_issue.issue_date = "2026-08-03"
    mock_issue.newspaper = mock_newspaper

    mock_article = MagicMock()
    mock_article.id = 42869
    mock_article.issue = mock_issue

    mock_photo = MagicMock()
    mock_photo.id = 9607
    mock_photo.article_id = 42869
    mock_photo.article = mock_article

    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = mock_photo

    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_res
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    mock_factory = MagicMock()
    mock_factory.return_value = mock_session

    res_date, res_np, res_art_id, res_iss_id = await resolve_attached_asset_context(
        session_factory=mock_factory,
        attached_photo_id=9607,
    )

    assert res_date == "2026-08-03"
    assert res_np == "Hindustan Times"
    assert res_art_id == 42869
    assert res_iss_id == 107


