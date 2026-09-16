"""Unit tests for query condensation, coreference resolution, and anti-hallucination guardrails."""

import pytest
from unittest.mock import AsyncMock

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
    # Issue 1: Existential queries with ambiguous pronouns must NOT be skipped by existential check
    assert needs_condensation("Are there any reports on its front page?", history) is True
    assert needs_condensation("Is there any article about their chief minister?", history) is True
    assert needs_condensation("Are there any reports on floods in Assam?", history) is False


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
    """Verify resolve_attached_asset_context queries session for photo and extracts full attributes."""
    from unittest.mock import AsyncMock, MagicMock

    from app.agent.condenser import resolve_attached_asset_context

    mock_newspaper = MagicMock()
    mock_newspaper.name = "The Goan"

    mock_issue = MagicMock()
    mock_issue.id = 97
    mock_issue.issue_date = "2026-08-05"
    mock_issue.newspaper = mock_newspaper

    mock_article = MagicMock()
    mock_article.id = 41142
    mock_article.headline = "Ahead of protest, CM meets Shah on quota for STs"
    mock_article.issue = mock_issue

    mock_photo = MagicMock()
    mock_photo.id = 7645
    mock_photo.article_id = 41142
    mock_photo.caption = "Chief Minister Pramod Sawant with Union Home Minister Amit Shah in New Delhi on Tuesday."
    mock_photo.visual_type = "photo"
    mock_photo.page_id = 101
    mock_photo.article = mock_article

    mock_photo_res = MagicMock()
    mock_photo_res.scalar_one_or_none.return_value = mock_photo

    mock_page_res = MagicMock()
    mock_page_res.scalar_one_or_none.return_value = 1

    mock_session = AsyncMock()
    # First query is for Photo, second query is for Page number
    mock_session.execute.side_effect = [mock_photo_res, mock_page_res]
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    mock_factory = MagicMock()
    mock_factory.return_value = mock_session

    res = await resolve_attached_asset_context(
        session_factory=mock_factory,
        attached_photo_id=7645,
    )

    assert isinstance(res, dict)
    assert res.get("issue_date") == "2026-08-05"
    assert res.get("newspaper_name") == "The Goan"
    assert res.get("article_id") == 41142
    assert res.get("issue_id") == 97
    assert res.get("photo_id") == 7645
    assert res.get("headline") == "Ahead of protest, CM meets Shah on quota for STs"
    assert "Pramod Sawant" in res.get("caption", "")
    assert res.get("visual_type") == "photo"
    assert res.get("page_number") == 1


@pytest.mark.asyncio
async def test_condense_query_with_attached_asset_prevents_context_leakage():
    """Verify that an attached asset strictly isolates the query from prior turn's unrelated entities."""
    history = [
        {
            "role": "user",
            "content": "Drugs can give brief high but make life low: Modi to youth",
        },
        {
            "role": "assistant",
            "content": (
                "In Hindustan Times on 2026-08-03, Prime Minister Narendra Modi addressed youth "
                "during the launch of Nasha Mukt Bharat Sankalp Abhiyan."
            ),
            "citations": [
                {
                    "photo_id": 9607,
                    "article_id": 42869,
                    "headline": "Drugs can give brief high but make life low: Modi to youth",
                    "newspaper_name": "Hindustan Times",
                    "issue_date": "2026-08-03",
                    "page_number": 5,
                }
            ],
        },
    ]

    attached_asset = {
        "photo_id": 7645,
        "article_id": 41142,
        "issue_date": "2026-08-05",
        "newspaper_name": "The Goan",
        "headline": "Ahead of protest, CM meets Shah on quota for STs",
        "caption": "Chief Minister Pramod Sawant with Union Home Minister Amit Shah in New Delhi on Tuesday.",
        "visual_type": "photo",
        "page_number": 1,
    }

    # User asks "name the pearson in this photo" (with realistic typo)
    resolved = await condense_conversational_query(
        query="name the pearson in this photo",
        chat_history=history,
        provider=None,  # Tests deterministic fallback grounding
        attached_asset=attached_asset,
    )

    # Must target The Goan and the attached photo/headline
    assert "The Goan" in resolved
    assert "2026-08-05" in resolved
    assert "photo #7645" in resolved
    assert "Ahead of protest, CM meets Shah on quota for STs" in resolved

    # MUST NOT contain any leakage from Turn 1
    assert "Hindustan Times" not in resolved
    assert "Modi" not in resolved
    assert "2026-08-03" not in resolved
    assert "brief high" not in resolved


@pytest.mark.asyncio
async def test_condense_query_with_mock_llm_provider_decision():
    """Verify LLM reformulator prompt includes attached asset block and decision rules."""
    from unittest.mock import AsyncMock, MagicMock

    from app.providers.base import ModelResponse

    history = [
        {
            "role": "user",
            "content": "Drugs can give brief high but make life low: Modi to youth",
        },
        {
            "role": "assistant",
            "content": "In Hindustan Times on 2026-08-03, Narendra Modi launched Nasha Mukt Abhiyan.",
        },
    ]

    attached_asset = {
        "photo_id": 7645,
        "article_id": 41142,
        "issue_date": "2026-08-05",
        "newspaper_name": "The Goan",
        "headline": "Ahead of protest, CM meets Shah on quota for STs",
        "caption": "Chief Minister Pramod Sawant with Union Home Minister Amit Shah in New Delhi on Tuesday.",
        "visual_type": "photo",
        "page_number": 1,
    }

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=ModelResponse(
            text="What is the name of the person in photo #7645 from The Goan dated 2026-08-05 regarding Chief Minister Pramod Sawant and Amit Shah?",
        )
    )

    resolved = await condense_conversational_query(
        query="name the pearson in this photo",
        chat_history=history,
        provider=mock_provider,
        attached_asset=attached_asset,
    )

    # Check that LLM complete was called
    assert mock_provider.complete.called
    call_args = mock_provider.complete.call_args[1]
    messages = call_args["messages"]

    # Verify system prompt has decision rules
    system_content = messages[0].content
    assert "ATTACHED ASSET REFERENCE" in system_content
    assert "CONVERSATION CONTINUATION" in system_content
    assert "DO NOT inherit or contaminate" in system_content

    # Verify user prompt includes attached asset details
    user_content = messages[1].content
    assert "CURRENT WORKSPACE ATTACHED ASSET:" in user_content
    assert "Photo (ID: #7645)" in user_content
    assert "The Goan" in user_content
    assert "2026-08-05 (Page 1)" in user_content
    assert "Ahead of protest, CM meets Shah on quota for STs" in user_content
    assert "Pramod Sawant" in user_content

    assert resolved == "What is the name of the person in photo #7645 from The Goan dated 2026-08-05 regarding Chief Minister Pramod Sawant and Amit Shah?"


@pytest.mark.asyncio
async def test_end_to_end_attached_asset_planning_with_dirty_history():
    """Verify that planner schedules inspect_visual_asset on The Goan (not Hindustan Times) despite dirty history."""
    dirty_history = [
        {
            "role": "user",
            "content": "Drugs can give brief high but make life low: Modi to youth",
        },
        {
            "role": "assistant",
            "content": (
                "In Hindustan Times on 2026-08-03, Prime Minister Narendra Modi addressed youth "
                "during the launch of Nasha Mukt Bharat Sankalp Abhiyan."
            ),
            "citations": [
                {
                    "photo_id": 9607,
                    "article_id": 42869,
                    "headline": "Drugs can give brief high but make life low: Modi to youth",
                    "newspaper_name": "Hindustan Times",
                    "issue_date": "2026-08-03",
                    "page_number": 5,
                }
            ],
        },
    ]

    attached_asset = {
        "photo_id": 7645,
        "article_id": 41142,
        "issue_date": "2026-08-05",
        "newspaper_name": "The Goan",
        "headline": "Ahead of protest, CM meets Shah on quota for STs",
        "caption": "Chief Minister Pramod Sawant with Union Home Minister Amit Shah in New Delhi on Tuesday.",
        "visual_type": "photo",
        "page_number": 1,
    }

    # 1. Condenser resolves query cleanly
    condensed_q = await condense_conversational_query(
        query="name the pearson in this photo",
        chat_history=dirty_history,
        provider=None,  # deterministic fallback
        attached_asset=attached_asset,
    )

    assert "The Goan" in condensed_q
    assert "2026-08-05" in condensed_q
    assert "photo #7645" in condensed_q
    assert "Hindustan Times" not in condensed_q
    assert "Modi" not in condensed_q

    # 2. QueryPlanner builds plan with deterministic fallback
    mock_p = AsyncMock()
    mock_p.complete = AsyncMock(side_effect=Exception("Deterministic test"))
    planner = QueryPlanner(provider=mock_p)
    plan_result = await planner.plan_query_async(
        condensed_q,
        enable_web_search=False,
        active_issue_date="2026-08-05",
        active_newspapers=["The Goan"],
        attached_article_id=41142,
        attached_photo_id=7645,
    )

    tool_names = [call.tool_name for call in plan_result.tool_calls]
    assert "inspect_visual_asset" in tool_names

    # Verify arguments of inspect_visual_asset
    visual_call = next(call for call in plan_result.tool_calls if call.tool_name == "inspect_visual_asset")
    assert visual_call.arguments.get("photo_id") == 7645
    assert visual_call.arguments.get("newspaper_name") == "The Goan"
    assert visual_call.arguments.get("issue_date") == "2026-08-05"

    # Verify NO tool call targets Hindustan Times or 2026-08-03
    for call in plan_result.tool_calls:
        args = call.arguments
        assert args.get("newspaper_name") != "Hindustan Times"
        assert args.get("issue_date") != "2026-08-03"
        assert args.get("date_from") != "2026-08-03"
        assert args.get("date_to") != "2026-08-03"


def test_normal_text_query_existential_there_does_not_trigger_condensation():
    """Verify existential 'is there', 'are there', 'was there' do NOT trigger condensation."""
    dirty_history = [
        {"role": "user", "content": "Show me Hindustan Times on 2026-08-03"},
        {
            "role": "assistant",
            "content": "Narendra Modi launched Nasha Mukt Yuva for Viksit Bharat Sankalp Abhiyan in Hindustan Times on 2026-08-03.",
            "citations": [{"newspaper_name": "Hindustan Times", "issue_date": "2026-08-03", "page_number": 1}],
        },
    ]

    assert needs_condensation("Is there any news about education policy in Maharashtra?", dirty_history) is False
    assert needs_condensation("Are there any articles about renewable energy?", dirty_history) is False
    assert needs_condensation("Was there any coverage of the budget?", dirty_history) is False
    assert needs_condensation("Tell me about the ST quota protest in Goa", dirty_history) is False


def test_normal_text_query_newspaper_switch_purges_stale_date_in_guardrail3():
    """Verify switching newspaper in normal text query purges prior newspaper's date."""
    dirty_history = [
        {"role": "user", "content": "Show me Hindustan Times on 2026-08-03"},
        {
            "role": "assistant",
            "content": "Narendra Modi launched Nasha Mukt Yuva for Viksit Bharat Sankalp Abhiyan in Hindustan Times on 2026-08-03.",
            "citations": [{"newspaper_name": "Hindustan Times", "issue_date": "2026-08-03", "page_number": 1}],
        },
    ]

    ctx = extract_active_issue_from_history(dirty_history, current_query="Summarize the front page of The Goan")
    assert ctx.get("newspaper_name") == "The Goan"
    assert "issue_date" not in ctx  # Crucial: must NOT inherit 2026-08-03 from Hindustan Times!
    assert ctx.get("target_newspapers") == ["The Goan"]


@pytest.mark.asyncio
async def test_normal_text_query_topic_switch_e2e_no_leakage():
    """Verify normal text topic switch query produces clean plan without prior turn constraints."""
    dirty_history = [
        {"role": "user", "content": "Show me Hindustan Times on 2026-08-03"},
        {
            "role": "assistant",
            "content": "Narendra Modi launched Nasha Mukt Yuva for Viksit Bharat Sankalp Abhiyan in Hindustan Times on 2026-08-03.",
            "citations": [{"newspaper_name": "Hindustan Times", "issue_date": "2026-08-03", "page_number": 1}],
        },
    ]

    query = "Tell me about the ST quota protest in Goa"
    # 1. Not flagged as follow-up needing condensation
    assert needs_condensation(query, dirty_history) is False

    # 2. Standalone planning: active_date and active_newspapers must be None/empty
    mock_p = AsyncMock()
    mock_p.complete = AsyncMock(side_effect=Exception("Deterministic test"))
    planner = QueryPlanner(provider=mock_p)
    plan_result = await planner.plan_query_async(
        query,
        enable_web_search=False,
        active_issue_date=None,
        active_newspapers=None,
    )

    for call in plan_result.tool_calls:
        args = call.arguments
        assert args.get("newspaper_name") != "Hindustan Times"
        assert args.get("issue_date") != "2026-08-03"
        assert args.get("date_from") != "2026-08-03"
        assert args.get("date_to") != "2026-08-03"


def test_normal_text_query_legitimate_followup_preserves_context():
    """Verify true conversational follow-up queries continue to trigger condensation."""
    dirty_history = [
        {"role": "user", "content": "Show me Hindustan Times on 2026-08-03"},
        {
            "role": "assistant",
            "content": "Narendra Modi launched Nasha Mukt Yuva for Viksit Bharat Sankalp Abhiyan in Hindustan Times on 2026-08-03.",
            "citations": [{"newspaper_name": "Hindustan Times", "issue_date": "2026-08-03", "page_number": 1}],
        },
    ]

    assert needs_condensation("what else did he say about youth?", dirty_history) is True
    assert needs_condensation("Summarize the front page", dirty_history) is True
    assert needs_condensation("Did any other newspaper cover this event?", dirty_history) is True


@pytest.mark.asyncio
async def test_condenser_shared_coverage_followup():
    """Verify follow-up queries like 'list all those article that are similar' retain shared context."""
    shared_history = [
        {
            "role": "user",
            "content": "give me the similar articles from newspaper of the Goan and the morning standard both dated 1/8/2026",
        },
        {
            "role": "assistant",
            "content": "Here is the shared coverage between The Goan and The Morning Standard on 2026-08-01.",
            "citations": [
                {"newspaper_name": "The Goan", "issue_date": "2026-08-01", "page_number": 5},
                {"newspaper_name": "The Morning Standard", "issue_date": "2026-08-01", "page_number": 7},
            ],
        },
    ]

    followup_query = "list all those article that are similar"
    assert needs_condensation(followup_query, shared_history) is True

    resolved = await condense_conversational_query(
        query=followup_query,
        chat_history=shared_history,
        provider=None,  # triggers deterministic heuristic fallback
    )

    assert "The Goan" in resolved
    assert "The Morning Standard" in resolved
    assert "2026-08-01" in resolved

    from app.agent.extractor import extract_parameters_from_query
    params = extract_parameters_from_query(resolved)
    assert params.get("is_shared") is True
    assert params.get("newspaper_name") == "The Goan"
    assert params.get("comparison_newspaper") == "The Morning Standard"
    assert params.get("issue_date") == "2026-08-01"




