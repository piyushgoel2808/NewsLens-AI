"""Unit and integration tests for Reflexive Evaluator and Adaptive Closed-Loop Re-Planning."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.evaluator import EvidenceEvaluator
from app.agent.planner import QueryPlanner
from app.agent.state import AgentState, ToolExecutionRecord
from app.providers.base import ModelResponse


@pytest.fixture
def mock_engines():
    entity_search = MagicMock()
    web_search = MagicMock()
    tool_maker = MagicMock()
    sql_analytics = MagicMock()
    return entity_search, web_search, tool_maker, sql_analytics


def _make_state(**kwargs) -> AgentState:
    base: AgentState = {
        "query": "What is the new education policy?",
        "original_query": "What is the new education policy?",
        "chat_history": [],
        "archetype": "factual_lookup",
        "plan": [],
        "tool_executions": [],
        "evidence_items": [],
        "synthesized_answer": "",
        "citations": [],
        "cost_usd": 0.0,
        "latency_ms": 0,
        "user_id": None,
        "model_override": None,
        "enable_web_search": False,
        "web_search_results": [],
        "active_issue_id": None,
        "active_newspaper_name": None,
        "active_issue_date": None,
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
        "evaluation_verdict": None,
        "recovery_attempts": 0,
        "gap_diagnosis": None,
    }
    base.update(kwargs)
    return base


@pytest.mark.asyncio
async def test_evaluator_fast_floor_skips_llm_on_high_confidence(mock_engines):
    """Verify that rich, grounded broadsheet evidence bypasses the LLM-as-Judge immediately (0ms added)."""
    entity_search, web_search, tool_maker, sql_analytics = mock_engines
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state = _make_state(
        query="Tell me about the BRICS summit security in Hindustan Times",
        archetype="factual_lookup",
    )
    rich_evidence = [
        {
            "article_id": 43033,
            "headline": "Security heightened, key areas beautified for Brics",
            "newspaper_name": "Hindustan Times",
            "issue_date": "2026-09-11",
            "snippet": "Elaborate security arrangements were put in place across the capital ahead of the high-profile BRICS leaders summit.",
            "prominence_score": 0.92,
            "pages": [10],
        },
        {
            "article_id": 43034,
            "headline": "Traffic advisories issued for Brics delegates",
            "newspaper_name": "Hindustan Times",
            "issue_date": "2026-09-11",
            "snippet": "Delhi Police issued traffic advisories restricting vehicle movements along core routes for the summit.",
            "prominence_score": 0.85,
            "pages": [11],
        },
    ]

    with patch.object(evaluator, "_evaluate_with_llm_judge", new=AsyncMock()) as mock_judge:
        verdict = await evaluator.evaluate_evidence_async(rich_evidence, state)
        assert verdict.is_sufficient is True
        assert verdict.recommended_action == "proceed_to_synthesis"
        assert verdict.quality_score >= 0.70
        # Judge should NOT be invoked for fast-floor pass
        assert not mock_judge.called


@pytest.mark.asyncio
async def test_evaluator_fast_floor_grounded_by_attached_asset(mock_engines):
    """Verify that an attached article match in evidence immediately passes fast-floor."""
    entity_search, web_search, tool_maker, sql_analytics = mock_engines
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state = _make_state(
        query="Summarize this article",
        attached_article_id=43033,
    )
    evidence = [
        {
            "article_id": 43033,
            "headline": "Security heightened for Brics",
            "newspaper_name": "Hindustan Times",
            "issue_date": "2026-09-11",
            "snippet": "Full text of article 43033...",
            "prominence_score": 0.9,
            "pages": [10],
        }
    ]

    with patch.object(evaluator, "_evaluate_with_llm_judge", new=AsyncMock()) as mock_judge:
        verdict = await evaluator.evaluate_evidence_async(evidence, state)
        assert verdict.is_sufficient is True
        assert verdict.recommended_action == "proceed_to_synthesis"
        assert not mock_judge.called


@pytest.mark.asyncio
async def test_evaluator_llm_judge_diagnoses_gap_and_routes_replan(mock_engines):
    """Verify that when evidence is empty, LLM judge diagnoses the gap and routes to replan_static_tools."""
    entity_search, web_search, tool_maker, sql_analytics = mock_engines
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state = _make_state(
        query="Find coverage of tourism policy in The Goan",
        tool_executions=[
            {
                "tool_name": "hybrid_search",
                "tool_input": {"query": "tourism policy", "newspaper_name": "The Goan", "date_from": "2026-09-11", "date_to": "2026-09-11"},
                "results_count": 0,
                "execution_time_ms": 50,
            }
        ],
    )

    mock_judge_response = json.dumps({
        "is_sufficient": False,
        "quality_score": 0.1,
        "gap_diagnosis": "Search returned 0 results due to overly narrow date constraint 2026-09-11 for The Goan.",
        "recommended_action": "replan_static_tools",
        "corrective_hints": {
            "suggested_tool": "hybrid_search",
            "suggested_query": "Goa tourism policy guidelines",
            "broaden_keywords": True,
        },
    })

    with patch("app.agent.evaluator.get_registry") as mock_reg:
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=ModelResponse(text=mock_judge_response))
        mock_reg.return_value.get_provider.return_value = mock_provider
        mock_reg.return_value.get_chat_provider.return_value = mock_provider

        verdict = await evaluator.evaluate_evidence_async([], state)
        assert verdict.is_sufficient is False
        assert verdict.recommended_action == "replan_static_tools"
        assert "narrow date constraint" in verdict.gap_reason
        assert verdict.corrective_hints.get("suggested_tool") == "hybrid_search"


@pytest.mark.asyncio
async def test_evaluator_guardrail_prevents_dynamic_tool_on_text_summarization(mock_engines):
    """Verify that dynamic_analysis is never recommended for text summarization even if LLM hallucinated it."""
    entity_search, web_search, tool_maker, sql_analytics = mock_engines
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state = _make_state(
        query="Security heightened, key areas beautified for Brics?, find this article and summrize ir",
        archetype="factual_lookup",
    )

    # Simulate LLM returning synthesize_dynamic_tool erroneously for text summarization
    mock_judge_response = json.dumps({
        "is_sufficient": False,
        "quality_score": 0.2,
        "gap_diagnosis": "Article text retrieved is long and requires summarization.",
        "recommended_action": "synthesize_dynamic_tool",
    })

    with patch("app.agent.evaluator.get_registry") as mock_reg:
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=ModelResponse(text=mock_judge_response))
        mock_reg.return_value.get_provider.return_value = mock_provider
        mock_reg.return_value.get_chat_provider.return_value = mock_provider

        verdict = await evaluator.evaluate_evidence_async([], state)
        assert verdict.is_sufficient is False
        # Guardrail MUST override synthesize_dynamic_tool to replan_static_tools
        assert verdict.recommended_action == "replan_static_tools"


@pytest.mark.asyncio
async def test_adaptive_replanner_anti_repetition():
    """Verify that QueryPlanner.replan_with_feedback_async avoids re-executing identical failing calls."""
    planner = QueryPlanner()
    query = "Find the Brics security article in Hindustan Times"

    previous_plan = [
        {"tool_name": "hybrid_search", "arguments": {"query": "Brics security", "newspaper_name": "Hindustan Times", "date_from": "2026-08-01", "date_to": "2026-08-01"}}
    ]
    tool_executions: list[ToolExecutionRecord] = [
        {
            "tool_name": "hybrid_search",
            "tool_input": {"query": "Brics security", "newspaper_name": "Hindustan Times", "date_from": "2026-08-01", "date_to": "2026-08-01"},
            "results_count": 0,
            "execution_time_ms": 60,
        }
    ]

    # Mock provider returning the exact same call (simulating a dumb LLM)
    mock_llm_output = json.dumps({
        "thought_process": "Retrying search with Hindustan Times",
        "archetype": "factual_lookup",
        "tool_calls": [
            {
                "tool_name": "hybrid_search",
                "arguments": {"query": "Brics security", "newspaper_name": "Hindustan Times", "date_from": "2026-08-01", "date_to": "2026-08-01"},
                "purpose": "Retry search",
            }
        ],
    })

    with patch("app.agent.planner.get_registry") as mock_reg:
        mock_prov = MagicMock()
        mock_prov.complete = AsyncMock(return_value=ModelResponse(text=mock_llm_output))
        mock_reg.return_value.get_provider.return_value = mock_prov
        mock_reg.return_value.get_chat_provider.return_value = mock_prov

        replan_res = await planner.replan_with_feedback_async(
            query=query,
            previous_plan=previous_plan,
            tool_executions=tool_executions,
            gap_diagnosis="Zero articles found on date 2026-08-01.",
        )

        assert len(replan_res.tool_calls) >= 1
        top_call = replan_res.tool_calls[0]
        # Anti-repetition guard should have stripped the failing date_from/date_to and expanded top_k
        assert "date_from" not in top_call.arguments
        assert "date_to" not in top_call.arguments
        assert top_call.arguments.get("top_k", 0) >= 8


@pytest.mark.asyncio
async def test_adaptive_replanner_heuristic_fallback_on_missing_coverage():
    """Verify deterministic fallback when LLM is unavailable and gap diagnosis specifies missing newspaper."""
    planner = QueryPlanner()
    query = "Compare coverage between The Goan and The Morning Standard"

    with patch("app.agent.planner.get_registry", side_effect=Exception("LLM offline")):
        replan_res = await planner.replan_with_feedback_async(
            query=query,
            previous_plan=[],
            tool_executions=[],
            gap_diagnosis="Retrieved evidence is completely missing articles from: The Morning Standard.",
        )

        assert len(replan_res.tool_calls) >= 1
        call = replan_res.tool_calls[0]
        assert call.tool_name == "hybrid_search"
        assert call.arguments.get("newspaper_name") == "The Morning Standard"


@pytest.mark.asyncio
async def test_evaluator_flags_missing_date_filter_on_date_query(mock_engines):
    """Verify evaluator flags missing date filter when user explicitly requests a date."""
    entity_search, web_search, tool_maker, sql_analytics = mock_engines
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state = _make_state(
        query="IS ANY NEWSPAPER AVAILABLE FOR DATED 28/04/2026",
        archetype="quantitative_trend",
    )
    # Evidence item has no date filter (Active Filters: None) and Overview
    evidence = [
        {
            "article_id": 0,
            "headline": "Issue Count Analysis: 24 issues found",
            "newspaper_name": "Archive",
            "issue_date": "Overview",
            "snippet": "=== RELATIONAL ISSUE COUNT AUDIT ===\n• Total Matching Issues: 24\n• Active Filters: None",
            "prominence_score": 1.0,
            "source_tool": "sql_analytics",
            "metadata": {"count": 24, "filters": {}},
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert not verdict.is_sufficient
    assert any("temporal_mismatch" in g for g in verdict.detected_gaps)
    assert verdict.recommended_action == "replan_static_tools"
    assert verdict.corrective_hints.get("issue_date") == "2026-04-28"


@pytest.mark.asyncio
async def test_evaluator_accepts_grounded_zero_count_availability_evidence(mock_engines):
    """Verify evaluator accepts verified zero-count audit specifically targeting the queried date as grounded proof."""
    entity_search, web_search, tool_maker, sql_analytics = mock_engines
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state = _make_state(
        query="IS ANY NEWSPAPER AVAILABLE FOR DATED 28/04/2026",
        archetype="quantitative_trend",
    )
    evidence = [
        {
            "article_id": 0,
            "headline": "Archive Availability Audit: 0 issues found for 2026-04-28",
            "newspaper_name": "Archive",
            "issue_date": "2026-04-28",
            "snippet": (
                "=== RELATIONAL ISSUE COUNT AUDIT ===\n"
                "• Target Date: 2026-04-28\n"
                "• Total Matching Issues: 0\n"
                "• Verification Status: No newspaper issues are available in the archive for 2026-04-28.\n"
                "• Archive Coverage Range: 2026-08-01 to 2026-09-11"
            ),
            "prominence_score": 1.0,
            "source_tool": "sql_analytics",
            "metadata": {"count": 0, "target_date": "2026-04-28"},
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert verdict.is_sufficient
    assert verdict.quality_score >= 0.70

