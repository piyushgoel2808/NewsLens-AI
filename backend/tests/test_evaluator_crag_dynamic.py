"""Unit tests for EvidenceEvaluator qualitative gap audit and ToolMaker dynamic fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.evaluator import (
    EvidenceEvaluator,
    _has_quantitative_payload,
)
from app.agent.state import AgentState
from app.agent.tool_maker import ToolMakerResult


@pytest.fixture
def mock_dependencies():
    entity_search = MagicMock()
    web_search = MagicMock()
    tool_maker = MagicMock()
    sql_analytics = MagicMock()
    return entity_search, web_search, tool_maker, sql_analytics


def test_audit_evidence_sufficiency_empty(mock_dependencies):
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)
    state: AgentState = {
        "query": "Tell me about the new health policy in The Goan",
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
    }

    verdict = evaluator.audit_evidence_sufficiency([], state)
    assert verdict.is_sufficient is False
    assert "empty_retrieval" in verdict.detected_gaps
    assert verdict.quality_score == 0.0


def test_audit_evidence_sufficiency_missing_comparison_newspaper(mock_dependencies):
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "give me the similar articles from newspaper of the Goan and the morning standard both dated 1/8/2026",
        "chat_history": [],
        "archetype": "cross_newspaper_comparison",
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
        "active_newspaper_name": "The Goan",
        "active_issue_date": "2026-08-01",
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    # Evidence only contains articles from The Goan
    evidence = [
        {
            "article_id": 40500,
            "headline": "Goa Tourism Curbs Announced",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-01",
            "snippet": "The Goa government has announced strict new regulations across all coastal tourist hubs.",
            "prominence_score": 0.9,
            "pages": [1],
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert verdict.is_sufficient is False
    assert any("missing_newspaper_coverage" in g for g in verdict.detected_gaps)
    assert "The Morning Standard" in verdict.gap_reason
    assert verdict.quality_score == 0.35


def test_audit_evidence_sufficiency_comparative_balance_satisfied(mock_dependencies):
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "give me the similar articles from newspaper of the Goan and the morning standard both dated 1/8/2026",
        "chat_history": [],
        "archetype": "cross_newspaper_comparison",
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
        "active_newspaper_name": "The Goan",
        "active_issue_date": "2026-08-01",
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    evidence = [
        {
            "article_id": 40500,
            "headline": "Goa Tourism Curbs Announced",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-01",
            "snippet": "The Goa government announced strict new regulations across all tourist hubs.",
            "prominence_score": 0.9,
            "pages": [1],
        },
        {
            "article_id": 41200,
            "headline": "Delhi University Compulsory Retirement",
            "newspaper_name": "The Morning Standard",
            "issue_date": "2026-08-01",
            "snippet": "A Delhi University professor was sent into compulsory retirement following an inquiry.",
            "prominence_score": 0.9,
            "pages": [1],
        },
    ]

    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert verdict.is_sufficient is True
    assert verdict.quality_score >= 0.70


def test_audit_evidence_sufficiency_quantitative_absence(mock_dependencies):
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "How many photos appear in each section of The Goan on August 5, 2026?",
        "chat_history": [],
        "archetype": "quantitative_trend",
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
        "active_newspaper_name": "The Goan",
        "active_issue_date": "2026-08-05",
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    # Only plain text snippets, no markdown table and no metadata metrics
    evidence = [
        {
            "article_id": 401,
            "headline": "Photo Exhibition in Panaji",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-05",
            "snippet": "Panaji hosted a photo exhibition showcasing local wildlife and cultural landscapes.",
            "prominence_score": 0.8,
            "pages": [3],
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert verdict.is_sufficient is False
    assert "missing_quantitative_payload" in verdict.detected_gaps
    assert "quantitative aggregates" in verdict.gap_reason.lower()


def test_audit_evidence_sufficiency_quantitative_present_via_table(mock_dependencies):
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "How many photos appear in each section of The Goan on August 5, 2026?",
        "chat_history": [],
        "archetype": "quantitative_trend",
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
        "active_newspaper_name": "The Goan",
        "active_issue_date": "2026-08-05",
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    evidence = [
        {
            "article_id": 0,
            "headline": "Photo Distribution Breakdown",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-05",
            "snippet": "| Section | Photo Count |\n|---|---|\n| Front Page | 8 |\n| Sports | 15 |\n",
            "prominence_score": 1.0,
            "source_tool": "sql_analytics",
        }
    ]

    assert _has_quantitative_payload(evidence) is True
    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert verdict.is_sufficient is True


@pytest.mark.asyncio
async def test_evaluate_and_fallback_triggers_tool_maker_with_gap_diagnosis(mock_dependencies):
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies

    # Configure dynamic tool maker mock
    tool_maker.generate_and_execute = AsyncMock(
        return_value=ToolMakerResult(
            success=True,
            code="async def analyze(db, q, ctx): return {'summary': 'Found 20 items', 'data': [], 'metadata': {}}",
            evidence_items=[
                {
                    "article_id": 9999,
                    "headline": "Recovered Shared Wire Story",
                    "newspaper_name": "The Morning Standard",
                    "issue_date": "2026-08-01",
                    "snippet": "Syndicated PTI wire story published simultaneously in both editions.",
                    "source_tool": "dynamic_tool",
                    "prominence_score": 0.95,
                }
            ],
            execution_time_ms=120,
        )
    )

    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "give me the similar articles from newspaper of the Goan and the morning standard both dated 1/8/2026",
        "chat_history": [],
        "archetype": "cross_newspaper_comparison",
        "plan": [{"tool_name": "sql_analytics", "arguments": {}}],
        "tool_executions": [
            {
                "tool_name": "sql_analytics",
                "tool_input": {"newspaper_name": "The Goan"},
                "results_count": 1,
                "execution_time_ms": 25,
            }
        ],
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
        "active_newspaper_name": "The Goan",
        "active_issue_date": "2026-08-01",
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    # Initial evidence has only The Goan
    initial_evidence = [
        {
            "article_id": 40500,
            "headline": "Goa Tourism Curbs Announced",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-01",
            "snippet": "The Goa government announced strict new regulations across all tourist hubs.",
            "prominence_score": 0.9,
            "pages": [1],
        }
    ]

    fallback_items, tool_records = await evaluator.evaluate_and_fallback(initial_evidence, state)

    # Verify ToolMaker was called with the gap diagnosis
    assert tool_maker.generate_and_execute.called
    call_kwargs = tool_maker.generate_and_execute.call_args[1]
    assert "The Morning Standard" in call_kwargs.get("gap_diagnosis", "")
    assert len(call_kwargs.get("attempted_tools", [])) == 1

    # Verify fallback items contain the synthesized item at the front
    assert len(fallback_items) == 2
    assert fallback_items[0]["article_id"] == 9999
    assert fallback_items[0]["headline"] == "Recovered Shared Wire Story"

    # Verify ToolExecutionRecord was created
    assert any(rec["tool_name"] == "crag_dynamic_tool_fallback" for rec in tool_records)
    crag_rec = next(rec for rec in tool_records if rec["tool_name"] == "crag_dynamic_tool_fallback")
    assert crag_rec["results_count"] == 1
    assert "The Morning Standard" in crag_rec["tool_input"]["gap_diagnosis"]


@pytest.mark.asyncio
async def test_evaluate_and_fallback_passes_sufficient_evidence_without_tool_maker(mock_dependencies):
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    tool_maker.generate_and_execute = AsyncMock()

    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "give me the similar articles from newspaper of the Goan and the morning standard both dated 1/8/2026",
        "chat_history": [],
        "archetype": "cross_newspaper_comparison",
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
        "active_newspaper_name": "The Goan",
        "active_issue_date": "2026-08-01",
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    balanced_evidence = [
        {
            "article_id": 40500,
            "headline": "Goa Tourism Curbs Announced",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-01",
            "snippet": "The Goa government announced strict new regulations across all tourist hubs.",
            "prominence_score": 0.9,
            "pages": [1],
        },
        {
            "article_id": 41200,
            "headline": "Delhi University Compulsory Retirement",
            "newspaper_name": "The Morning Standard",
            "issue_date": "2026-08-01",
            "snippet": "A Delhi University professor was sent into compulsory retirement following an inquiry.",
            "prominence_score": 0.9,
            "pages": [1],
        },
    ]

    res_evidence, tool_records = await evaluator.evaluate_and_fallback(balanced_evidence, state)
    assert len(res_evidence) == 2
    # ToolMaker should NOT have been invoked
    assert not tool_maker.generate_and_execute.called


def test_audit_evidence_sufficiency_quantitative_zero_count_routes_to_dynamic_tool(mock_dependencies):
    """Verify that when a quantitative query receives 0 count from static tools, it routes to dynamic tool synthesis."""
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "NO OF NEWSPAPER IN AUGUST",
        "chat_history": [],
        "archetype": "quantitative_trend",
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
    }

    evidence = [
        {
            "article_id": 0,
            "headline": "Archive Availability Audit: 0 issues found for 2026-08-01",
            "newspaper_name": "Archive",
            "issue_date": "2026-08-01",
            "snippet": (
                "=== RELATIONAL ISSUE COUNT AUDIT ===\n"
                "• Target Date: 2026-08-01\n"
                "• Total Matching Issues: 0\n"
                "• Newspaper Scope: All Newspapers\n"
                "• Verification Status: No newspaper issues are available in the archive for 2026-08-01.\n"
                "• Archive Coverage Range: 2026-08-01 to 2026-09-11\n"
                "• Available Publications in Archive: The Goan, The Hindu\n"
            ),
            "prominence_score": 1.0,
            "source_tool": "sql_analytics",
            "metadata": {
                "count": 0,
                "target_date": "2026-08-01",
                "archive_range": {"start": "2026-08-01", "end": "2026-09-11"},
            },
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert verdict.is_sufficient is False
    assert verdict.recommended_action == "synthesize_dynamic_tool"
    assert "zero_count_aggregate_gap" in verdict.detected_gaps


def test_audit_evidence_sufficiency_availability_zero_count_is_sufficient(mock_dependencies):
    """Verify that an explicit single-date availability query with 0 issues IS accepted as sufficient."""
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "IS ANY NEWSPAPER AVAILABLE FOR DATED 08/11/2026",
        "chat_history": [],
        "archetype": "quantitative_trend",
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
    }

    evidence = [
        {
            "article_id": 0,
            "headline": "Archive Availability Audit: 0 issues found for 2026-11-08",
            "newspaper_name": "Archive",
            "issue_date": "2026-11-08",
            "snippet": (
                "=== RELATIONAL ISSUE COUNT AUDIT ===\n"
                "• Target Date: 2026-11-08\n"
                "• Total Matching Issues: 0\n"
                "• Newspaper Scope: All Newspapers\n"
                "• Verification Status: No newspaper issues are available in the archive for 2026-11-08.\n"
                "• Archive Coverage Range: 2026-08-01 to 2026-09-11\n"
                "• Available Publications in Archive: The Goan, The Hindu\n"
            ),
            "prominence_score": 1.0,
            "source_tool": "sql_analytics",
            "metadata": {
                "count": 0,
                "target_date": "2026-11-08",
                "archive_range": {"start": "2026-08-01", "end": "2026-09-11"},
            },
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert verdict.is_sufficient is True
    assert verdict.recommended_action == "proceed_to_synthesis"


def test_audit_evidence_sufficiency_date_range_scope_mismatch_routes_to_dynamic_tool(mock_dependencies):
    """Verify that when a date-range query only receives single-day evidence, it routes to dynamic tool."""
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "NO OF NEWSPAPER IN AUGUST",
        "chat_history": [],
        "archetype": "quantitative_trend",
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
    }

    # Evidence has positive count (5 issues), but ONLY for a single day (2026-08-01), not the full month
    evidence = [
        {
            "article_id": 0,
            "headline": "Issue Count Analysis: 5 issues found for 2026-08-01",
            "newspaper_name": "Archive",
            "issue_date": "2026-08-01",
            "snippet": (
                "=== RELATIONAL ISSUE COUNT AUDIT ===\n"
                "• Total Matching Issues: 5\n"
                "• Target Date: 2026-08-01\n"
                "• Newspaper(s): The Goan\n"
            ),
            "prominence_score": 1.0,
            "source_tool": "sql_analytics",
            "metadata": {
                "count": 5,
                "total_issues": 5,
                "target_date": "2026-08-01",
                "filters": {"issue_date": "2026-08-01"},
            },
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(evidence, state)
    assert verdict.is_sufficient is False
    assert verdict.recommended_action == "synthesize_dynamic_tool"
    assert any("temporal_range_scope_mismatch" in g for g in verdict.detected_gaps)


def test_audit_evidence_sufficiency_error_evidence_routes_to_dynamic_tool(mock_dependencies):
    """Verify that when static retrieval returns an error/failure snippet, it routes to dynamic tool."""
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "LIST DISTINCT NEWSPAPER NAMES AVAILABLE IN  SEPTEMBER  2026 ",
        "chat_history": [],
        "archetype": "article_catalog",
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
    }

    err_evidence = [
        {
            "article_id": 0,
            "headline": "Issue Summary Error: No issue found for Business Standard on 2026-09-01",
            "newspaper_name": "Business Standard",
            "issue_date": "2026-09-01",
            "pages": [1],
            "snippet": "⚠️ No issue found for Business Standard on 2026-09-01",
            "prominence_score": 1.0,
            "source_tool": "sql_analytics",
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(err_evidence, state)
    assert verdict.is_sufficient is False
    assert verdict.recommended_action == "synthesize_dynamic_tool"
    assert "tool_execution_error_gap" in verdict.detected_gaps


def test_audit_evidence_sufficiency_archive_newspaper_scope_mismatch_routes_to_dynamic(mock_dependencies):
    """Verify that when an archive-wide distinct newspaper query receives only single-pub evidence, it routes to dynamic tool."""
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state: AgentState = {
        "query": "LIST DISTINCT NEWSPAPER NAMES AVAILABLE IN  SEPTEMBER  2026 ",
        "chat_history": [],
        "archetype": "quantitative_trend",
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
    }

    mismatch_evidence = [
        {
            "article_id": 0,
            "headline": "Issue & Newspaper Count Analysis: 1 newspapers (2 issues) found for 2026-09-01 to 2026-09-30",
            "newspaper_name": "Business Standard",
            "issue_date": "Overview",
            "pages": [1],
            "snippet": "=== RELATIONAL ISSUE COUNT AUDIT ===\n• Total Matching Issues: 2\n• Target Date / Range: 2026-09-01 to 2026-09-30\n• Newspaper(s): Business Standard",
            "prominence_score": 1.0,
            "source_tool": "sql_analytics",
            "metadata": {"count": 2, "total_issues": 2, "filters": {"newspaper_name": "Business Standard"}},
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(mismatch_evidence, state)
    assert verdict.is_sufficient is False
    assert verdict.recommended_action == "synthesize_dynamic_tool"
    assert "archive_newspaper_scope_mismatch" in verdict.detected_gaps


def test_audit_evidence_sufficiency_rejects_nan_metric_and_routes_to_dynamic(mock_dependencies):
    """Verify that EvidenceEvaluator detects NaN/nan in evidence snippet and routes to dynamic tool synthesis."""
    entity_search, web_search, tool_maker, sql_analytics = mock_dependencies
    evaluator = EvidenceEvaluator(entity_search, web_search, tool_maker, sql_analytics)

    state = {
        "query": "WHAT IS THE AVG LENGTH OF ARTICLES IN NEWSPAPER THE GOAN DATED 1/8/2026",
        "archetype": "analytical_computation",
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
        "active_newspaper_name": "The Goan",
        "active_issue_date": "2026-08-01",
        "attached_article_id": None,
        "attached_photo_id": None,
        "attached_asset": None,
        "error": None,
    }

    nan_evidence = [
        {
            "article_id": 0,
            "headline": "Analytical Computation: WHAT IS THE AVG LENGTH OF ARTICLES IN NEWSPAPER THE GOAN DAT",
            "newspaper_name": "Archive Analytics",
            "issue_date": "2026-08-01",
            "pages": [1],
            "snippet": "The average article length in The Goan on 2026-08-01 is nan words.",
            "prominence_score": 1.0,
            "source_tool": "dynamic_analysis",
            "metadata": {"avg_word_count": float("nan")},
        }
    ]

    verdict = evaluator.audit_evidence_sufficiency(nan_evidence, state)
    assert verdict.is_sufficient is False
    assert verdict.recommended_action == "synthesize_dynamic_tool"
    assert "tool_execution_error_gap" in verdict.detected_gaps
    assert "NaN" in verdict.gap_reason



