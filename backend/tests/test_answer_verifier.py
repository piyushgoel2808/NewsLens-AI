"""Unit tests for AnswerVerifier LLM Critic and Fact-Checking engine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.answer_verifier import (
    AnswerVerifier,
)
from app.providers.base import ModelResponse


@pytest.fixture
def mock_chat_provider():
    provider = MagicMock()
    provider.provider_name = "mock_provider"
    provider._model = "mock_model"
    return provider


@pytest.mark.asyncio
async def test_answer_verifier_accepts_grounded_answer(mock_chat_provider):
    """Verify that a faithful answer is accepted by LLM Judge."""
    judge_payload = {
        "is_valid": True,
        "has_hallucination": False,
        "has_contradiction": False,
        "evidence_gap_detected": False,
        "quality_score": 0.95,
        "factual_errors": [],
        "critique": "All numbers and dates correspond directly to retrieved evidence.",
        "recommended_action": "accept",
        "refined_answer": None,
    }
    mock_chat_provider.complete = AsyncMock(
        return_value=ModelResponse(text=json.dumps(judge_payload), input_tokens=100, output_tokens=50)
    )

    verifier = AnswerVerifier(provider=mock_chat_provider)
    evidence = [
        {
            "article_id": 101,
            "headline": "Goa Cabinet Approves New Industrial Policy",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-01",
            "snippet": "The Goa state cabinet on Friday formally approved the Goa Industrial Promotion Policy 2026.",
        }
    ]
    draft = "On 2026-08-01, The Goan reported that the Goa cabinet approved the Industrial Promotion Policy 2026."

    res = await verifier.verify_answer_async(
        query="What policy did Goa cabinet approve?",
        draft_answer=draft,
        evidence_items=evidence,
    )

    assert res.is_valid is True
    assert res.has_hallucination is False
    assert res.recommended_action == "accept"
    assert res.quality_score == 0.95


@pytest.mark.asyncio
async def test_answer_verifier_flags_hallucination_and_provides_refinement(mock_chat_provider):
    """Verify that AnswerVerifier detects ungrounded assertions and provides refined answer."""
    judge_payload = {
        "is_valid": False,
        "has_hallucination": True,
        "has_contradiction": True,
        "evidence_gap_detected": False,
        "quality_score": 0.3,
        "factual_errors": [
            "Asserted 150 crore funding allocated, but evidence mentions no monetary figures."
        ],
        "critique": "Draft hallucinated financial figures and added speculative corporate advice.",
        "recommended_action": "refine_answer",
        "refined_answer": "The Goan reported that the Goa cabinet approved the Industrial Policy 2026. No budget figures were disclosed.",
    }
    mock_chat_provider.complete = AsyncMock(
        return_value=ModelResponse(text=json.dumps(judge_payload), input_tokens=100, output_tokens=50)
    )

    verifier = AnswerVerifier(provider=mock_chat_provider)
    evidence = [
        {
            "article_id": 101,
            "headline": "Goa Cabinet Approves New Industrial Policy",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-01",
            "snippet": "The Goa state cabinet approved the Industrial Promotion Policy 2026.",
        }
    ]
    draft = "The Goa cabinet approved the policy with 150 crore rupees allocated. Investigate implications on corporate strategy."

    res = await verifier.verify_answer_async(
        query="What policy did Goa cabinet approve?",
        draft_answer=draft,
        evidence_items=evidence,
    )

    assert res.is_valid is False
    assert res.has_hallucination is True
    assert res.recommended_action == "refine_answer"
    assert res.refined_answer is not None
    assert "No budget figures were disclosed" in res.refined_answer


@pytest.mark.asyncio
async def test_answer_verifier_detects_evidence_gap_and_recommends_dynamic_fallback(mock_chat_provider):
    """Verify that AnswerVerifier detects evidence gaps and routes to dynamic code execution."""
    judge_payload = {
        "is_valid": False,
        "has_hallucination": True,
        "has_contradiction": False,
        "evidence_gap_detected": True,
        "quality_score": 0.2,
        "factual_errors": [
            "Retrieved evidence contains no data on word counts or article distribution for sports."
        ],
        "critique": "The evidence does not contain sports article metrics to answer this question.",
        "recommended_action": "fallback_to_dynamic_tool",
        "dynamic_tool_hint": "Execute dynamic SQL query computing average word_count in articles where section = 'Sports'",
        "refined_answer": None,
    }
    mock_chat_provider.complete = AsyncMock(
        return_value=ModelResponse(text=json.dumps(judge_payload), input_tokens=100, output_tokens=50)
    )

    verifier = AnswerVerifier(provider=mock_chat_provider)
    evidence = [
        {
            "article_id": 201,
            "headline": "Local Football Match Ends in Draw",
            "newspaper_name": "Herald",
            "issue_date": "2026-08-05",
            "snippet": "The inter-college tournament ended 1-1.",
        }
    ]
    draft = "The average word count of sports articles across all newspapers is approximately 650 words."

    res = await verifier.verify_answer_async(
        query="What is the average word count of sports articles across all newspapers?",
        draft_answer=draft,
        evidence_items=evidence,
    )

    assert res.is_valid is False
    assert res.evidence_gap_detected is True
    assert res.recommended_action == "fallback_to_dynamic_tool"
    assert "average word_count" in (res.dynamic_tool_hint or "")


def test_answer_verifier_fast_floor_zero_issue_contradiction():
    """Verify deterministic fast-floor catches 0-count issue contradictions immediately."""
    verifier = AnswerVerifier()
    evidence = [
        {
            "article_id": 0,
            "headline": "Archive Availability Audit: 0 issues found for 2026-04-28",
            "newspaper_name": "Archive",
            "issue_date": "2026-04-28",
            "snippet": "Total Matching Issues: 0\nVerification Status: No newspaper issues are available in the archive for 2026-04-28.",
            "source_tool": "sql_analytics",
            "metadata": {
                "count": 0,
                "target_date": "2026-04-28",
                "archive_range": {"start": "2026-08-01", "end": "2026-09-11"},
                "archive_newspapers": ["The Goan", "Hindustan Times"],
            },
        }
    ]

    draft = "Newspaper Availability for 28/04/2026: Yes. Total Matching Issues: 24."
    res = verifier._fast_groundedness_check(draft, evidence)

    assert res is not None
    assert res.is_valid is False
    assert res.has_contradiction is True
    assert res.recommended_action == "refine_answer"
    assert "No newspaper issues are available in the archive for 2026-04-28" in (res.refined_answer or "")
    assert "2026-08-01 to 2026-09-11" in (res.refined_answer or "")


def test_answer_verifier_catches_publication_scope_mismatch():
    """Verify that AnswerVerifier detects when an archive-wide query is narrowed to a single publication."""
    verifier = AnswerVerifier()
    query = "LIST DISTINCT NEWSPAPER NAMES AVAILABLE IN  SEPTEMBER  2026 "
    draft = (
        "⚡ EXECUTIVE SUMMARY: QUANTITATIVE ANALYSIS OF NEWSPAPER AVAILABILITY FOR BUSINESS STANDARD ON 2026-09-01\n"
        "The query seeks to analyze the number of distinct newspaper names available in September 2026, specifically for the publication Business Standard.\n"
        "Total Matching Issues for Business Standard on 2026-09-01: 0\n"
        "Newspaper Availability for Business Standard on 2026-09-01: No"
    )
    evidence = [
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

    res = verifier._fast_groundedness_check(draft_answer=draft, evidence_items=evidence, query=query)
    assert res is not None
    assert res.is_valid is False
    assert res.recommended_action == "fallback_to_dynamic_tool"
    assert "Publication scope mismatch" in res.critique


def test_answer_verifier_catches_nan_in_draft_answer():
    """Verify that AnswerVerifier detects 'nan words' in draft answers and routes to dynamic fallback."""
    verifier = AnswerVerifier()
    query = "WHAT IS THE AVG LENGTH OF ARTICLES IN NEWSPAPER THE GOAN DATED 1/8/2026"
    draft = (
        "⚡ EXECUTIVE SUMMARY: QUANTITATIVE ANALYSIS OF ARTICLE LENGTH IN THE GOAN ON 2026-08-01\n"
        "The query seeks to analyze the average article length in The Goan on 2026-08-01.\n"
        "Average Article Length in The Goan on 2026-08-01: nan words\n"
        "The average article length in The Goan on 2026-08-01 is nan words, indicating a consistent trend in article length."
    )
    evidence = [
        {
            "article_id": 0,
            "headline": "Analytical Computation: WHAT IS THE AVG LENGTH OF ARTICLES IN NEWSPAPER THE GOAN DAT",
            "newspaper_name": "Archive Analytics",
            "issue_date": "2026-08-01",
            "snippet": "The average article length in The Goan on 2026-08-01 is nan words.",
            "source_tool": "dynamic_analysis",
        }
    ]

    res = verifier._fast_groundedness_check(draft_answer=draft, evidence_items=evidence, query=query)
    assert res is not None
    assert res.is_valid is False
    assert res.evidence_gap_detected is True
    assert res.recommended_action == "fallback_to_dynamic_tool"
    assert "nan" in res.critique.lower()


