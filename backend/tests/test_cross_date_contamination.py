"""Tests for cross-date context contamination and stale asset leakage prevention."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.condenser import (
    condense_conversational_query,
    extract_active_issue_from_history,
)
from app.agent.executor import ToolExecutor
from app.agent.state import AgentState
from app.agent.tool_factory import reconcile_and_sanitize_arguments
from app.models.article import Article
from app.models.newspaper import Issue, Newspaper


def test_extract_active_issue_drops_stale_asset_on_date_conflict():
    """Verify that when query specifies a date (e.g. 2026-08-01), an attached asset from 2026-08-05 is dropped."""
    query = "do the Goan dated 1/8/2026 have any pm modi or amit shah photo ?"
    res = extract_active_issue_from_history(
        chat_history=[],
        current_query=query,
        attached_article_id=41142,
        attached_issue_date="2026-08-05",
        attached_newspaper_name="The Goan",
        attached_headline="Ahead of protest, CM meets Shah on quota for STs",
    )

    assert res.get("issue_date") == "2026-08-01"
    assert res.get("newspaper_name") == "The Goan"
    assert res.get("article_id") is None
    assert res.get("photo_id") is None
    assert res.get("headline") is None


def test_extract_active_issue_retains_asset_when_no_date_conflict():
    """Verify that when query does not specify a conflicting date, the attached asset is retained."""
    query = "who is the person in this photo?"
    res = extract_active_issue_from_history(
        chat_history=[],
        current_query=query,
        attached_article_id=41142,
        attached_photo_id=9607,
        attached_issue_date="2026-08-05",
        attached_newspaper_name="The Goan",
        attached_headline="Ahead of protest, CM meets Shah on quota for STs",
    )

    assert res.get("issue_date") == "2026-08-05"
    assert res.get("newspaper_name") == "The Goan"
    assert res.get("article_id") == 41142
    assert res.get("photo_id") == 9607
    assert res.get("headline") == "Ahead of protest, CM meets Shah on quota for STs"


def test_extract_active_issue_retains_asset_when_same_date():
    """Verify that when query specifies the same date as the attached asset, it is retained."""
    query = "in the Goan dated 2026-08-05 what does this article say?"
    res = extract_active_issue_from_history(
        chat_history=[],
        current_query=query,
        attached_article_id=41142,
        attached_issue_date="2026-08-05",
        attached_newspaper_name="The Goan",
        attached_headline="Ahead of protest, CM meets Shah on quota for STs",
    )

    assert res.get("issue_date") == "2026-08-05"
    assert res.get("article_id") == 41142


@pytest.mark.asyncio
async def test_condense_query_excludes_stale_asset_block_on_date_conflict():
    """Verify that condense_conversational_query does not include attached asset from conflicting date."""
    mock_provider = MagicMock()
    captured_messages = []

    async def mock_complete(messages, **kwargs):
        captured_messages.extend(messages)
        res = MagicMock()
        res.text = "Does The Goan dated August 1, 2026, have any photos of PM Modi or Amit Shah?"
        return res

    mock_provider.complete = AsyncMock(side_effect=mock_complete)

    attached = {
        "article_id": 41142,
        "issue_date": "2026-08-05",
        "newspaper_name": "The Goan",
        "headline": "Ahead of protest, CM meets Shah on quota for STs",
    }

    _ = await condense_conversational_query(
        query="do the Goan dated 1/8/2026 have any pm modi or amit shah photo ?",
        chat_history=[],
        attached_asset=attached,
        provider=mock_provider,
    )

    assert len(captured_messages) > 0
    user_prompt = captured_messages[1].content
    # The prompt MUST NOT include the attached asset from August 5
    assert "Ahead of protest" not in user_prompt
    assert "2026-08-05" not in user_prompt


def test_tool_factory_reconcile_does_not_inject_stale_article_id():
    """Verify tool_factory does not inject attached_article_id if attached_issue_date conflicts."""
    extracted = {
        "newspaper_name": "The Goan",
        "issue_date": "2026-08-01",
        "attached_article_id": 41142,
        "attached_issue_date": "2026-08-05",
    }
    sanitized = reconcile_and_sanitize_arguments(
        tool_name="inspect_visual_asset",
        args={"query": "PM Modi or Amit Shah photo"},
        extracted=extracted,
        query="do the Goan dated 1/8/2026 have any pm modi or amit shah photo ?",
    )

    assert "article_id" not in sanitized
    assert sanitized.get("issue_date") == "2026-08-01"
    assert sanitized.get("newspaper_name") == "The Goan"


@pytest.mark.asyncio
async def test_executor_rejects_foreign_article_and_does_not_overwrite_query_date():
    """Verify that ToolExecutor._execute_inspect_visual_asset rejects an article from 2026-08-05
    when the user query explicitly requests 2026-08-01, and NEVER overwrites issue_date to 2026-08-05.
    """
    mock_np = Newspaper(id=1, name="The Goan")
    mock_issue_aug5 = Issue(id=99, newspaper_id=1, issue_date="2026-08-05", newspaper=mock_np)
    mock_article_aug5 = Article(
        id=41142,
        issue_id=99,
        headline="Ahead of protest, CM meets Shah on quota for STs",
        issue=mock_issue_aug5,
    )

    # Mock DB session
    mock_session = AsyncMock()

    async def mock_execute(stmt):
        mock_res = MagicMock()
        stmt_str = str(stmt)
        if "articles.id = :id" in stmt_str or "articles_1.id = :id" in stmt_str or "WHERE articles.id" in stmt_str or "article_id" in stmt_str:
            mock_res.scalar_one_or_none.return_value = mock_article_aug5
        else:
            mock_res.scalar_one_or_none.return_value = None
            mock_res.scalars.return_value.all.return_value = []
            mock_res.scalars.return_value.first.return_value = None
        return mock_res

    mock_session.execute = AsyncMock(side_effect=mock_execute)

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

    executor = ToolExecutor(
        session_factory=mock_session_factory,
        hybrid_search=MagicMock(),
        entity_search=MagicMock(),
        timeline_builder=MagicMock(),
        sql_analytics=MagicMock(),
        coverage_analyzer=MagicMock(),
        web_search=MagicMock(),
    )

    state: AgentState = {
        "query": "do the Goan dated 1/8/2026 have any pm modi or amit shah photo ?",
        "original_query": "do the Goan dated 1/8/2026 have any pm modi or amit shah photo ?",
        "chat_history": [],
        "archetype": "factual_lookup",
        "plan": [],
        "tool_executions": [],
        "evidence_items": [],
        "synthesized_answer": "",
        "citations": [],
        "cost_usd": 0.0,
        "latency_ms": 0,
        "active_newspaper_name": "The Goan",
        "active_issue_date": "2026-08-01",
        "attached_article_id": 41142,
    }

    results, count = await executor._execute_inspect_visual_asset(
        args={
            "query": "PM Modi or Amit Shah photo",
            "newspaper_name": "The Goan",
            "issue_date": "2026-08-01",
            "article_id": 41142,
        },
        state=state,
    )

    # Because Article 41142 is from 2026-08-05 and the query is for 2026-08-01:
    # 1. Article 41142 MUST BE REJECTED.
    # 2. It must not return Article 41142 or photos from August 5.
    assert count == 0
    assert len(results) == 0
