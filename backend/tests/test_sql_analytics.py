"""Unit tests for SQLAnalyticsEngine."""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.article import Article
from app.models.newspaper import Issue, Newspaper, Page
from app.retrieval.sql_analytics import SQLAnalyticsEngine


@pytest.mark.asyncio
async def test_get_issue_summary() -> None:
    paper = Newspaper(
        id=1,
        name="The Financial Chronicle",
        country="India",
        default_language="en",
    )
    issue = Issue(
        id=10,
        newspaper_id=1,
        newspaper=paper,
        issue_date=datetime.date(2026, 7, 7),
        edition="National",
        language="en",
        total_pages=4,
        ingestion_status="indexed",
    )
    p1 = Page(
        id=101,
        issue_id=10,
        page_number=1,
        printed_page_number="Cover Wrap",
        is_advertisement_page=True,
    )
    p2 = Page(
        id=102,
        issue_id=10,
        page_number=2,
        printed_page_number="1",
        is_advertisement_page=False,
    )
    issue.pages = [p1, p2]

    art1 = Article(
        id=501,
        issue_id=10,
        primary_page_id=p2.id,
        headline="GDP GROWTH HITS 8.2 PERCENT",
        section="Economy",
        article_type="lead_story",
        byline_author="Staff Reporter",
        prominence_score=0.95,
        word_count=450,
    )
    art2 = Article(
        id=502,
        issue_id=10,
        primary_page_id=p1.id,
        headline="LUXURY WATCHES SALE",
        section="Commercial",
        article_type="advertisement",
        byline_author=None,
        prominence_score=0.20,
        word_count=80,
    )

    mock_db = AsyncMock()
    # First query returns issue, second returns articles
    mock_res_issue = MagicMock()
    mock_res_issue.scalars.return_value.first.return_value = issue
    mock_res_art = MagicMock()
    mock_res_art.scalars.return_value.all.return_value = [art1, art2]

    mock_db.execute.side_effect = [mock_res_issue, mock_res_art]

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    summary = await engine.get_issue_summary(
        newspaper_name="Financial Chronicle",
        issue_date="2026-07-07",
    )

    assert summary["newspaper"] == "The Financial Chronicle"
    assert summary["issue_date"] == "2026-07-07"
    assert summary["total_articles"] == 2
    assert summary["total_pages"] == 2
    assert summary["section_breakdown"]["Economy"] == 1
    assert summary["section_breakdown"]["Commercial"] == 1

    lead_art = next(
        a for a in summary["articles"]
        if a["headline"] == "GDP GROWTH HITS 8.2 PERCENT"
    )
    assert lead_art["printed_page"] == "1"
    assert lead_art["page_number"] == 2
    assert lead_art["byline_author"] == "Staff Reporter"

    # Test with page_filter="1"
    mock_db.execute.side_effect = [mock_res_issue, mock_res_art]
    p_summary = await engine.get_issue_summary(
        newspaper_name="Financial Chronicle",
        issue_date="2026-07-07",
        page_filter="1",
    )
    assert p_summary["total_articles"] == 1
    assert p_summary["total_issue_articles"] == 2
    assert p_summary["articles"][0]["headline"] == "GDP GROWTH HITS 8.2 PERCENT"


@pytest.mark.asyncio
async def test_count_articles() -> None:
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalar.return_value = 14
    mock_db.execute.return_value = mock_res

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    res = await engine.count_articles(
        newspaper_name="Financial Chronicle",
        section="Economy",
    )

    assert res["count"] == 14
    assert res["filters"]["newspaper_name"] == "Financial Chronicle"
    assert res["filters"]["section"] == "Economy"


@pytest.mark.asyncio
async def test_list_issue_articles() -> None:
    mock_db = AsyncMock()
    paper = Newspaper(id=1, name="Financial Chronicle")
    issue = Issue(id=10, newspaper=paper, issue_date=datetime.date(2026, 7, 7), pages=[])
    art1 = Article(
        id=501,
        headline="TECH STOCKS RALLY",
        section="Markets",
        article_type="news",
        word_count=300,
    )
    art2 = Article(
        id=502,
        headline="LOCAL SPORTS RECAP",
        section="Sports",
        article_type="news",
        word_count=200,
    )

    mock_res_issue = MagicMock()
    mock_res_issue.scalars.return_value.first.return_value = issue
    mock_res_art = MagicMock()
    mock_res_art.scalars.return_value.all.return_value = [art1, art2]
    mock_db.execute.side_effect = [mock_res_issue, mock_res_art]

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    articles = await engine.list_issue_articles(
        newspaper_name="Financial Chronicle",
        section="Markets",
    )

    assert len(articles) == 1
    assert articles[0]["headline"] == "TECH STOCKS RALLY"
