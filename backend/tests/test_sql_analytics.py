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


@pytest.mark.asyncio
async def test_list_issue_articles_nonexistent_issue_strict_error() -> None:
    mock_db = AsyncMock()
    mock_res_empty = MagicMock()
    mock_res_empty.scalars.return_value.first.return_value = None
    mock_db.execute.return_value = mock_res_empty

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)

    # 1. Non-existent issue_id
    res_id = await engine.list_issue_articles(issue_id=9999)
    assert "error" in res_id
    assert "Issue #9999 was not found" in res_id["error"]

    # 2. Non-existent newspaper name + date
    res_np_dt = await engine.list_issue_articles(
        newspaper_name="The Economic Times",
        issue_date="2026-08-27",
    )
    assert "error" in res_np_dt
    assert "No issue found for 'The Economic Times'" in res_np_dt["error"]

    # 3. Query string fallback extraction
    res_query = await engine.get_issue_summary(
        query="summrizze the whole newspaper of THE ECONOMICS times issue 84 dated 27/8/2026"
    )
    assert "error" in res_query
    assert "No issue found for 'The Economic Times'" in res_query["error"]


@pytest.mark.asyncio
async def test_list_issue_articles_category_filter() -> None:
    mock_db = AsyncMock()
    paper = Newspaper(id=1, name="The New York Times")
    issue = Issue(id=87, newspaper=paper, issue_date=datetime.date(2026, 8, 27), pages=[])
    art_news = Article(
        id=601,
        headline="Wheat crisis unfolds",
        section="National",
        article_type="lead_story",
        word_count=500,
    )
    art_sports = Article(
        id=602,
        headline="Tennis era comparison: Federer and Nadal",
        section="Sports",
        article_type="news",
        word_count=350,
    )

    mock_res_issue = MagicMock()
    mock_res_issue.scalars.return_value.first.return_value = issue
    mock_res_art = MagicMock()
    mock_res_art.scalars.return_value.all.return_value = [art_news, art_sports]
    mock_db.execute.side_effect = [mock_res_issue, mock_res_art]

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    summary = await engine.get_issue_summary(
        newspaper_name="The New York Times",
        category_filter="Sports",
    )

    assert summary["total_articles"] == 1
    assert summary["total_issue_articles"] == 2
    assert summary["articles"][0]["headline"] == "Tennis era comparison: Federer and Nadal"
    assert summary["articles"][0]["section"] == "Sports"


@pytest.mark.asyncio
async def test_list_issue_articles_secondary_topic_and_keyword_fallback() -> None:
    """Verify category_filter retrieves articles matching secondary topics or content keywords."""
    from app.models.entity import ArticleTopic, Topic

    mock_db = AsyncMock()
    paper = Newspaper(id=1, name="The Economic Times")
    issue = Issue(id=84, newspaper=paper, issue_date=datetime.date(2026, 8, 27), pages=[])

    # Article 1: Business news with secondary Sports topic
    t_sports = Topic(id=99, name="Sports", taxonomy_path="Newsroom > Sports")
    art_deal = Article(
        id=701,
        headline="Rajasthan Royals Deal: CCI Seeks More Details",
        section="Corporate & Industry",
        article_type="news",
        word_count=400,
    )
    at1 = ArticleTopic(article_id=701, topic_id=99, confidence=0.85)
    at1.topic = t_sports
    art_deal.article_topics = [at1]

    # Article 2: Sports news on inside page with keyword in subheadline
    art_cricket = Article(
        id=702,
        headline="India Turn the Screw",
        subheadline="SSC Test Sri Lanka's resistance fades after Mendis fifties as India close in on series win",
        section="National",
        article_type="news",
        word_count=450,
    )
    art_cricket.article_topics = []

    # Article 3: Pure politics news
    art_pol = Article(
        id=703,
        headline="Parliament Monsoon Session Concludes",
        subheadline="Opposition MPs raise questions on inflation",
        section="National",
        article_type="news",
        word_count=300,
    )
    art_pol.article_topics = []

    mock_res_issue = MagicMock()
    mock_res_issue.scalars.return_value.first.return_value = issue
    mock_res_art = MagicMock()
    mock_res_art.scalars.return_value.all.return_value = [art_deal, art_cricket, art_pol]
    mock_db.execute.side_effect = [mock_res_issue, mock_res_art]

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    res = await engine.list_issue_articles(
        newspaper_name="The Economic Times",
        category_filter="Sports",
    )

    art_list = res["articles"] if isinstance(res, dict) else res
    headlines = [a["headline"] for a in art_list]
    assert "Rajasthan Royals Deal: CCI Seeks More Details" in headlines
    assert "India Turn the Screw" in headlines
    assert "Parliament Monsoon Session Concludes" not in headlines


