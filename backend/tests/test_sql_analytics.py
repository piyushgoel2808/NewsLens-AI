"""Unit tests for SQLAnalyticsEngine."""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.asyncio
async def test_get_newspaper_coverage_difference() -> None:
    from unittest.mock import AsyncMock, patch

    engine = SQLAnalyticsEngine(session_factory=MagicMock())

    source_summary = {
        "newspaper": "The Goan",
        "issue_date": "2026-08-01",
        "articles": [
            {
                "id": 1,
                "headline": "Beware! AI-enabled traffic challans go live from today",
                "section": "Front Page",
                "page_number": 1,
                "printed_page": "1",
                "summary": "Goa traffic police launch AI challans.",
            },
            {
                "id": 2,
                "headline": "Modi, Burnham talk better bilateral ties",
                "section": "National",
                "page_number": 2,
                "printed_page": "2",
                "summary": "Prime Minister holds bilateral discussions.",
            },
        ],
    }

    comp_summary = {
        "newspaper": "The Morning Standard",
        "issue_date": "2026-08-01",
        "articles": [
            {
                "id": 101,
                "headline": "MODI, BURNHAM TALK BETTER TIES",
                "section": "Front Page",
                "page_number": 1,
                "printed_page": "1",
                "summary": "Talks on trade and ties.",
            },
            {
                "id": 102,
                "headline": "Mega allocation for offshore oil & gas exploration",
                "section": "National",
                "page_number": 3,
                "printed_page": "3",
                "summary": "Government announces energy investments.",
            },
        ],
    }

    with patch.object(engine, "list_issue_articles", AsyncMock(side_effect=[source_summary, comp_summary])):
        diff = await engine.get_newspaper_coverage_difference(
            source_newspaper="The Goan",
            comparison_newspaper="The Morning Standard",
            issue_date="2026-08-01",
        )

        assert diff["source_newspaper"] == "The Goan"
        assert diff["comparison_newspaper"] == "The Morning Standard"
        assert diff["exclusive_count"] == 1
        assert diff["shared_count"] == 1
        assert diff["exclusive_articles"][0]["headline"] == "Beware! AI-enabled traffic challans go live from today"
        assert diff["shared_articles"][0]["source_headline"] == "Modi, Burnham talk better bilateral ties"


@pytest.mark.asyncio
async def test_get_photo_counts_by_section() -> None:
    """Verify get_photo_counts_by_section returns aggregated counts and section breakdown."""
    mock_db = AsyncMock()
    mock_res = MagicMock()
    mock_res.all.return_value = [("National", 19), ("Health", 12), ("Front Page", 11)]
    mock_db.execute.return_value = mock_res

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    result = await engine.get_photo_counts_by_section(
        newspaper_name="The Goan",
        issue_date="2026-08-05",
    )

    assert result["total_photos"] == 42
    assert len(result["section_counts"]) == 3
    assert result["by_section"]["National"] == 19
    assert result["by_section"]["Health"] == 12
    assert result["by_section"]["Front Page"] == 11
    assert result["filters"]["newspaper_name"] == "The Goan"
    assert result["filters"]["issue_date"] == "2026-08-05"


@pytest.mark.asyncio
async def test_get_newspaper_shared_coverage() -> None:
    """Verify 2-tier hybrid shared coverage matching and metadata."""
    mock_session_factory = MagicMock()
    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)

    source_summary = {
        "newspaper": "The Goan",
        "issue_date": "2026-08-01",
        "articles": [
            {
                "id": 1,
                "headline": "SC stays stray animal compensation order",
                "page_number": 5,
                "section": "National",
                "byline_author": "PTI",
                "word_count": 250,
            },
            {
                "id": 2,
                "headline": "Local Goa Panchayat meets on beach cleaning",
                "page_number": 2,
                "section": "Goa",
                "byline_author": "Staff Reporter",
                "word_count": 180,
            },
        ],
    }

    comp_summary = {
        "newspaper": "The Morning Standard",
        "issue_date": "2026-08-01",
        "articles": [
            {
                "id": 101,
                "headline": "Apex court puts hold on stray animal compensation",
                "page_number": 7,
                "section": "Nation",
                "byline_author": "PTI",
                "word_count": 260,
            },
            {
                "id": 102,
                "headline": "Delhi Metro extends yellow line services",
                "page_number": 3,
                "section": "City",
                "byline_author": "Express News",
                "word_count": 150,
            },
        ],
    }

    with patch.object(engine, "list_issue_articles", side_effect=[source_summary, comp_summary]):
        res = await engine.get_newspaper_shared_coverage(
            newspaper_a="The Goan",
            newspaper_b="The Morning Standard",
            issue_date="2026-08-01",
        )

        assert res["newspaper_a"] == "The Goan"
        assert res["newspaper_b"] == "The Morning Standard"
        assert res["issue_date"] == "2026-08-01"
        assert res["shared_count"] == 1
        assert len(res["shared_stories"]) == 1

        story = res["shared_stories"][0]
        assert story["article_id_a"] == 1
        assert story["article_id_b"] == 101
        assert story["headline_a"] == "SC stays stray animal compensation order"
        assert story["headline_b"] == "Apex court puts hold on stray animal compensation"
        assert story["newspaper_a"] == "The Goan"
        assert story["newspaper_b"] == "The Morning Standard"
        assert "stray" in story["shared_keywords"]
        assert "animal" in story["shared_keywords"]
        assert "compensation" in story["shared_keywords"]


@pytest.mark.asyncio
async def test_count_advertisements() -> None:
    from collections import namedtuple

    mock_db = AsyncMock()
    mock_res = MagicMock()

    Row = namedtuple("Row", ["id", "headline", "section", "article_type", "word_count", "issue_date", "newspaper_name", "page_number"])
    mock_rows = [
        Row(42923, "[Advertisement] Test Ad 1", "Advertisements & Notices", "advertisement", 400, datetime.date(2026, 9, 11), "Hindustan Times", 1),
        Row(42931, "[Advertisement] Test Ad 2", "Advertisements & Notices", "advertisement", 150, datetime.date(2026, 9, 11), "Hindustan Times", 3),
    ]
    mock_res.all.return_value = mock_rows
    mock_db.execute.return_value = mock_res

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    res = await engine.count_advertisements(newspaper_name="Hindustan Times", issue_date="2026-09-11")

    assert res["count"] == 2
    assert len(res["advertisements"]) == 2
    assert res["advertisements"][0]["headline"] == "[Advertisement] Test Ad 1"
    assert res["advertisements"][0]["page_number"] == 1
    assert res["advertisements"][1]["page_number"] == 3


@pytest.mark.asyncio
async def test_count_issues() -> None:
    mock_db = AsyncMock()
    mock_res_count = MagicMock()
    mock_res_count.scalar.return_value = 8

    mock_res_det = MagicMock()
    mock_res_det.all.return_value = [(1, "2026-08-01", "The Goan", 12)]

    mock_db.execute.side_effect = [mock_res_count, mock_res_det]

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    res = await engine.count_issues(newspaper_name="The Goan", issue_date="2026-08-01")

    assert res["count"] == 8
    assert res["filters"]["newspaper_name"] == "The Goan"
    assert res["filters"]["issue_date"] == "2026-08-01"
    assert "The Goan" in res["newspapers"]


@pytest.mark.asyncio
async def test_count_issues_zero_hits_returns_archive_range() -> None:
    mock_db = AsyncMock()
    mock_res_count = MagicMock()
    mock_res_count.scalar.return_value = 0

    mock_res_range = MagicMock()
    mock_res_range.one_or_none.return_value = ("2026-08-01", "2026-09-11")

    mock_res_nps = MagicMock()
    mock_res_nps.all.return_value = [("The Goan",), ("Hindustan Times",)]

    mock_db.execute.side_effect = [mock_res_count, mock_res_range, mock_res_nps]

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    engine = SQLAnalyticsEngine(session_factory=mock_session_factory)
    res = await engine.count_issues(issue_date="28/04/2026")

    assert res["count"] == 0
    assert res["filters"]["issue_date"] == "2026-04-28"
    assert res["archive_range"] == {"start": "2026-08-01", "end": "2026-09-11"}
    assert "The Goan" in res["archive_newspapers"]






