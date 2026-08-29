"""Unit tests for Narrative Trajectory Engine and Timeline API."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.models.article import Article, ArticlePage
from app.models.newspaper import Issue, Newspaper
from app.retrieval.timeline_builder import (
    NarrativeTrajectoryResponse,
    NewspaperPerspective,
    TimelineBuilder,
    TimelineMilestone,
)


@pytest.mark.asyncio
async def test_narrative_trajectory_models() -> None:
    """Verify Pydantic models serialize and deserialize correctly."""
    p = NewspaperPerspective(
        newspaper_name="Mint",
        issue_date="2026-08-01",
        pdf_page=1,
        headline="Tata Power Signs Solar Agreement",
        key_takeaway="Tata Power secured 500MW solar contract in Odisha.",
        angle="Financial Impact",
        bboxes=[{"x0": 10, "y0": 20, "x1": 100, "y1": 200}],
        issue_id=1,
        article_id=101,
    )
    assert p.newspaper_name == "Mint"
    assert p.angle == "Financial Impact"

    m = TimelineMilestone(
        milestone_id="m_1",
        date="2026-08-01",
        canonical_event="Tata Power signs 500MW solar agreement with Odisha government.",
        event_phase="Breaking",
        perspectives=[p],
        discrepancies=["Mint reported 500MW while Business Standard reported 450MW"],
    )
    assert m.event_phase == "Breaking"
    assert len(m.discrepancies) == 1

    resp = NarrativeTrajectoryResponse(
        query="Tata Power Solar Deal",
        topic_summary="Tata Power announced solar deal in early August.",
        date_range=["2026-08-01", "2026-08-05"],
        milestones=[m],
        latency_ms=45,
        cost_usd=0.001,
        cached=False,
    )
    assert len(resp.milestones) == 1
    dump = resp.model_dump()
    assert dump["query"] == "Tata Power Solar Deal"
    assert dump["milestones"][0]["perspectives"][0]["newspaper_name"] == "Mint"


@pytest.mark.asyncio
async def test_timeline_builder_build_timeline() -> None:
    """Test legacy build_timeline grouping."""
    paper = Newspaper(id=1, name="Mint", country="India", default_language="en")
    issue = Issue(
        id=10,
        newspaper_id=1,
        newspaper=paper,
        issue_date=datetime.date(2026, 8, 1),
    )
    art = Article(
        id=101,
        issue_id=10,
        headline="Tata Power Deal",
        summary="Deal signed.",
        prominence_score=0.9,
        issue=issue,
    )
    ap = ArticlePage(
        id=1,
        article_id=101,
        page_id=5,
        page_number=1,
        bbox_json=[{"x0": 0, "y0": 0, "x1": 10, "y1": 10}],
    )
    art.article_pages = [ap]

    mock_db = MagicMock()
    mock_db_res = MagicMock()
    mock_db_res.scalars.return_value.all.return_value = [art]
    mock_db.execute = AsyncMock(return_value=mock_db_res)

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__.return_value = mock_db

    builder = TimelineBuilder(session_factory=mock_factory)
    res = await builder.build_timeline(query="Tata Power")
    assert res.total_articles == 1
    assert res.total_dates == 1
    assert res.date_groups[0].date == "2026-08-01"


@pytest.mark.asyncio
async def test_timeline_builder_build_narrative_trajectory_heuristic_fallback() -> None:
    """Test build_narrative_trajectory heuristic fallback when LLM is unavailable."""
    paper1 = Newspaper(id=1, name="Mint", country="India", default_language="en")
    paper2 = Newspaper(id=2, name="Business Standard", country="India", default_language="en")
    issue1 = Issue(id=10, newspaper_id=1, newspaper=paper1, issue_date=datetime.date(2026, 8, 1))
    issue2 = Issue(id=11, newspaper_id=2, newspaper=paper2, issue_date=datetime.date(2026, 8, 1))

    art1 = Article(
        id=101,
        issue_id=10,
        headline="Tata Power Signs Solar Agreement",
        summary="500MW project initiated.",
        prominence_score=0.9,
        section="Companies",
        issue=issue1,
    )
    art2 = Article(
        id=102,
        issue_id=11,
        headline="Odisha Grants Clean Energy Mandate",
        summary="Tata Power wins contract.",
        prominence_score=0.85,
        section="Industry",
        issue=issue2,
    )
    art1.article_pages = [ArticlePage(id=1, article_id=101, page_id=5, page_number=1, bbox_json=[])]
    art2.article_pages = [ArticlePage(id=2, article_id=102, page_id=6, page_number=2, bbox_json=[])]

    mock_db = MagicMock()
    mock_db_res = MagicMock()
    mock_db_res.scalars.return_value.all.return_value = [art1, art2]
    mock_db.execute = AsyncMock(return_value=mock_db_res)

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__.return_value = mock_db

    mock_cache = MagicMock()
    mock_cache.get_query = AsyncMock(return_value=None)
    mock_cache.set_query = AsyncMock(return_value=True)

    progress_events = []

    async def on_progress(stage: str, data: dict) -> None:
        progress_events.append(stage)

    builder = TimelineBuilder(session_factory=mock_factory, cache_store=mock_cache)
    res = await builder.build_narrative_trajectory(
        query="Tata Power solar",
        use_cache=False,
        on_progress=on_progress,
    )

    assert res.query == "Tata Power solar"
    assert len(res.milestones) == 1
    m = res.milestones[0]
    assert m.date == "2026-08-01"
    assert len(m.perspectives) == 2
    assert hasattr(m, "active_entities")
    assert "fetching_articles" in progress_events
    assert "clustering_dates" in progress_events
    assert "completed" in progress_events


@pytest.mark.asyncio
async def test_timeline_builder_redis_caching() -> None:
    """Verify that cached queries return immediately with cached=True."""
    mock_cache = MagicMock()
    mock_cache.get_query = AsyncMock(
        return_value={
            "query": "Tata Power",
            "topic_summary": "Cached summary",
            "date_range": ["2026-08-01", "2026-08-02"],
            "milestones": [],
            "latency_ms": 0,
            "cost_usd": 0.0,
            "cached": True,
        }
    )

    mock_factory = MagicMock()
    builder = TimelineBuilder(session_factory=mock_factory, cache_store=mock_cache)

    res = await builder.build_narrative_trajectory(query="Tata Power", use_cache=True)
    assert res.cached is True
    assert res.topic_summary == "Cached summary"
    mock_cache.get_query.assert_called_once()


@pytest.mark.asyncio
async def test_timeline_api_endpoints() -> None:
    """Test POST /api/query/timeline and /api/query/timeline/stream."""
    app = create_app()

    mock_session_factory = MagicMock()
    mock_db = MagicMock()
    mock_db_res = MagicMock()
    mock_db_res.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_db_res)
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    with patch("app.api.routers.query.get_session_factory", return_value=mock_session_factory):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. REST Endpoint
            resp = await client.post(
                "/api/query/timeline",
                json={"query": "Telecom Spectrum Auction"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "milestones" in data
            assert data["query"] == "Telecom Spectrum Auction"

            # 2. SSE Stream Endpoint
            stream_resp = await client.post(
                "/api/query/timeline/stream",
                json={"query": "Telecom Spectrum Auction"},
            )
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers.get("content-type", "")
            stream_content = stream_resp.text
            assert "event: stage" in stream_content or "event: result" in stream_content
