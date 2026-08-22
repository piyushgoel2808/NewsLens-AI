"""Unit tests for Corpus and Issue APIs: GET /api/newspapers and GET /api/issues."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.models.base import get_db


@dataclass
class _MockNewspaperRow:
    id: int
    name: str
    publisher: str
    default_language: str
    country: str
    issue_count: int
    earliest_issue: str
    latest_issue: str
    article_count: int = 10


@dataclass
class _MockIssue:
    id: int
    newspaper_id: int
    newspaper: Any
    issue_date: str
    edition: str
    language: str
    total_pages: int
    pages: list[Any]
    ingestion_status: str
    created_at: Any


@pytest.mark.asyncio
async def test_list_newspapers_endpoint() -> None:
    app = create_app()
    mock_db = MagicMock()
    mock_res = MagicMock()
    mock_res.all.return_value = [
        _MockNewspaperRow(
            id=1,
            name="The Daily Record",
            publisher="Record Media",
            default_language="en",
            country="IN",
            issue_count=2,
            earliest_issue="2026-08-01",
            latest_issue="2026-08-21",
            article_count=10,
        )
    ]
    mock_db.execute = AsyncMock(return_value=mock_res)

    async def override_get_db() -> AsyncGenerator[MagicMock, None]:
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/newspapers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "The Daily Record"
        assert data[0]["article_count"] == 10


@pytest.mark.asyncio
async def test_list_issues_endpoint() -> None:
    app = create_app()

    mock_db = MagicMock()
    mock_res = MagicMock()
    mock_np = MagicMock()
    mock_np.name = "The Daily Record"

    created = MagicMock()
    created.isoformat.return_value = "2026-08-21T00:00:00"

    mock_issue = _MockIssue(
        id=1,
        newspaper_id=1,
        newspaper=mock_np,
        issue_date="2026-08-21",
        edition="morning",
        language="en",
        total_pages=4,
        pages=[MagicMock(), MagicMock()],
        ingestion_status="completed",
        created_at=created,
    )
    mock_res.all.return_value = [(mock_issue, 5, 12)]
    mock_res.scalars.return_value.all.return_value = [mock_issue]
    mock_db.execute = AsyncMock(return_value=mock_res)

    async def override_get_db() -> AsyncGenerator[MagicMock, None]:
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/issues")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["newspaper_name"] == "The Daily Record"
        assert data[0]["ingestion_status"] == "completed"


@pytest.mark.asyncio
async def test_get_article_details_endpoint() -> None:
    app = create_app()

    mock_db = MagicMock()
    mock_res = MagicMock()
    mock_np = MagicMock()
    mock_np.name = "The Daily Record"

    mock_issue = MagicMock()
    mock_issue.newspaper = mock_np
    mock_issue.issue_date = "2026-08-21"

    mock_chunk = MagicMock()
    mock_chunk.id = 101
    mock_chunk.chunk_index = 0
    mock_chunk.text = "[The Daily Record, 2026-08-21] Test chunk text."
    mock_chunk.token_count = 12
    mock_chunk.embedding_vector_id = "vec-123"

    mock_page = MagicMock()
    mock_page.page_number = 1

    mock_article = MagicMock()
    mock_article.id = 1
    mock_article.issue_id = 10
    mock_article.issue = mock_issue
    mock_article.headline = "MARKET SURGE"
    mock_article.subheadline = "Stocks reach high"
    mock_article.byline_author = "Reporter"
    mock_article.section = "Business"
    mock_article.article_type = "news"
    mock_article.prominence_score = 0.95
    mock_article.word_count = 150
    mock_article.summary = "Summary text"
    mock_article.full_text = "Full article text..."
    mock_article.article_pages = [mock_page]
    mock_article.chunks = [mock_chunk]

    mock_res.scalar_one_or_none.return_value = mock_article
    mock_db.execute = AsyncMock(return_value=mock_res)

    async def override_get_db() -> AsyncGenerator[MagicMock, None]:
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/articles/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 1
        assert data["headline"] == "MARKET SURGE"
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["embedding_vector_id"] == "vec-123"


@pytest.mark.asyncio
async def test_inspect_issue_ingestion_endpoint() -> None:
    app = create_app()

    mock_db = MagicMock()
    mock_issue_res = MagicMock()
    mock_np = MagicMock()
    mock_np.name = "The Daily Record"

    mock_page = MagicMock()
    mock_page.id = 10
    mock_page.page_number = 1
    mock_page.width_px = 2400
    mock_page.height_px = 3300
    mock_page.ocr_confidence = 0.92
    mock_page.raster_object_key = "pages/10.png"
    mock_page.ingestion_status = "segmented"

    mock_article = MagicMock()
    mock_article.id = 1
    mock_article.headline = "MARKET SURGE"
    mock_article.section = "Business"
    mock_article.article_type = "news"
    mock_article.prominence_score = 0.95
    mock_article.word_count = 150
    mock_article.summary = "Summary"
    mock_article.full_text = "Full text"
    mock_article.article_pages = [mock_page]

    mock_issue = MagicMock()
    mock_issue.id = 5
    mock_issue.newspaper_id = 1
    mock_issue.newspaper = mock_np
    mock_issue.issue_date = "2026-08-21"
    mock_issue.edition = "morning"
    mock_issue.language = "en"
    mock_issue.ingestion_status = "completed"
    mock_issue.pages = [mock_page]
    mock_issue.articles = [mock_article]

    mock_issue_res.scalar_one_or_none.return_value = mock_issue

    # Total chunks query
    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 1

    # Chunks select query
    mock_chunks_res = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.id = 101
    mock_chunk.article_id = 1
    mock_chunk.chunk_index = 0
    mock_chunk.text = "Chunk text"
    mock_chunk.token_count = 10
    mock_chunk.embedding_vector_id = "vec-123"
    mock_chunks_res.all.return_value = [(mock_chunk, "MARKET SURGE")]

    mock_db.execute = AsyncMock(side_effect=[mock_issue_res, mock_count_res, mock_chunks_res])

    async def override_get_db() -> AsyncGenerator[MagicMock, None]:
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/issues/5/inspection?chunk_limit=50&chunk_offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["issue"]["id"] == 5
        assert len(data["pages"]) == 1
        assert data["pages"][0]["ocr_fallback_triggered"] is True
        assert len(data["articles"]) == 1
        assert len(data["chunks"]) == 1
        assert data["pagination"]["total"] == 1
