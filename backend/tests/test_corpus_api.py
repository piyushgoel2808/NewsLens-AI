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

    mock_res.scalars.return_value.all.return_value = [
        _MockIssue(
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
    ]
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
