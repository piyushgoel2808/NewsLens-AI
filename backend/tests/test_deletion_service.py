"""Unit tests for the 3-Tier Hard Deletion Blueprint and REST API endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.ingestion.deletion_service import DeletionService
from app.models.base import get_db


@dataclass
class _MockPage:
    id: int
    issue_id: int
    page_number: int
    raster_object_key: str | None = None


@dataclass
class _MockIssue:
    id: int
    newspaper_id: int
    source_zip_id: int | None = None
    ingestion_status: str = "completed"


class TestDeletionService:
    """Tests for DeletionService 3-tier deletion orchestrator."""

    @pytest.mark.asyncio
    async def test_delete_issue_3_tiers(self) -> None:
        """Verify Qdrant, MinIO, and MySQL purge sequence."""
        mock_db = MagicMock()
        mock_issue = _MockIssue(id=10, newspaper_id=1, source_zip_id=42)
        mock_pages = [
            _MockPage(
                id=101,
                issue_id=10,
                page_number=1,
                raster_object_key="issues/10/pages/p1.png",
            ),
            _MockPage(
                id=102,
                issue_id=10,
                page_number=2,
                raster_object_key="issues/10/pages/p2.png",
            ),
        ]

        mock_db.get = AsyncMock(return_value=mock_issue)
        mock_pages_res = MagicMock()
        mock_pages_res.scalars.return_value.all.return_value = mock_pages
        mock_db.execute = AsyncMock(return_value=mock_pages_res)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()

        mock_qdrant = MagicMock()
        mock_qdrant.delete_by_filter = AsyncMock()

        mock_minio = MagicMock()
        mock_minio.delete = AsyncMock()
        mock_minio.delete_prefix = AsyncMock(return_value=2)

        service = DeletionService(db=mock_db, minio=mock_minio, qdrant=mock_qdrant)
        result = await service.delete_issue(10)

        assert result["status"] == "deleted"
        assert result["issue_id"] == 10

        # Tier 1: Qdrant
        mock_qdrant.delete_by_filter.assert_awaited_once_with({"issue_id": 10})

        # Tier 2: MinIO
        assert mock_minio.delete_prefix.await_count >= 2
        assert mock_minio.delete.await_count == 2

        # Tier 3: MySQL
        assert mock_db.delete.await_count >= 1
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_issue_returns_not_found(self) -> None:
        """Verify handling for missing issue."""
        mock_db = MagicMock()
        mock_db.get = AsyncMock(return_value=None)
        service = DeletionService(db=mock_db, minio=MagicMock(), qdrant=MagicMock())
        result = await service.delete_issue(999999)
        assert result["status"] == "not_found"


class TestDeleteAPIEndpoints:
    """Tests for DELETE /api/issues/{id} and DELETE /api/ingest/jobs/{id}."""

    @pytest.mark.asyncio
    async def test_delete_issue_endpoint_success(self) -> None:
        """Verify 200 response on successful issue deletion."""
        app = create_app()
        mock_db = MagicMock()
        mock_issue = _MockIssue(id=10, newspaper_id=1)
        mock_db.get = AsyncMock(return_value=mock_issue)
        mock_pages_res = MagicMock()
        mock_pages_res.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_pages_res)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()

        async def override_get_db() -> AsyncGenerator[Any, None]:
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/api/issues/10")
            assert resp.status_code == 200
            json_body = resp.json()
            assert json_body["status"] == "deleted"
            assert json_body["issue_id"] == 10

    @pytest.mark.asyncio
    async def test_delete_issue_endpoint_not_found(self) -> None:
        """Verify 404 response on missing issue."""
        app = create_app()
        mock_db = MagicMock()
        mock_db.get = AsyncMock(return_value=None)

        async def override_get_db() -> AsyncGenerator[Any, None]:
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/api/issues/999999")
            assert resp.status_code == 404
            assert "not found" in resp.json()["detail"].lower()
