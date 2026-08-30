"""Unit tests for the PageReingestionService and single-page re-ingest API endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.ingestion.page_reingestion import PageReingestionService, check_is_advertisement_text
from app.models.base import get_db
from app.models.newspaper import Issue


def create_dummy_pdf_bytes() -> bytes:
    """Create a minimal 2-page PDF in memory."""
    doc = pymupdf.open()
    page1 = doc.new_page(width=600, height=800)
    page1.insert_text((50, 100), "Headline: Major Tech Breakthrough Announced Today\nBy John Doe\nEngineers unveiled a new architecture.", fontsize=14)
    page2 = doc.new_page(width=600, height=800)
    page2.insert_text((50, 100), "Markets rally on rate cut hopes.", fontsize=14)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPageReingestionService:
    """Tests for PageReingestionService orchestration logic."""

    def test_check_is_advertisement_text(self) -> None:
        """Verify commercial advertisement detection heuristics."""
        assert check_is_advertisement_text("Public Notice for e-tender 2026") is True
        assert check_is_advertisement_text("Terms & Conditions apply. Limited period offer.") is True
        assert check_is_advertisement_text("Smarter steels for people and planet") is True
        assert check_is_advertisement_text("RBI cuts repo rate by 25 basis points in monetary policy") is False

    @pytest.mark.asyncio
    async def test_reingest_page_raises_on_missing_issue(self) -> None:
        """Verify ValueError is raised if issue_id does not exist."""
        mock_db = MagicMock()
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_res)

        service = PageReingestionService(db=mock_db)
        with pytest.raises(ValueError, match="Issue 9999 not found"):
            await service.reingest_page(issue_id=9999, page_number=1)

    @pytest.mark.asyncio
    async def test_reingest_page_raises_on_missing_page(self) -> None:
        """Verify ValueError is raised if page_number does not exist for the issue."""
        mock_db = MagicMock()
        mock_issue = MagicMock(spec=Issue)
        mock_issue.id = 10
        mock_issue.newspaper = None
        mock_issue.issue_date = "2026-08-30"
        mock_issue.language = "en"
        mock_issue.source_zip_id = 42

        # First query returns issue, second returns None for page
        res_issue = MagicMock()
        res_issue.scalar_one_or_none.return_value = mock_issue

        res_page = MagicMock()
        res_page.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(side_effect=[res_issue, res_page])

        service = PageReingestionService(db=mock_db)
        with pytest.raises(ValueError, match="Page 99 not found for Issue 10"):
            await service.reingest_page(issue_id=10, page_number=99)


class TestPageReingestAPI:
    """Tests for the REST API endpoint POST /api/issues/{issue_id}/pages/{page_number}/reingest."""

    @pytest.mark.asyncio
    async def test_api_reingest_returns_404_on_invalid_issue(self) -> None:
        """Test API handles missing issue with HTTP 404."""
        app = create_app()
        mock_db = MagicMock()
        res = MagicMock()
        res.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=res)

        async def override_get_db():
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/issues/9999/pages/1/reingest")
            assert resp.status_code == 404
            assert "Issue 9999 not found" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_api_reingest_success_mock(self) -> None:
        """Test API successfully returns summary dictionary on successful re-ingestion."""
        app = create_app()

        async def override_get_db():
            yield MagicMock()

        app.dependency_overrides[get_db] = override_get_db

        dummy_result: dict[str, Any] = {
            "status": "success",
            "issue_id": 91,
            "page_number": 1,
            "printed_page_number": "1",
            "is_advertisement_page": False,
            "articles_count": 2,
            "photos_count": 1,
            "chunks_count": 3,
            "vectors_count": 3,
            "articles": [
                {
                    "id": 101,
                    "headline": "Bulk drug exporters fret as China tightens screws",
                    "article_type": "news",
                    "section": "Front Page",
                    "word_count": 19,
                    "prominence_score": 0.75,
                }
            ],
        }

        with patch("app.ingestion.page_reingestion.PageReingestionService.reingest_page", new=AsyncMock(return_value=dummy_result)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/issues/91/pages/1/reingest")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "success"
                assert data["page_number"] == 1
                assert data["articles_count"] == 2
                assert data["vectors_count"] == 3
