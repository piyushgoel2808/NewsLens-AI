"""Tests for Photo analysis endpoint: POST /api/photos/{photo_id}/analyze."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.ingestion.visual_extractor import VisualClassification, VisualExtractionResult
from app.models.article import Photo
from app.models.base import get_db


@pytest.mark.asyncio
async def test_analyze_photo_asset_404_for_nonexistent_photo() -> None:
    """Verify POST /api/photos/999999/analyze returns 404 when photo does not exist."""
    app = create_app()

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: None))

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_db] = _override_db
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/photos/999999/analyze")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_analyze_photo_asset_success() -> None:
    """Verify POST /api/photos/{id}/analyze runs VLM analysis and updates photo record."""
    app = create_app()

    mock_photo = Photo(
        id=6199,
        article_id=101,
        page_id=7,
        object_key="photos/7/photo_1.png",
        caption="Welder working on car chassis in Mexico",
        visual_type="photo",
        vlm_description=None,
    )

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: mock_photo))
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_db] = _override_db
    transport = ASGITransport(app=app)

    fake_image_bytes = b"fake_png_image_bytes"
    mock_classification = VisualClassification(visual_type="photo", contains_data=False, confidence=0.95)
    mock_extraction = VisualExtractionResult(
        summary="A mechanic is welding the chassis of a vintage car in an open garage.",
        key_metrics=["Subject: Mechanic welding chassis", "Vehicle: Vintage sedan"],
        confidence=0.95,
        visual_type="photo",
    )

    with (
        patch("app.storage.minio_store.MinioStore.get", new_callable=AsyncMock) as mock_minio_get,
        patch("app.ingestion.visual_extractor.VisualDataExtractor.process_image_crop", new_callable=AsyncMock) as mock_process,
    ):
        mock_minio_get.return_value = fake_image_bytes
        mock_process.return_value = (mock_classification, mock_extraction)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/photos/6199/analyze")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == 6199
            assert "mechanic is welding the chassis" in data["vlm_description"]
            assert "Key Visual Elements:" in data["vlm_description"]
            assert data["visual_type"] == "photo"
