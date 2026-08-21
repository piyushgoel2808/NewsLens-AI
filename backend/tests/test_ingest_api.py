"""API endpoint tests for the Ingestion router."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.ingestion.intake import IntakeResult
from app.models.base import get_db

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_db_session() -> AsyncMock:
    session = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_upload_endpoint(mock_db_session: AsyncMock) -> None:
    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncMock, None]:
        yield mock_db_session

    app.dependency_overrides[get_db] = override_get_db

    pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
    pdf_bytes = pdf_path.read_bytes()

    with (
        patch("app.api.main.init_db"),
        patch("app.api.main.close_db", new_callable=AsyncMock),
        patch("app.storage.minio_store.MinioStore.startup", new_callable=AsyncMock),
        patch("app.storage.qdrant_store.QdrantStore._ensure_collection", new_callable=AsyncMock),
        patch(
            "app.ingestion.intake.IntakeService.process_upload", new_callable=AsyncMock
        ) as mock_process,
        patch(
            "app.api.routers.ingest.run_ingestion_pipeline", new_callable=AsyncMock
        ) as mock_pipeline,
    ):
        mock_process.return_value = IntakeResult(
            job_id=42,
            total_files=1,
            issues_created=[101],
            skipped_duplicates=[],
        )
        mock_pipeline.return_value = {
            "issue_id": 101,
            "total_pages": 1,
            "pages": [],
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            files = {"file": ("frontpage.pdf", pdf_bytes, "application/pdf")}
            data = {
                "newspaper_name": "The Daily Chronicle",
                "issue_date": "1929-10-24",
                "edition": "morning",
                "language": "en",
            }
            resp = await client.post("/api/ingest/upload", files=files, data=data)

            assert resp.status_code == 201
            body = resp.json()
            assert body["job_id"] == 42
            assert body["issues_created"] == [101]


@pytest.mark.asyncio
async def test_get_ingestion_job_not_found(mock_db_session: AsyncMock) -> None:
    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncMock, None]:
        yield mock_db_session

    app.dependency_overrides[get_db] = override_get_db

    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = mock_res

    with (
        patch("app.api.main.init_db"),
        patch("app.api.main.close_db", new_callable=AsyncMock),
        patch("app.storage.minio_store.MinioStore.startup", new_callable=AsyncMock),
        patch("app.storage.qdrant_store.QdrantStore._ensure_collection", new_callable=AsyncMock),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/ingest/jobs/999")
            assert resp.status_code == 404
