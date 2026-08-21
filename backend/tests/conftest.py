"""Shared pytest fixtures for NewsLens-AI backend tests."""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

# Set testing env vars before any app imports
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("MYSQL_HOST", "localhost")
os.environ.setdefault("MYSQL_DB", "newslens_test")
os.environ.setdefault("APP_DEBUG", "true")
os.environ.setdefault("MODEL_CONFIG_PATH", "../model_config.yaml")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_settings():
    """Return a Settings instance with test values."""
    from app.core.config import Settings

    return Settings(
        mysql_host="localhost",
        mysql_db="newslens_test",
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        testing=True,
    )


@pytest.fixture
async def app_client() -> AsyncGenerator[AsyncClient, None]:
    """Async test client for the FastAPI app."""
    from unittest.mock import patch

    # Patch startup dependencies so tests don't need real services
    with (
        patch("app.api.main.init_db"),
        patch("app.api.main.close_db", new_callable=AsyncMock),
        patch("app.storage.minio_store.MinioStore.startup", new_callable=AsyncMock),
        patch("app.storage.qdrant_store.QdrantStore._ensure_collection", new_callable=AsyncMock),
    ):
        from app.api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
