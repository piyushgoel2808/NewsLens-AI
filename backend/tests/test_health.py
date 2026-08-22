"""Tests for the /health endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Test GET /health — always 200, status field reflects dependency health."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, app_client: AsyncClient) -> None:
        with (
            patch(
                "app.api.routers.health._check_mysql",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 5},
            ),
            patch(
                "app.api.routers.health._check_qdrant",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 3},
            ),
            patch(
                "app.api.routers.health._check_minio",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 4},
            ),
            patch(
                "app.api.routers.health._check_redis",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 1},
            ),
        ):
            response = await app_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_all_up_returns_healthy(self, app_client: AsyncClient) -> None:
        with (
            patch(
                "app.api.routers.health._check_mysql",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 5},
            ),
            patch(
                "app.api.routers.health._check_qdrant",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 3},
            ),
            patch(
                "app.api.routers.health._check_minio",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 4},
            ),
            patch(
                "app.api.routers.health._check_redis",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 1},
            ),
        ):
            response = await app_client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_one_down_returns_degraded(self, app_client: AsyncClient) -> None:
        with (
            patch(
                "app.api.routers.health._check_mysql",
                new_callable=AsyncMock,
                return_value={"status": "down", "error": "Connection refused"},
            ),
            patch(
                "app.api.routers.health._check_qdrant",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 3},
            ),
            patch(
                "app.api.routers.health._check_minio",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 4},
            ),
            patch(
                "app.api.routers.health._check_redis",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 1},
            ),
        ):
            response = await app_client.get("/health")
        assert response.status_code == 200  # Always 200
        data = response.json()
        assert data["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_response_has_version_field(self, app_client: AsyncClient) -> None:
        with (
            patch(
                "app.api.routers.health._check_mysql",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 5},
            ),
            patch(
                "app.api.routers.health._check_qdrant",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 3},
            ),
            patch(
                "app.api.routers.health._check_minio",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 4},
            ),
            patch(
                "app.api.routers.health._check_redis",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 1},
            ),
        ):
            response = await app_client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_response_has_dependencies_field(self, app_client: AsyncClient) -> None:
        with (
            patch(
                "app.api.routers.health._check_mysql",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 5},
            ),
            patch(
                "app.api.routers.health._check_qdrant",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 3},
            ),
            patch(
                "app.api.routers.health._check_minio",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 4},
            ),
            patch(
                "app.api.routers.health._check_redis",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 1},
            ),
        ):
            response = await app_client.get("/health")
        data = response.json()
        assert "dependencies" in data
        deps = data["dependencies"]
        assert "mysql" in deps
        assert "qdrant" in deps
        assert "minio" in deps
        assert "redis" in deps

    @pytest.mark.asyncio
    async def test_all_down_returns_degraded_not_500(self, app_client: AsyncClient) -> None:
        with (
            patch(
                "app.api.routers.health._check_mysql",
                new_callable=AsyncMock,
                return_value={"status": "down", "error": "refused"},
            ),
            patch(
                "app.api.routers.health._check_qdrant",
                new_callable=AsyncMock,
                return_value={"status": "down", "error": "refused"},
            ),
            patch(
                "app.api.routers.health._check_minio",
                new_callable=AsyncMock,
                return_value={"status": "down", "error": "refused"},
            ),
            patch(
                "app.api.routers.health._check_redis",
                new_callable=AsyncMock,
                return_value={"status": "down", "error": "refused"},
            ),
        ):
            response = await app_client.get("/health")
        assert response.status_code == 200  # Never 500 from health check
        assert response.json()["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_response_has_x_request_id_header(self, app_client: AsyncClient) -> None:
        with (
            patch(
                "app.api.routers.health._check_mysql",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 5},
            ),
            patch(
                "app.api.routers.health._check_qdrant",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 3},
            ),
            patch(
                "app.api.routers.health._check_minio",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 4},
            ),
            patch(
                "app.api.routers.health._check_redis",
                new_callable=AsyncMock,
                return_value={"status": "up", "latency_ms": 1},
            ),
        ):
            response = await app_client.get("/health")
        assert "x-request-id" in response.headers
