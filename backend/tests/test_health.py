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

    @pytest.mark.asyncio
    async def test_check_celery_worker_up(self) -> None:
        from app.api.routers.health import _check_celery

        with patch(
            "app.ingestion.celery_app.celery_app.control.ping",
            return_value=[{"celery@worker1": {"ok": "pong"}}],
        ):
            result = await _check_celery()
        assert result["status"] == "up"
        assert result["active_workers"] == 1
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_check_celery_worker_idle(self) -> None:
        from app.api.routers.health import _check_celery

        with patch(
            "app.ingestion.celery_app.celery_app.control.ping",
            return_value=[],
        ):
            result = await _check_celery()
        assert result["status"] == "idle"
        assert result["active_workers"] == 0

    @pytest.mark.asyncio
    async def test_check_celery_broken_pipe_handled_gracefully(self) -> None:
        from app.api.routers.health import _check_celery

        with (
            patch(
                "app.ingestion.celery_app.celery_app.control.ping",
                side_effect=BrokenPipeError("Broken pipe"),
            ),
            patch("app.ingestion.celery_app.celery_app.close") as mock_close,
        ):
            result = await _check_celery()
        assert result["status"] == "down"
        assert "Broken pipe" in result["error"]
        assert mock_close.called

    def test_celery_transport_hardening_options(self) -> None:
        from app.ingestion.celery_app import celery_app

        conf = celery_app.conf
        assert conf.broker_connection_retry_on_startup is True
        assert conf.broker_transport_options["socket_keepalive"] is True
        assert conf.broker_transport_options["health_check_interval"] == 15
        assert conf.broker_transport_options["retry_on_timeout"] is True
        assert conf.worker_prefetch_multiplier == 1

