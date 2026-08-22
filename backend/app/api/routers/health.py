"""Health check endpoint for NewsLens-AI.

GET /health — pings all downstream dependencies and returns their status.
Always returns 200 (never 503) so load balancers don't drop the pod for
a degraded-but-running dependency. The 'status' field conveys the severity.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


async def _check_mysql() -> dict[str, Any]:
    settings = get_settings()
    t0 = time.monotonic()
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(settings.database.async_url, pool_pre_ping=True)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return {"status": "up", "latency_ms": round((time.monotonic() - t0) * 1000)}
    except Exception as e:
        logger.warning("MySQL health check failed", extra={"error": str(e)})
        return {"status": "down", "error": str(e)[:200]}


async def _check_qdrant() -> dict[str, Any]:
    settings = get_settings()
    t0 = time.monotonic()
    try:
        from app.storage.qdrant_store import QdrantStore

        store = QdrantStore(settings.qdrant)
        reachable = await store.ping()
        await store.close()
        if reachable:
            return {"status": "up", "latency_ms": round((time.monotonic() - t0) * 1000)}
        return {"status": "down", "error": "Qdrant ping failed"}
    except Exception as e:
        return {"status": "down", "error": str(e)[:200]}


async def _check_minio() -> dict[str, Any]:
    settings = get_settings()
    t0 = time.monotonic()
    try:
        from app.storage.minio_store import MinioStore

        store = MinioStore(settings.minio)
        reachable = await store.ping()
        if reachable:
            return {"status": "up", "latency_ms": round((time.monotonic() - t0) * 1000)}
        return {"status": "down", "error": "MinIO ping failed"}
    except Exception as e:
        return {"status": "down", "error": str(e)[:200]}


async def _check_redis() -> dict[str, Any]:
    settings = get_settings()
    t0 = time.monotonic()
    try:
        import redis.asyncio as aioredis

        r = aioredis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        return {"status": "up", "latency_ms": round((time.monotonic() - t0) * 1000)}
    except Exception as e:
        return {"status": "down", "error": str(e)[:200]}


@router.get("/health", summary="Health check", tags=["health"])
async def health_check() -> dict[str, Any]:
    """Check the health of all NewsLens-AI downstream dependencies.

    Returns:
        JSON object with overall status and per-dependency details.
        Status values: 'healthy' (all up) | 'degraded' (some down).
    """
    mysql_result, qdrant_result, minio_result, redis_result = await asyncio.gather(
        _check_mysql(),
        _check_qdrant(),
        _check_minio(),
        _check_redis(),
        return_exceptions=False,
    )
    deps = {
        "mysql": mysql_result,
        "qdrant": qdrant_result,
        "minio": minio_result,
        "redis": redis_result,
    }
    all_up = all(isinstance(d, dict) and d.get("status") == "up" for d in deps.values())
    return {
        "status": "healthy" if all_up else "degraded",
        "dependencies": deps,
        "version": "0.1.0",
    }
