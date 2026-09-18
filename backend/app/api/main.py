"""FastAPI application factory for NewsLens-AI.

Manages the full application lifecycle:
- Startup: logging, DB, MinIO, Qdrant
- Shutdown: clean connection teardown
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.base import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: run startup tasks, yield, then shutdown."""
    settings = get_settings()
    setup_logging(settings.app_log_level)
    logger = get_logger(__name__)

    logger.info("Starting NewsLens-AI", extra={"version": "0.1.0", "debug": settings.app_debug})

    # Initialize async DB engine
    init_db(settings.database.async_url)
    logger.info("Database engine initialized")

    # Initialize Object Store (MinIO or GCS, create buckets if missing)
    try:
        from app.storage import get_object_store

        store = get_object_store(settings)
        await store.startup()
        app.state.minio = store
    except Exception as e:
        logger.warning("Object store startup failed (continuing)", extra={"error": str(e)})
        app.state.minio = None

    # Initialize Qdrant (create collection if missing)
    try:
        from app.storage.qdrant_store import QdrantStore

        qdrant = QdrantStore(settings.qdrant)
        await qdrant._ensure_collection()
        app.state.qdrant = qdrant
        logger.info("Qdrant collection ready")
    except Exception as e:
        logger.warning("Qdrant startup failed (continuing)", extra={"error": str(e)})
        app.state.qdrant = None

    # Pre-warm embedding & neural reranker in background to absorb cold-start latency
    async def _prewarm_models() -> None:
        try:
            from app.providers.registry import get_registry
            from app.retrieval.reranker import CrossEncoderReranker

            reg = get_registry()
            embed_provider = reg.get_provider("embedding")
            if embed_provider and hasattr(embed_provider, "embed_one"):
                logger.info("Pre-warming embedding model in background...")
                await embed_provider.embed_one("NewsLens AI initialization query")
                logger.info("Embedding model pre-warmed successfully")

            reranker = CrossEncoderReranker()
            await reranker._get_model()
            logger.info("Reranker model pre-warmed successfully")
        except Exception as prewarm_err:
            logger.warning("Model pre-warming non-critical warning", extra={"error": str(prewarm_err)})

    if not settings.testing and os.environ.get("ENABLE_MODEL_PREWARM", "false").lower() in ("1", "true"):
        asyncio.create_task(_prewarm_models())

    logger.info("NewsLens-AI startup complete")
    yield

    # Shutdown
    logger.info("Shutting down NewsLens-AI")
    await close_db()
    if app.state.qdrant:
        await app.state.qdrant.close()
    logger.info("NewsLens-AI shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="NewsLens-AI",
        version="0.1.0",
        description="Newspaper Intelligence Agentic RAG System",
        lifespan=lifespan,
        docs_url="/docs" if settings.app_debug else None,
        redoc_url="/redoc" if settings.app_debug else None,
    )

    # CORS
    cors_kwargs: dict[str, Any] = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    if "*" in settings.cors_origins:
        cors_kwargs["allow_origins"] = ["*"]
        cors_kwargs["allow_credentials"] = False
    else:
        cors_kwargs["allow_origins"] = settings.cors_origins
        cors_kwargs["allow_credentials"] = True
    app.add_middleware(CORSMiddleware, **cors_kwargs)

    # Prometheus Metrics Middleware
    from app.core.metrics import PrometheusMiddleware, generate_prometheus_metrics

    app.add_middleware(PrometheusMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Prometheus metrics exposition endpoint."""
        metrics_data, content_type = generate_prometheus_metrics()
        return Response(content=metrics_data, media_type=content_type)

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Any) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger = get_logger("newslens.exception")
        logger.exception(
            "Unhandled exception",
            extra={
                "path": str(request.url),
                "method": request.method,
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    # Register routers
    from app.api.routers.articles import router as articles_router
    from app.api.routers.health import router as health_router
    from app.api.routers.ingest import router as ingest_router
    from app.api.routers.metadata import router as metadata_router
    from app.api.routers.models import router as models_router
    from app.api.routers.newspapers import router as newspapers_router
    from app.api.routers.query import router as query_router
    from app.api.routers.settings import router as settings_router

    app.include_router(health_router)
    app.include_router(models_router, prefix="/api")
    app.include_router(ingest_router)
    app.include_router(articles_router)
    app.include_router(metadata_router)
    app.include_router(query_router)
    app.include_router(newspapers_router)
    app.include_router(settings_router)

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
