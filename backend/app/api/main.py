"""FastAPI application factory for NewsLens-AI.

Manages the full application lifecycle:
- Startup: logging, DB, MinIO, Qdrant
- Shutdown: clean connection teardown
"""
from __future__ import annotations

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

    # Initialize MinIO (create buckets if missing)
    try:
        from app.storage.minio_store import MinioStore
        minio = MinioStore(settings.minio)
        await minio.startup()
        app.state.minio = minio
    except Exception as e:
        logger.warning("MinIO startup failed (continuing)", extra={"error": str(e)})
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
    origins = ["*"] if settings.app_debug else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    from app.api.routers.health import router as health_router
    from app.api.routers.ingest import router as ingest_router
    from app.api.routers.models import router as models_router

    app.include_router(health_router)
    app.include_router(models_router, prefix="/api")
    app.include_router(ingest_router)

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
