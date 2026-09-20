"""Celery application configuration for NewsLens-AI asynchronous tasks."""

from __future__ import annotations

from celery import Celery  # type: ignore[import-untyped]

__all__ = ["celery_app"]

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "newslens_ai",
    broker=settings.redis.url,
    backend=settings.redis.url,
    include=["app.ingestion.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour limit for huge archives
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "socket_keepalive": True,
        "socket_timeout": 10.0,
        "socket_connect_timeout": 10.0,
        "health_check_interval": 15,
        "retry_on_timeout": True,
        "max_connections": 20,
    },
    result_backend_transport_options={
        "socket_keepalive": True,
        "socket_timeout": 10.0,
        "socket_connect_timeout": 10.0,
        "health_check_interval": 15,
        "retry_on_timeout": True,
    },
    worker_prefetch_multiplier=1,
    worker_cancel_long_running_tasks_on_connection_loss=True,
)

