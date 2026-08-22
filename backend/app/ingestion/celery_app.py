"""Celery application configuration for NewsLens-AI asynchronous tasks."""

from __future__ import annotations

from celery import Celery

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
)
