"""SQLAlchemy ORM model: IngestionJob."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, Integer, func
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

_SOURCE_TYPES = ("single_pdf", "multi_pdf", "folder", "zip")
_JOB_STATUSES = ("pending", "running", "completed", "failed", "partial")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(
        Enum(*_SOURCE_TYPES, name="source_type_enum"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum(*_JOB_STATUSES, name="job_status_enum"),
        default="pending",
        nullable=False,
    )
    total_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # List of {file, stage, error} dicts
    error_log: Mapped[list[Any] | None] = mapped_column(JSON)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
