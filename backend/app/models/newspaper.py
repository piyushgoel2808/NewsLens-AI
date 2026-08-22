"""SQLAlchemy ORM models: Newspaper, Issue, Page."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Newspaper(Base):
    __tablename__ = "newspapers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    publisher: Mapped[str | None] = mapped_column(String(255))
    default_language: Mapped[str | None] = mapped_column(String(10))
    country: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    issues: Mapped[list[Issue]] = relationship(
        "Issue", back_populates="newspaper", cascade="all, delete-orphan"
    )


class Issue(Base):
    __tablename__ = "issues"
    __mapper_args__ = {"confirm_deleted_rows": False}
    __table_args__ = (
        UniqueConstraint(
            "newspaper_id", "issue_date", "edition", name="uq_issue_newspaper_date_edition"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    newspaper_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("newspapers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    issue_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    edition: Mapped[str | None] = mapped_column(String(100))
    language: Mapped[str | None] = mapped_column(String(10))
    total_pages: Mapped[int | None] = mapped_column(Integer)
    source_zip_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("ingestion_jobs.id", ondelete="SET NULL")
    )
    ingestion_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    newspaper: Mapped[Newspaper] = relationship("Newspaper", back_populates="issues")
    pages: Mapped[list[Page]] = relationship(
        "Page", back_populates="issue", cascade="all, delete-orphan"
    )
    articles: Mapped[list[Any]] = relationship(
        "Article", back_populates="issue", cascade="all, delete-orphan"
    )


class Page(Base):
    __tablename__ = "pages"
    __mapper_args__ = {"confirm_deleted_rows": False}
    __table_args__ = (UniqueConstraint("issue_id", "page_number", name="uq_page_issue_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issue_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("issues.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raster_object_key: Mapped[str | None] = mapped_column(String(512))  # MinIO key
    width_px: Mapped[int | None] = mapped_column(Integer)
    height_px: Mapped[int | None] = mapped_column(Integer)
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    printed_page_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_advertisement_page: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Status state machine:
    # pending → rasterized → layout_done → ocr_done → segmented →
    # classified → metadata_done → embedded → indexed → failed
    ingestion_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)

    issue: Mapped[Issue] = relationship("Issue", back_populates="pages")
