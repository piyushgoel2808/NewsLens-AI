"""SQLAlchemy ORM models: Article, ArticlePage, ArticleChunk, Photo, Table."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.mysql import JSON, LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

_ARTICLE_TYPES = (
    "news",
    "editorial",
    "sidebar",
    "advertisement",
    "photo_caption",
    "table_content",
    "continuation",
    "unknown",
)


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issue_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("issues.id", ondelete="CASCADE"), nullable=False, index=True
    )
    primary_page_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("pages.id", ondelete="SET NULL"), index=True
    )
    headline: Mapped[str | None] = mapped_column(String(1024))
    subheadline: Mapped[str | None] = mapped_column(String(1024))
    byline_author: Mapped[str | None] = mapped_column(String(512))
    section: Mapped[str | None] = mapped_column(String(255), index=True)
    article_type: Mapped[str] = mapped_column(
        Enum(*_ARTICLE_TYPES, name="article_type_enum"),
        default="unknown",
        nullable=False,
    )
    language: Mapped[str | None] = mapped_column(String(10), index=True)
    # Computed from: page number, headline font size, column span, word count, presence of photo
    prominence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, index=True)
    word_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    full_text: Mapped[str | None] = mapped_column(LONGTEXT)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    issue: Mapped[Issue] = relationship("Issue", back_populates="articles")  # type: ignore[name-defined]
    article_pages: Mapped[list[ArticlePage]] = relationship(
        "ArticlePage", back_populates="article", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[ArticleChunk]] = relationship(
        "ArticleChunk", back_populates="article", cascade="all, delete-orphan"
    )
    photos: Mapped[list[Photo]] = relationship(
        "Photo", back_populates="article", cascade="all, delete-orphan"
    )
    tables: Mapped[list[ArticleTable]] = relationship(
        "ArticleTable", back_populates="article", cascade="all, delete-orphan"
    )
    article_entities: Mapped[list[ArticleEntity]] = relationship(  # type: ignore[name-defined]
        "ArticleEntity", back_populates="article", cascade="all, delete-orphan"
    )
    article_topics: Mapped[list[ArticleTopic]] = relationship(  # type: ignore[name-defined]
        "ArticleTopic", back_populates="article", cascade="all, delete-orphan"
    )
    article_events: Mapped[list[ArticleEvent]] = relationship(  # type: ignore[name-defined]
        "ArticleEvent", back_populates="article", cascade="all, delete-orphan"
    )


class ArticlePage(Base):
    """Junction: one article can span multiple pages."""

    __tablename__ = "article_pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # bboxes on this page for this article
    bbox_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    block_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    article: Mapped[Article] = relationship("Article", back_populates="article_pages")


class ArticleChunk(Base):
    """Sub-chunk of an article's full_text for embedding/retrieval."""

    __tablename__ = "article_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # UUID string pointing to the Qdrant point ID; MySQL never stores the raw vector
    embedding_vector_id: Mapped[str | None] = mapped_column(String(255))

    article: Mapped[Article] = relationship("Article", back_populates="chunks")


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("articles.id", ondelete="SET NULL")
    )
    page_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bbox_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    caption: Mapped[str | None] = mapped_column(Text)
    object_key: Mapped[str | None] = mapped_column(String(512))  # MinIO key

    article: Mapped[Article | None] = relationship("Article", back_populates="photos")


class ArticleTable(Base):
    """Extracted table from a page."""

    __tablename__ = "tables"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("articles.id", ondelete="SET NULL")
    )
    page_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bbox_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    extracted_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # Structured table data
    object_key: Mapped[str | None] = mapped_column(String(512))

    article: Mapped[Article | None] = relationship("Article", back_populates="tables")
