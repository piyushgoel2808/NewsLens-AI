"""SQLAlchemy models for NewsLens-AI.

Import all models here so Alembic's env.py can detect them via Base.metadata.
"""

from app.models.article import Article, ArticleChunk, ArticlePage, ArticleTable, Photo  # noqa: F401
from app.models.base import Base  # noqa: F401
from app.models.entity import (  # noqa: F401
    ArticleEntity,
    ArticleEvent,
    ArticleTopic,
    Entity,
    Event,
    Topic,
)
from app.models.ingestion import IngestionJob  # noqa: F401
from app.models.newspaper import Issue, Newspaper, Page  # noqa: F401
from app.models.query import QueryLog  # noqa: F401

__all__ = [
    "Base",
    "Newspaper",
    "Issue",
    "Page",
    "Article",
    "ArticlePage",
    "ArticleChunk",
    "Photo",
    "ArticleTable",
    "Entity",
    "ArticleEntity",
    "Topic",
    "ArticleTopic",
    "Event",
    "ArticleEvent",
    "IngestionJob",
    "QueryLog",
]
