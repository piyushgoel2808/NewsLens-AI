"""SQLAlchemy ORM models: Entity, Topic, Event and their article join tables."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import Date, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

_ENTITY_TYPES = ("person", "org", "location", "misc")


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("name", "type", name="uq_entity_name_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    type: Mapped[str] = mapped_column(Enum(*_ENTITY_TYPES, name="entity_type_enum"), nullable=False)
    # Self-referential FK for entity merging/canonicalization
    canonical_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("entities.id", ondelete="SET NULL")
    )

    article_entities: Mapped[list[ArticleEntity]] = relationship(
        "ArticleEntity", back_populates="entity"
    )


class ArticleEntity(Base):
    __tablename__ = "article_entities"
    __mapper_args__ = {"confirm_deleted_rows": False}
    __table_args__ = (UniqueConstraint("article_id", "entity_id", name="uq_article_entity"),)

    article_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    entity_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("entities.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
        index=True,
    )
    mention_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    salience_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    article: Mapped[Any] = relationship(
        "Article", back_populates="article_entities"
    )
    entity: Mapped[Entity] = relationship("Entity", back_populates="article_entities")


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    taxonomy_path: Mapped[str | None] = mapped_column(
        String(1024)
    )  # e.g. "Politics > Elections > General"

    article_topics: Mapped[list[ArticleTopic]] = relationship(
        "ArticleTopic", back_populates="topic"
    )


class ArticleTopic(Base):
    __tablename__ = "article_topics"
    __mapper_args__ = {"confirm_deleted_rows": False}

    article_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    topic_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
        index=True,
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    article: Mapped[Any] = relationship(
        "Article", back_populates="article_topics"
    )
    topic: Mapped[Topic] = relationship("Topic", back_populates="article_topics")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(1024), nullable=False)
    canonical_date: Mapped[date | None] = mapped_column(Date)
    description: Mapped[str | None] = mapped_column(Text)
    # Self-referential FK for event clustering
    event_cluster_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="SET NULL")
    )

    article_events: Mapped[list[ArticleEvent]] = relationship(
        "ArticleEvent", back_populates="event"
    )


class ArticleEvent(Base):
    __tablename__ = "article_events"
    __mapper_args__ = {"confirm_deleted_rows": False}

    article_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("events.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
        index=True,
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    article: Mapped[Any] = relationship(
        "Article", back_populates="article_events"
    )
    event: Mapped[Event] = relationship("Event", back_populates="article_events")
