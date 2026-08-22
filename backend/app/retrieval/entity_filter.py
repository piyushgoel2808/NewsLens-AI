"""Entity-Based Search Engine for structured entity and taxonomy queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.article import Article
from app.models.entity import ArticleEntity, ArticleTopic, Entity, Topic
from app.models.newspaper import Issue

logger = get_logger(__name__)


@dataclass
class EntitySearchResult:
    """An article result matched via entity and taxonomy filters."""

    article_id: int
    headline: str
    byline_author: str | None
    section: str | None
    article_type: str
    prominence_score: float
    entity_name: str
    entity_type: str
    mention_count: int
    salience_score: float
    newspaper_name: str
    issue_date: str
    pages: list[int]
    summary: str
    issue_id: int = 0
    bboxes: list[dict[str, Any]] = field(default_factory=list)


class EntitySearchEngine:
    """Retrieves articles based on structured entity mentions and topic taxonomy."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def search_by_entity(
        self,
        entity_name: str | None = None,
        entity_type: str | None = None,
        topic_name: str | None = None,
        min_salience: float = 0.0,
        newspaper_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        top_k: int = 10,
    ) -> list[EntitySearchResult]:
        """Search articles containing specified entities or topic taxonomy."""
        async with self._session_factory() as db:
            stmt = (
                select(ArticleEntity)
                .join(Entity, ArticleEntity.entity_id == Entity.id)
                .join(Article, ArticleEntity.article_id == Article.id)
                .join(Issue, Article.issue_id == Issue.id)
                .options(
                    selectinload(ArticleEntity.entity),
                    selectinload(ArticleEntity.article)
                    .selectinload(Article.issue)
                    .selectinload(Issue.newspaper),
                    selectinload(ArticleEntity.article).selectinload(Article.article_pages),
                )
            )

            if entity_name:
                stmt = stmt.where(Entity.name.ilike(f"%{entity_name}%"))
            if entity_type:
                stmt = stmt.where(Entity.type == entity_type)
            if min_salience > 0.0:
                stmt = stmt.where(ArticleEntity.salience_score >= min_salience)
            if newspaper_id:
                stmt = stmt.where(Issue.newspaper_id == newspaper_id)
            if date_from:
                stmt = stmt.where(Issue.issue_date >= date_from)
            if date_to:
                stmt = stmt.where(Issue.issue_date <= date_to)

            if topic_name:
                stmt = (
                    stmt.join(ArticleTopic, ArticleTopic.article_id == Article.id)
                    .join(Topic, ArticleTopic.topic_id == Topic.id)
                    .where(Topic.name.ilike(f"%{topic_name}%"))
                )

            stmt = stmt.order_by(
                desc(ArticleEntity.salience_score),
                desc(ArticleEntity.mention_count),
            ).limit(top_k)

            res = await db.execute(stmt)
            records = res.scalars().all()

            results: list[EntitySearchResult] = []
            for ae in records:
                art = ae.article
                if not art:
                    continue

                np_name = (
                    art.issue.newspaper.name if art.issue and art.issue.newspaper else "Daily News"
                )
                issue_date = str(art.issue.issue_date) if art.issue else ""
                pages_list = (
                    sorted({ap.page_number for ap in art.article_pages})
                    if art.article_pages
                    else []
                )

                bboxes_list: list[dict[str, Any]] = []
                if art.article_pages:
                    for ap in art.article_pages:
                        if ap.bbox_json:
                            if isinstance(ap.bbox_json, list):
                                bboxes_list.extend(ap.bbox_json)
                            elif isinstance(ap.bbox_json, dict):
                                bboxes_list.append(ap.bbox_json)

                results.append(
                    EntitySearchResult(
                        article_id=art.id,
                        headline=art.headline or "Untitled",
                        byline_author=art.byline_author,
                        section=art.section,
                        article_type=art.article_type,
                        prominence_score=art.prominence_score,
                        entity_name=ae.entity.name if ae.entity else "",
                        entity_type=ae.entity.type if ae.entity else "misc",
                        mention_count=ae.mention_count,
                        salience_score=ae.salience_score,
                        newspaper_name=np_name,
                        issue_date=issue_date,
                        pages=pages_list,
                        summary=art.summary or (art.full_text[:250] if art.full_text else ""),
                        issue_id=art.issue_id,
                        bboxes=bboxes_list,
                    )
                )

            logger.info(
                "Entity search executed",
                extra={
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "hits": len(results),
                },
            )

            return results
