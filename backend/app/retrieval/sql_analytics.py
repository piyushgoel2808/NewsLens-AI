"""SQL Analytics Engine: Quantitative trends, mention frequencies, and distribution metrics."""
from __future__ import annotations

from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.models.article import Article, ArticlePage
from app.models.entity import ArticleEntity, Entity
from app.models.newspaper import Issue

logger = get_logger(__name__)


class SQLAnalyticsEngine:
    """Executes safe, parameterized aggregation queries for quantitative news trends."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_entity_mention_trends(
        self,
        entity_name: str,
        group_by_period: str = "month",  # "day", "month", "year"
    ) -> list[dict[str, Any]]:
        """Compute mention frequency trends over time for a specific entity."""
        async with self._session_factory() as db:
            stmt = (
                select(
                    Issue.issue_date,
                    func.count(ArticleEntity.article_id).label("article_count"),
                    func.sum(ArticleEntity.mention_count).label("total_mentions"),
                    func.avg(ArticleEntity.salience_score).label("avg_salience"),
                )
                .join(Article, ArticleEntity.article_id == Article.id)
                .join(Issue, Article.issue_id == Issue.id)
                .join(Entity, ArticleEntity.entity_id == Entity.id)
                .where(Entity.name.ilike(f"%{entity_name}%"))
                .group_by(Issue.issue_date)
                .order_by(Issue.issue_date)
            )

            res = await db.execute(stmt)
            rows = res.all()

            return [
                {
                    "date": str(r.issue_date),
                    "article_count": r.article_count,
                    "total_mentions": int(r.total_mentions or 0),
                    "avg_salience": round(float(r.avg_salience or 0.0), 3),
                }
                for r in rows
            ]

    async def get_topic_distribution(
        self,
        newspaper_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compute article volume breakdown across topic categories."""
        async with self._session_factory() as db:
            stmt = (
                select(
                    Article.section,
                    Article.article_type,
                    func.count(Article.id).label("count"),
                    func.avg(Article.prominence_score).label("avg_prominence"),
                )
                .join(Issue, Article.issue_id == Issue.id)
            )

            if newspaper_id:
                stmt = stmt.where(Issue.newspaper_id == newspaper_id)

            stmt = stmt.group_by(Article.section, Article.article_type).order_by(desc("count"))

            res = await db.execute(stmt)
            rows = res.all()

            return [
                {
                    "section": r.section or "General",
                    "article_type": r.article_type,
                    "count": r.count,
                    "avg_prominence": round(float(r.avg_prominence or 0.0), 3),
                }
                for r in rows
            ]

    async def get_frontpage_prominence_ratio(
        self,
        newspaper_id: int | None = None,
    ) -> dict[str, Any]:
        """Compute percentage of articles appearing on Page 1 vs inside pages."""
        async with self._session_factory() as db:
            stmt = (
                select(
                    ArticlePage.page_number,
                    func.count(ArticlePage.article_id.distinct()).label("article_count"),
                )
                .join(Article, ArticlePage.article_id == Article.id)
                .join(Issue, Article.issue_id == Issue.id)
            )

            if newspaper_id:
                stmt = stmt.where(Issue.newspaper_id == newspaper_id)

            stmt = stmt.group_by(ArticlePage.page_number).order_by(ArticlePage.page_number)
            res = await db.execute(stmt)
            rows = res.all()

            page_distribution = {r.page_number: r.article_count for r in rows}
            total_articles = sum(page_distribution.values())
            page_1_count = page_distribution.get(1, 0)
            ratio = (page_1_count / total_articles) if total_articles > 0 else 0.0

            return {
                "total_articles": total_articles,
                "frontpage_articles": page_1_count,
                "frontpage_ratio": round(ratio, 4),
                "page_distribution": page_distribution,
            }
