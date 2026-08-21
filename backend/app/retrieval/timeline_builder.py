"""Timeline Builder: Chronological event trajectory and milestone aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import asc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.article import Article
from app.models.newspaper import Issue

logger = get_logger(__name__)


@dataclass
class TimelineMilestone:
    """A specific event or news report on a particular date."""

    article_id: int
    headline: str
    byline_author: str | None
    section: str | None
    summary: str
    prominence_score: float
    pages: list[int]


@dataclass
class TimelineDateGroup:
    """All news coverage and milestones on a single calendar date."""

    date: str  # YYYY-MM-DD
    newspaper_name: str
    articles_count: int
    milestones: list[TimelineMilestone] = field(default_factory=list)


@dataclass
class TimelineResult:
    """Full chronological trajectory across the requested period."""

    query: str
    total_dates: int
    total_articles: int
    date_groups: list[TimelineDateGroup] = field(default_factory=list)


class TimelineBuilder:
    """Builds chronological news coverage timelines."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def build_timeline(
        self,
        query: str | None = None,
        newspaper_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> TimelineResult:
        """Construct a structured chronological timeline of articles."""
        async with self._session_factory() as db:
            stmt = (
                select(Article)
                .join(Issue, Article.issue_id == Issue.id)
                .options(
                    selectinload(Article.issue).selectinload(Issue.newspaper),
                    selectinload(Article.article_pages),
                )
            )

            if newspaper_id:
                stmt = stmt.where(Issue.newspaper_id == newspaper_id)
            if date_from:
                stmt = stmt.where(Issue.issue_date >= date_from)
            if date_to:
                stmt = stmt.where(Issue.issue_date <= date_to)
            if query:
                stmt = stmt.where(
                    (Article.headline.ilike(f"%{query}%"))
                    | (Article.summary.ilike(f"%{query}%"))
                    | (Article.full_text.ilike(f"%{query}%"))
                )

            stmt = stmt.order_by(asc(Issue.issue_date), asc(Article.primary_page_id)).limit(limit)
            res = await db.execute(stmt)
            articles = res.scalars().all()

            # Group articles by (date, newspaper)
            grouped: dict[tuple[str, str], list[TimelineMilestone]] = {}
            for art in articles:
                issue_date = str(art.issue.issue_date) if art.issue else "Unknown Date"
                np_name = (
                    art.issue.newspaper.name
                    if art.issue and art.issue.newspaper
                    else "Daily News"
                )
                key = (issue_date, np_name)
                if key not in grouped:
                    grouped[key] = []

                pages_list = (
                    sorted({ap.page_number for ap in art.article_pages})
                    if art.article_pages
                    else []
                )
                summary = art.summary or (art.full_text[:250] if art.full_text else "")

                grouped[key].append(
                    TimelineMilestone(
                        article_id=art.id,
                        headline=art.headline or "Untitled",
                        byline_author=art.byline_author,
                        section=art.section,
                        summary=summary,
                        prominence_score=art.prominence_score,
                        pages=pages_list,
                    )
                )

            date_groups: list[TimelineDateGroup] = []
            for (dt, np_name), milestones in grouped.items():
                date_groups.append(
                    TimelineDateGroup(
                        date=dt,
                        newspaper_name=np_name,
                        articles_count=len(milestones),
                        milestones=milestones,
                    )
                )

            logger.info(
                "Timeline built",
                extra={
                    "query": query,
                    "dates_count": len(date_groups),
                    "articles_count": len(articles),
                },
            )

            return TimelineResult(
                query=query or "All Coverage",
                total_dates=len(date_groups),
                total_articles=len(articles),
                date_groups=date_groups,
            )
