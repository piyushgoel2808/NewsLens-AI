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
            stmt = select(
                Article.section,
                Article.article_type,
                func.count(Article.id).label("count"),
                func.avg(Article.prominence_score).label("avg_prominence"),
            ).join(Issue, Article.issue_id == Issue.id)

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

    async def list_issue_articles(
        self,
        issue_id: int | None = None,
        newspaper_name: str | None = None,
        issue_date: str | None = None,
        section: str | None = None,
        page_filter: str | int | None = None,
        exclude_page_filter: str | int | None = None,
        category_filter: str | None = None,
        page_number: int | None = None,
    ) -> Any:
        """Return a structured manifest of all articles in an issue with section/type breakdowns."""
        from sqlalchemy.orm import selectinload
        async with self._session_factory() as db:
            issue: Issue | None = None
            if issue_id:
                stmt = (
                    select(Issue)
                    .where(Issue.id == issue_id)
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)
            elif newspaper_name and issue_date:
                from app.models.newspaper import Newspaper
                stmt = (
                    select(Issue)
                    .join(Newspaper)
                    .where(
                        Newspaper.name.ilike(f"%{newspaper_name}%"),
                        Issue.issue_date == issue_date,
                    )
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)
            elif newspaper_name:
                from app.models.newspaper import Newspaper
                stmt = (
                    select(Issue)
                    .join(Newspaper)
                    .where(Newspaper.name.ilike(f"%{newspaper_name}%"))
                    .order_by(desc(Issue.issue_date))
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)
            else:
                # Latest issue
                stmt = (
                    select(Issue)
                    .order_by(desc(Issue.id))
                    .limit(1)
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)

            if not issue:
                return {"error": "Issue not found", "articles": []}

            art_stmt = (
                select(Article)
                .where(Article.issue_id == issue.id)
                .options(selectinload(Article.category))
                .order_by(Article.primary_page_id, desc(Article.prominence_score))
            )
            art_res = await db.execute(art_stmt)
            articles = art_res.scalars().all()

            section_counts: dict[str, int] = {}
            type_counts: dict[str, int] = {}
            category_counts: dict[str, int] = {}
            manifest: list[dict[str, Any]] = []

            page_folio_map = {
                p.id: p.printed_page_number or str(p.page_number) for p in issue.pages
            }
            page_num_map = {p.id: p.page_number for p in issue.pages}

            for a in articles:
                sec = a.section or "General"
                section_counts[sec] = section_counts.get(sec, 0) + 1
                atype = a.article_type or "news"
                type_counts[atype] = type_counts.get(atype, 0) + 1
                cat_name = a.category.name if a.category else (a.section or "General")
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

                p_num = page_num_map.get(a.primary_page_id, 1) if a.primary_page_id else 1
                folio = (
                    page_folio_map.get(a.primary_page_id, str(p_num))
                    if a.primary_page_id
                    else str(p_num)
                )

                manifest.append(
                    {
                        "id": a.id,
                        "headline": a.headline,
                        "section": sec,
                        "category": cat_name,
                        "article_type": atype,
                        "byline_author": a.byline_author,
                        "page_number": p_num,
                        "printed_page": folio,
                        "word_count": a.word_count,
                        "prominence_score": a.prominence_score,
                    }
                )

            filtered_manifest = manifest

            # 1. Apply section or category filter if requested
            if section:
                sec_target = section.strip().lower()
                filtered_manifest = [
                    m for m in filtered_manifest
                    if sec_target in str(m.get("section", "")).lower()
                ]
            elif category_filter:
                cat_target = category_filter.strip().lower()
                filtered_manifest = [
                    m for m in filtered_manifest
                    if cat_target in str(m.get("category", "")).lower()
                    or cat_target in str(m.get("section", "")).lower()
                ]

            # 2. Apply positive page filter if requested
            if page_number is not None:
                filtered_manifest = [m for m in filtered_manifest if m.get("page_number") == page_number]
            elif page_filter is not None:
                p_raw = str(page_filter).strip().lower()
                p_target = p_raw.replace("page", "").replace("pg", "").strip()
                printed_matches = [
                    m
                    for m in filtered_manifest
                    if str(m["printed_page"]).lower() == p_target
                    or str(m["printed_page"]).lower() == f"page {p_target}"
                ]
                filtered_manifest = (
                    printed_matches
                    if printed_matches
                    else [m for m in filtered_manifest if str(m["page_number"]) == p_target]
                )

            # 3. Apply negative page exclusion filter (hard safety net)
            if exclude_page_filter is not None:
                excl_raw = str(exclude_page_filter).strip().lower()
                excl_target = excl_raw.replace("page", "").replace("pg", "").strip()
                filtered_manifest = [
                    m for m in filtered_manifest
                    if str(m["page_number"]) != excl_target
                    and str(m["printed_page"]).lower() != excl_target
                    and str(m["printed_page"]).lower() != f"page {excl_target}"
                ]

            if section is not None or page_number is not None:
                return filtered_manifest

            return {
                "issue_id": issue.id,
                "newspaper": issue.newspaper.name if issue.newspaper else "Newspaper",
                "issue_date": str(issue.issue_date),
                "page_filter": str(page_filter) if page_filter else None,
                "exclude_page_filter": str(exclude_page_filter) if exclude_page_filter else None,
                "category_filter": category_filter,
                "total_articles": len(filtered_manifest),
                "total_issue_articles": len(articles),
                "total_pages": len(issue.pages),
                "section_breakdown": section_counts,
                "type_breakdown": type_counts,
                "category_breakdown": category_counts,
                "articles": filtered_manifest,
            }

    async def count_articles(
        self,
        newspaper_name: str | None = None,
        issue_date: str | None = None,
        section: str | None = None,
        article_type: str | None = None,
    ) -> dict[str, Any]:
        """Return exact article count matching filters."""
        from app.models.newspaper import Newspaper

        async with self._session_factory() as db:
            stmt = select(func.count(Article.id)).join(Issue, Article.issue_id == Issue.id)
            if newspaper_name:
                stmt = stmt.join(Newspaper, Issue.newspaper_id == Newspaper.id).where(
                    Newspaper.name.ilike(f"%{newspaper_name}%")
                )
            if issue_date:
                stmt = stmt.where(Issue.issue_date == issue_date)
            if section:
                stmt = stmt.where(Article.section.ilike(f"%{section}%"))
            if article_type:
                stmt = stmt.where(Article.article_type == article_type)

            res = await db.execute(stmt)
            count = res.scalar() or 0
            return {
                "count": count,
                "filters": {
                    "newspaper_name": newspaper_name,
                    "issue_date": issue_date,
                    "section": section,
                    "article_type": article_type,
                },
            }

    async def get_issue_summary(
        self,
        newspaper_name: str | None = None,
        issue_date: str | None = None,
        issue_id: int | None = None,
        page_filter: str | int | None = None,
        exclude_page_filter: str | int | None = None,
        category_filter: str | None = None,
    ) -> Any:
        """Alias for list_issue_articles to maintain backward compatibility."""
        return await self.list_issue_articles(
            issue_id=issue_id,
            newspaper_name=newspaper_name,
            issue_date=issue_date,
            page_filter=page_filter,
            exclude_page_filter=exclude_page_filter,
            category_filter=category_filter,
        )
