"""SQL Analytics Engine: Quantitative trends, mention frequencies, and distribution metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.article import Article, ArticlePage
from app.models.entity import ArticleEntity, ArticleTopic, Entity
from app.models.newspaper import Issue

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "category_aliases.yaml"
_USER_QUERY_SYNONYMS: dict[str, str] = {}
_CATEGORY_KEYWORDS: dict[str, list[str]] = {}
_CANONICAL_CATEGORIES: list[str] = []

if _CONFIG_PATH.exists():
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            _USER_QUERY_SYNONYMS = {
                k.lower().strip(): v for k, v in cfg.get("user_query_synonyms", {}).items()
            }
            _CATEGORY_KEYWORDS = cfg.get("category_keywords", {})
            _CANONICAL_CATEGORIES = cfg.get("canonical_categories", [])
    except Exception as e:
        logger.warning("Failed to load category_aliases.yaml in sql_analytics", extra={"error": str(e)})



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
        query: str | None = None,
    ) -> Any:
        """Return a structured manifest of all articles in an issue with section/type breakdowns."""
        from app.models.newspaper import Newspaper

        # Defensive fallback: if query is provided and explicit parameters are missing, extract them
        if query and not (issue_id or newspaper_name or issue_date):
            from app.agent.planner import extract_parameters_from_query
            extracted = extract_parameters_from_query(query)
            if not issue_id and extracted.get("issue_id"):
                issue_id = extracted["issue_id"]
            if not newspaper_name and extracted.get("newspaper_name"):
                newspaper_name = extracted["newspaper_name"]
            if not issue_date and extracted.get("issue_date"):
                issue_date = extracted["issue_date"]
            if not category_filter and extracted.get("category_filter"):
                category_filter = extracted["category_filter"]

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
                # If issue found, but newspaper_name was also specified and does not match:
                if issue and newspaper_name and issue.newspaper:
                    if newspaper_name.lower() not in issue.newspaper.name.lower() and issue.newspaper.name.lower() not in newspaper_name.lower():
                        issue = None

            # Fallback 1: Resolve by (newspaper_name, issue_date)
            if not issue and newspaper_name and issue_date:
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

            # Fallback 2: Resolve by newspaper_name (latest issue)
            if not issue and newspaper_name:
                stmt = (
                    select(Issue)
                    .join(Newspaper)
                    .where(Newspaper.name.ilike(f"%{newspaper_name}%"))
                    .order_by(desc(Issue.issue_date), desc(Issue.id))
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)

            # Fallback 3: Resolve by issue_date (latest issue on that date)
            if not issue and issue_date:
                stmt = (
                    select(Issue)
                    .where(Issue.issue_date == issue_date)
                    .order_by(desc(Issue.id))
                    .options(
                        selectinload(Issue.newspaper),
                        selectinload(Issue.pages),
                    )
                )
                res = await db.execute(stmt)
                issue = res.scalars().first() if hasattr(res, "scalars") else (res.scalar_one_or_none() if hasattr(res, "scalar_one_or_none") else None)

            # Fallback 4: Resolve latest overall issue if no filters given
            if not issue and not issue_id and not newspaper_name and not issue_date:
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
                if issue_id and not newspaper_name and not issue_date:
                    return {
                        "error": f"Issue #{issue_id} was not found in the newspaper archive.",
                        "articles": [],
                    }
                elif newspaper_name and issue_date:
                    return {
                        "error": f"No issue found for '{newspaper_name}' on date {issue_date}.",
                        "articles": [],
                    }
                elif newspaper_name:
                    return {
                        "error": f"No issues found in archive for newspaper '{newspaper_name}'.",
                        "articles": [],
                    }
                return {"error": "No newspaper issues found in archive.", "articles": []}

            art_stmt = (
                select(Article)
                .where(Article.issue_id == issue.id)
                .options(
                    selectinload(Article.category),
                    selectinload(Article.article_topics).selectinload(ArticleTopic.topic),
                )
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
                art_topics = [at.topic.name for at in (a.article_topics or []) if at.topic]

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
                        "subheadline": a.subheadline,
                        "section": sec,
                        "printed_section": a.printed_section,
                        "category": cat_name,
                        "topics": art_topics,
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
                    or sec_target in str(m.get("printed_section", "") or "").lower()
                ]
            elif category_filter:
                cat_raw = category_filter.strip().lower()
                resolved_cat = _USER_QUERY_SYNONYMS.get(cat_raw, cat_raw).lower()
                canon_key = next(
                    (c for c in _CANONICAL_CATEGORIES if c.lower() in (cat_raw, resolved_cat)),
                    None,
                )
                kw_cluster = _CATEGORY_KEYWORDS.get(canon_key, []) if canon_key else [cat_raw, resolved_cat]

                matched: list[dict[str, Any]] = []
                matched_ids: set[int] = set()

                for m in filtered_manifest:
                    m_id = m.get("id")
                    m_cat = str(m.get("category", "")).lower()
                    m_sec = str(m.get("section", "")).lower()
                    m_psec = str(m.get("printed_section", "") or "").lower()
                    m_topics = [t.lower() for t in m.get("topics", [])]
                    hl_sub = f"{m.get('headline', '')} {m.get('subheadline', '')}".lower()

                    structured_match = (
                        cat_raw in m_cat
                        or resolved_cat in m_cat
                        or cat_raw in m_sec
                        or resolved_cat in m_sec
                        or cat_raw in m_psec
                        or resolved_cat in m_psec
                        or any(cat_raw in t or resolved_cat in t for t in m_topics)
                    )
                    keyword_match = any(
                        re.search(r"\b" + re.escape(kw.lower()) + r"\b", hl_sub)
                        for kw in kw_cluster
                        if len(kw) >= 3
                    )

                    if (structured_match or keyword_match) and m_id not in matched_ids:
                        matched.append(m)
                        if m_id is not None:
                            matched_ids.add(m_id)

                filtered_manifest = matched

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
        query: str | None = None,
    ) -> Any:
        """Alias for list_issue_articles to maintain backward compatibility."""
        return await self.list_issue_articles(
            issue_id=issue_id,
            newspaper_name=newspaper_name,
            issue_date=issue_date,
            page_filter=page_filter,
            exclude_page_filter=exclude_page_filter,
            category_filter=category_filter,
            query=query,
        )

    async def get_newspaper_coverage_difference(
        self,
        source_newspaper: str,
        comparison_newspaper: str,
        issue_date: str | None = None,
    ) -> dict[str, Any]:
        """Compute verified differential coverage: articles in source_newspaper absent from comparison_newspaper."""
        source_summary = await self.list_issue_articles(
            newspaper_name=source_newspaper,
            issue_date=issue_date,
        )
        if not isinstance(source_summary, dict) or "error" in source_summary:
            err = source_summary.get("error", "Source issue not found") if isinstance(source_summary, dict) else "Source error"
            return {"error": f"Source publication '{source_newspaper}': {err}"}

        comp_summary = await self.list_issue_articles(
            newspaper_name=comparison_newspaper,
            issue_date=issue_date,
        )
        if not isinstance(comp_summary, dict) or "error" in comp_summary:
            err = comp_summary.get("error", "Comparison issue not found") if isinstance(comp_summary, dict) else "Comparison error"
            return {"error": f"Comparison publication '{comparison_newspaper}': {err}"}

        source_articles = source_summary.get("articles", [])
        comp_articles = comp_summary.get("articles", [])

        # Build token sets for comparison headlines
        comp_word_sets: list[set[str]] = []
        for ca in comp_articles:
            chl = (ca.get("headline") or "").strip().lower()
            c_words = set(w.strip("?:!.,\"'()[]{}<>-") for w in chl.split() if len(w) > 3)
            comp_word_sets.append(c_words)

        exclusive_articles: list[dict[str, Any]] = []
        shared_articles: list[dict[str, Any]] = []

        # Noise tokens to filter out
        noise_hls = {
            "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
            "in short >>", "in short", "news in brief", "panaji", "margao", "vasco", "mapusa",
        }

        for sa in source_articles:
            hl = (sa.get("headline") or "").strip()
            if len(hl) < 10 or hl.lower() in noise_hls:
                continue

            hl_words = set(w.strip("?:!.,\"'()[]{}<>-") for w in hl.lower().split() if len(w) > 3)
            if not hl_words:
                continue

            max_overlap = 0.0
            matched_comp_hl = None
            for idx, c_words in enumerate(comp_word_sets):
                if not c_words:
                    continue
                overlap = len(hl_words & c_words) / max(1, min(len(hl_words), len(c_words)))
                if overlap > max_overlap:
                    max_overlap = overlap
                    matched_comp_hl = comp_articles[idx].get("headline")

            if max_overlap >= 0.50:
                shared_articles.append({
                    "source_headline": hl,
                    "matched_headline": matched_comp_hl,
                    "overlap_score": round(max_overlap, 2),
                })
            else:
                exclusive_articles.append({
                    "id": sa.get("id"),
                    "headline": hl,
                    "page_number": sa.get("page_number", 1),
                    "printed_page": sa.get("printed_page", "1"),
                    "section": sa.get("section", "General"),
                    "category": sa.get("category", "General"),
                    "snippet": sa.get("summary") or sa.get("snippet", ""),
                })

        return {
            "source_newspaper": source_summary.get("newspaper", source_newspaper),
            "comparison_newspaper": comp_summary.get("newspaper", comparison_newspaper),
            "issue_date": str(source_summary.get("issue_date", issue_date)),
            "total_source_articles": len(source_articles),
            "total_comparison_articles": len(comp_articles),
            "exclusive_count": len(exclusive_articles),
            "shared_count": len(shared_articles),
            "exclusive_articles": exclusive_articles,
            "shared_articles": shared_articles[:10],
        }

