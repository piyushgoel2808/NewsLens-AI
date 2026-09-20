"""Declarative Broadsheet Schema & Softly-Decoupled Archive Context Provider.

Provides:
1. STATIC_BROADSHEET_SCHEMA: 100% declarative schema catalog of broadsheet MySQL tables,
   columns, and relationships. Requires zero database connections or network calls.
2. get_archive_and_schema_context: Asynchronously retrieves live archive date bounds and
   publication rosters, cached with a 5-minute TTL. Softly decoupled: if database access is
   unavailable, uninitialized, or errors, it immediately falls back to safe static archive defaults.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# ---------------------------------------------------------------------------
# 1. Declarative Broadsheet Database Schema Catalog (100% Static & In-Memory)
# ---------------------------------------------------------------------------

STATIC_BROADSHEET_SCHEMA = """### 🗄️ BROADSHEET RELATIONAL DATABASE SCHEMA

The archive is stored in a relational MySQL database with the following core tables:

1. `newspapers`:
   - Primary Columns: `id` (INT PK), `name` (VARCHAR, e.g. 'The Goan', 'Hindustan Times', 'Mint', 'Business Standard'), `publisher`, `country`, `default_language`.
   - Purpose: Publication brand catalog.

2. `issues`:
   - Primary Columns: `id` (INT PK), `newspaper_id` (INT FK -> newspapers.id), `issue_date` (DATE 'YYYY-MM-DD'), `edition`, `total_pages` (INT).
   - Purpose: Physical broadsheet print issues and editions.

3. `article_categories`:
   - Primary Columns: `id` (INT PK), `name` (VARCHAR, e.g. 'Business & Markets', 'Economy & Policy', 'Politics', 'Health', 'Sports', 'Entertainment', 'Technology', 'Science & Environment', 'Crime & Law', 'Opinion/Editorial', 'World/International', 'Lifestyle').
   - Purpose: Canonical newsroom taxonomy. Note: The `articles` table has `category_id`, NOT a raw category string.

4. `articles`:
   - Primary Columns: `id` (INT PK), `issue_id` (INT FK -> issues.id), `category_id` (INT FK -> article_categories.id), `headline` (VARCHAR), `subheadline`, `byline_author` (VARCHAR), `section` (VARCHAR), `printed_section`, `article_type` (ENUM: 'news', 'editorial', 'opinion', 'analysis', 'advertisement', 'sidebar', etc.), `word_count` (INT), `prominence_score` (FLOAT), `primary_page_id` (INT FK -> pages.id).
   - CRITICAL SCHEMA RULE: The `articles` table has NO `page_number` column. To filter or group articles by `page_number`, ALWAYS JOIN `article_pages` on `article_pages.article_id = articles.id`!
   - Purpose: Individual news stories, editorials, and commercial notices. Full text embeddings are indexed in the vector store for `hybrid_search`.

5. `article_pages`:
   - Primary Columns: `id` (INT PK), `article_id` (INT FK -> articles.id), `page_id` (INT FK -> pages.id), `page_number` (INT), `block_order` (INT).
   - Purpose: Maps articles to pages. Crucial: ALWAYS join `article_pages` to filter articles by page number (e.g. `JOIN article_pages ap ON a.id = ap.article_id WHERE ap.page_number = 6`).

6. `pages`:
   - Primary Columns: `id` (INT PK), `issue_id` (INT FK -> issues.id), `page_number` (INT), `is_advertisement_page` (BOOLEAN).
   - Purpose: Broadsheet newspaper pages.

7. `photos`:
   - Primary Columns: `id` (INT PK), `article_id` (INT FK -> articles.id), `issue_id` (INT FK -> issues.id), `page_number` (INT), `visual_type` (VARCHAR: 'photo', 'infographic', 'chart', 'map'), `caption` (TEXT).
   - Purpose: Visual assets and infographics linked to articles and pages."""

ARCHIVE_SCHEMA: dict[str, frozenset[str]] = {
    "newspapers": frozenset({"id", "name"}),
    "issues": frozenset({"id", "newspaper_id", "issue_date", "total_pages"}),
    "articles": frozenset({
        "id",
        "issue_id",
        "headline",
        "subheadline",
        "byline_author",
        "section",
        "article_type",
        "prominence_score",
        "word_count",
        "summary",
        "full_text",
        "category_id",
        "primary_page_id",
    }),
    "article_pages": frozenset({
        "id",
        "article_id",
        "page_id",
        "page_number",
        "block_order",
    }),
    "article_categories": frozenset({"id", "name"}),
    "pages": frozenset({"id", "issue_id", "page_number", "is_advertisement_page"}),
    "photos": frozenset({"id", "article_id", "page_id", "caption", "visual_type"}),
    "entities": frozenset({"id", "name", "type"}),
    "article_entities": frozenset({
        "article_id",
        "entity_id",
        "mention_count",
        "salience_score",
    }),
}

KNOWN_COLUMN_HALLUCINATIONS: dict[str, str] = {
    "page_number": "article_pages.page_number (join `article_pages` on `articles.id = article_pages.article_id` where `article_pages.page_number = :page_num`)",
    "page": "article_pages.page_number (join `article_pages` on `articles.id = article_pages.article_id`)",
    "pages": "article_pages.page_number (join `article_pages` on `articles.id = article_pages.article_id`)",
    "published_at": "issues.issue_date (join `issues` on `articles.issue_id = issues.id`)",
    "publish_date": "issues.issue_date (join `issues` on `articles.issue_id = issues.id`)",
    "publication_date": "issues.issue_date (join `issues` on `articles.issue_id = issues.id`)",
    "date": "issues.issue_date (join `issues` on `articles.issue_id = issues.id`)",
    "article_date": "issues.issue_date (join `issues` on `articles.issue_id = issues.id`)",
    "section_name": "articles.section",
    "category": "article_categories.name (join `article_categories` on `articles.category_id = article_categories.id`)",
    "category_name": "article_categories.name (join `article_categories` on `articles.category_id = article_categories.id`)",
    "newspaper": "newspapers.name (join `newspapers` on `issues.newspaper_id = newspapers.id`)",
    "newspaper_name": "newspapers.name (join `newspapers` on `issues.newspaper_id = newspapers.id`)",
    "source": "newspapers.name (join `newspapers` on `issues.newspaper_id = newspapers.id`)",
    "edition": "newspapers.name or issues.issue_date",
    "city": "newspapers.name",
    "sentiment": "prominence_score or custom calculation",
    "is_ad": "articles.article_type = 'advertisement' or pages.is_advertisement_page",
    "ad": "articles.article_type = 'advertisement'",
}


# ---------------------------------------------------------------------------
# 2. Static Archive Defaults (Fallback for Offline CI/CD and DB Disconnections)
# ---------------------------------------------------------------------------

STATIC_CANONICAL_PUBLICATIONS = [
    "Business Standard",
    "Hindustan Times",
    "Mint",
    "THE ECONOMIC TIMES",
    "The Goan",
    "The Guardian",
    "The Hindu",
    "The Indian Express",
    "The Morning Standard",
    "THE NEW YORK TIMES",
]

STATIC_ARCHIVE_DATE_MIN = "2026-08-01"
STATIC_ARCHIVE_DATE_MAX = "2026-09-11"

STATIC_CANONICAL_CATEGORIES = [
    "Business & Markets",
    "Crime & Law",
    "Economy & Policy",
    "Entertainment",
    "Health",
    "Lifestyle",
    "Opinion/Editorial",
    "Politics",
    "Science & Environment",
    "Sports",
    "Technology",
    "World/International",
]


@dataclass(frozen=True)
class ArchiveMetadata:
    """Structured container for dynamic archive coverage bounds and schema context."""

    min_date: str
    max_date: str
    publications: list[str]
    categories: list[str]
    context_str: str
    total_issues: int = 0
    publication_roster: tuple[tuple[str, int, str, str], ...] = ()


def get_fallback_archive_metadata() -> ArchiveMetadata:
    """Return safe static archive metadata when database is offline or uninitialized."""
    context_lines = [
        STATIC_BROADSHEET_SCHEMA,
        "",
        "### 📅 ACTIVE ARCHIVE COVERAGE & BOUNDARIES",
        f"- Verified Date Range: {STATIC_ARCHIVE_DATE_MIN} to {STATIC_ARCHIVE_DATE_MAX}",
        f"- Available Broadsheet Publications ({len(STATIC_CANONICAL_PUBLICATIONS)}): {', '.join(STATIC_CANONICAL_PUBLICATIONS)}",
        f"- Active Taxonomy Categories: {', '.join(STATIC_CANONICAL_CATEGORIES)}",
    ]
    return ArchiveMetadata(
        min_date=STATIC_ARCHIVE_DATE_MIN,
        max_date=STATIC_ARCHIVE_DATE_MAX,
        publications=list(STATIC_CANONICAL_PUBLICATIONS),
        categories=list(STATIC_CANONICAL_CATEGORIES),
        context_str="\n".join(context_lines),
    )


def get_known_publications() -> list[str]:
    """Return the list of active broadsheet publications (cached or static fallback)."""
    meta = _ARCHIVE_CACHE.get("metadata")
    if meta and isinstance(meta, ArchiveMetadata) and meta.publications:
        return list(meta.publications)
    return list(STATIC_CANONICAL_PUBLICATIONS)


# ---------------------------------------------------------------------------
# 3. Cached Live Archive Bounds Provider (Soft Decoupled)
# ---------------------------------------------------------------------------

_ARCHIVE_CACHE: dict[str, Any] = {
    "cached_at": 0.0,
    "context_str": "",
    "metadata": None,
}
_CACHE_TTL_SECONDS = 300.0  # 5 minutes


async def get_archive_metadata(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> ArchiveMetadata:
    """Return structured ArchiveMetadata introspected from MySQL, cached for 5 minutes.

    Guarantees:
    - Zero crashing: Never raises an exception even if the database is None or unreachable.
    - Zero lag: In-memory 5-minute cache avoids querying MySQL repeatedly during agent runs.
    - Soft decoupling: Immediately serves safe static defaults if database connection fails.
    """
    now = time.monotonic()
    cached_meta = _ARCHIVE_CACHE.get("metadata")
    if cached_meta and (now - _ARCHIVE_CACHE["cached_at"]) < _CACHE_TTL_SECONDS:
        return cached_meta

    min_date = STATIC_ARCHIVE_DATE_MIN
    max_date = STATIC_ARCHIVE_DATE_MAX
    publications = list(STATIC_CANONICAL_PUBLICATIONS)
    categories = list(STATIC_CANONICAL_CATEGORIES)
    total_issues = 0
    pub_roster: list[tuple[str, int, str, str]] = []

    if session_factory is not None:
        try:
            from sqlalchemy import func

            from app.models.article import ArticleCategory
            from app.models.newspaper import Issue, Newspaper

            async with session_factory() as db:
                # Query date range and total completed issues
                stmt = select(
                    func.min(Issue.issue_date),
                    func.max(Issue.issue_date),
                    func.count(Issue.id),
                ).where(Issue.ingestion_status.in_(("completed", "indexed", "ready")))
                res = await db.execute(stmt)
                row = res.one_or_none()
                if inspect.isawaitable(row):
                    row = await row
                d_min, d_max, total_cnt = row or (None, None, 0)
                if d_min:
                    min_date = str(d_min)
                if d_max:
                    max_date = str(d_max)
                if total_cnt:
                    total_issues = int(total_cnt)

                # Query publication-level grouped roster (name, issue count, min date, max date)
                roster_stmt = (
                    select(
                        Newspaper.name,
                        func.count(Issue.id),
                        func.min(Issue.issue_date),
                        func.max(Issue.issue_date),
                    )
                    .join(Issue, Issue.newspaper_id == Newspaper.id)
                    .where(Issue.ingestion_status.in_(("completed", "indexed", "ready")))
                    .group_by(Newspaper.id, Newspaper.name)
                    .order_by(Newspaper.name)
                )
                roster_res = await db.execute(roster_stmt)
                roster_rows = roster_res.all()
                if inspect.isawaitable(roster_rows):
                    roster_rows = await roster_rows

                db_pubs: list[str] = []
                for r in roster_rows:
                    p_name = str(r[0])
                    p_cnt = int(r[1])
                    p_min = str(r[2]) if r[2] else ""
                    p_max = str(r[3]) if r[3] else ""
                    pub_roster.append((p_name, p_cnt, p_min, p_max))
                    db_pubs.append(p_name)

                if db_pubs:
                    publications = db_pubs

                # Query canonical categories
                cat_stmt = select(ArticleCategory.name).order_by(ArticleCategory.name)
                cat_res = await db.execute(cat_stmt)
                cat_scalars = cat_res.scalars()
                if inspect.isawaitable(cat_scalars):
                    cat_scalars = await cat_scalars
                db_cats = list(cat_scalars.all() if hasattr(cat_scalars, "all") else [])
                if db_cats:
                    categories = db_cats
        except Exception:
            # Silently fall back to safe canonical defaults on any DB exception
            pass

    context_lines = [
        STATIC_BROADSHEET_SCHEMA,
        "",
        "### 📅 ACTIVE ARCHIVE COVERAGE & BOUNDARIES",
        f"- Total Issues in Archive: {total_issues or len(pub_roster)} across {len(publications)} publications ({min_date} to {max_date})",
        "- Publications & Verified Dates:",
    ]
    if pub_roster:
        for p_name, p_cnt, p_min, p_max in pub_roster:
            span_str = f"{p_min} to {p_max}" if p_min != p_max else p_min
            context_lines.append(f"  * {p_name}: {p_cnt} issue{'s' if p_cnt != 1 else ''} ({span_str})")
    else:
        context_lines.append(f"  * Available Publications: {', '.join(publications)}")
    context_lines.append(f"- Active Taxonomy Categories: {', '.join(categories)}")

    context_str = "\n".join(context_lines)
    meta = ArchiveMetadata(
        min_date=min_date,
        max_date=max_date,
        publications=publications,
        categories=categories,
        context_str=context_str,
        total_issues=total_issues or len(pub_roster),
        publication_roster=tuple(pub_roster),
    )
    _ARCHIVE_CACHE["cached_at"] = now
    _ARCHIVE_CACHE["context_str"] = context_str
    _ARCHIVE_CACHE["metadata"] = meta
    return meta


async def get_archive_and_schema_context(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> str:
    """Return a unified markdown context block containing the schema and archive bounds.

    Maintained for backward compatibility; delegates to get_archive_metadata.
    """
    meta = await get_archive_metadata(session_factory)
    return meta.context_str


__all__ = [
    "STATIC_BROADSHEET_SCHEMA",
    "ARCHIVE_SCHEMA",
    "KNOWN_COLUMN_HALLUCINATIONS",
    "STATIC_CANONICAL_PUBLICATIONS",
    "STATIC_ARCHIVE_DATE_MIN",
    "STATIC_ARCHIVE_DATE_MAX",
    "STATIC_CANONICAL_CATEGORIES",
    "ArchiveMetadata",
    "get_fallback_archive_metadata",
    "get_known_publications",
    "get_archive_metadata",
    "get_archive_and_schema_context",
]
