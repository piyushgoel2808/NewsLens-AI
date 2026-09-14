"""Declarative Broadsheet Schema & Softly-Decoupled Archive Context Provider.

Provides:
1. STATIC_BROADSHEET_SCHEMA: 100% declarative schema catalog of broadsheet MySQL tables,
   columns, and relationships. Requires zero database connections or network calls.
2. get_archive_and_schema_context: Asynchronously retrieves live archive date bounds and
   publication rosters, cached with a 5-minute TTL. Softly decoupled: if database access is
   unavailable, uninitialized, or errors, it immediately falls back to safe static archive defaults.
"""

from __future__ import annotations

import contextlib
import time
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
   - Primary Columns: `id` (INT PK), `issue_id` (INT FK -> issues.id), `category_id` (INT FK -> article_categories.id), `headline` (VARCHAR), `subheadline`, `byline_author` (VARCHAR), `section` (VARCHAR), `printed_section`, `article_type` (ENUM: 'news', 'editorial', 'opinion', 'analysis', 'advertisement', 'sidebar', etc.), `word_count` (INT), `prominence_score` (FLOAT), `page_number` (INT via primary_page_id).
   - Purpose: Individual news stories, editorials, and commercial notices. Full text embeddings are indexed in the vector store for `hybrid_search`.

5. `pages`:
   - Primary Columns: `id` (INT PK), `issue_id` (INT FK -> issues.id), `page_number` (INT), `is_advertisement_page` (BOOLEAN).
   - Purpose: Broadsheet newspaper pages.

6. `photos`:
   - Primary Columns: `id` (INT PK), `article_id` (INT FK -> articles.id), `issue_id` (INT FK -> issues.id), `page_number` (INT), `visual_type` (VARCHAR: 'photo', 'infographic', 'chart', 'map'), `caption` (TEXT).
   - Purpose: Visual assets and infographics linked to articles and pages."""


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


# ---------------------------------------------------------------------------
# 3. Cached Live Archive Bounds Provider (Soft Decoupled)
# ---------------------------------------------------------------------------

_ARCHIVE_CACHE: dict[str, Any] = {
    "cached_at": 0.0,
    "context_str": "",
}
_CACHE_TTL_SECONDS = 300.0  # 5 minutes


async def get_archive_and_schema_context(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> str:
    """Return a unified markdown context block containing the schema and archive bounds.

    Guarantees:
    - Zero crashing: Never raises an exception even if the database is None or unreachable.
    - Zero lag: In-memory 5-minute cache avoids querying MySQL repeatedly during agent runs.
    - Soft decoupling: Immediately serves safe static defaults if database connection fails.
    """
    now = time.monotonic()
    if _ARCHIVE_CACHE["context_str"] and (now - _ARCHIVE_CACHE["cached_at"]) < _CACHE_TTL_SECONDS:
        return _ARCHIVE_CACHE["context_str"]

    min_date = STATIC_ARCHIVE_DATE_MIN
    max_date = STATIC_ARCHIVE_DATE_MAX
    publications = list(STATIC_CANONICAL_PUBLICATIONS)
    categories = list(STATIC_CANONICAL_CATEGORIES)

    if session_factory is not None:
        try:
            from app.models.article import ArticleCategory
            from app.models.newspaper import Issue, Newspaper
            from sqlalchemy import func

            async with session_factory() as db:
                # Query date range and distinct publications
                stmt = select(
                    func.min(Issue.issue_date),
                    func.max(Issue.issue_date),
                ).where(Issue.ingestion_status.in_(("completed", "indexed", "ready")))
                res = await db.execute(stmt)
                d_min, d_max = res.one_or_none() or (None, None)
                if d_min:
                    min_date = str(d_min)
                if d_max:
                    max_date = str(d_max)

                # Query distinct active newspapers
                np_stmt = (
                    select(Newspaper.name)
                    .join(Issue, Issue.newspaper_id == Newspaper.id)
                    .where(Issue.ingestion_status.in_(("completed", "indexed", "ready")))
                    .distinct()
                    .order_by(Newspaper.name)
                )
                np_res = await db.execute(np_stmt)
                db_pubs = list(np_res.scalars().all())
                if db_pubs:
                    publications = db_pubs

                # Query canonical categories
                cat_stmt = select(ArticleCategory.name).order_by(ArticleCategory.name)
                cat_res = await db.execute(cat_stmt)
                db_cats = list(cat_res.scalars().all())
                if db_cats:
                    categories = db_cats
        except Exception:
            # Silently fall back to safe canonical defaults on any DB exception
            pass

    context_lines = [
        STATIC_BROADSHEET_SCHEMA,
        "",
        "### 📅 ACTIVE ARCHIVE COVERAGE & BOUNDARIES",
        f"- Verified Date Range: {min_date} to {max_date}",
        f"- Available Broadsheet Publications ({len(publications)}): {', '.join(publications)}",
        f"- Active Taxonomy Categories: {', '.join(categories)}",
    ]

    context_str = "\n".join(context_lines)
    _ARCHIVE_CACHE["cached_at"] = now
    _ARCHIVE_CACHE["context_str"] = context_str
    return context_str


__all__ = [
    "STATIC_BROADSHEET_SCHEMA",
    "STATIC_CANONICAL_PUBLICATIONS",
    "STATIC_ARCHIVE_DATE_MIN",
    "STATIC_ARCHIVE_DATE_MAX",
    "STATIC_CANONICAL_CATEGORIES",
    "get_archive_and_schema_context",
]
