"""MySQL FULLTEXT search implementation.

Implements SearchIndex using MySQL's FULLTEXT index on articles(headline, full_text).
This provides lexical/BM25-style retrieval as the keyword search leg of hybrid RAG.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.storage.base import FullTextSearchResult

logger = get_logger(__name__)

_META_NOISE_PATTERN = re.compile(
    r"\b(?:explain|summari[sz]e|tell\s+me\s+about|what\s+happened|article|story|headline|"
    r"newspaper|broadsheet|edition|dated?|published\s+on|page\s*\d+|pg\s*\d+|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b",
    re.IGNORECASE,
)


def _clean_fulltext_query(query: str) -> str:
    """Strip conversational command words, dates, and page noise to sharpen BM25 keyword precision."""
    # If query contains a quoted headline, prioritize the quoted phrase
    quoted = re.search(r"[\"“']([^\"”']{8,150})[\"”']", query)
    if quoted:
        return quoted.group(1).strip()
    cleaned = _META_NOISE_PATTERN.sub(" ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.!,'\"“”")
    return cleaned if len(cleaned) >= 2 else query.strip()


class MySQLFullTextSearch:
    """SearchIndex backed by MySQL FULLTEXT (NATURAL LANGUAGE mode) with ILIKE fallback."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[FullTextSearchResult]:
        """Run a FULLTEXT search on articles with automatic ILIKE fallback for short terms/headlines."""
        if not query.strip():
            return []

        clean_q = _clean_fulltext_query(query)
        filters = filters or {}
        newspaper_id = filters.get("newspaper_id")
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        page_number = filters.get("page_number")

        join_clauses: list[str] = []
        if newspaper_id or date_from or date_to:
            join_clauses.append("JOIN issues i ON a.issue_id = i.id")
        if page_number is not None:
            join_clauses.append("JOIN article_pages ap ON a.id = ap.article_id")

        joins = " ".join(join_clauses)
        filter_conditions: list[str] = []
        params: dict[str, Any] = {"query": clean_q, "limit": top_k}

        if newspaper_id:
            filter_conditions.append("i.newspaper_id = :newspaper_id")
            params["newspaper_id"] = newspaper_id
        if date_from:
            filter_conditions.append("i.issue_date >= :date_from")
            params["date_from"] = date_from
        if date_to:
            filter_conditions.append("i.issue_date <= :date_to")
            params["date_to"] = date_to
        if page_number is not None:
            filter_conditions.append("ap.page_number = :page_number")
            params["page_number"] = int(page_number)

        ft_conditions = [
            "MATCH(a.headline, a.full_text) AGAINST (:query IN NATURAL LANGUAGE MODE) > 0",
            *filter_conditions,
        ]
        where_clause = " AND ".join(ft_conditions)

        sql = text(f"""
            SELECT
                a.id AS article_id,
                a.headline,
                LEFT(a.full_text, 200) AS snippet,
                MATCH(a.headline, a.full_text) AGAINST (:query IN NATURAL LANGUAGE MODE) AS score
            FROM articles a
            {joins}
            WHERE {where_clause}
            ORDER BY score DESC
            LIMIT :limit
        """)  # nosec

        async with self._session_factory() as session:
            result = await session.execute(sql, params)
            rows = result.fetchall()

            # Fallback: if FULLTEXT returned 0 hits (e.g. short acronym < 4 chars or exact substring),
            # run a fast parameterized LIKE fallback on headline
            if not rows and len(clean_q) >= 2:
                like_params = dict(params)
                like_params["like_pat"] = f"%{clean_q[:60]}%"
                like_conditions = ["a.headline LIKE :like_pat", *filter_conditions]
                like_where = " AND ".join(like_conditions)
                like_sql = text(f"""
                    SELECT
                        a.id AS article_id,
                        a.headline,
                        LEFT(a.full_text, 200) AS snippet,
                        1.0 AS score
                    FROM articles a
                    {joins}
                    WHERE {like_where}
                    ORDER BY a.prominence_score DESC
                    LIMIT :limit
                """)  # nosec
                like_res = await session.execute(like_sql, like_params)
                rows = like_res.fetchall()

        results = [
            FullTextSearchResult(
                article_id=row.article_id,
                headline=row.headline,
                snippet=row.snippet or "",
                score=float(row.score),
            )
            for row in rows
        ]
        logger.info(
            "MySQL FULLTEXT search",
            extra={"query": clean_q[:50], "hits": len(results)},
        )
        return results
