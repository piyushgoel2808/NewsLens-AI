"""MySQL FULLTEXT search implementation.

Implements SearchIndex using MySQL's FULLTEXT index on articles(headline, full_text).
This provides lexical/BM25-style retrieval as the keyword search leg of hybrid RAG.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.storage.base import FullTextSearchResult

logger = get_logger(__name__)


class MySQLFullTextSearch:
    """SearchIndex backed by MySQL FULLTEXT (NATURAL LANGUAGE mode)."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[FullTextSearchResult]:
        """Run a FULLTEXT search on articles.

        Supports filters: newspaper_id (int), date_from (str YYYY-MM-DD),
        date_to (str YYYY-MM-DD).

        Args:
            query: Search terms (MySQL NATURAL LANGUAGE mode).
            top_k: Max results.
            filters: Optional structured filters.

        Returns:
            List of FullTextSearchResult ordered by relevance score.
        """
        if not query.strip():
            return []

        filters = filters or {}
        newspaper_id = filters.get("newspaper_id")
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")

        # Build SQL dynamically but safely (parameterised, no f-string injection)
        joins = (
            "JOIN issues i ON a.issue_id = i.id" if (newspaper_id or date_from or date_to) else ""
        )
        conditions = [
            "MATCH(a.headline, a.full_text) AGAINST (:query IN NATURAL LANGUAGE MODE) > 0"
        ]
        params: dict[str, Any] = {"query": query, "limit": top_k}

        if newspaper_id:
            conditions.append("i.newspaper_id = :newspaper_id")
            params["newspaper_id"] = newspaper_id
        if date_from:
            conditions.append("i.issue_date >= :date_from")
            params["date_from"] = date_from
        if date_to:
            conditions.append("i.issue_date <= :date_to")
            params["date_to"] = date_to

        where_clause = " AND ".join(conditions)

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
        """)  # nosec (no user-controlled fragments; joins/conditions are code-defined)

        async with self._session_factory() as session:
            result = await session.execute(sql, params)
            rows = result.fetchall()

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
            extra={"query": query[:50], "hits": len(results)},
        )
        return results
