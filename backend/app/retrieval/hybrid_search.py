"""Hybrid Reciprocal Rank Fusion (RRF) Search Engine.

Combines dense semantic vector retrieval (Qdrant) and sparse keyword retrieval
(MySQL FULLTEXT) using Reciprocal Rank Fusion (RRF):
    RRF_score(d) = sum_{m in Models} 1 / (k + rank_m(d))
with standard smoothing parameter k = 60.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.article import Article
from app.models.newspaper import Issue
from app.providers.base import EmbeddingProvider
from app.providers.registry import get_registry
from app.storage.mysql_fulltext import MySQLFullTextSearch
from app.storage.qdrant_store import QdrantStore

logger = get_logger(__name__)


@dataclass
class SearchFilter:
    """Filters for hybrid retrieval."""

    newspaper_id: int | None = None
    date_from: str | None = None  # YYYY-MM-DD
    date_to: str | None = None  # YYYY-MM-DD
    article_type: str | None = None
    min_prominence: float | None = None


@dataclass
class HybridSearchResult:
    """Unified result from hybrid search with combined RRF score and citations."""

    article_id: int
    headline: str
    subheadline: str | None
    byline_author: str | None
    section: str | None
    article_type: str
    prominence_score: float
    rrf_score: float
    vector_rank: int | None
    keyword_rank: int | None
    snippet: str
    newspaper_name: str
    issue_date: str
    pages: list[int]
    matched_chunks: list[dict[str, Any]] = field(default_factory=list)


class HybridSearchEngine:
    """Hybrid Search Engine combining Qdrant dense vectors and MySQL FULLTEXT."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        qdrant: QdrantStore | None = None,
        embed_provider: EmbeddingProvider | None = None,
        k: int = 60,
    ) -> None:
        self._session_factory = session_factory
        self._settings = get_settings()
        self._qdrant = qdrant or QdrantStore(self._settings.qdrant)
        self._provider = embed_provider
        self._ft_search = MySQLFullTextSearch(session_factory=session_factory)
        self._k = k

    def _get_embedding_provider(self) -> EmbeddingProvider:
        if self._provider:
            return self._provider
        registry = get_registry()
        provider = registry.get_provider("embedding")
        if not isinstance(provider, EmbeddingProvider):
            raise TypeError(f"Provider {provider} does not implement EmbeddingProvider")
        return provider

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilter | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> list[HybridSearchResult]:
        """Execute hybrid search using Reciprocal Rank Fusion."""
        if not query.strip():
            return []

        search_filters_dict: dict[str, Any] = {}
        if filters:
            if filters.newspaper_id:
                search_filters_dict["newspaper_id"] = filters.newspaper_id
            if filters.date_from or filters.date_to:
                date_range: dict[str, str] = {}
                if filters.date_from:
                    date_range["gte"] = filters.date_from
                if filters.date_to:
                    date_range["lte"] = filters.date_to
                search_filters_dict["issue_date"] = date_range

        # 1. Run Vector Search (Dense)
        provider = self._get_embedding_provider()
        query_vector = await provider.embed_one(query)

        vector_results = await self._qdrant.search(
            query_vector=query_vector,
            top_k=top_k * 3,  # Fetch more for reciprocal fusion
            filters=search_filters_dict if search_filters_dict else None,
        )

        # 2. Run FULLTEXT Search (Sparse)
        ft_filters: dict[str, Any] = {}
        if filters:
            if filters.newspaper_id:
                ft_filters["newspaper_id"] = filters.newspaper_id
            if filters.date_from:
                ft_filters["date_from"] = filters.date_from
            if filters.date_to:
                ft_filters["date_to"] = filters.date_to

        keyword_results = await self._ft_search.search(
            query=query,
            top_k=top_k * 3,
            filters=ft_filters if ft_filters else None,
        )

        # 3. Compute RRF Scores
        # Map: article_id -> {vector_rank, keyword_rank, rrf_score, chunks}
        scores: dict[int, dict[str, Any]] = {}

        for rank, v_res in enumerate(vector_results, start=1):
            art_id = v_res.article_id
            if art_id is None:
                continue
            if art_id not in scores:
                scores[art_id] = {
                    "vector_rank": rank,
                    "keyword_rank": None,
                    "rrf_score": 0.0,
                    "chunks": [],
                }
            scores[art_id]["vector_rank"] = rank
            scores[art_id]["rrf_score"] += dense_weight * (1.0 / (self._k + rank))
            scores[art_id]["chunks"].append(v_res.payload)

        for rank, kw_res in enumerate(keyword_results, start=1):
            art_id = kw_res.article_id
            if art_id is None:
                continue
            if art_id not in scores:
                scores[art_id] = {
                    "vector_rank": None,
                    "keyword_rank": rank,
                    "rrf_score": 0.0,
                    "chunks": [],
                }
            scores[art_id]["keyword_rank"] = rank
            scores[art_id]["rrf_score"] += sparse_weight * (1.0 / (self._k + rank))

        if not scores:
            return []

        # Sort candidate article IDs by RRF score descending
        sorted_candidates = sorted(
            scores.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )[:top_k]

        candidate_ids = [item[0] for item in sorted_candidates]

        # 4. Fetch full Article records from MySQL with issues and pages
        async with self._session_factory() as db:
            stmt = (
                select(Article)
                .where(Article.id.in_(candidate_ids))
                .options(
                    selectinload(Article.issue).selectinload(Issue.newspaper),
                    selectinload(Article.article_pages),
                )
            )
            db_articles_res = await db.execute(stmt)
            articles_map = {a.id: a for a in db_articles_res.scalars().all()}

        # 5. Format unified HybridSearchResult list in RRF order
        final_results: list[HybridSearchResult] = []
        for art_id, score_info in sorted_candidates:
            article = articles_map.get(art_id)
            if not article:
                continue

            # Check post-retrieval filters if specified
            if filters and filters.article_type and article.article_type != filters.article_type:
                continue
            if (
                filters
                and filters.min_prominence
                and article.prominence_score < filters.min_prominence
            ):
                continue

            np_name = (
                article.issue.newspaper.name
                if article.issue and article.issue.newspaper
                else "Daily News"
            )
            issue_date = str(article.issue.issue_date) if article.issue else ""
            pages_list = (
                sorted({ap.page_number for ap in article.article_pages})
                if article.article_pages
                else []
            )

            # Snippet: summary or lead paragraph
            snippet = article.summary or (article.full_text[:300] if article.full_text else "")

            final_results.append(
                HybridSearchResult(
                    article_id=article.id,
                    headline=article.headline or "Untitled",
                    subheadline=article.subheadline,
                    byline_author=article.byline_author,
                    section=article.section,
                    article_type=article.article_type,
                    prominence_score=article.prominence_score,
                    rrf_score=round(score_info["rrf_score"], 6),
                    vector_rank=score_info["vector_rank"],
                    keyword_rank=score_info["keyword_rank"],
                    snippet=snippet,
                    newspaper_name=np_name,
                    issue_date=issue_date,
                    pages=pages_list,
                    matched_chunks=score_info["chunks"],
                )
            )

        logger.info(
            "Hybrid search executed",
            extra={
                "query": query[:40],
                "vector_hits": len(vector_results),
                "keyword_hits": len(keyword_results),
                "fused_results": len(final_results),
            },
        )

        return final_results
