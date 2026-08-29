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
from app.retrieval.reranker import CrossEncoderReranker
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
    category_id: int | None = None
    category_name: str | None = None
    min_prominence: float | None = None
    page_number: int | None = None
    printed_page: str | None = None
    exclude_pages: list[int] = field(default_factory=list)
    exclude_printed_pages: list[str] = field(default_factory=list)
    has_photo: bool | None = None
    has_table: bool | None = None


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
    issue_id: int = 0
    bboxes: list[dict[str, Any]] = field(default_factory=list)
    printed_pages: list[str] = field(default_factory=list)
    matched_chunks: list[dict[str, Any]] = field(default_factory=list)
    rerank_score: float | None = None
    parent_article_text: str | None = None
    has_visual_data: bool = False
    visual_type: str | None = None




class HybridSearchEngine:
    """Hybrid Search Engine combining Qdrant dense vectors, MySQL FULLTEXT, and Cross-Encoder Reranking."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        qdrant: QdrantStore | None = None,
        embed_provider: EmbeddingProvider | None = None,
        reranker: CrossEncoderReranker | None = None,
        k: int = 60,
    ) -> None:
        self._session_factory = session_factory
        self._settings = get_settings()
        self._qdrant = qdrant or QdrantStore(self._settings.qdrant)
        self._provider = embed_provider
        self._reranker = reranker or CrossEncoderReranker()
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
        rerank: bool = True,
    ) -> list[HybridSearchResult]:
        """Execute two-stage hybrid search using Reciprocal Rank Fusion and Cross-Encoder neural reranking."""
        if not query.strip():
            return []

        search_filters_dict: dict[str, Any] = {}
        if filters:
            if filters.newspaper_id:
                search_filters_dict["newspaper_id"] = filters.newspaper_id
            if filters.page_number:
                search_filters_dict["page_numbers"] = filters.page_number
            if filters.printed_page:
                search_filters_dict["printed_pages"] = filters.printed_page
            if filters.has_photo is not None:
                search_filters_dict["has_photo"] = filters.has_photo
            if filters.has_table is not None:
                search_filters_dict["has_table"] = filters.has_table
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
            score_threshold=0.30,
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

        # Two-stage retrieval cascade:
        # Fetch Top 75 candidates from RRF hybrid search to pass into second-stage Cross-Encoder
        candidate_pool_size = max(75, top_k * 3) if rerank else top_k
        sorted_candidates = sorted(
            scores.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )[:candidate_pool_size]

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
            if filters and filters.category_id and article.category_id != filters.category_id:
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
            printed_pages_list = (
                [
                    ap.printed_page_number
                    for ap in sorted(article.article_pages, key=lambda p: p.page_number)
                    if ap.printed_page_number
                ]
                if article.article_pages
                else []
            )

            # Hard Safety-Net Invariant: Assert no returned article has an excluded page
            if filters and filters.exclude_pages and any(p in filters.exclude_pages for p in pages_list):
                continue
            if filters and filters.exclude_printed_pages:
                excl_clean = {p.strip().lower() for p in filters.exclude_printed_pages}
                if any(
                    p.lower() in excl_clean or f"page {p.lower()}" in excl_clean
                    for p in printed_pages_list
                ):
                    continue

            # Check positive page filter if specified
            if filters and filters.page_number and filters.page_number not in pages_list:
                continue
            if filters and filters.printed_page:
                p_filter_clean = filters.printed_page.strip().lower()
                matches_printed = any(
                    p.lower() == p_filter_clean or f"page {p_filter_clean}" == p.lower()
                    for p in printed_pages_list
                )
                matches_pdf_page = any(str(p) == p_filter_clean for p in pages_list)
                if not (matches_printed or matches_pdf_page):
                    continue

            # Snippet & Parent-Document (Small-to-Big) Context Strategy:
            # 1. Capture exact matched chunks for citation precision
            # 2. Enrich with parent article text / full summary for non-truncated synthesis
            exact_match_snippet = ""
            if score_info["chunks"]:
                chunk_texts = [
                    c.get("chunk_text") or c.get("raw_text") or ""
                    for c in score_info["chunks"]
                    if (c.get("chunk_text") or c.get("raw_text"))
                ]
                exact_match_snippet = "\n\n".join(chunk_texts[:2]) if chunk_texts else ""

            parent_full_text = article.full_text or article.summary or ""

            if exact_match_snippet:
                snippet = (
                    f"[Exact Chunk Match]:\n{exact_match_snippet}\n\n"
                    f"[Article Parent Context]:\n{parent_full_text[:1200]}"
                    if len(parent_full_text) > len(exact_match_snippet)
                    else exact_match_snippet
                )
            else:
                snippet = article.summary or (article.full_text[:600] if article.full_text else "")

            bboxes_list: list[dict[str, Any]] = []
            if article.article_pages:
                for ap in article.article_pages:
                    if ap.bbox_json:
                        if isinstance(ap.bbox_json, list):
                            bboxes_list.extend(ap.bbox_json)
                        elif isinstance(ap.bbox_json, dict):
                            bboxes_list.append(ap.bbox_json)

            # Detect if matched chunks or article contain visual infographic / chart data
            has_vis = any(c.get("has_visual_data") or c.get("chunk_type") == "visual" for c in score_info["chunks"])
            v_type = next((c.get("visual_type") for c in score_info["chunks"] if c.get("visual_type")), None)

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
                    issue_id=article.issue_id,
                    bboxes=bboxes_list,
                    printed_pages=printed_pages_list,
                    matched_chunks=score_info["chunks"],
                    parent_article_text=parent_full_text,
                    has_visual_data=has_vis,
                    visual_type=v_type,
                )
            )

        # Second-Stage Neural Reranking:
        # Pass candidate pool into CrossEncoder to compute interaction scores and extract Top K
        if rerank and final_results:
            candidates_data = [
                {
                    "result_obj": r,
                    "headline": r.headline,
                    "snippet": r.snippet,
                    "rrf_score": r.rrf_score,
                    "prominence_score": r.prominence_score,
                }
                for r in final_results
            ]
            reranked_data = await self._reranker.rerank(
                query=query,
                candidates=candidates_data,
                top_k=top_k,
            )
            top_results: list[HybridSearchResult] = []
            for item in reranked_data:
                res_obj: HybridSearchResult = item["result_obj"]
                res_obj.rerank_score = item.get("rerank_score")
                top_results.append(res_obj)
            final_results = top_results
        else:
            final_results = final_results[:top_k]

        logger.info(
            "Hybrid search executed with two-stage cascade",
            extra={
                "query": query[:40],
                "vector_hits": len(vector_results),
                "keyword_hits": len(keyword_results),
                "fused_results": len(final_results),
                "reranked": rerank,
            },
        )

        return final_results
