"""Unit tests for Hybrid Search Engine and Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.retrieval.hybrid_search import HybridSearchEngine, HybridSearchResult
from app.storage.base import FullTextSearchResult, VectorSearchResult


class MockEmbedProvider:
    """Mock embedding provider for hybrid search test."""

    async def embed_one(self, text: str) -> list[float]:
        return [0.1] * 768


class TestHybridSearch:
    """Test suite for HybridSearchEngine."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_list(self) -> None:
        engine = HybridSearchEngine(session_factory=MagicMock())
        results = await engine.search("")
        assert results == []

    @pytest.mark.asyncio
    async def test_rrf_scoring_and_fusion(self) -> None:
        mock_qdrant = MagicMock()
        mock_qdrant.search = AsyncMock(
            return_value=[
                VectorSearchResult(
                    id="v1",
                    score=0.85,
                    payload={
                        "headline": "A1",
                        "newspaper_name": "Paper A",
                        "issue_date": "2026-08-21",
                        "pages": [1],
                    },
                    article_id=1,
                ),
                VectorSearchResult(
                    id="v2",
                    score=0.75,
                    payload={
                        "headline": "A2",
                        "newspaper_name": "Paper B",
                        "issue_date": "2026-08-21",
                        "pages": [2],
                    },
                    article_id=2,
                ),
            ]
        )

        mock_session_factory = MagicMock()
        mock_db = MagicMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_db

        # Mock DB articles
        mock_art_1 = MagicMock(
            id=1,
            headline="Headline 1",
            subheadline=None,
            byline_author="Author 1",
            section="Finance",
            article_type="news",
            prominence_score=0.9,
            summary="Summary 1",
            full_text="Full text 1",
            issue=MagicMock(issue_date="2026-08-21", newspaper=MagicMock(name="Daily News")),
            pages=[MagicMock(page_number=1)],
        )
        mock_art_2 = MagicMock(
            id=2,
            headline="Headline 2",
            subheadline=None,
            byline_author="Author 2",
            section="General",
            article_type="news",
            prominence_score=0.7,
            summary="Summary 2",
            full_text="Full text 2",
            issue=MagicMock(issue_date="2026-08-21", newspaper=MagicMock(name="Daily News")),
            pages=[MagicMock(page_number=2)],
        )

        mock_db_res = MagicMock()
        mock_db_res.scalars.return_value.all.return_value = [mock_art_1, mock_art_2]
        mock_db.execute = AsyncMock(return_value=mock_db_res)

        engine = HybridSearchEngine(
            session_factory=mock_session_factory,
            qdrant=mock_qdrant,
            embed_provider=MockEmbedProvider(),
            k=60,
        )

        # Mock sparse search
        engine._ft_search.search = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                FullTextSearchResult(
                    article_id=2,
                    headline="Headline 2",
                    score=3.5,
                    snippet="Snip 2",
                ),
                FullTextSearchResult(
                    article_id=1,
                    headline="Headline 1",
                    score=1.2,
                    snippet="Snip 1",
                ),
            ]
        )

        results = await engine.search("financial market news", top_k=5)

        assert len(results) == 2
        # Verify RRF properties
        assert all(isinstance(r, HybridSearchResult) for r in results)
        assert results[0].rrf_score > 0.0
        # Both article 1 and article 2 appear in vector and keyword results
        assert results[0].vector_rank is not None
        assert results[0].keyword_rank is not None
