"""Unit tests for ArticleEmbedder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import ArticleEmbedder
from app.providers.base import ProviderCapability


class MockEmbedProvider:
    """Mock embedding provider for unit tests."""

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(embedding_dim=768)

    @property
    def provider_name(self) -> str:
        return "mock_embed"

    @property
    def embedding_dim(self) -> int:
        return 768

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    async def embed_one(self, text: str) -> list[float]:
        return [0.1] * 768


class TestArticleEmbedder:
    """Test suite for ArticleEmbedder."""

    @pytest.mark.asyncio
    async def test_embed_and_index_chunks(self) -> None:
        mock_db = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.add = MagicMock()

        mock_qdrant = MagicMock()
        mock_qdrant.upsert = AsyncMock()

        embedder = ArticleEmbedder(
            db=mock_db,
            qdrant=mock_qdrant,
            embed_provider=MockEmbedProvider(),
        )

        chunks = [
            DocumentChunk(
                chunk_index=0,
                text="[Newspaper: Daily | Headline: TEST]\n\nFirst paragraph text content.",
                token_count=15,
                header_context="[Newspaper: Daily | Headline: TEST]",
                raw_text="First paragraph text content.",
            ),
            DocumentChunk(
                chunk_index=1,
                text="[Newspaper: Daily | Headline: TEST]\n\nSecond paragraph text content.",
                token_count=15,
                header_context="[Newspaper: Daily | Headline: TEST]",
                raw_text="Second paragraph text content.",
            ),
        ]

        vector_ids = await embedder.embed_and_index_chunks(
            article_id=101,
            issue_id=1,
            newspaper_name="The Daily Record",
            issue_date="2026-08-21",
            headline="TEST STORY",
            section="Front Page",
            article_type="news",
            prominence_score=0.85,
            page_numbers=[1],
            entities=["John Smith", "New Delhi"],
            topics=["Politics"],
            chunks=chunks,
        )

        assert len(vector_ids) == 2
        assert mock_qdrant.upsert.called
        assert mock_db.add.call_count == 2
        assert mock_db.flush.called

        # Inspect points payload passed to Qdrant
        points_arg = mock_qdrant.upsert.call_args[0][0]
        assert len(points_arg) == 2
        assert points_arg[0].payload["article_id"] == 101
        assert points_arg[0].payload["headline"] == "TEST STORY"
        assert len(points_arg[0].vector) == 768
