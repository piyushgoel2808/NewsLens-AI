"""Dense Vector Embedding and Qdrant Indexing Engine for Newspaper Chunks."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.chunker import DocumentChunk
from app.models.article import ArticleChunk
from app.providers.base import EmbeddingProvider
from app.providers.registry import get_registry
from app.storage.base import VectorPoint
from app.storage.qdrant_store import QdrantStore

logger = get_logger(__name__)


class ArticleEmbedder:
    """Embeds article chunks and synchronizes vector points into Qdrant and MySQL."""

    def __init__(
        self,
        db: AsyncSession,
        qdrant: QdrantStore | None = None,
        embed_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._db = db
        self._settings = get_settings()
        self._qdrant = qdrant or QdrantStore(self._settings.qdrant)
        self._provider = embed_provider

    def _get_embedding_provider(self) -> EmbeddingProvider:
        """Resolve configured embedding provider."""
        if self._provider:
            return self._provider
        registry = get_registry()
        provider = registry.get_provider("embedding")
        if not isinstance(provider, EmbeddingProvider):
            raise TypeError(f"Provider {provider} does not implement EmbeddingProvider")
        return provider

    async def embed_and_index_chunks(
        self,
        article_id: int,
        issue_id: int,
        newspaper_name: str,
        issue_date: str,
        headline: str,
        section: str | None,
        article_type: str,
        prominence_score: float,
        page_numbers: list[int],
        entities: list[str],
        topics: list[str],
        chunks: list[DocumentChunk],
        printed_pages: list[str] | None = None,
    ) -> list[str]:
        """Embed a batch of document chunks, upsert to Qdrant, and persist ArticleChunk records."""
        if not chunks:
            return []

        # 1. Embed chunk texts
        texts_to_embed = [c.text for c in chunks]
        provider = self._get_embedding_provider()
        vectors = await provider.embed(texts_to_embed)

        vector_ids: list[str] = []
        qdrant_points: list[VectorPoint] = []

        # 2. Prepare points for Qdrant and records for MySQL
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            vector_ids.append(point_id)

            payload = {
                "article_id": article_id,
                "issue_id": issue_id,
                "newspaper_name": newspaper_name,
                "issue_date": issue_date,
                "headline": headline,
                "section": section or "General",
                "article_type": article_type,
                "prominence_score": prominence_score,
                "chunk_index": chunk.chunk_index,
                "page_numbers": page_numbers,
                "printed_pages": printed_pages or [],
                "entities": entities,
                "topics": topics,
                "chunk_text": chunk.text,
                "raw_text": chunk.raw_text,
            }

            qdrant_points.append(
                VectorPoint(
                    id=point_id,
                    vector=vectors[i],
                    payload=payload,
                )
            )

            # Create MySQL record
            chunk_record = ArticleChunk(
                article_id=article_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
                embedding_vector_id=point_id,
            )
            self._db.add(chunk_record)

        # 3. Upsert to Qdrant
        await self._qdrant.upsert(qdrant_points)
        await self._db.flush()

        logger.info(
            "Article chunks embedded and indexed in Qdrant",
            extra={
                "article_id": article_id,
                "chunks_count": len(chunks),
                "collection": self._settings.qdrant.collection_name,
            },
        )

        return vector_ids
