"""Qdrant vector store implementation.

Implements VectorStore using the qdrant-client async API.
Creates the collection on startup if it doesn't exist.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import QdrantSettings
from app.core.logging import get_logger
from app.storage.base import VectorPoint, VectorSearchResult

logger = get_logger(__name__)


class QdrantStore:
    """VectorStore backed by Qdrant."""

    def __init__(
        self,
        settings: QdrantSettings,
        embedding_dim: int = 1024,
    ) -> None:
        self._settings = settings
        self._embedding_dim = embedding_dim
        self._collection = settings.collection_name
        self._client = AsyncQdrantClient(
            host=settings.host,
            port=settings.port,
            api_key=settings.api_key,
            check_compatibility=False,
        )

    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        try:
            existing = await self._client.get_collections()
            names = [c.name for c in existing.collections]
            if self._collection not in names:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=self._embedding_dim,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection",
                    extra={
                        "collection": self._collection,
                        "dim": self._embedding_dim,
                    },
                )
        except Exception as e:
            logger.error("Failed to ensure Qdrant collection", extra={"error": str(e)})
            raise

    async def upsert(self, points: list[VectorPoint]) -> None:
        """Upsert a batch of VectorPoints into Qdrant."""
        if not points:
            return
        qdrant_points = [
            qmodels.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
        ]
        await self._client.upsert(
            collection_name=self._collection,
            points=qdrant_points,
        )
        logger.info("Qdrant upsert", extra={"count": len(points)})

    def _build_filter(self, filters: dict[str, Any]) -> qmodels.Filter | None:
        """Convert a simple filter dict to a Qdrant Filter."""
        must: list[qmodels.Condition] = []
        for key, value in filters.items():
            if isinstance(value, dict):
                # Range filter: {"issue_date": {"gte": "2024-01-01", "lte": "2024-12-31"}}
                range_kwargs = {}
                if "gte" in value:
                    range_kwargs["gte"] = value["gte"]
                if "lte" in value:
                    range_kwargs["lte"] = value["lte"]
                if "gt" in value:
                    range_kwargs["gt"] = value["gt"]
                if "lt" in value:
                    range_kwargs["lt"] = value["lt"]
                must.append(
                    qmodels.FieldCondition(
                        key=key,
                        range=qmodels.Range(**range_kwargs),
                    )
                )
            else:
                # Exact match filter
                must.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=value),
                    )
                )
        return qmodels.Filter(must=must) if must else None

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Find similar vectors. Applies payload filters if provided."""
        qdrant_filter = self._build_filter(filters) if filters else None
        query_response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            VectorSearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {},
                article_id=r.payload.get("article_id") if r.payload else None,
            )
            for r in query_response.points
        ]

    async def delete(self, ids: list[str]) -> None:
        """Delete points by ID."""
        if not ids:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.PointIdsList(points=list(ids)),
        )

    async def delete_by_filter(self, filters: dict[str, Any]) -> None:
        """Delete points matching payload filters (e.g. {'issue_id': 42})."""
        qdrant_filter = self._build_filter(filters)
        if not qdrant_filter:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(filter=qdrant_filter),
        )
        logger.info("Qdrant delete by filter", extra={"filters": filters})

    async def collection_info(self) -> dict[str, Any]:
        """Return collection metadata."""
        info = await self._client.get_collection(self._collection)
        return {
            "name": self._collection,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    async def ping(self) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self._client.close()
