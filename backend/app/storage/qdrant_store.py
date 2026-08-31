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
                # Check if it's an exact match date/value (gte == lte)
                if "gte" in value and "lte" in value and value["gte"] == value["lte"]:
                    must.append(
                        qmodels.FieldCondition(
                            key=key,
                            match=qmodels.MatchValue(value=value["gte"]),
                        )
                    )
                    continue

                # Check if any value is a date string (e.g. YYYY-MM-DD)
                is_date = any(isinstance(v, str) and "-" in v for v in value.values())
                if is_date:
                    from datetime import datetime, timezone
                    dt_kwargs: dict[str, Any] = {}
                    for bound, val in value.items():
                        if isinstance(val, str):
                            try:
                                if bound in ("gte", "gt"):
                                    if len(val) == 10:
                                        d = datetime.fromisoformat(val)
                                        dt_kwargs[bound] = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                                    else:
                                        dt_kwargs[bound] = datetime.fromisoformat(val)
                                elif bound in ("lte", "lt"):
                                    if len(val) == 10:
                                        d = datetime.fromisoformat(val)
                                        dt_kwargs[bound] = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
                                    else:
                                        dt_kwargs[bound] = datetime.fromisoformat(val)
                            except Exception:
                                pass
                        elif isinstance(val, (int, float)):
                            dt_kwargs[bound] = val
                    if dt_kwargs:
                        must.append(
                            qmodels.FieldCondition(
                                key=key,
                                range=qmodels.DatetimeRange(**dt_kwargs),
                            )
                        )
                    continue

                # Standard numeric range filter
                range_kwargs = {}
                for bound in ("gte", "lte", "gt", "lt"):
                    if bound in value:
                        range_kwargs[bound] = value[bound]
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

    async def set_payload_by_filter(self, payload: dict[str, Any], filters: dict[str, Any]) -> None:
        """Update payload fields for all points matching a filter selector."""
        qdrant_filter = self._build_filter(filters)
        if not qdrant_filter:
            return
        await self._client.set_payload(
            collection_name=self._collection,
            payload=payload,
            points=qmodels.FilterSelector(filter=qdrant_filter),
        )
        logger.info(
            "Qdrant updated payload by filter",
            extra={"filters": filters, "payload_keys": list(payload.keys())},
        )

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
