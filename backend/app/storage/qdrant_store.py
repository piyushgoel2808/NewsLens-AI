"""Qdrant vector store implementation.

Implements VectorStore using the qdrant-client async API.
Creates the collection on startup if it doesn't exist.
"""

from __future__ import annotations

import contextlib
from datetime import UTC
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
        self._collection_v2 = getattr(settings, "collection_name_v2", "article_chunks_v2")
        if settings.host.startswith("http://") or settings.host.startswith("https://"):
            self._client = AsyncQdrantClient(
                url=settings.host,
                port=settings.port if settings.port not in (80, 443) else None,
                api_key=settings.api_key,
                check_compatibility=False,
            )
        elif getattr(settings, "https", False):
            self._client = AsyncQdrantClient(
                url=f"https://{settings.host}:{settings.port}",
                api_key=settings.api_key,
                check_compatibility=False,
            )
        else:
            self._client = AsyncQdrantClient(
                host=settings.host,
                port=settings.port,
                api_key=settings.api_key,
                check_compatibility=False,
            )

    def get_collection_name(self, embedding_dim: int | None = None) -> str:
        """Resolve collection name based on vector dimension (768 -> v2, default 1024 -> v1)."""
        dim = embedding_dim if embedding_dim is not None else self._embedding_dim
        if dim == 768:
            return self._collection_v2
        return self._collection

    async def _ensure_single_collection(self, collection_name: str, dim: int) -> None:
        """Create a specific collection if it doesn't exist, and create payload indexes."""
        try:
            existing = await self._client.get_collections()
            names = [c.name for c in existing.collections]
            if collection_name not in names:
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=dim,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection",
                    extra={
                        "collection": collection_name,
                        "dim": dim,
                    },
                )

            # Ensure payload indexes for search filters
            indexed_fields: dict[str, qmodels.PayloadSchemaType] = {
                "page_numbers": qmodels.PayloadSchemaType.INTEGER,
                "newspaper_name": qmodels.PayloadSchemaType.KEYWORD,
                "issue_date": qmodels.PayloadSchemaType.KEYWORD,
                "has_photo": qmodels.PayloadSchemaType.BOOL,
                "has_table": qmodels.PayloadSchemaType.BOOL,
                "article_id": qmodels.PayloadSchemaType.INTEGER,
                "issue_id": qmodels.PayloadSchemaType.INTEGER,
            }
            for field, schema_type in indexed_fields.items():
                with contextlib.suppress(Exception):
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=schema_type,
                    )
        except Exception as e:
            logger.error(
                "Failed to ensure Qdrant collection",
                extra={"collection": collection_name, "error": str(e)},
            )
            raise

    async def _ensure_collection(self) -> None:
        """Ensure both article_chunks (1024d) and article_chunks_v2 (768d) collections exist."""
        # 1. Local/Hybrid collection (1024d)
        await self._ensure_single_collection(self._collection, 1024)
        # 2. Cloud collection (768d)
        if self._collection_v2 and self._collection_v2 != self._collection:
            await self._ensure_single_collection(self._collection_v2, 768)

    async def upsert(
        self,
        points: list[VectorPoint],
        collection_name: str | None = None,
    ) -> None:
        """Upsert a batch of VectorPoints into Qdrant. Auto-routes by vector dimension if not specified."""
        if not points:
            return
        target_collection = collection_name
        if not target_collection:
            dim = len(points[0].vector) if points[0].vector else self._embedding_dim
            target_collection = self.get_collection_name(dim)

        qdrant_points = [
            qmodels.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
        ]
        await self._client.upsert(
            collection_name=target_collection,
            points=qdrant_points,
        )
        logger.info(
            "Qdrant upsert",
            extra={"count": len(points), "collection": target_collection},
        )

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
                    from datetime import datetime
                    dt_kwargs: dict[str, Any] = {}
                    for bound, val in value.items():
                        if isinstance(val, str):
                            try:
                                if bound in ("gte", "gt"):
                                    if len(val) == 10:
                                        d = datetime.fromisoformat(val)
                                        dt_kwargs[bound] = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=UTC)
                                    else:
                                        dt_kwargs[bound] = datetime.fromisoformat(val)
                                elif bound in ("lte", "lt"):
                                    if len(val) == 10:
                                        d = datetime.fromisoformat(val)
                                        dt_kwargs[bound] = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=UTC)
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
        collection_name: str | None = None,
    ) -> list[VectorSearchResult]:
        """Find similar vectors in target collection (auto-routed by vector dimension if not specified)."""
        target_collection = collection_name or self.get_collection_name(len(query_vector))
        qdrant_filter = self._build_filter(filters) if filters else None
        query_response = await self._client.query_points(
            collection_name=target_collection,
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

    def _target_collections(self, collection_name: str | None = None) -> list[str]:
        """Return list of collections to operate on."""
        if collection_name:
            return [collection_name]
        cols = [self._collection]
        if self._collection_v2 and self._collection_v2 not in cols:
            cols.append(self._collection_v2)
        return cols

    async def delete(self, ids: list[str], collection_name: str | None = None) -> None:
        """Delete points by ID from target collection(s)."""
        if not ids:
            return
        for col in self._target_collections(collection_name):
            with contextlib.suppress(Exception):
                await self._client.delete(
                    collection_name=col,
                    points_selector=qmodels.PointIdsList(points=list(ids)),
                )

    async def delete_by_filter(self, filters: dict[str, Any], collection_name: str | None = None) -> None:
        """Delete points matching payload filters from target collection(s)."""
        qdrant_filter = self._build_filter(filters)
        if not qdrant_filter:
            return
        for col in self._target_collections(collection_name):
            with contextlib.suppress(Exception):
                await self._client.delete(
                    collection_name=col,
                    points_selector=qmodels.FilterSelector(filter=qdrant_filter),
                )
        logger.info("Qdrant delete by filter", extra={"filters": filters})

    async def set_payload_by_filter(
        self, payload: dict[str, Any], filters: dict[str, Any], collection_name: str | None = None
    ) -> None:
        """Update payload fields for all points matching a filter selector in target collection(s)."""
        qdrant_filter = self._build_filter(filters)
        if not qdrant_filter:
            return
        for col in self._target_collections(collection_name):
            with contextlib.suppress(Exception):
                await self._client.set_payload(
                    collection_name=col,
                    payload=payload,
                    points=qmodels.FilterSelector(filter=qdrant_filter),
                )
        logger.info(
            "Qdrant updated payload by filter",
            extra={"filters": filters, "payload_keys": list(payload.keys())},
        )

    async def collection_info(self, collection_name: str | None = None) -> dict[str, Any]:
        """Return collection metadata."""
        col = collection_name or self._collection
        info = await self._client.get_collection(col)
        return {
            "name": col,
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
