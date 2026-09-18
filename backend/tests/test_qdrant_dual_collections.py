"""Unit tests for QdrantStore dual collection management and routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qmodels

from app.core.config import QdrantSettings
from app.storage.base import VectorPoint
from app.storage.qdrant_store import QdrantStore


class TestQdrantDualCollections:
    """Test suite for dual Qdrant collections: article_chunks (1024d) vs article_chunks_v2 (768d)."""

    def test_get_collection_name_by_dim(self) -> None:
        settings = QdrantSettings(
            collection_name="article_chunks",
            collection_name_v2="article_chunks_v2",
        )
        store = QdrantStore(settings, embedding_dim=1024)

        # 1024-dim (Local PyTorch BGE-M3) -> article_chunks
        assert store.get_collection_name(1024) == "article_chunks"
        # 768-dim (Cloud Google Gemini 001) -> article_chunks_v2
        assert store.get_collection_name(768) == "article_chunks_v2"
        # Default fallback without argument -> article_chunks (since default embedding_dim=1024)
        assert store.get_collection_name() == "article_chunks"

    @pytest.mark.asyncio
    async def test_upsert_auto_routing_by_dim(self) -> None:
        settings = QdrantSettings(
            collection_name="article_chunks",
            collection_name_v2="article_chunks_v2",
        )
        store = QdrantStore(settings)
        store._client = MagicMock()
        store._client.upsert = AsyncMock()

        # 1. Upsert 768-dim point -> should route to article_chunks_v2
        point_768 = [VectorPoint(id="p1", vector=[0.1] * 768, payload={"headline": "Gemini 768"})]
        await store.upsert(point_768)

        store._client.upsert.assert_called_once()
        call_kwargs = store._client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "article_chunks_v2"

        # 2. Upsert 1024-dim point -> should route to article_chunks
        store._client.upsert.reset_mock()
        point_1024 = [VectorPoint(id="p2", vector=[0.2] * 1024, payload={"headline": "BGE 1024"})]
        await store.upsert(point_1024)

        store._client.upsert.assert_called_once()
        call_kwargs = store._client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "article_chunks"

    @pytest.mark.asyncio
    async def test_search_auto_routing_by_dim(self) -> None:
        settings = QdrantSettings(
            collection_name="article_chunks",
            collection_name_v2="article_chunks_v2",
        )
        store = QdrantStore(settings)
        store._client = MagicMock()

        mock_points_res = MagicMock()
        mock_points_res.points = []
        store._client.query_points = AsyncMock(return_value=mock_points_res)

        # 1. Search with 768d query vector -> routes to article_chunks_v2
        await store.search(query_vector=[0.1] * 768, top_k=5)
        store._client.query_points.assert_called_once()
        assert store._client.query_points.call_args[1]["collection_name"] == "article_chunks_v2"

        # 2. Search with 1024d query vector -> routes to article_chunks
        store._client.query_points.reset_mock()
        await store.search(query_vector=[0.2] * 1024, top_k=5)
        store._client.query_points.assert_called_once()
        assert store._client.query_points.call_args[1]["collection_name"] == "article_chunks"

    @pytest.mark.asyncio
    async def test_delete_by_filter_cross_collection(self) -> None:
        settings = QdrantSettings(
            collection_name="article_chunks",
            collection_name_v2="article_chunks_v2",
        )
        store = QdrantStore(settings)
        store._client = MagicMock()
        store._client.delete = AsyncMock()

        # Delete by filter without collection specified should delete from BOTH collections
        await store.delete_by_filter({"issue_id": 42})

        assert store._client.delete.call_count == 2
        collections_deleted = [call[1]["collection_name"] for call in store._client.delete.call_args_list]
        assert "article_chunks" in collections_deleted
        assert "article_chunks_v2" in collections_deleted
