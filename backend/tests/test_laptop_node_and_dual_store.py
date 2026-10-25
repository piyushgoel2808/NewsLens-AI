"""Unit & integration tests for Sovereign Laptop Node, Dual Object Storage, and EmbeddingGemma 2."""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.api.routers.settings import get_laptop_node_runtime_state
from app.core.config import QdrantSettings, get_settings
from app.providers.base import ProviderError
from app.providers.embedding_gemma_provider import EmbeddingGemma2Provider, _normalize_l2
from app.providers.ollama_provider import OllamaProvider
from app.providers.registry import ModelRegistry
from app.retrieval.reranker import CrossEncoderReranker
from app.storage.dual_store import DualSyncObjectStore
from app.storage.qdrant_store import QdrantStore


# ---------------------------------------------------------------------------
# Test 1: DualSyncObjectStore writes to both and reads local-first
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dual_sync_object_store_writes_both_and_reads_local_first() -> None:
    local_mock = AsyncMock()
    cloud_mock = AsyncMock()

    local_mock.put = AsyncMock(return_value="page_1.png")
    cloud_mock.put = AsyncMock(return_value="page_1.png")
    local_mock.get = AsyncMock(return_value=b"local-image-data")
    cloud_mock.get = AsyncMock(return_value=b"cloud-fallback-data")
    local_mock.delete_prefix = AsyncMock(return_value=5)
    cloud_mock.delete_prefix = AsyncMock(return_value=5)

    dual_store = DualSyncObjectStore(primary_local=local_mock, secondary_cloud=cloud_mock)

    # 1. Put writes to both stores concurrently
    res_key = await dual_store.put("pages", "page_1.png", b"test-data")
    assert res_key == "page_1.png"
    assert local_mock.put.called
    assert cloud_mock.put.called

    # 2. Local-first read returns local data
    data = await dual_store.get("pages", "page_1.png")
    assert data == b"local-image-data"
    assert not cloud_mock.get.called

    # 3. Fallback to cloud if local fails or misses
    local_mock.get = AsyncMock(side_effect=Exception("Local MinIO Miss"))
    fallback_data = await dual_store.get("pages", "page_1.png")
    assert fallback_data == b"cloud-fallback-data"
    assert cloud_mock.get.called

    # 4. delete_prefix calls both stores
    del_count = await dual_store.delete_prefix("pages", "issue_1/")
    assert del_count == 5

    # 5. put raises if both fail
    local_mock.put = AsyncMock(side_effect=Exception("Local MinIO Down"))
    cloud_mock.put = AsyncMock(side_effect=Exception("Cloud R2 Down"))
    with pytest.raises(RuntimeError):
        await dual_store.put("pages", "fail.png", b"fail")

    # 5b. put succeeds in degraded mode (local fail, cloud success) and purges local copy
    local_mock.put = AsyncMock(side_effect=Exception("Local MinIO Full"))
    cloud_mock.put = AsyncMock(return_value="degraded.png")
    local_mock.delete = AsyncMock()
    degraded_key = await dual_store.put("pages", "degraded.png", b"degraded-data")
    assert degraded_key == "degraded.png"
    assert local_mock.delete.called

    # 6. list_objects returns deduplicated union from both tiers
    local_mock.list_objects = AsyncMock(return_value=["page_1.png", "page_2.png"])
    cloud_mock.list_objects = AsyncMock(return_value=["page_2.png", "page_3.png"])
    objs = await dual_store.list_objects("pages")
    assert objs == ["page_1.png", "page_2.png", "page_3.png"]

    # 7. presign_url checks cloud existence first
    cloud_mock.exists = AsyncMock(return_value=True)
    cloud_mock.presign_url = AsyncMock(return_value="https://r2.cloud.com/signed.png")
    presigned = await dual_store.presign_url("pages", "signed.png")
    assert presigned == "https://r2.cloud.com/signed.png"


# ---------------------------------------------------------------------------
# Test 2: EmbeddingGemma2Provider 768d, Asymmetric Prefixes & Qdrant v2
# ---------------------------------------------------------------------------
def test_embedding_gemma2_provider_768d_and_asymmetric_prefixes() -> None:
    provider = EmbeddingGemma2Provider(dimensions=768)

    # 1. Check asymmetric retrieval prefixes
    query_fmt = provider._format_text("monetary policy rate", task_type="RETRIEVAL_QUERY")
    assert query_fmt == "task: search result | query: monetary policy rate"

    doc_fmt = provider._format_text("The Reserve Bank paused rates.", task_type="RETRIEVAL_DOCUMENT")
    assert doc_fmt == "title: broadsheet article | text: The Reserve Bank paused rates."

    # 2. Check Matryoshka truncation and L2 normalization
    raw_vec = [1.0] * 1024
    norm_vec = _normalize_l2(raw_vec, target_dim=768)
    assert len(norm_vec) == 768
    norm = math.sqrt(sum(x * x for x in norm_vec))
    assert pytest.approx(norm, 1e-4) == 1.0

    # 3. Verify QdrantStore collection routing (768d -> article_chunks_v2)
    qdrant_settings = QdrantSettings(
        collection_name="article_chunks",
        collection_name_v2="article_chunks_v2",
    )
    store = QdrantStore(settings=qdrant_settings, embedding_dim=1024)
    assert store.get_collection_name(1024) == "article_chunks"
    assert store.get_collection_name(768) == "article_chunks_v2"


# ---------------------------------------------------------------------------
# Test 3: Laptop Node Toggle requires Admin PIN
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_laptop_node_toggle_requires_admin_pin() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Missing or incorrect PIN returns 403 Forbidden
        bad_resp = await client.post(
            "/api/settings/laptop-node/toggle",
            json={"enabled": True},
            headers={"X-Admin-Pin": "wrong-pin"},
        )
        assert bad_resp.status_code == 403

        # Valid PIN (1234) succeeds
        good_resp = await client.post(
            "/api/settings/laptop-node/toggle",
            json={"enabled": True, "tunnel_url": "http://localhost:11434"},
            headers={"X-Admin-Pin": "1234"},
        )
        assert good_resp.status_code == 200
        data = good_resp.json()
        assert data["enabled"] is True

        # Toggle OFF succeeds
        off_resp = await client.post(
            "/api/settings/laptop-node/toggle",
            json={"enabled": False},
            headers={"X-Admin-Pin": "1234"},
        )
        assert off_resp.status_code == 200
        assert off_resp.json()["enabled"] is False


# ---------------------------------------------------------------------------
# Test 4: Laptop Node Status discovers local Ollama models
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_laptop_node_status_discovers_local_ollama_models() -> None:
    state = get_laptop_node_runtime_state()
    state["enabled"] = True
    state["tunnel_url"] = "http://localhost:11434"

    mock_tags_response = MagicMock()
    mock_tags_response.status_code = 200
    mock_tags_response.json.return_value = {
        "models": [
            {"name": "llama3.1:8b"},
            {"name": "deepseek-r1:14b"},
            {"name": "qwen3-vl:latest"},
            {"name": "embeddinggemma-2"},
        ]
    }

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get = AsyncMock(return_value=mock_tags_response)

    app = create_app()
    transport = ASGITransport(app=app)
    with patch("app.api.routers.settings.httpx.AsyncClient", return_value=mock_client):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/settings/laptop-node/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["enabled"] is True
            assert data["online"] is True
            assert data["tunnel_url"] is None  # Unauthenticated hides tunnel_url
            assert "llama3.1:8b" in data["available_models"]
            assert "deepseek-r1:14b" in data["available_models"]

            # Authenticated admin status reveals tunnel_url
            admin_resp = await client.get("/api/settings/laptop-node/status", headers={"X-Admin-Pin": "1234"})
            assert admin_resp.status_code == 200
            admin_data = admin_resp.json()
            assert admin_data["tunnel_url"] == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Test 5: Laptop model override routing and semaphore protection
# ---------------------------------------------------------------------------
def test_laptop_model_override_routing_and_semaphore() -> None:
    settings = get_settings()
    registry = ModelRegistry(settings)

    state = get_laptop_node_runtime_state()
    state["enabled"] = True
    state["tunnel_url"] = "http://localhost:11434"

    # Resolving laptop:deepseek-r1:14b returns an OllamaProvider with that model
    provider = registry.get_chat_provider("laptop:deepseek-r1:14b")
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "deepseek-r1:14b"
    assert provider._is_laptop_node is True

    # When laptop node is turned OFF, raises ProviderError
    state["enabled"] = False
    with pytest.raises(ProviderError, match="turned OFF"):
        registry.get_chat_provider("laptop:deepseek-r1:14b")


# ---------------------------------------------------------------------------
# Test 6: Render Free Tier skips heavy PyTorch pre-warming and reranker
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_render_free_tier_skips_heavy_pytorch_prewarm() -> None:
    reranker = CrossEncoderReranker()

    candidates = [
        {"headline": "Market Rally", "snippet": "Sensex surges 500 points", "rrf_score": 0.8},
        {"headline": "Weather Report", "snippet": "Rain expected in Mumbai", "rrf_score": 0.4},
    ]

    with patch.object(get_settings(), "render_free_tier", True):
        # 1. Loading model synchronously returns None immediately (0 MB RAM)
        model = reranker._load_model_sync()
        assert model is None

        # 2. Reranker executes pure Python HeuristicReranker
        results = await reranker.rerank("sensex rally", candidates, top_k=2)
        assert len(results) == 2
        assert results[0]["headline"] == "Market Rally"
        assert "rerank_score" in results[0]
