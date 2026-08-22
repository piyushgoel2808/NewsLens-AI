"""Tests for Redis Cache Store with deterministic hashing and graceful degradation."""

from __future__ import annotations

import pytest

from app.storage.cache_store import (
    CacheStore,
    compute_embedding_cache_key,
    compute_query_cache_key,
)


def test_deterministic_query_cache_key() -> None:
    """Verify identical queries with different spacing/casing produce identical keys."""
    key1 = compute_query_cache_key(
        query="  What are the key policy decisions?  ",
        model_id="GROQ_LLAMA",
        date_filters="2026-08-20",
        issue_ids=[1, 2],
    )
    key2 = compute_query_cache_key(
        query="what are the key policy decisions?",
        model_id="groq_llama",
        date_filters="2026-08-20",
        issue_ids=[2, 1],  # different order
    )
    key3 = compute_query_cache_key(
        query="What are the key policy decisions?",
        model_id="groq_llama",
        date_filters="2026-08-22",  # different date
    )

    assert key1 == key2
    assert key1.startswith("newslens:query:")
    assert key1 != key3


def test_deterministic_embedding_cache_key() -> None:
    """Verify embedding cache key generation."""
    k1 = compute_embedding_cache_key("  inflation target  ", model="bge-m3")
    k2 = compute_embedding_cache_key("inflation target", model="BGE-M3")
    assert k1 == k2
    assert k1.startswith("newslens:embed:")


@pytest.mark.asyncio
async def test_cache_store_graceful_degradation_on_unreachable_redis() -> None:
    """Verify CacheStore degrades gracefully and does not throw exceptions when Redis is down."""
    # Point to invalid port to simulate Redis being unreachable
    store = CacheStore(redis_url="redis://localhost:9999/0")

    # get_query should return None safely without raising
    res = await store.get_query("newslens:query:test_non_existent")
    assert res is None

    # set_query should return False safely without raising
    saved = await store.set_query("newslens:query:test_key", {"answer": "Sample"}, ttl_seconds=10)
    assert saved is False

    # get_embedding and set_embedding should also degrade safely
    embed = await store.get_embedding("test text")
    assert embed is None

    embed_saved = await store.set_embedding("test text", [0.1, 0.2, 0.3])
    assert embed_saved is False

    await store.close()
