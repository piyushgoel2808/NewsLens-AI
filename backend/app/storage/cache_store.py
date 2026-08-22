"""Redis-backed Semantic & Exact Query Cache Store with Graceful Degradation.

Provides deterministic SHA-256 key hashing for queries, filters, and embedding vectors.
All cache lookups and storage operations degrade gracefully on Redis timeouts or connection
failures without crashing ongoing agent queries.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import record_cache_event

logger = get_logger(__name__)


def compute_query_cache_key(
    query: str,
    model_id: str = "",
    date_filters: str = "",
    issue_ids: list[int] | None = None,
) -> str:
    """Compute a deterministic SHA-256 cache key from normalized inputs."""
    norm_query = " ".join(query.strip().lower().split())
    norm_model = (model_id or "").strip().lower()
    norm_dates = (date_filters or "").strip().lower()
    sorted_issue_ids = sorted(issue_ids) if issue_ids else []

    composite = json.dumps(
        {
            "q": norm_query,
            "m": norm_model,
            "d": norm_dates,
            "i": sorted_issue_ids,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()
    return f"newslens:query:{digest}"


def compute_embedding_cache_key(text: str, model: str = "") -> str:
    """Compute deterministic SHA-256 key for an embedding text snippet."""
    norm_text = text.strip()
    composite = f"{norm_text}::{model.strip().lower()}"
    digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()
    return f"newslens:embed:{digest}"


class CacheStore:
    """Async Redis cache with safe fallback and graceful degradation."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or get_settings().redis_url
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Lazily initialize and return the async Redis connection."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.Redis.from_url(
                    self._redis_url,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                    decode_responses=False,
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize Redis client; cache disabled",
                    extra={"error": str(e)},
                )
                self._client = None
        return self._client

    async def get_query(self, cache_key: str) -> dict[str, Any] | None:
        """Fetch cached agent query answer safely. Returns None on cache miss or Redis error."""
        try:
            client = await self._get_client()
            if client is None:
                record_cache_event("query", "miss")
                return None

            data_bytes = await client.get(cache_key)
            if not data_bytes:
                record_cache_event("query", "miss")
                return None

            payload: dict[str, Any] = json.loads(data_bytes.decode("utf-8"))
            record_cache_event("query", "hit")
            return payload
        except Exception as e:
            logger.warning(
                "Redis get_query encountered an error; bypassing cache gracefully",
                extra={"key": cache_key, "error": str(e)},
            )
            record_cache_event("query", "error")
            return None

    async def set_query(
        self,
        cache_key: str,
        payload: dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> bool:
        """Store synthesized query result in cache safely."""
        try:
            client = await self._get_client()
            if client is None:
                return False

            data_str = json.dumps(payload)
            await client.set(cache_key, data_str.encode("utf-8"), ex=ttl_seconds)
            record_cache_event("query", "set")
            return True
        except Exception as e:
            logger.warning(
                "Redis set_query encountered an error; skipping cache store",
                extra={"key": cache_key, "error": str(e)},
            )
            record_cache_event("query", "error")
            return False

    async def get_embedding(self, text: str, model: str = "") -> list[float] | None:
        """Retrieve cached text embedding vector."""
        key = compute_embedding_cache_key(text, model)
        try:
            client = await self._get_client()
            if client is None:
                record_cache_event("embedding", "miss")
                return None

            data_bytes = await client.get(key)
            if not data_bytes:
                record_cache_event("embedding", "miss")
                return None

            vector: list[float] = json.loads(data_bytes.decode("utf-8"))
            record_cache_event("embedding", "hit")
            return vector
        except Exception as e:
            logger.warning(
                "Redis get_embedding failed; bypassing cache",
                extra={"key": key, "error": str(e)},
            )
            record_cache_event("embedding", "error")
            return None

    async def set_embedding(
        self,
        text: str,
        vector: list[float],
        model: str = "",
        ttl_seconds: int = 86400,
    ) -> bool:
        """Store embedding vector in cache."""
        key = compute_embedding_cache_key(text, model)
        try:
            client = await self._get_client()
            if client is None:
                return False

            data_str = json.dumps(vector)
            await client.set(key, data_str.encode("utf-8"), ex=ttl_seconds)
            record_cache_event("embedding", "set")
            return True
        except Exception as e:
            logger.warning(
                "Redis set_embedding failed; skipping cache",
                extra={"key": key, "error": str(e)},
            )
            record_cache_event("embedding", "error")
            return False

    async def close(self) -> None:
        """Close active Redis connections."""
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None
