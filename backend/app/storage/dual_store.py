"""Dual-Sync Object Store: Local Docker MinIO + Free Cloud S3 (Cloudflare R2)."""
from __future__ import annotations

import asyncio
import contextlib

from app.core.logging import get_logger
from app.storage.base import ObjectStore
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)


class DualSyncObjectStore(ObjectStore):
    """Writes to both Local Docker MinIO and Cloud S3 (R2); reads local-first with cloud fallback."""

    def __init__(self, primary_local: MinioStore, secondary_cloud: MinioStore) -> None:
        self._local = primary_local
        self._cloud = secondary_cloud

    async def startup(self) -> None:
        results = await asyncio.gather(
            self._local.startup(),
            self._cloud.startup(),
            return_exceptions=True,
        )
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                tier = "local_minio" if idx == 0 else "cloud_r2"
                logger.warning(
                    "DualSyncObjectStore startup error",
                    extra={"tier": tier, "error": str(res)},
                )

    async def ping(self) -> bool:
        results = await asyncio.gather(
            self._local.ping(),
            self._cloud.ping(),
            return_exceptions=True,
        )
        return any(r is True for r in results)

    async def ensure_bucket(self, bucket: str) -> None:
        results = await asyncio.gather(
            self._local.ensure_bucket(bucket),
            self._cloud.ensure_bucket(bucket),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) == len(results):
            raise RuntimeError(f"DualSyncObjectStore failed to ensure bucket '{bucket}' on all tiers")
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                tier = "local_minio" if idx == 0 else "cloud_r2"
                logger.warning(
                    "DualSyncObjectStore ensure_bucket error",
                    extra={"tier": tier, "bucket": bucket, "error": str(res)},
                )

    async def put(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        local_res, cloud_res = await asyncio.gather(
            self._local.put(bucket, key, data, content_type),
            self._cloud.put(bucket, key, data, content_type),
            return_exceptions=True,
        )
        local_failed = isinstance(local_res, Exception)
        cloud_failed = isinstance(cloud_res, Exception)

        if local_failed and cloud_failed:
            raise RuntimeError(f"DualSyncObjectStore put failed on all tiers for {bucket}/{key}")

        # If local write failed but cloud succeeded, remove any stale local copy
        # so subsequent get() calls cannot serve outdated data
        if local_failed:
            logger.warning(
                "DualSyncObjectStore put failed on local tier (cloud succeeded); purging stale local copy",
                extra={"key": key, "error": str(local_res)},
            )
            with contextlib.suppress(Exception):
                await self._local.delete(bucket, key)

        if cloud_failed:
            logger.warning(
                "DualSyncObjectStore put failed on cloud tier (local succeeded); purging stale cloud copy",
                extra={"key": key, "error": str(cloud_res)},
            )
            with contextlib.suppress(Exception):
                await self._cloud.delete(bucket, key)
        return key

    async def get(self, bucket: str, key: str) -> bytes:
        # 1. Try fast local Docker MinIO first
        try:
            data = await self._local.get(bucket, key)
            if data is not None:
                return data
        except Exception:
            pass
        # 2. Fallback to Cloud S3 (Cloudflare R2)
        return await self._cloud.get(bucket, key)

    async def exists(self, bucket: str, key: str) -> bool:
        try:
            if await self._local.exists(bucket, key):
                return True
        except Exception:
            pass
        return await self._cloud.exists(bucket, key)

    async def delete(self, bucket: str, key: str) -> None:
        results = await asyncio.gather(
            self._local.delete(bucket, key),
            self._cloud.delete(bucket, key),
            return_exceptions=True,
        )
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                tier = "local_minio" if idx == 0 else "cloud_r2"
                target = self._local if idx == 0 else self._cloud
                logger.warning(
                    "DualSyncObjectStore delete failure on tier; retrying once",
                    extra={"tier": tier, "key": key, "error": str(res)},
                )
                try:
                    await target.delete(bucket, key)
                except Exception as retry_err:
                    logger.error(
                        "DualSyncObjectStore delete retry failed; tier remains out of sync",
                        extra={"tier": tier, "key": key, "error": str(retry_err)},
                    )
                    raise RuntimeError(
                        f"DualSyncObjectStore delete failed on tier {tier} for {bucket}/{key}: {retry_err}"
                    ) from retry_err

    async def delete_prefix(self, bucket: str, prefix: str) -> int:
        results = await asyncio.gather(
            self._local.delete_prefix(bucket, prefix),
            self._cloud.delete_prefix(bucket, prefix),
            return_exceptions=True,
        )
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                tier = "local_minio" if idx == 0 else "cloud_r2"
                raise RuntimeError(
                    f"DualSyncObjectStore delete_prefix failed on tier {tier} for {bucket}/{prefix}: {res}"
                ) from res
        counts = [r for r in results if isinstance(r, int)]
        return max(counts) if counts else 0

    async def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        results = await asyncio.gather(
            self._local.list_objects(bucket, prefix),
            self._cloud.list_objects(bucket, prefix),
            return_exceptions=True,
        )
        lists = [r for r in results if isinstance(r, list)]
        if not lists:
            raise RuntimeError(f"DualSyncObjectStore list failed on all tiers for {bucket}/{prefix}")
        return sorted({k for lst in lists for k in lst})

    async def presign_url(self, bucket: str, key: str, expires_seconds: int = 3600) -> str:
        # Check which tier actually holds the object before presigning
        try:
            if await self._cloud.exists(bucket, key):
                return await self._cloud.presign_url(bucket, key, expires_seconds)
        except Exception:
            pass
        return await self._local.presign_url(bucket, key, expires_seconds)
