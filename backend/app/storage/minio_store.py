"""MinIO object store implementation.

Wraps the synchronous MinIO Python client in asyncio executors so it's
safe to call from async FastAPI/Celery code without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable
from datetime import timedelta
from typing import TypeVar

from minio import Minio
from minio.error import S3Error

from app.core.config import MinioSettings
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class MinioStore:
    """ObjectStore backed by MinIO / S3-compatible storage."""

    def __init__(self, settings: MinioSettings) -> None:
        self._settings = settings
        self._client = Minio(
            endpoint=settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
        )

    async def _run(self, fn: Callable[[], T]) -> T:
        """Run a synchronous MinIO call in the default thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)

    async def ensure_bucket(self, bucket: str) -> None:
        """Create the bucket if it does not already exist."""

        def _ensure() -> None:
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info("Created MinIO bucket", extra={"bucket": bucket})

        await self._run(_ensure)

    async def put(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload bytes as an object."""

        def _put() -> None:
            self._client.put_object(
                bucket_name=bucket,
                object_name=key,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )

        await self._run(_put)

    async def get(self, bucket: str, key: str) -> bytes:
        """Download an object and return its bytes."""

        def _get() -> bytes:
            response = self._client.get_object(bucket_name=bucket, object_name=key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        return await self._run(_get)

    async def delete(self, bucket: str, key: str) -> None:
        """Delete an object."""
        await self._run(lambda: self._client.remove_object(bucket, key))

    async def delete_prefix(self, bucket: str, prefix: str) -> int:
        """Delete all objects with given prefix (e.g. 'issues/42/')."""

        def _delete_prefix() -> int:
            objects = self._client.list_objects(bucket, prefix=prefix, recursive=True)
            count = 0
            for obj in objects:
                self._client.remove_object(bucket, obj.object_name)
                count += 1
            return count

        return await self._run(_delete_prefix)

    async def presign_url(
        self,
        bucket: str,
        key: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Generate a pre-signed GET URL."""

        def _presign() -> str:
            return self._client.presigned_get_object(
                bucket_name=bucket,
                object_name=key,
                expires=timedelta(seconds=expires_seconds),
            )

        return await self._run(_presign)

    async def exists(self, bucket: str, key: str) -> bool:
        """Return True if the object exists."""

        def _exists() -> bool:
            try:
                self._client.stat_object(bucket, key)
                return True
            except S3Error as e:
                if e.code in ("NoSuchKey", "NoSuchBucket"):
                    return False
                raise

        return await self._run(_exists)

    async def ping(self) -> bool:
        """Return True if MinIO is reachable."""
        try:
            await self._run(self._client.list_buckets)
            return True
        except Exception:
            return False

    async def startup(self) -> None:
        """Ensure all required buckets exist. Call at app startup."""
        await self.ensure_bucket(self._settings.bucket_pages)
        await self.ensure_bucket(self._settings.bucket_originals)
        logger.info("MinIO buckets ready")
