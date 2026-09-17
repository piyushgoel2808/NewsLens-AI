"""Google Cloud Storage (GCS) object store implementation.

Implements the ObjectStore protocol for production GCP deployments.
Wraps the synchronous google-cloud-storage Python client in asyncio executors
so it is safe to call from async FastAPI/Celery code without blocking the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from typing import Any, TypeVar

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class GoogleCloudStorageStore:
    """ObjectStore backed by Google Cloud Storage."""

    def __init__(self, settings: Settings) -> None:
        from google.cloud import storage

        self._settings = settings
        project = settings.gcs_project_id or settings.gcp_project_id
        if project:
            self._client = storage.Client(project=project)
        else:
            self._client = storage.Client()
        logger.info(
            "Initialized GoogleCloudStorageStore",
            extra={"project": self._client.project},
        )

    async def _run(self, fn: Callable[[], T]) -> T:
        """Run a synchronous GCS call in the default thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)

    async def ensure_bucket(self, bucket: str) -> None:
        """Create the bucket if it does not already exist."""

        def _ensure() -> None:
            try:
                b = self._client.bucket(bucket)
                if not b.exists():
                    self._client.create_bucket(b, location="asia-south1")
                    logger.info("Created GCS bucket", extra={"bucket": bucket})
            except Exception as e:
                logger.warning(
                    "ensure_bucket check failed (may already exist or insufficient permissions to create)",
                    extra={"bucket": bucket, "error": str(e)},
                )

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
            b = self._client.bucket(bucket)
            blob = b.blob(key)
            blob.upload_from_string(data, content_type=content_type)

        await self._run(_put)

    async def get(self, bucket: str, key: str) -> bytes:
        """Download an object and return its bytes. Returns b'' if key does not exist."""

        def _get() -> bytes:
            try:
                b = self._client.bucket(bucket)
                blob = b.blob(key)
                if not blob.exists():
                    logger.warning("Object not found in GCS", extra={"bucket": bucket, "key": key})
                    return b""
                return blob.download_as_bytes()
            except Exception as e:
                logger.warning("Failed to download object from GCS", extra={"bucket": bucket, "key": key, "error": str(e)})
                return b""

        return await self._run(_get)

    async def delete(self, bucket: str, key: str) -> None:
        """Delete an object."""

        def _delete() -> None:
            b = self._client.bucket(bucket)
            blob = b.blob(key)
            if blob.exists():
                blob.delete()

        await self._run(_delete)

    async def delete_prefix(self, bucket: str, prefix: str) -> int:
        """Delete all objects with given prefix (e.g. 'issues/42/')."""

        def _delete_prefix() -> int:
            b = self._client.bucket(bucket)
            blobs = list(self._client.list_blobs(bucket, prefix=prefix))
            count = len(blobs)
            if count > 0:
                b.delete_blobs(blobs)
            return count

        return await self._run(_delete_prefix)

    async def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """List object keys in a bucket matching prefix."""

        def _list() -> list[str]:
            b = self._client.bucket(bucket)
            return [blob.name for blob in b.list_blobs(prefix=prefix)]

        return await self._run(_list)

    async def presign_url(
        self,
        bucket: str,
        key: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Generate a pre-signed GET URL."""

        def _presign() -> str:
            b = self._client.bucket(bucket)
            blob = b.blob(key)
            try:
                return blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(seconds=expires_seconds),
                    method="GET",
                )
            except Exception as e:
                logger.warning(
                    "Signed URL generation failed; falling back to direct media link",
                    extra={"error": str(e), "bucket": bucket, "key": key},
                )
                return f"https://storage.googleapis.com/{bucket}/{key}"

        return await self._run(_presign)

    async def exists(self, bucket: str, key: str) -> bool:
        """Return True if the object exists."""

        def _exists() -> bool:
            try:
                b = self._client.bucket(bucket)
                blob = b.blob(key)
                return blob.exists()
            except Exception:
                return False

        return await self._run(_exists)

    async def ping(self) -> bool:
        """Return True if GCS is reachable."""

        def _ping() -> bool:
            try:
                test_bucket = self._settings.gcs_bucket_pages or "newslens-pages"
                b = self._client.bucket(test_bucket)
                return b.exists()
            except Exception:
                return False

        return await self._run(_ping)

    async def startup(self) -> None:
        """Ensure all required buckets exist. Call at app startup."""
        pages = self._settings.gcs_bucket_pages
        orig = self._settings.gcs_bucket_originals
        if pages:
            await self.ensure_bucket(pages)
        if orig:
            await self.ensure_bucket(orig)
        logger.info("GCS buckets verified", extra={"pages": pages, "originals": orig})
