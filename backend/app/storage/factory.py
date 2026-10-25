"""Storage factory for NewsLens-AI.

Instantiates either MinioStore (for local development) or GoogleCloudStorageStore
(for production GCP deployments) based on Settings.storage_backend.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.storage.base import ObjectStore

logger = get_logger(__name__)


def get_object_store(settings: Settings | None = None) -> ObjectStore:
    """Return the configured ObjectStore implementation."""
    s = settings or get_settings()
    backend = (getattr(s, "storage_backend", "minio") or "minio").lower()

    if backend == "gcs":
        from app.storage.gcs_store import GoogleCloudStorageStore

        return GoogleCloudStorageStore(s)

    from app.storage.minio_store import MinioStore

    if backend == "dual":
        if not getattr(s, "cloud_s3_endpoint", None):
            logger.warning(
                "STORAGE_BACKEND is configured as 'dual' but CLOUD_S3_ENDPOINT is not set. Falling back to local MinioStore."
            )
            return MinioStore(s.minio)

        if not s.cloud_s3_access_key or not s.cloud_s3_secret_key:
            logger.warning(
                "STORAGE_BACKEND is 'dual' with CLOUD_S3_ENDPOINT set, but CLOUD_S3_ACCESS_KEY or CLOUD_S3_SECRET_KEY is empty."
            )

        from app.storage.dual_store import DualSyncObjectStore

        local_store = MinioStore(s.minio)
        cloud_store = MinioStore(
            endpoint=s.cloud_s3_endpoint,
            access_key=s.cloud_s3_access_key or "",
            secret_key=s.cloud_s3_secret_key or "",
            secure=getattr(s, "cloud_s3_secure", True),
        )
        return DualSyncObjectStore(local_store, cloud_store)

    return MinioStore(s.minio)
