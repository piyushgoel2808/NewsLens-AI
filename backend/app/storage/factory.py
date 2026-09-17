"""Storage factory for NewsLens-AI.

Instantiates either MinioStore (for local development) or GoogleCloudStorageStore
(for production GCP deployments) based on Settings.storage_backend.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.storage.base import ObjectStore


def get_object_store(settings: Settings | None = None) -> ObjectStore:
    """Return the configured ObjectStore implementation."""
    s = settings or get_settings()
    backend = (getattr(s, "storage_backend", "minio") or "minio").lower()

    if backend == "gcs":
        from app.storage.gcs_store import GoogleCloudStorageStore

        return GoogleCloudStorageStore(s)

    from app.storage.minio_store import MinioStore

    return MinioStore(s.minio)
