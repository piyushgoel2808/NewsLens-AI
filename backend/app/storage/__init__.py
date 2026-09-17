"""Storage module for NewsLens-AI."""

from app.storage.factory import get_object_store
from app.storage.minio_store import MinioStore

__all__ = ["get_object_store", "MinioStore"]
