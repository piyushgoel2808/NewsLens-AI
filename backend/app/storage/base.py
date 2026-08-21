"""Storage interface abstractions for NewsLens-AI.

Three backends:
- VectorStore  — semantic vector search (Qdrant)
- ObjectStore  — binary blob storage (MinIO / S3)
- SearchIndex  — full-text / lexical search (MySQL FULLTEXT or OpenSearch)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class VectorPoint:
    """A vector + payload to upsert into the vector store."""

    id: str               # UUID string (matches article_chunks.embedding_vector_id)
    vector: list[float]
    payload: dict[str, Any]  # {article_id, newspaper_id, issue_date, section, language, ...}


@dataclass
class VectorSearchResult:
    """A single result from a vector similarity search."""

    id: str            # Qdrant point ID
    score: float       # Cosine similarity (0-1)
    payload: dict[str, Any]
    article_id: int | None = None
    chunk_index: int | None = None


@dataclass
class FullTextSearchResult:
    """A single result from full-text search."""

    article_id: int
    headline: str | None
    snippet: str  # Matched text excerpt (first 200 chars)
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Interface for a vector similarity search store."""

    async def upsert(self, points: list[VectorPoint]) -> None:
        """Upsert a batch of vectors with their payloads."""
        ...

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Find the top_k most similar vectors.

        Args:
            query_vector: The query embedding.
            top_k: Number of results to return.
            filters: Payload filters, e.g. {\"newspaper_id\": 1}.
            score_threshold: Minimum similarity score to include.
        """
        ...

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by their IDs."""
        ...

    async def collection_info(self) -> dict[str, Any]:
        """Return metadata about the collection (point count, vector size, etc.)."""
        ...


@runtime_checkable
class ObjectStore(Protocol):
    """Interface for binary object storage (S3-compatible)."""

    async def put(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload an object."""
        ...

    async def get(self, bucket: str, key: str) -> bytes:
        """Download an object."""
        ...

    async def delete(self, bucket: str, key: str) -> None:
        """Delete an object."""
        ...

    async def presign_url(
        self,
        bucket: str,
        key: str,
        expires_seconds: int = 3600,
    ) -> str:
        """Generate a pre-signed URL for temporary public access."""
        ...

    async def exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists without downloading it."""
        ...

    async def ensure_bucket(self, bucket: str) -> None:
        """Create the bucket if it doesn't exist."""
        ...


@runtime_checkable
class SearchIndex(Protocol):
    """Interface for full-text / lexical search."""

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[FullTextSearchResult]:
        """Run a full-text search query."""
        ...
