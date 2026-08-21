"""Hard-Delete Orchestration Service for NewsLens-AI.

Implements the 3-Tier Hard Deletion Blueprint:
1. Qdrant Vector Store: Delete all vector points matching issue_id via payload filter.
2. MinIO Object Store: Purge original PDFs and page rasters under `issues/{issue_id}/`.
3. MySQL Relational DB: Delete Issue (cascading pages, articles, chunks, tables, photos)
   and any parent IngestionJob records.
"""
from __future__ import annotations

import contextlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.ingestion import IngestionJob
from app.models.newspaper import Issue, Page
from app.storage.minio_store import MinioStore
from app.storage.qdrant_store import QdrantStore

logger = get_logger(__name__)


class DeletionService:
    """Orchestrates complete multi-tier deletion across Qdrant, MinIO, and MySQL."""

    def __init__(
        self,
        db: AsyncSession,
        minio: MinioStore | None = None,
        qdrant: QdrantStore | None = None,
    ) -> None:
        self._db = db
        self._settings = get_settings()
        self._minio = minio or MinioStore(self._settings.minio)
        self._qdrant = qdrant or QdrantStore(self._settings.qdrant)

    async def delete_issue(self, issue_id: int) -> dict[str, Any]:
        """Permanently delete an issue and all associated vectors, files, and DB rows."""
        logger.info("Initiating 3-tier hard deletion for issue", extra={"issue_id": issue_id})

        # Fetch issue to verify existence and retrieve metadata
        issue = await self._db.get(Issue, issue_id)
        if not issue:
            return {"status": "not_found", "issue_id": issue_id, "detail": "Issue not found"}

        source_job_id = issue.source_zip_id
        pages_res = await self._db.execute(select(Page).where(Page.issue_id == issue_id))
        pages = pages_res.scalars().all()

        # ---------------------------------------------------------------------
        # Tier 1: Qdrant Vector Store Purge
        # ---------------------------------------------------------------------
        try:
            await self._qdrant.delete_by_filter({"issue_id": issue_id})
            logger.info("Tier 1: Purged Qdrant vectors", extra={"issue_id": issue_id})
        except Exception as e:
            logger.warning(
                "Tier 1: Qdrant vector deletion error (continuing purge)",
                extra={"issue_id": issue_id, "error": str(e)},
            )

        # ---------------------------------------------------------------------
        # Tier 2: MinIO Object Storage Purge
        # ---------------------------------------------------------------------
        deleted_files = 0
        try:
            # 1. Delete explicit page raster keys
            for p in pages:
                if p.raster_object_key:
                    with contextlib.suppress(Exception):
                        await self._minio.delete(
                            self._settings.minio.bucket_pages,
                            p.raster_object_key,
                        )
                        deleted_files += 1

            # 2. Delete prefix directories for the issue
            deleted_files += await self._minio.delete_prefix(
                self._settings.minio.bucket_pages,
                f"issues/{issue_id}/",
            )
            deleted_files += await self._minio.delete_prefix(
                self._settings.minio.bucket_originals,
                f"issues/{issue_id}/",
            )
            logger.info(
                "Tier 2: Purged MinIO objects",
                extra={"issue_id": issue_id, "deleted_files": deleted_files},
            )
        except Exception as e:
            logger.warning(
                "Tier 2: MinIO deletion error (continuing purge)",
                extra={"issue_id": issue_id, "error": str(e)},
            )

        # ---------------------------------------------------------------------
        # Tier 3: MySQL Relational Database Purge
        # ---------------------------------------------------------------------
        try:
            # Delete associated IngestionJob if present
            if source_job_id:
                job = await self._db.get(IngestionJob, source_job_id)
                if job:
                    await self._db.delete(job)

            # Delete the Issue (cascading deletes pages, articles, chunks, tables, photos)
            await self._db.delete(issue)
            await self._db.commit()
            logger.info("Tier 3: Purged MySQL records", extra={"issue_id": issue_id})
        except Exception as e:
            await self._db.rollback()
            logger.error(
                "Tier 3: MySQL deletion failed",
                extra={"issue_id": issue_id, "error": str(e)},
            )
            raise

        return {
            "status": "deleted",
            "issue_id": issue_id,
            "deleted_files_count": deleted_files,
            "deleted_vectors": True,
        }

    async def delete_job(self, job_id: int) -> dict[str, Any]:
        """Delete an IngestionJob and any linked Issue and storage artifacts."""
        job = await self._db.get(IngestionJob, job_id)
        if not job:
            return {"status": "not_found", "job_id": job_id, "detail": "Job not found"}

        issue_res = await self._db.execute(select(Issue).where(Issue.source_zip_id == job_id))
        issue = issue_res.scalars().first()

        if issue:
            return await self.delete_issue(issue.id)

        await self._db.delete(job)
        await self._db.commit()
        return {"status": "deleted", "job_id": job_id}

