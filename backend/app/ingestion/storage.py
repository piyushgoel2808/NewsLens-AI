"""Storage, Compression & Maintenance Subsystem for NewsLens-AI Ingestion.

Consolidates:
1. PDF lossless stream compression and object deduplication (via PyMuPDF).
2. 3-Tier Hard Deletion Orchestrator (Qdrant vectors, MinIO artifacts, MySQL relational models).
3. Debug JSON artifacts and inspection telemetry exporter.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.ingestion import IngestionJob
from app.models.newspaper import Issue, Page
from app.storage import get_object_store
from app.storage.base import ObjectStore
from app.storage.qdrant_store import QdrantStore

logger = get_logger(__name__)

# =============================================================================
# PDF Pre-Ingestion Compression Layer
# =============================================================================


def compress_pdf_bytes(pdf_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    """Compress PDF bytes using PyMuPDF lossless stream deflation and object deduplication.

    Args:
        pdf_bytes: Raw input PDF bytes.

    Returns:
        tuple of (processed_bytes, compression_metadata_dict).
        If compression fails, is not beneficial, or input is invalid, returns the original
        bytes with status explanation in metadata.
    """
    original_size = len(pdf_bytes) if pdf_bytes else 0
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        return pdf_bytes, {
            "original_bytes": original_size,
            "compressed_bytes": original_size,
            "saved_bytes": 0,
            "reduction_pct": 0.0,
            "status": "skipped_not_pdf",
        }

    doc: pymupdf.Document | None = None
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        if doc.is_encrypted:
            return pdf_bytes, {
                "original_bytes": original_size,
                "compressed_bytes": original_size,
                "saved_bytes": 0,
                "reduction_pct": 0.0,
                "status": "skipped_encrypted",
            }

        # Lossless optimization:
        # - garbage=4: merge duplicate objects, unreferenced stream cleaning
        # - clean=True: sanitize content streams
        # - deflate=True: deflate uncompressed streams
        # - deflate_images=True: compress uncompressed image streams
        # - deflate_fonts=True: compress uncompressed font streams
        compressed = doc.tobytes(
            garbage=4,
            deflate=True,
            clean=True,
            deflate_images=True,
            deflate_fonts=True,
        )

        comp_size = len(compressed)
        if comp_size == 0 or comp_size >= original_size:
            return pdf_bytes, {
                "original_bytes": original_size,
                "compressed_bytes": original_size,
                "saved_bytes": 0,
                "reduction_pct": 0.0,
                "status": "skipped_no_reduction",
            }

        saved_bytes = original_size - comp_size
        reduction_pct = round((saved_bytes / original_size) * 100.0, 2)

        return compressed, {
            "original_bytes": original_size,
            "compressed_bytes": comp_size,
            "saved_bytes": saved_bytes,
            "reduction_pct": reduction_pct,
            "status": "compressed",
        }
    except Exception as exc:
        logger.warning(
            "PDF compression failed; falling back to original bytes",
            extra={"error": str(exc), "original_bytes": original_size},
        )
        return pdf_bytes, {
            "original_bytes": original_size,
            "compressed_bytes": original_size,
            "saved_bytes": 0,
            "reduction_pct": 0.0,
            "status": "failed",
            "error": str(exc),
        }
    finally:
        if doc is not None:
            doc.close()


def compress_pdf(pdf_bytes: bytes) -> bytes:
    """Convenience wrapper returning only the compressed PDF bytes."""
    compressed_bytes, _ = compress_pdf_bytes(pdf_bytes)
    return compressed_bytes


# =============================================================================
# 3-Tier Hard Deletion Orchestration Service
# =============================================================================


class DeletionService:
    """Orchestrates complete multi-tier deletion across Qdrant, MinIO, and MySQL."""

    def __init__(
        self,
        db: AsyncSession,
        minio: ObjectStore | None = None,
        qdrant: QdrantStore | None = None,
    ) -> None:
        self._db = db
        self._settings = get_settings()
        self._minio = minio or get_object_store(self._settings)
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
            # Dissociate foreign key to avoid FK constraint conflict/autoflush issues
            issue.source_zip_id = None

            # Delete the Issue (cascading deletes pages, articles, chunks, tables, photos)
            await self._db.delete(issue)

            # Delete associated IngestionJob if present
            if source_job_id:
                job = await self._db.get(IngestionJob, source_job_id)
                if job:
                    await self._db.delete(job)

            await self._db.commit()
            logger.info("Tier 3: Purged MySQL records", extra={"issue_id": issue_id})
        except Exception as e:
            import inspect

            if hasattr(self._db, "rollback"):
                r = self._db.rollback()
                if inspect.isawaitable(r):
                    await r
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


# =============================================================================
# Debug Artifacts & Telemetry Exporter
# =============================================================================


def _slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    clean = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", clean)


class DebugArtifactsExporter:
    """Exports structured debug JSON files for an ingested newspaper issue."""

    def __init__(self, base_output_dir: str | Path = "debug_output") -> None:
        self.base_output_dir = Path(base_output_dir)

    def get_issue_debug_dir(
        self,
        issue_id: int,
        newspaper_name: str,
        issue_date: str,
        edition: str = "morning",
    ) -> Path:
        """Construct the output directory path for an issue."""
        np_slug = _slugify(newspaper_name) or "daily"
        date_slug = _slugify(str(issue_date)) or "unknown_date"
        edition_slug = _slugify(edition) or "default"
        folder_name = f"{np_slug}_{date_slug}_{edition_slug}_issue_{issue_id}"
        target_dir = self.base_output_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def export_issue_artifacts(
        self,
        issue_id: int,
        newspaper_name: str,
        issue_date: str,
        edition: str = "morning",
        page_extractions: list[dict[str, Any]] | None = None,
        rag_chunks: list[dict[str, Any]] | None = None,
        articles: list[dict[str, Any]] | None = None,
        advertisements: list[dict[str, Any]] | None = None,
        summary_metrics: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Export all 5 structured debug JSON files for an issue.

        Returns a dictionary mapping artifact names to their absolute file paths.
        """
        target_dir = self.get_issue_debug_dir(
            issue_id=issue_id,
            newspaper_name=newspaper_name,
            issue_date=issue_date,
            edition=edition,
        )

        exported_files: dict[str, str] = {}

        # 1. OCR Extracted Text
        ocr_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "total_pages": len(page_extractions or []),
            "pages": page_extractions or [],
        }
        ocr_path = target_dir / "ocr_extracted_text.json"
        with ocr_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(ocr_payload), f, indent=2, ensure_ascii=False)
        exported_files["ocr_extracted_text"] = str(ocr_path.resolve())

        # 2. RAG Chunks
        chunks_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "total_chunks": len(rag_chunks or []),
            "chunks": rag_chunks or [],
        }
        chunks_path = target_dir / "rag_chunks.json"
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(chunks_payload), f, indent=2, ensure_ascii=False)
        exported_files["rag_chunks"] = str(chunks_path.resolve())

        # 3. Articles Manifest
        articles_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "total_articles": len(articles or []),
            "articles": articles or [],
        }
        articles_path = target_dir / "articles_manifest.json"
        with articles_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(articles_payload), f, indent=2, ensure_ascii=False)
        exported_files["articles_manifest"] = str(articles_path.resolve())

        # 4. Identified Advertisements
        ads_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "total_advertisements": len(advertisements or []),
            "advertisements": advertisements or [],
        }
        ads_path = target_dir / "identified_advertisements.json"
        with ads_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(ads_payload), f, indent=2, ensure_ascii=False)
        exported_files["identified_advertisements"] = str(ads_path.resolve())

        # 5. Ingestion Summary
        summary_payload = {
            "issue_id": issue_id,
            "newspaper_name": newspaper_name,
            "issue_date": str(issue_date),
            "edition": edition,
            "exported_at": datetime.now(UTC).isoformat(),
            "output_directory": str(target_dir.resolve()),
            "total_pages": len(page_extractions or []),
            "total_articles": len(articles or []),
            "total_chunks": len(rag_chunks or []),
            "total_advertisements": len(advertisements or []),
            "metrics": summary_metrics or {},
            "generated_files": exported_files,
        }
        summary_path = target_dir / "ingestion_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(self._sanitize(summary_payload), f, indent=2, ensure_ascii=False)
        exported_files["ingestion_summary"] = str(summary_path.resolve())

        logger.info(
            "Debug artifacts exported successfully",
            extra={
                "issue_id": issue_id,
                "output_dir": str(target_dir),
                "total_articles": len(articles or []),
                "total_chunks": len(rag_chunks or []),
            },
        )

        return exported_files

    def _sanitize(self, data: Any) -> Any:
        """Recursively convert non-serializable objects into JSON-compatible types."""
        from collections.abc import Mapping, Sequence
        from enum import Enum

        if data is None or isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, bytes):
            return f"<bytes len={len(data)}>"
        if isinstance(data, Enum):
            return data.value
        if is_dataclass(data) and not isinstance(data, type):
            return self._sanitize(asdict(data))
        if isinstance(data, Mapping):
            return {str(k): self._sanitize(v) for k, v in data.items()}
        if isinstance(data, (list, tuple, set, Sequence)) and not isinstance(
            data, (str, bytes, bytearray)
        ):
            return [self._sanitize(item) for item in data]
        if hasattr(data, "isoformat") and callable(data.isoformat):
            return data.isoformat()
        if hasattr(data, "__dict__"):
            return self._sanitize(dict(data.__dict__))
        return str(data)


__all__ = [
    "DebugArtifactsExporter",
    "DeletionService",
    "compress_pdf",
    "compress_pdf_bytes",
]
