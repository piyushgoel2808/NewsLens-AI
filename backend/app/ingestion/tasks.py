"""Celery and synchronous pipeline execution tasks for Phase 1 PDF Ingestion."""
from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.celery_app import celery_app
from app.ingestion.detector import PDFPageDetector
from app.ingestion.rasterizer import PDFRasterizer
from app.models.newspaper import Issue, Page
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)


async def run_ingestion_pipeline(
    issue_id: int,
    pdf_bytes: bytes,
    dpi: int = 300,
    minio: MinioStore | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end Phase 1 ingestion pipeline for an issue.

    Steps:
    1. Rasterize all PDF pages to PNG (at specified DPI) and upload to MinIO.
    2. Analyze page text layers (digital vs scanned detection).
    3. Update MySQL Page records and Issue state.
    """
    settings = get_settings()
    store = minio or MinioStore(settings.minio)

    if session_factory is None:
        engine = create_async_engine(settings.database.async_url, echo=False)
        maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    else:
        maker = session_factory

    async with maker() as db:
        # Step 1: Rasterize PDF pages
        rasterizer = PDFRasterizer(db=db, minio=store)
        rendered_pages = await rasterizer.rasterize_pdf_bytes(
            pdf_bytes=pdf_bytes,
            issue_id=issue_id,
            dpi=dpi,
        )

        # Step 2: Detect text layer and extract structured blocks
        detector = PDFPageDetector()
        analysis_results = detector.analyze_document_bytes(pdf_bytes)

        # Step 3: Update Page records with text analysis details
        for analysis in analysis_results:
            stmt = select(Page).where(
                Page.issue_id == issue_id,
                Page.page_number == analysis.page_number,
            )
            res = await db.execute(stmt)
            page = res.scalar_one_or_none()
            if page:
                if analysis.requires_ocr:
                    page.ingestion_status = "scanned_ready_for_ocr"
                else:
                    page.ingestion_status = "digital_text_extracted"

        # Update Issue record
        issue_stmt = select(Issue).where(Issue.id == issue_id)
        issue_res = await db.execute(issue_stmt)
        issue = issue_res.scalar_one_or_none()
        if issue:
            issue.ingestion_status = "ready_for_segmentation"

        await db.commit()

        logger.info(
            "Phase 1 ingestion pipeline completed successfully",
            extra={
                "issue_id": issue_id,
                "rendered_pages": len(rendered_pages),
                "analyzed_pages": len(analysis_results),
            },
        )

        return {
            "issue_id": issue_id,
            "total_pages": len(rendered_pages),
            "pages": [
                {
                    "page_number": p.page_number,
                    "width_px": p.width_px,
                    "height_px": p.height_px,
                    "object_key": p.object_key,
                    "type": analysis_results[i].page_type.value,
                    "requires_ocr": analysis_results[i].requires_ocr,
                    "char_count": analysis_results[i].character_count,
                    "block_count": len(analysis_results[i].blocks),
                }
                for i, p in enumerate(rendered_pages)
            ],
        }


@celery_app.task(name="app.ingestion.tasks.process_issue_ingestion_task")  # type: ignore[untyped-decorator]
def process_issue_ingestion_task(
    issue_id: int,
    pdf_bytes: bytes,
    dpi: int = 300,
) -> dict[str, Any]:
    """Celery worker task to process an issue asynchronously."""
    return asyncio.run(run_ingestion_pipeline(issue_id, pdf_bytes, dpi=dpi))
