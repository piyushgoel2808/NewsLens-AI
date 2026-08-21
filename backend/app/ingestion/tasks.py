"""Celery and synchronous pipeline execution tasks for PDF Ingestion, OCR & Layout Analysis."""
from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.celery_app import celery_app
from app.ingestion.detector import PDFPageDetector
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.ocr_service import OCRService
from app.ingestion.rasterizer import PDFRasterizer
from app.models.newspaper import Issue, Page
from app.providers.base import OCRBlock
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)


async def run_ingestion_pipeline(
    issue_id: int,
    pdf_bytes: bytes,
    dpi: int = 300,
    minio: MinioStore | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end Phase 1 & 2 ingestion pipeline for an issue.

    Steps:
    1. Rasterize all PDF pages to PNG (at specified DPI) and upload to MinIO.
    2. Analyze page text layers (digital vs scanned detection).
    3. Run OCR on scanned/image-only pages.
    4. Run layout analysis & reading order resolution on all pages.
    5. Update MySQL Page records and Issue state.
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

        # Step 2: Detect text layer
        detector = PDFPageDetector()
        analysis_results = detector.analyze_document_bytes(pdf_bytes)

        ocr_service = OCRService(db=db, minio=store)
        layout_analyzer = LayoutAnalyzer()

        pages_summary: list[dict[str, Any]] = []

        # Fetch issue upfront
        issue_stmt = select(Issue).where(Issue.id == issue_id)
        issue_res = await db.execute(issue_stmt)
        issue = issue_res.scalar_one_or_none()
        issue_lang = issue.language if issue else "en"

        # Step 3 & 4: OCR and Layout Analysis per page
        for i, rendered in enumerate(rendered_pages):
            page_num = rendered.page_number
            analysis = analysis_results[i]

            stmt = select(Page).where(
                Page.issue_id == issue_id,
                Page.page_number == page_num,
            )
            res = await db.execute(stmt)
            page = res.scalar_one_or_none()
            if not page:
                continue

            extracted_ocr_blocks: list[OCRBlock] = []

            # Run OCR if page is scanned
            if analysis.requires_ocr:
                try:
                    ocr_res = await ocr_service.process_page_ocr(
                        page_id=page.id,
                        image_bytes=rendered.image_bytes,
                        lang_hint=issue_lang,
                    )
                    extracted_ocr_blocks = ocr_res.blocks
                    page.ocr_confidence = ocr_res.mean_confidence
                except Exception as e:
                    logger.warning(
                        "OCR fallback on page",
                        extra={"page_id": page.id, "error": str(e)},
                    )

            # Run layout analysis
            layout_res = await layout_analyzer.analyze_page(
                page_number=page_num,
                width_px=rendered.width_px,
                height_px=rendered.height_px,
                image_bytes=rendered.image_bytes,
                digital_blocks=analysis.blocks if not analysis.requires_ocr else None,
                ocr_blocks=extracted_ocr_blocks if analysis.requires_ocr else None,
            )

            page.ingestion_status = "layout_done"

            pages_summary.append(
                {
                    "page_number": page_num,
                    "width_px": rendered.width_px,
                    "height_px": rendered.height_px,
                    "object_key": rendered.object_key,
                    "type": analysis.page_type.value,
                    "requires_ocr": analysis.requires_ocr,
                    "ocr_confidence": page.ocr_confidence,
                    "char_count": analysis.character_count,
                    "layout_elements": len(layout_res.elements),
                    "reading_blocks": len(layout_res.reading_order),
                    "layout_source": layout_res.source,
                }
            )

        # Update Issue record
        issue_stmt = select(Issue).where(Issue.id == issue_id)
        issue_res = await db.execute(issue_stmt)
        issue = issue_res.scalar_one_or_none()
        if issue:
            issue.ingestion_status = "ready_for_segmentation"

        await db.commit()

        logger.info(
            "Phase 2 ingestion pipeline completed successfully",
            extra={
                "issue_id": issue_id,
                "rendered_pages": len(rendered_pages),
            },
        )

        return {
            "issue_id": issue_id,
            "total_pages": len(rendered_pages),
            "pages": pages_summary,
        }


@celery_app.task(name="app.ingestion.tasks.process_issue_ingestion_task")  # type: ignore[untyped-decorator]
def process_issue_ingestion_task(
    issue_id: int,
    pdf_bytes: bytes,
    dpi: int = 300,
) -> dict[str, Any]:
    """Celery worker task to process an issue asynchronously."""
    return asyncio.run(run_ingestion_pipeline(issue_id, pdf_bytes, dpi=dpi))
