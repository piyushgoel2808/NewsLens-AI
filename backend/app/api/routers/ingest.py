"""FastAPI Ingestion Router for NewsLens-AI."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.ingestion.intake import IntakeService
from app.ingestion.tasks import run_ingestion_pipeline
from app.models.base import get_db
from app.models.ingestion import IngestionJob
from app.models.newspaper import Page

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


@router.post(
    "/ingest/upload",
    summary="Upload PDF or ZIP archive of newspaper issues",
    status_code=status.HTTP_201_CREATED,
)
async def upload_newspaper_document(
    file: UploadFile = File(...),
    newspaper_name: str = Form(..., description="Name of the newspaper (e.g. 'The Daily Times')"),
    issue_date: date = Form(..., description="Publication date of the issue (YYYY-MM-DD)"),
    edition: str = Form("morning", description="Edition identifier (e.g. 'morning', 'evening')"),
    language: str = Form("en", description="Primary ISO 639-1 language code (e.g. 'en', 'hi')"),
    force: bool = Form(False, description="Force re-ingestion if duplicate exists"),
    sync_processing: bool = Form(
        True,
        description="If True, processes rasterization and text extraction synchronously in request",
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Upload and ingest a single PDF or a ZIP archive containing multiple issue PDFs."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "upload.pdf"
    intake = IntakeService(db=db)

    try:
        intake_res = await intake.process_upload(
            file_bytes=content,
            filename=filename,
            newspaper_name=newspaper_name,
            issue_date=issue_date,
            edition=edition,
            language=language,
            force=force,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pipeline_results: list[dict[str, Any]] = []

    # Run processing pipeline on newly created issues
    if filename.lower().endswith(".pdf") and intake_res.issues_created:
        for issue_id in intake_res.issues_created:
            if sync_processing:
                try:
                    res = await run_ingestion_pipeline(issue_id=issue_id, pdf_bytes=content)
                    pipeline_results.append(res)
                except Exception as e:
                    logger.error(
                        "Sync pipeline execution failed",
                        extra={"issue_id": issue_id, "error": str(e)},
                    )
                    pipeline_results.append(
                        {"issue_id": issue_id, "error": str(e), "status": "failed"}
                    )
            else:
                asyncio.create_task(run_ingestion_pipeline(issue_id=issue_id, pdf_bytes=content))
                pipeline_results.append(
                    {"issue_id": issue_id, "status": "processing_in_background"}
                )

    is_duplicate_skipped = bool(intake_res.skipped_duplicates and not intake_res.issues_created)
    return {
        "message": (
            "Upload skipped (duplicate issue already exists. Check 'Force Re-ingest' to overwrite)"
            if is_duplicate_skipped
            else "Upload processed successfully"
        ),
        "status": "skipped_duplicate" if is_duplicate_skipped else "success",
        "is_duplicate": is_duplicate_skipped,
        "job_id": intake_res.job_id,
        "total_files": intake_res.total_files,
        "issues_created": intake_res.issues_created,
        "issue_id": intake_res.issues_created[0] if intake_res.issues_created else None,
        "skipped_duplicates": intake_res.skipped_duplicates,
        "pipeline_results": pipeline_results,
    }


@router.get("/ingest/jobs/{job_id}", summary="Get status of an ingestion job")
async def get_ingestion_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Fetch status and progress for a specific ingestion job."""
    stmt = select(IngestionJob).where(IngestionJob.id == job_id)
    res = await db.execute(stmt)
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"Ingestion job {job_id} not found.")

    return {
        "id": job.id,
        "source_type": job.source_type,
        "status": job.status,
        "total_files": job.total_files,
        "processed_files": job.processed_files,
        "failed_files": job.failed_files,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_log": job.error_log or [],
    }


@router.get("/ingest/jobs", summary="List past ingestion jobs")
async def list_ingestion_jobs(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List recent ingestion jobs."""
    stmt = select(IngestionJob).order_by(desc(IngestionJob.id)).limit(limit)
    res = await db.execute(stmt)
    jobs = res.scalars().all()
    return [
        {
            "id": j.id,
            "source_type": j.source_type,
            "status": j.status,
            "total_files": j.total_files,
            "processed_files": j.processed_files,
            "failed_files": j.failed_files,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]


@router.get("/pages/{page_id}/layout", summary="Get extracted layout and reading order of a page")
async def get_page_layout(
    page_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve spatial layout elements and reading order sequence for a page."""
    stmt = select(Page).where(Page.id == page_id).options(selectinload(Page.issue))
    res = await db.execute(stmt)
    page = res.scalar_one_or_none()
    if not page:
        raise HTTPException(status_code=404, detail=f"Page {page_id} not found.")

    return {
        "page_id": page.id,
        "issue_id": page.issue_id,
        "page_number": page.page_number,
        "width_px": page.width_px,
        "height_px": page.height_px,
        "raster_object_key": page.raster_object_key,
        "ocr_confidence": page.ocr_confidence,
        "ingestion_status": page.ingestion_status,
    }


@router.delete("/ingest/jobs/{job_id}", summary="Delete an ingestion job and linked assets")
async def delete_ingestion_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Purge an ingestion job and all associated issue artifacts."""
    from app.ingestion.deletion_service import DeletionService

    service = DeletionService(db=db)
    result = await service.delete_job(job_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return result
