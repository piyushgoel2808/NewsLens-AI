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
from app.models.newspaper import Issue, Page

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


@router.post(
    "/ingest/inspect-preview",
    summary="Lightweight pre-upload inspection to detect newspaper and multi-page consensus date",
)
async def inspect_upload_preview(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Inspect uploaded PDF and return detected brand, date consensus, and existing publications."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    from app.ingestion.consensus_extractor import extract_newspaper_and_date_consensus
    from app.models.newspaper import Newspaper

    all_news_res = await db.execute(select(Newspaper))
    existing_newspapers = [
        {"id": n.id, "name": n.name, "default_language": n.default_language}
        for n in all_news_res.scalars().all()
    ]
    existing_names: list[str] = [str(n["name"]) for n in existing_newspapers if n["name"]]

    filename = file.filename or "upload.pdf"
    det_brand, det_date, telemetry = extract_newspaper_and_date_consensus(
        pdf_bytes=content,
        max_pages=15,
        existing_newspaper_names=existing_names,
        filename=filename,
    )

    is_new = bool(det_brand and det_brand not in existing_names)

    return {
        "detected_newspaper": det_brand or (existing_names[0] if existing_names else "Daily News"),
        "detected_date": str(det_date) if det_date else str(date.today()),
        "is_new_newspaper": is_new,
        "existing_newspapers": existing_newspapers,
        "telemetry": telemetry,
    }


@router.post(
    "/ingest/upload",
    summary="Upload PDF or ZIP archive of newspaper issues",
    status_code=status.HTTP_201_CREATED,
)
async def upload_newspaper_document(
    file: UploadFile = File(...),
    newspaper_name: str = Form("auto", description="Newspaper title or 'auto' for consensus"),
    issue_date: date | None = Form(None, description="Publication date or None for consensus"),
    edition: str = Form("morning", description="Edition identifier (e.g. 'morning', 'evening')"),
    language: str = Form("en", description="Primary ISO 639-1 language code (e.g. 'en', 'hi')"),
    parser_engine: str = Form(
        "docling",
        description="Parser engine: docling, mineru, gemini_vision, tesseract_vlm",
    ),
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

    effective_date = issue_date or date.today()
    effective_name = newspaper_name if newspaper_name and newspaper_name != "auto" else "auto"

    try:
        intake_res = await intake.process_upload(
            file_bytes=content,
            filename=filename,
            newspaper_name=effective_name,
            issue_date=effective_date,
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
            pdf_payload = intake_res.compressed_contents.get(issue_id, content)
            if sync_processing:
                try:
                    res = await run_ingestion_pipeline(
                        issue_id=issue_id,
                        pdf_bytes=pdf_payload,
                        parser_engine=parser_engine,
                    )
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
                asyncio.create_task(
                    run_ingestion_pipeline(
                        issue_id=issue_id,
                        pdf_bytes=pdf_payload,
                        parser_engine=parser_engine,
                    )
                )
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


@router.get(
    "/ingest/issues/{issue_id}/debug-artifacts",
    summary="Get list of exported debug artifacts for an issue",
)
async def get_issue_debug_artifacts(
    issue_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List available debug artifacts (OCR text, RAG chunks, articles manifest, advertisements)."""
    from app.ingestion.debug_exporter import DebugArtifactsExporter

    stmt = select(Issue).where(Issue.id == issue_id).options(selectinload(Issue.newspaper))
    res = await db.execute(stmt)
    issue = res.scalar_one_or_none()
    if not issue:
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found.")

    np_name = issue.newspaper.name if issue.newspaper else "daily"
    exporter = DebugArtifactsExporter()
    issue_dir = exporter.get_issue_debug_dir(
        issue_id=issue.id,
        newspaper_name=np_name,
        issue_date=str(issue.issue_date),
        edition=issue.edition or "morning",
    )

    artifacts = {}
    for filename in [
        "ocr_extracted_text.json",
        "rag_chunks.json",
        "articles_manifest.json",
        "identified_advertisements.json",
        "ingestion_summary.json",
    ]:
        file_path = issue_dir / filename
        key = filename.replace(".json", "")
        artifacts[key] = {
            "exists": file_path.exists(),
            "path": str(file_path.resolve()) if file_path.exists() else None,
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        }

    return {
        "issue_id": issue_id,
        "newspaper_name": np_name,
        "issue_date": str(issue.issue_date),
        "debug_directory": str(issue_dir.resolve()),
        "artifacts": artifacts,
    }


@router.get(
    "/ingest/issues/{issue_id}/debug-artifacts/{artifact_name}",
    summary="Fetch contents of a specific debug artifact JSON file",
)
async def get_issue_debug_artifact_content(
    issue_id: int,
    artifact_name: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Fetch content of a specific debug artifact (e.g. ocr_extracted_text, rag_chunks, etc.)."""
    import json

    from app.ingestion.debug_exporter import DebugArtifactsExporter

    clean_name = artifact_name.replace(".json", "").strip().lower()
    allowed_artifacts = {
        "ocr_extracted_text",
        "rag_chunks",
        "articles_manifest",
        "identified_advertisements",
        "ingestion_summary",
    }
    if clean_name not in allowed_artifacts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid artifact name '{artifact_name}'. Allowed: {sorted(allowed_artifacts)}",
        )

    stmt = select(Issue).where(Issue.id == issue_id).options(selectinload(Issue.newspaper))
    res = await db.execute(stmt)
    issue = res.scalar_one_or_none()
    if not issue:
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found.")

    np_name = issue.newspaper.name if issue.newspaper else "daily"
    exporter = DebugArtifactsExporter()
    issue_dir = exporter.get_issue_debug_dir(
        issue_id=issue.id,
        newspaper_name=np_name,
        issue_date=str(issue.issue_date),
        edition=issue.edition or "morning",
    )
    target_file = issue_dir / f"{clean_name}.json"
    if not target_file.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Artifact '{clean_name}.json' not found for issue {issue_id}. "
                "Ingestion may not have completed."
            ),
        )

    with target_file.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data
