"""FastAPI router for Newspaper Corpus, Issues, Pages, and Image Proxying."""
from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.models.article import Article
from app.models.base import get_db
from app.models.newspaper import Issue, Newspaper, Page
from app.storage.minio_store import MinioStore

router = APIRouter(tags=["corpus"])


@router.get("/api/newspapers", summary="List all newspapers with aggregated metrics")
async def list_newspapers(
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all registered newspapers with issue counts, article counts, and date spans."""
    stmt = (
        select(
            Newspaper.id,
            Newspaper.name,
            Newspaper.publisher,
            Newspaper.default_language,
            Newspaper.country,
            func.count(Issue.id.distinct()).label("issue_count"),
            func.min(Issue.issue_date).label("earliest_issue"),
            func.max(Issue.issue_date).label("latest_issue"),
            func.count(Article.id.distinct()).label("article_count"),
        )
        .outerjoin(Issue, Issue.newspaper_id == Newspaper.id)
        .outerjoin(Article, Article.issue_id == Issue.id)
        .group_by(
            Newspaper.id,
            Newspaper.name,
            Newspaper.publisher,
            Newspaper.default_language,
            Newspaper.country,
        )
        .order_by(Newspaper.name)
    )

    res = await db.execute(stmt)
    rows = res.all()

    return [
        {
            "id": r.id,
            "name": r.name,
            "publisher": r.publisher,
            "default_language": r.default_language,
            "country": r.country,
            "issue_count": r.issue_count,
            "article_count": r.article_count,
            "earliest_issue": str(r.earliest_issue) if r.earliest_issue else None,
            "latest_issue": str(r.latest_issue) if r.latest_issue else None,
        }
        for r in rows
    ]


@router.get("/api/issues", summary="List issues with optional filtering")
async def list_issues(
    newspaper_id: int | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List issues matching date, newspaper, and status criteria."""
    stmt = (
        select(Issue)
        .options(selectinload(Issue.newspaper), selectinload(Issue.pages))
        .order_by(desc(Issue.issue_date))
    )

    if newspaper_id:
        stmt = stmt.where(Issue.newspaper_id == newspaper_id)
    if date_from:
        stmt = stmt.where(Issue.issue_date >= date_from)
    if date_to:
        stmt = stmt.where(Issue.issue_date <= date_to)
    if status:
        stmt = stmt.where(Issue.ingestion_status == status)

    stmt = stmt.limit(limit)
    res = await db.execute(stmt)
    issues = res.scalars().all()

    return [
        {
            "id": iss.id,
            "newspaper_id": iss.newspaper_id,
            "newspaper_name": iss.newspaper.name if iss.newspaper else "Daily News",
            "issue_date": str(iss.issue_date),
            "edition": iss.edition,
            "language": iss.language,
            "total_pages": iss.total_pages or len(iss.pages),
            "ingestion_status": iss.ingestion_status,
            "created_at": iss.created_at.isoformat() if iss.created_at else None,
        }
        for iss in issues
    ]


@router.get("/api/issues/{issue_id}", summary="Get detailed issue overview with pages and articles")
async def get_issue_details(
    issue_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve full issue metadata, page scans list, and article manifest."""
    stmt = (
        select(Issue)
        .where(Issue.id == issue_id)
        .options(
            selectinload(Issue.newspaper),
            selectinload(Issue.pages),
            selectinload(Issue.articles).selectinload(Article.article_pages),
        )
    )
    res = await db.execute(stmt)
    issue = res.scalar_one_or_none()

    if not issue:
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")

    pages_data = [
        {
            "id": p.id,
            "page_number": p.page_number,
            "width_px": p.width_px,
            "height_px": p.height_px,
            "ocr_confidence": p.ocr_confidence,
            "raster_object_key": p.raster_object_key,
            "ingestion_status": p.ingestion_status,
            "image_url": f"/api/pages/{p.id}/image",
        }
        for p in sorted(issue.pages, key=lambda x: x.page_number)
    ]

    articles_data = [
        {
            "id": a.id,
            "headline": a.headline or "Untitled",
            "section": a.section,
            "article_type": a.article_type,
            "prominence_score": a.prominence_score,
            "word_count": a.word_count,
            "summary": a.summary,
            "pages": sorted({ap.page_number for ap in a.article_pages}),
        }
        for a in issue.articles
    ]

    return {
        "id": issue.id,
        "newspaper_id": issue.newspaper_id,
        "newspaper_name": issue.newspaper.name if issue.newspaper else "Daily News",
        "issue_date": str(issue.issue_date),
        "edition": issue.edition,
        "language": issue.language,
        "total_pages": len(pages_data),
        "ingestion_status": issue.ingestion_status,
        "pages": pages_data,
        "articles": articles_data,
    }


@router.get("/api/pages/{page_id}/image", summary="Stream original 300 DPI page scan image")
async def get_page_image(
    page_id: int,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream high-resolution raster image PNG from MinIO object storage."""
    stmt = select(Page).where(Page.id == page_id)
    res = await db.execute(stmt)
    page = res.scalar_one_or_none()

    if not page or not page.raster_object_key:
        raise HTTPException(status_code=404, detail=f"Page {page_id} image scan not found")

    settings = get_settings()
    minio = MinioStore(settings.minio)
    image_bytes = await minio.get(
        bucket=settings.minio.bucket_pages,
        key=page.raster_object_key,
    )

    if not image_bytes:
        raise HTTPException(status_code=404, detail="Image object missing from storage")

    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )
