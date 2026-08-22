"""FastAPI router for Newspaper Corpus, Issues, Pages, and Image Proxying."""

from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.models.article import Article, ArticleChunk
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


@router.get("/api/issues", summary="List issues matching filters with article and chunk counts")
async def list_issues(
    newspaper_id: int | None = Query(None, description="Filter by newspaper ID"),
    date_from: str | None = Query(None, description="Filter issues from date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="Filter issues up to date (YYYY-MM-DD)"),
    status: str | None = Query(None, description="Filter by status (pending|parsed|failed)"),
    limit: int = Query(50, ge=1, le=200, description="Max issues to return"),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List issues matching date, newspaper, and status criteria."""
    stmt = (
        select(
            Issue,
            func.count(distinct(Article.id)).label("article_count"),
            func.count(distinct(ArticleChunk.id)).label("chunk_count"),
        )
        .outerjoin(Article, Article.issue_id == Issue.id)
        .outerjoin(ArticleChunk, ArticleChunk.article_id == Article.id)
        .options(selectinload(Issue.newspaper), selectinload(Issue.pages))
        .group_by(Issue.id)
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
    rows = res.all()

    return [
        {
            "id": iss.id,
            "newspaper_id": iss.newspaper_id,
            "newspaper_name": iss.newspaper.name if iss.newspaper else "Daily News",
            "issue_date": str(iss.issue_date),
            "edition": iss.edition,
            "language": iss.language,
            "total_pages": iss.total_pages or len(iss.pages),
            "article_count": art_cnt,
            "chunk_count": chk_cnt,
            "ingestion_status": iss.ingestion_status,
            "created_at": iss.created_at.isoformat() if iss.created_at else None,
        }
        for iss, art_cnt, chk_cnt in rows
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
        "article_count": len(articles_data),
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


@router.get(
    "/api/issues/{issue_id}/inspection",
    summary="Complete Ingestion & Chunking Transparency Inspector",
)
async def inspect_issue_ingestion(
    issue_id: int,
    chunk_limit: int = Query(50, ge=1, le=200, description="Max number of chunks to return"),
    chunk_offset: int = Query(0, ge=0, description="Chunk offset pagination index"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve transparency breakdown of page extraction modes, OCR fallback, and chunks."""
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

    pages_data = []
    for p in sorted(issue.pages, key=lambda x: x.page_number):
        is_ocr_used = p.ocr_confidence is not None and p.ocr_confidence > 0.0
        pages_data.append(
            {
                "id": p.id,
                "page_number": p.page_number,
                "printed_page_number": p.printed_page_number,
                "is_advertisement_page": p.is_advertisement_page,
                "width_px": p.width_px,
                "height_px": p.height_px,
                "ocr_confidence": p.ocr_confidence,
                "raster_object_key": p.raster_object_key,
                "ingestion_status": p.ingestion_status,
                "extraction_mode": (
                    "Advertisement"
                    if p.is_advertisement_page
                    else ("OCR" if is_ocr_used else "Digital Native")
                ),
                "ocr_fallback_triggered": is_ocr_used,
                "ocr_fallback_reason": (
                    "Corrupted Font / Gibberish Detected" if is_ocr_used else None
                ),
                "image_url": f"/api/pages/{p.id}/image",
            }
        )

    articles_data = [
        {
            "id": a.id,
            "headline": a.headline or "Untitled",
            "section": a.section,
            "article_type": a.article_type,
            "is_advertisement": a.article_type == "advertisement",
            "prominence_score": a.prominence_score,
            "word_count": a.word_count,
            "summary": a.summary,
            "full_text_preview": (
                (a.full_text[:300] + "...")
                if a.full_text and len(a.full_text) > 300
                else a.full_text
            ),
            "pages": sorted({ap.page_number for ap in a.article_pages}),
            "printed_pages": [
                ap.printed_page_number
                for ap in sorted(a.article_pages, key=lambda ap: ap.page_number)
                if ap.printed_page_number
            ],
            "bboxes": [
                bbox
                for ap in a.article_pages
                if ap.bbox_json and isinstance(ap.bbox_json, dict)
                for bbox in ap.bbox_json.get("bboxes", [])
            ],
            "page_bboxes": {
                ap.page_number: ap.bbox_json.get("bboxes", [])
                for ap in a.article_pages
                if ap.bbox_json and isinstance(ap.bbox_json, dict)
            },
        }
        for a in issue.articles
    ]

    # Fetch paginated chunks across the issue
    total_chunks_stmt = (
        select(func.count(ArticleChunk.id))
        .join(Article, Article.id == ArticleChunk.article_id)
        .where(Article.issue_id == issue_id)
    )
    total_chunks_res = await db.execute(total_chunks_stmt)
    total_chunks = total_chunks_res.scalar() or 0

    chunks_stmt = (
        select(ArticleChunk, Article.headline)
        .join(Article, Article.id == ArticleChunk.article_id)
        .where(Article.issue_id == issue_id)
        .order_by(Article.id, ArticleChunk.chunk_index)
        .offset(chunk_offset)
        .limit(chunk_limit)
    )
    chunks_res = await db.execute(chunks_stmt)
    chunk_rows = chunks_res.all()

    chunks_data = [
        {
            "id": chunk.id,
            "article_id": chunk.article_id,
            "headline": headline or "Untitled",
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "embedding_vector_id": chunk.embedding_vector_id,
        }
        for chunk, headline in chunk_rows
    ]

    return {
        "issue": {
            "id": issue.id,
            "newspaper_id": issue.newspaper_id,
            "newspaper_name": issue.newspaper.name if issue.newspaper else "Daily News",
            "issue_date": str(issue.issue_date),
            "edition": issue.edition,
            "language": issue.language,
            "total_pages": len(pages_data),
            "article_count": len(articles_data),
            "total_chunks": total_chunks,
            "ingestion_status": issue.ingestion_status,
        },
        "pages": pages_data,
        "articles": articles_data,
        "chunks": chunks_data,
        "pagination": {
            "total": total_chunks,
            "limit": chunk_limit,
            "offset": chunk_offset,
            "has_more": (chunk_offset + chunk_limit) < total_chunks,
        },
    }


@router.delete("/api/issues/{issue_id}", summary="Permanently delete an issue, vectors, and files")
async def delete_issue(
    issue_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Execute 3-Tier Hard Deletion Blueprint across Qdrant, MinIO, and MySQL."""
    from app.ingestion.deletion_service import DeletionService

    service = DeletionService(db=db)
    result = await service.delete_issue(issue_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
    return result
