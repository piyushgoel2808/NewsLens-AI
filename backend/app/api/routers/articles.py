"""FastAPI router for Article queries and inspection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.article import Article
from app.models.base import get_db
from app.models.newspaper import Issue

router = APIRouter(prefix="/api", tags=["articles"])


@router.get("/articles/{article_id}", summary="Get detailed article with photos, tables, and pages")
async def get_article_details(
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve full article text, metadata, page bounding boxes, and media assets."""
    stmt = (
        select(Article)
        .where(Article.id == article_id)
        .options(
            selectinload(Article.article_pages),
            selectinload(Article.photos),
            selectinload(Article.tables),
            selectinload(Article.chunks),
            selectinload(Article.issue).selectinload(Issue.newspaper),
        )
    )
    res = await db.execute(stmt)
    article = res.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail=f"Article {article_id} not found.")

    chunks_data = [
        {
            "id": c.id,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "token_count": c.token_count,
            "embedding_vector_id": c.embedding_vector_id,
        }
        for c in sorted(article.chunks, key=lambda x: x.chunk_index)
    ]

    return {
        "id": article.id,
        "issue_id": article.issue_id,
        "newspaper_name": (
            article.issue.newspaper.name if article.issue and article.issue.newspaper else "Unknown"
        ),
        "issue_date": str(article.issue.issue_date) if article.issue else "",
        "primary_page_id": article.primary_page_id,
        "headline": article.headline or "Untitled",
        "subheadline": article.subheadline,
        "byline_author": article.byline_author,
        "section": article.section,
        "article_type": article.article_type,
        "language": article.language,
        "prominence_score": article.prominence_score,
        "word_count": article.word_count,
        "summary": article.summary,
        "full_text": article.full_text,
        "pages": [
            {
                "page_id": ap.page_id,
                "page_number": ap.page_number,
                "bbox_json": ap.bbox_json,
                "block_order": ap.block_order,
            }
            for ap in sorted(article.article_pages, key=lambda x: x.block_order)
        ],
        "chunks": chunks_data,
        "photos": [
            {
                "id": ph.id,
                "caption": ph.caption,
                "object_key": ph.object_key,
                "bbox_json": ph.bbox_json,
                "image_url": f"/api/photos/{ph.id}/image",
            }
            for ph in article.photos
        ],
        "tables": [
            {
                "id": tb.id,
                "object_key": tb.object_key,
                "extracted_json": tb.extracted_json,
                "bbox_json": tb.bbox_json,
            }
            for tb in article.tables
        ],
    }


@router.get("/photos/{photo_id}/image", summary="Stream cropped article photo from MinIO")
async def get_photo_image(
    photo_id: int,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream cropped article image/photo PNG from MinIO object storage."""
    import io

    from app.core.config import get_settings
    from app.models.article import Photo
    from app.storage.minio_store import MinioStore

    stmt = select(Photo).where(Photo.id == photo_id)
    res = await db.execute(stmt)
    photo = res.scalar_one_or_none()

    if not photo or not photo.object_key:
        raise HTTPException(status_code=404, detail=f"Photo {photo_id} not found")

    settings = get_settings()
    minio = MinioStore(settings.minio)
    image_bytes = await minio.get(
        bucket=settings.minio.bucket_pages,
        key=photo.object_key,
    )

    if not image_bytes:
        raise HTTPException(status_code=404, detail="Photo object missing from storage")

    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get(
    "/issues/{issue_id}/articles",
    summary="List all articles in an issue ordered by prominence",
)
async def list_issue_articles(
    issue_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all articles in an issue sorted by prominence score."""
    stmt = (
        select(Article)
        .where(Article.issue_id == issue_id)
        .order_by(desc(Article.prominence_score))
        .options(selectinload(Article.article_pages))
    )
    res = await db.execute(stmt)
    articles = res.scalars().all()

    return [
        {
            "id": a.id,
            "headline": a.headline,
            "subheadline": a.subheadline,
            "byline_author": a.byline_author,
            "section": a.section,
            "article_type": a.article_type,
            "prominence_score": a.prominence_score,
            "word_count": a.word_count,
            "pages_spanned": [ap.page_number for ap in a.article_pages],
        }
        for a in articles
    ]
