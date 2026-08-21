"""PDF Rasterization service for NewsLens-AI.

Renders vector and scanned PDF pages to high-resolution PNG images (300 DPI)
using PyMuPDF (fitz). Uploads rasterized pages to MinIO `newslens-pages`
bucket and synchronizes page metadata (dimensions, object key, status) in MySQL.
"""
from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.newspaper import Issue, Page
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)

DEFAULT_DPI = 300
THUMBNAIL_DPI = 100


@dataclass
class RasterizedPage:
    """Represents a rendered page image with spatial metadata."""

    page_number: int
    image_bytes: bytes
    width_px: int
    height_px: int
    dpi: int
    object_key: str


class PDFRasterizer:
    """Renders PDF documents into image assets and persists them to MinIO and MySQL."""

    def __init__(self, db: AsyncSession, minio: MinioStore | None = None) -> None:
        self._db = db
        self._settings = get_settings()
        self._minio = minio or MinioStore(self._settings.minio)

    def render_page(
        self,
        doc: fitz.Document,
        page_index: int,
        dpi: int = DEFAULT_DPI,
    ) -> tuple[bytes, int, int]:
        """Render a single 0-indexed page from an open PyMuPDF document to PNG bytes."""
        page = doc.load_page(page_index)
        zoom = dpi / 72.0  # 72 points per inch in PDF spec
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pix.tobytes(output="png")
        return img_bytes, pix.width, pix.height

    async def rasterize_pdf_bytes(
        self,
        pdf_bytes: bytes,
        issue_id: int,
        dpi: int = DEFAULT_DPI,
    ) -> list[RasterizedPage]:
        """Rasterize all pages of a PDF and store records in MySQL and MinIO."""
        stmt = select(Issue).where(Issue.id == issue_id)
        res = await self._db.execute(stmt)
        issue = res.scalar_one_or_none()
        if not issue:
            raise ValueError(f"Issue with id {issue_id} not found.")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        issue.total_pages = total_pages
        logger.info(
            "Starting PDF rasterization",
            extra={"issue_id": issue_id, "total_pages": total_pages, "dpi": dpi},
        )

        results: list[RasterizedPage] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            img_bytes, width_px, height_px = self.render_page(doc, page_idx, dpi=dpi)

            # MinIO key: pages/{newspaper_id}/{issue_date}/{edition}/page_{num}.png
            edition_slug = (issue.edition or "default").lower().replace(" ", "_")
            object_key = (
                f"pages/{issue.newspaper_id}/{issue.issue_date}/{edition_slug}/page_{page_num}.png"
            )

            # Upload to MinIO
            await self._minio.put(
                bucket=self._settings.minio.bucket_pages,
                key=object_key,
                data=img_bytes,
                content_type="image/png",
            )

            # Get or create Page record
            page_stmt = select(Page).where(
                Page.issue_id == issue.id,
                Page.page_number == page_num,
            )
            page_res = await self._db.execute(page_stmt)
            page = page_res.scalar_one_or_none()

            if not page:
                page = Page(
                    issue_id=issue.id,
                    page_number=page_num,
                    raster_object_key=object_key,
                    width_px=width_px,
                    height_px=height_px,
                    ingestion_status="rasterized",
                )
                self._db.add(page)
            else:
                page.raster_object_key = object_key
                page.width_px = width_px
                page.height_px = height_px
                page.ingestion_status = "rasterized"

            results.append(
                RasterizedPage(
                    page_number=page_num,
                    image_bytes=img_bytes,
                    width_px=width_px,
                    height_px=height_px,
                    dpi=dpi,
                    object_key=object_key,
                )
            )

        issue.ingestion_status = "rasterized"
        await self._db.commit()
        doc.close()

        logger.info(
            "PDF rasterization completed",
            extra={"issue_id": issue_id, "rendered_pages": len(results)},
        )
        return results
