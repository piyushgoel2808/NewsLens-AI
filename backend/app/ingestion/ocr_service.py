"""OCR orchestration service for scanned newspaper pages."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.newspaper import Page
from app.providers.base import OCREngine, OCRResult, ProviderError
from app.providers.tesseract_ocr import TesseractOCR
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)


class OCRService:
    """Orchestrates OCR extraction on scanned newspaper pages."""

    def __init__(
        self,
        db: AsyncSession,
        ocr_engine: OCREngine | None = None,
        minio: MinioStore | None = None,
    ) -> None:
        self._db = db
        self._settings = get_settings()
        if ocr_engine:
            self._ocr = ocr_engine
        else:
            try:
                from app.providers.registry import get_registry

                provider = get_registry().get_provider("ocr")
                if isinstance(provider, OCREngine):
                    self._ocr = provider
                else:
                    self._ocr = TesseractOCR()
            except Exception:
                self._ocr = TesseractOCR()
        self._minio = minio or MinioStore(self._settings.minio)

    async def process_page_ocr(
        self,
        page_id: int,
        image_bytes: bytes | None = None,
        lang_hint: str | None = None,
    ) -> OCRResult:
        """Run OCR on a page image, persist metrics to MySQL, and return structured text blocks."""
        stmt = select(Page).where(Page.id == page_id)
        res = await self._db.execute(stmt)
        page = res.scalar_one_or_none()
        if not page:
            raise ValueError(f"Page with id {page_id} not found.")

        # If image_bytes not provided directly, retrieve from MinIO
        if image_bytes is None:
            if not page.raster_object_key:
                raise ValueError(
                    f"Page {page_id} has no raster_object_key and no image_bytes provided."
                )
            image_bytes = await self._minio.get(
                bucket=self._settings.minio.bucket_pages,
                key=page.raster_object_key,
            )

        logger.info(
            "Starting OCR on page image",
            extra={"page_id": page_id, "size_bytes": len(image_bytes)},
        )

        try:
            ocr_result = await self._ocr.ocr(image_bytes=image_bytes, lang_hint=lang_hint)
        except ProviderError as e:
            logger.error("OCR execution error", extra={"page_id": page_id, "error": str(e)})
            raise

        # Update MySQL Page record
        page.ocr_confidence = ocr_result.mean_confidence
        page.ingestion_status = "ocr_done"
        await self._db.commit()

        logger.info(
            "OCR completed on page",
            extra={
                "page_id": page_id,
                "confidence": round(ocr_result.mean_confidence, 4),
                "blocks_count": len(ocr_result.blocks),
            },
        )

        return ocr_result
