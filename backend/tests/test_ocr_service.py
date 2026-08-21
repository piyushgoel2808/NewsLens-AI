"""Unit tests for OCRService orchestration."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.ocr_service import OCRService
from app.models.newspaper import Page
from app.providers.base import OCRBlock, OCRResult


class TestOCRService:
    """Test OCRService processing and DB updates."""

    @pytest.mark.asyncio
    async def test_process_page_ocr_success(self) -> None:
        mock_db = AsyncMock(spec=AsyncSession)
        mock_minio = AsyncMock()
        mock_ocr_engine = AsyncMock()

        # Mock Page model
        mock_page = Page(
            id=42,
            issue_id=1,
            page_number=1,
            raster_object_key="pages/1/1929-10-24/morning/page_1.png",
            ocr_confidence=None,
            ingestion_status="rasterized",
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = mock_page
        mock_db.execute.return_value = mock_res

        mock_ocr_engine.ocr.return_value = OCRResult(
            blocks=[
                OCRBlock(text="HEADLINE", bbox=(10, 10, 200, 40), confidence=0.95),
                OCRBlock(text="Body text paragraph", bbox=(10, 50, 200, 150), confidence=0.88),
            ],
            full_text="HEADLINE Body text paragraph",
            mean_confidence=0.915,
            language="eng",
        )

        service = OCRService(db=mock_db, ocr_engine=mock_ocr_engine, minio=mock_minio)
        ocr_result = await service.process_page_ocr(page_id=42, image_bytes=b"fake-png-bytes")

        assert ocr_result.mean_confidence == 0.915
        assert len(ocr_result.blocks) == 2
        assert mock_page.ocr_confidence == 0.915
        assert mock_page.ingestion_status == "ocr_done"
        assert mock_db.commit.called
