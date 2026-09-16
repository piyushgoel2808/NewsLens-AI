"""Unit tests for PDFRasterizer service."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.rasterizer import PDFRasterizer
from app.models.newspaper import Issue

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestPDFRasterizer:
    """Test PDF page rasterization to PNG."""

    @pytest.mark.asyncio
    async def test_rasterize_single_page_pdf(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
        pdf_bytes = pdf_path.read_bytes()

        mock_db = AsyncMock(spec=AsyncSession)
        mock_minio = AsyncMock()

        # Mock issue query
        mock_issue = Issue(
            id=1,
            newspaper_id=1,
            issue_date=date(1929, 10, 24),
            edition="morning",
            total_pages=None,
        )
        mock_issue_res = MagicMock()
        mock_issue_res.scalar_one_or_none.return_value = mock_issue

        # Mock page query (none found -> creates new Page)
        mock_page_res = MagicMock()
        mock_page_res.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_issue_res, mock_page_res]

        rasterizer = PDFRasterizer(db=mock_db, minio=mock_minio)
        pages = await rasterizer.rasterize_pdf_bytes(pdf_bytes=pdf_bytes, issue_id=1, dpi=300)

        assert len(pages) == 1
        page = pages[0]
        assert page.page_number == 1
        assert page.dpi == 300
        assert page.width_px > 2000  # At 300 DPI, width is ~2479 px
        assert page.height_px > 3000
        assert page.image_bytes.startswith(b"\x89PNG")  # Valid PNG signature
        assert page.object_key == "pages/1/1929-10-24/morning/page_1.png"
        assert mock_minio.put.called

    @pytest.mark.asyncio
    async def test_rasterize_multi_page_pdf(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_multi_page_issue.pdf"
        pdf_bytes = pdf_path.read_bytes()

        mock_db = AsyncMock(spec=AsyncSession)
        mock_minio = AsyncMock()

        mock_issue = Issue(
            id=2,
            newspaper_id=1,
            issue_date=date(1929, 10, 25),
            edition="evening",
            total_pages=None,
        )
        mock_issue_res = MagicMock()
        mock_issue_res.scalar_one_or_none.return_value = mock_issue

        # Mock page query for each page (3 pages)
        mock_page_res1 = MagicMock()
        mock_page_res1.scalar_one_or_none.return_value = None
        mock_page_res2 = MagicMock()
        mock_page_res2.scalar_one_or_none.return_value = None
        mock_page_res3 = MagicMock()
        mock_page_res3.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [
            mock_issue_res,
            mock_page_res1,
            mock_page_res2,
            mock_page_res3,
        ]

        rasterizer = PDFRasterizer(db=mock_db, minio=mock_minio)
        pages = await rasterizer.rasterize_pdf_bytes(pdf_bytes=pdf_bytes, issue_id=2, dpi=150)

        assert len(pages) == 3
        assert [p.page_number for p in pages] == [1, 2, 3]
        assert mock_minio.put.call_count == 3
        assert mock_issue.total_pages == 3

    @pytest.mark.asyncio
    async def test_rasterize_single_page_method(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
        pdf_bytes = pdf_path.read_bytes()

        mock_db = AsyncMock(spec=AsyncSession)
        mock_minio = AsyncMock()

        mock_issue = Issue(
            id=1,
            newspaper_id=1,
            issue_date=date(1929, 10, 24),
            edition="morning",
            total_pages=None,
        )
        mock_issue_res = MagicMock()
        mock_issue_res.scalar_one_or_none.return_value = mock_issue

        mock_page_res = MagicMock()
        mock_page_res.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_issue_res, mock_page_res]

        rasterizer = PDFRasterizer(db=mock_db, minio=mock_minio)
        rendered = await rasterizer.rasterize_single_page(
            pdf_bytes=pdf_bytes,
            issue_id=1,
            page_number=1,
            dpi=300,
        )

        assert rendered.page_number == 1
        assert rendered.dpi == 300
        assert rendered.image_bytes.startswith(b"\x89PNG")
        assert rendered.object_key == "pages/1/1929-10-24/morning/page_1.png"
        assert mock_minio.put.called

    @pytest.mark.asyncio
    async def test_rasterize_default_150_dpi(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
        pdf_bytes = pdf_path.read_bytes()

        mock_db = AsyncMock(spec=AsyncSession)
        mock_minio = AsyncMock()

        mock_issue = Issue(
            id=3,
            newspaper_id=1,
            issue_date=date(1929, 10, 26),
            edition="morning",
            total_pages=None,
        )
        mock_issue_res = MagicMock()
        mock_issue_res.scalar_one_or_none.return_value = mock_issue

        mock_page_res = MagicMock()
        mock_page_res.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_issue_res, mock_page_res]

        rasterizer = PDFRasterizer(db=mock_db, minio=mock_minio)
        pages = await rasterizer.rasterize_pdf_bytes(pdf_bytes=pdf_bytes, issue_id=3)

        assert len(pages) == 1
        page = pages[0]
        assert page.dpi == 150
        assert 1000 < page.width_px < 1500  # At 150 DPI, A4 width is ~1240 px
        assert 1500 < page.height_px < 2000  # At 150 DPI, A4 height is ~1754 px


