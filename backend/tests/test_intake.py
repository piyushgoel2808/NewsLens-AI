"""Unit and integration tests for IntakeService and file validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.intake import IntakeService, compute_sha256, is_valid_pdf
from app.models.newspaper import Newspaper

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestIntakeValidation:
    """Test PDF and ZIP validation helpers."""

    def test_compute_sha256(self) -> None:
        data = b"Sample newspaper content"
        h = compute_sha256(data)
        assert len(h) == 64
        assert h == compute_sha256(data)

    def test_is_valid_pdf_positive(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
        assert is_valid_pdf(pdf_path.read_bytes()) is True

    def test_is_valid_pdf_negative(self) -> None:
        assert is_valid_pdf(b"Not a PDF file content") is False


class TestIntakeService:
    """Test IntakeService database records and deduplication."""

    @pytest.mark.asyncio
    async def test_process_upload_single_pdf(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
        pdf_bytes = pdf_path.read_bytes()

        mock_db = AsyncMock(spec=AsyncSession)
        mock_minio = AsyncMock()

        # Configure mock DB queries
        mock_newspaper_res = MagicMock()
        mock_newspaper_res.scalar_one_or_none.return_value = Newspaper(id=1, name="The Daily Bugle")

        mock_issue_res = MagicMock()
        mock_issue_res.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_newspaper_res, mock_issue_res]

        service = IntakeService(db=mock_db, minio=mock_minio)
        res = await service.process_upload(
            file_bytes=pdf_bytes,
            filename="frontpage.pdf",
            newspaper_name="The Daily Bugle",
            issue_date=date(1930, 5, 12),
            edition="morning",
            language="en",
        )

        assert res.total_files == 1
        assert len(res.issues_created) == 1
        assert len(res.skipped_duplicates) == 0
        assert 1 in res.compressed_contents
        assert len(res.compressed_contents[1]) < len(pdf_bytes)
        assert mock_minio.put.called
        # Verify minio was called with compressed bytes
        put_kwargs = mock_minio.put.call_args.kwargs
        assert put_kwargs["data"] == res.compressed_contents[1]

    @pytest.mark.asyncio
    async def test_process_upload_invalid_pdf_raises(self) -> None:
        mock_db = AsyncMock(spec=AsyncSession)
        mock_minio = AsyncMock()
        service = IntakeService(db=mock_db, minio=mock_minio)

        with pytest.raises(ValueError, match="not a valid PDF document"):
            await service.process_upload(
                file_bytes=b"Corrupted data stream",
                filename="corrupt.pdf",
                newspaper_name="Times",
                issue_date=date(2020, 1, 1),
            )

    @pytest.mark.asyncio
    async def test_process_upload_zip_archive(self) -> None:
        zip_path = FIXTURES_DIR / "sample_newspaper_archive.zip"
        zip_bytes = zip_path.read_bytes()

        mock_db = AsyncMock(spec=AsyncSession)
        mock_minio = AsyncMock()

        mock_newspaper_res = MagicMock()
        mock_newspaper_res.scalar_one_or_none.return_value = Newspaper(
            id=1, name="Archive Chronicle"
        )

        mock_issue_res1 = MagicMock()
        mock_issue_res1.scalar_one_or_none.return_value = None

        mock_issue_res2 = MagicMock()
        mock_issue_res2.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [
            mock_newspaper_res,
            mock_issue_res1,
            mock_newspaper_res,
            mock_issue_res2,
        ]

        service = IntakeService(db=mock_db, minio=mock_minio)
        res = await service.process_upload(
            file_bytes=zip_bytes,
            filename="archive.zip",
            newspaper_name="Archive Chronicle",
            issue_date=date(1929, 10, 24),
            edition="morning",
            language="en",
        )

        assert res.total_files == 2
        assert len(res.issues_created) == 2
        assert mock_minio.put.call_count == 2
