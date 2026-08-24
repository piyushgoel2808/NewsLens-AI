"""Unit tests for the Pre-Ingestion PDF Compression Layer."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from app.ingestion.compressor import compress_pdf, compress_pdf_bytes

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestPDFCompressor:
    """Test suite for compress_pdf_bytes and compress_pdf helper."""

    def test_compress_pdf_bytes_valid_digital_pdf(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
        assert pdf_path.exists()
        raw_bytes = pdf_path.read_bytes()

        compressed_bytes, stats = compress_pdf_bytes(raw_bytes)

        assert stats["status"] == "compressed"
        assert stats["original_bytes"] == len(raw_bytes)
        assert stats["compressed_bytes"] < len(raw_bytes)
        assert stats["reduction_pct"] > 0.0
        assert stats["saved_bytes"] > 0
        assert compressed_bytes.startswith(b"%PDF-")

        # Verify compressed PDF is valid and preserves document structure
        doc = pymupdf.open(stream=compressed_bytes, filetype="pdf")
        assert len(doc) == 1
        page_text = doc[0].get_text()
        assert len(page_text) > 0
        doc.close()

    def test_compress_pdf_bytes_scanned_pdf(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_scanned_page.pdf"
        assert pdf_path.exists()
        raw_bytes = pdf_path.read_bytes()

        compressed_bytes, stats = compress_pdf_bytes(raw_bytes)

        assert stats["status"] == "compressed"
        assert stats["compressed_bytes"] < len(raw_bytes)
        assert stats["reduction_pct"] > 50.0  # Significant reduction for scanned PDFs
        assert compressed_bytes.startswith(b"%PDF-")

    def test_compress_pdf_bytes_invalid_or_corrupted_stream(self) -> None:
        corrupt_data = b"This is plain text and definitely not a valid PDF file stream."
        processed_bytes, stats = compress_pdf_bytes(corrupt_data)

        assert processed_bytes == corrupt_data
        assert stats["status"] == "skipped_not_pdf"
        assert stats["reduction_pct"] == 0.0
        assert stats["saved_bytes"] == 0

    def test_compress_pdf_bytes_empty_stream(self) -> None:
        empty_data = b""
        processed_bytes, stats = compress_pdf_bytes(empty_data)

        assert processed_bytes == b""
        assert stats["status"] == "skipped_not_pdf"
        assert stats["original_bytes"] == 0
        assert stats["reduction_pct"] == 0.0

    def test_compress_pdf_convenience_wrapper(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_multi_page_issue.pdf"
        raw_bytes = pdf_path.read_bytes()

        result_bytes = compress_pdf(raw_bytes)

        assert isinstance(result_bytes, bytes)
        assert len(result_bytes) <= len(raw_bytes)
        assert result_bytes.startswith(b"%PDF-")
