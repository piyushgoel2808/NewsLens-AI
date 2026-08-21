"""Unit tests for PDFPageDetector digital vs scanned classification."""
from __future__ import annotations

from pathlib import Path

from app.ingestion.detector import PageType, PDFPageDetector

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestPDFPageDetector:
    """Test digital text layer detection and classification."""

    def test_detect_digital_page(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
        detector = PDFPageDetector()
        results = detector.analyze_document_bytes(pdf_path.read_bytes())

        assert len(results) == 1
        res = results[0]
        assert res.page_type in (PageType.DIGITAL, PageType.HYBRID)
        assert res.requires_ocr is False
        assert res.character_count > 100
        assert res.word_count > 20
        assert len(res.blocks) >= 3
        # Ensure headline candidate detected
        assert any(b.is_heading_candidate for b in res.blocks)

    def test_detect_scanned_page(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_scanned_page.pdf"
        detector = PDFPageDetector()
        results = detector.analyze_document_bytes(pdf_path.read_bytes())

        assert len(results) == 1
        res = results[0]
        assert res.page_type == PageType.SCANNED
        assert res.requires_ocr is True
        assert res.character_count < 80

    def test_detect_multi_page_document(self) -> None:
        pdf_path = FIXTURES_DIR / "sample_multi_page_issue.pdf"
        detector = PDFPageDetector()
        results = detector.analyze_document_bytes(pdf_path.read_bytes())

        assert len(results) == 3
        for i, res in enumerate(results, start=1):
            assert res.page_number == i
            assert res.requires_ocr is False
            assert len(res.blocks) >= 1
