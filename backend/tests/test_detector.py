"""Unit tests for PDFPageDetector digital vs scanned classification and gibberish detection."""
from __future__ import annotations

from pathlib import Path

from app.ingestion.detector import PageType, PDFPageDetector, is_text_gibberish

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestPDFPageDetector:
    """Test digital text layer detection, gibberish heuristics, and classification."""

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

    def test_is_text_gibberish_valid_text(self) -> None:
        valid_text = (
            "The stock market experienced a major surge yesterday as investors reacted to "
            "positive economic indicators. Central bank officials stated that inflation "
            "remains under control while employment figures showed steady improvement."
        )
        assert is_text_gibberish(valid_text) is False

    def test_is_text_gibberish_replacement_characters(self) -> None:
        corrupted_text = (
            "The market was \ufffd\ufffd\ufffd " * 20
            + " and continued to \ufffd\ufffd fall sharply."
        )
        assert is_text_gibberish(corrupted_text, threshold=0.10) is True

    def test_is_text_gibberish_font_mapping_b_repetition(self) -> None:
        # Font unmapped glyph loop (as seen in Business Standard PDF)
        b_loop_text = (
            "NEW DELHI | TUESDAY 7 JULY 2026\n"
            + "b" * 600
        )
        assert is_text_gibberish(b_loop_text) is True

    def test_is_text_gibberish_control_characters(self) -> None:
        # Unmapped TrueType subset codes (as seen in Indian Express PDF)
        control_text = "".join(chr(i % 30 + 1) for i in range(500))
        assert is_text_gibberish(control_text) is True
