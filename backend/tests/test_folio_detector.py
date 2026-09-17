"""Unit tests for FolioDetector ensuring 100% sequential integer page standardization."""

from __future__ import annotations

from app.ingestion.detector import DigitalTextBlock
from app.ingestion.metadata import FolioDetector
from app.providers.base import OCRBlock


class TestFolioDetector:
    """Test suite ensuring folio detector standardizes strictly on sequential integer page numbers."""

    def test_extract_printed_folio_standardizes_on_page_number(self) -> None:
        detector = FolioDetector()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="DELHI | MONDAY, JULY 7, 2026 | PAGE 12",
                bbox=(50.0, 20.0, 950.0, 50.0),
            ),
            DigitalTextBlock(
                block_id=1,
                text="Business news story content on page 12...",
                bbox=(50.0, 100.0, 450.0, 600.0),
            ),
        ]

        folio = detector.extract_printed_page_number(
            page_number=15,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=blocks,
        )
        assert folio == "15"

    def test_section_prefix_standardizes_on_page_number(self) -> None:
        detector = FolioDetector()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="BUSINESS STANDARD | COMPANIES | PAGE B-3",
                bbox=(50.0, 25.0, 900.0, 60.0),
            )
        ]

        folio = detector.extract_printed_page_number(
            page_number=18,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=blocks,
        )
        assert folio == "18"

    def test_extract_folio_standardizes_on_page_number(self) -> None:
        detector = FolioDetector()
        ocr_blocks = [
            OCRBlock(
                text="THE HINDU - OPINION | PAGE 7",
                bbox=(40.0, 30.0, 900.0, 70.0),
                confidence=0.96,
            ),
        ]

        folio = detector.extract_folio(
            blocks=ocr_blocks,
            height_px=1400.0,
            width_px=1000.0,
            page_number=10,
        )
        assert folio == "10"

    def test_advertisement_page_preserves_page_number(self) -> None:
        detector = FolioDetector()
        folio = detector.extract_printed_page_number(
            page_number=1,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=[],
            is_advertisement_page=True,
        )
        assert folio == "1"

    def test_extract_printed_page_number_direct(self) -> None:
        detector = FolioDetector()
        assert detector.extract_printed_page_number(page_number=5) == "5"
        assert detector.extract_printed_page_number(page_number=22) == "22"
