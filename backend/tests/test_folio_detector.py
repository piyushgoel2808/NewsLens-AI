"""Unit tests for FolioDetector."""

from __future__ import annotations

from app.ingestion.detector import DigitalTextBlock
from app.ingestion.folio_detector import FolioDetector
from app.providers.base import OCRBlock


class TestFolioDetector:
    """Test suite for printed newspaper page number (folio) detection."""

    def test_extract_printed_folio_from_header_line(self) -> None:
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
        assert folio == "12"

    def test_extract_printed_folio_from_section_prefix(self) -> None:
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
        assert folio in ("B-3", "B3")

    def test_extract_printed_folio_from_ocr_blocks(self) -> None:
        detector = FolioDetector()
        ocr_blocks = [
            OCRBlock(
                text="THE HINDU - OPINION | PAGE 7",
                bbox=(40.0, 30.0, 900.0, 70.0),
                confidence=0.96,
            ),
            OCRBlock(
                text="Editorial columns and letters to editor...",
                bbox=(50.0, 120.0, 450.0, 600.0),
                confidence=0.91,
            ),
        ]

        folio = detector.extract_printed_page_number(
            page_number=10,
            height_px=1400.0,
            width_px=1000.0,
            ocr_blocks=ocr_blocks,
        )
        assert folio == "7"

    def test_advertisement_page_defaults_to_cover_ad_wrap(self) -> None:
        detector = FolioDetector()
        folio = detector.extract_printed_page_number(
            page_number=1,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=[],
            is_advertisement_page=True,
        )
        assert folio == "Cover/Ad Wrap"

    def test_extrapolate_folio_from_previous_page(self) -> None:
        detector = FolioDetector()
        # Page 6 was detected on PDF p.10 -> PDF p.12 should be Page 8
        folio = detector.extract_printed_page_number(
            page_number=12,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=[],
            is_advertisement_page=False,
            last_known_folio_num=6,
            last_known_pdf_page=10,
        )
        assert folio == "8"

    def test_fallback_to_unnumbered_with_pdf_index(self) -> None:
        detector = FolioDetector()
        folio = detector.extract_printed_page_number(
            page_number=3,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=[],
            is_advertisement_page=False,
        )
        assert folio == "Unnumbered (PDF p.3)"

    def test_mint_header_with_date_extracts_page_number_not_date_day(self) -> None:
        """Verify THURSDAY, 30 JULY 2026 BENGALURU 13 extracts 13, not 30."""
        detector = FolioDetector()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="THURSDAY, 30 JULY 2026 BENGALURU 13",
                bbox=(50.0, 20.0, 950.0, 45.0),
            ),
            DigitalTextBlock(
                block_id=1,
                text="Main front page story text here...",
                bbox=(50.0, 100.0, 450.0, 600.0),
            ),
        ]

        folio = detector.extract_printed_page_number(
            page_number=13,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=blocks,
        )
        assert folio == "13"

    def test_mint_header_with_leading_page_number(self) -> None:
        """Verify 13 THURSDAY, 30 JULY 2026 BENGALURU extracts 13, not 30."""
        detector = FolioDetector()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="13 THURSDAY, 30 JULY 2026 BENGALURU",
                bbox=(50.0, 20.0, 950.0, 45.0),
            ),
        ]

        folio = detector.extract_printed_page_number(
            page_number=13,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=blocks,
        )
        assert folio == "13"

    def test_date_with_month_first_and_trailing_folio(self) -> None:
        """Verify JULY 30, 2026 | MUMBAI | 5 extracts 5."""
        detector = FolioDetector()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="JULY 30, 2026 | MUMBAI | 5",
                bbox=(50.0, 20.0, 950.0, 45.0),
            ),
        ]

        folio = detector.extract_printed_page_number(
            page_number=5,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=blocks,
        )
        assert folio == "5"

    def test_spatial_parsing_discards_body_ads_and_phone_numbers(self) -> None:
        """Verify body phone numbers (181, 182), prices (31), and letters (M) are discarded."""
        detector = FolioDetector()
        blocks = [
            # Top header block (y1 = 40 <= 1400 * 0.08 = 112)
            DigitalTextBlock(
                block_id=0,
                text="THURSDAY, 30 JULY 2026 BENGALURU 13",
                bbox=(50.0, 10.0, 950.0, 40.0),
            ),
            # Middle body advertisement with numbers 181, 182, 31 (y0=500, y1=600 -> DISCARDED)
            DigitalTextBlock(
                block_id=1,
                text="Call 181-182-31 for commercial real estate offers. Price Rs. 31 Lakhs M",
                bbox=(50.0, 500.0, 450.0, 600.0),
            ),
            # Lower middle body story (y0=700, y1=900 -> DISCARDED)
            DigitalTextBlock(
                block_id=2,
                text="Section M Report on Industrial Growth 181",
                bbox=(50.0, 700.0, 450.0, 900.0),
            ),
        ]

        folio = detector.extract_folio(
            blocks=blocks,
            height_px=1400.0,
            width_px=1000.0,
            page_number=13,
        )
        assert folio == "13"

    def test_extract_folio_with_dictionary_inputs(self) -> None:
        """Verify extract_folio works with generic dict-based bounding box inputs."""
        detector = FolioDetector()
        dict_blocks = [
            {
                "bbox": [50.0, 15.0, 950.0, 45.0],
                "text": "DELHI | THURSDAY, JULY 30, 2026 | PAGE 14",
            },
            {
                "bbox": [50.0, 400.0, 450.0, 800.0],
                "text": "Body text with false number 182 and ad copy",
            },
        ]

        folio = detector.extract_folio(
            blocks=dict_blocks,
            height_px=1400.0,
            width_px=1000.0,
            page_number=14,
        )
        assert folio == "14"

    def test_footer_zone_folio_extraction(self) -> None:
        """Verify folio in the bottom 5% footer zone is extracted correctly."""
        detector = FolioDetector()
        blocks = [
            # Body text (y0=200, y1=800 -> DISCARDED)
            DigitalTextBlock(
                block_id=0,
                text="Editorial columns and general opinions 181",
                bbox=(50.0, 200.0, 450.0, 800.0),
            ),
            # Bottom footer folio (y0=1350 >= 1400 * 0.95 = 1330)
            DigitalTextBlock(
                block_id=1,
                text="PAGE 7",
                bbox=(450.0, 1350.0, 550.0, 1380.0),
            ),
        ]

        folio = detector.extract_folio(
            blocks=blocks,
            height_px=1400.0,
            width_px=1000.0,
            page_number=7,
        )
        assert folio == "7"

    def test_full_page_spanning_ocr_block_rejection(self) -> None:
        """Verify full-page spanning OCR blocks (height span > 15%) are rejected."""
        detector = FolioDetector()
        blocks = [
            # Giant OCR block spanning entire page from 0 to 1400 (height span = 100%)
            OCRBlock(
                text="181 Full scanned page text 182 with random numbers 31",
                bbox=(0.0, 0.0, 1000.0, 1400.0),
                confidence=0.88,
            ),
        ]

        folio = detector.extract_folio(
            blocks=blocks,
            height_px=1400.0,
            width_px=1000.0,
            page_number=9,
        )
        # Giant block is rejected, falls back to unnumbered
        assert folio == "Unnumbered (PDF p.9)"

    def test_ocr_300_dpi_coordinates_with_body_numbers_and_phone_numbers(self) -> None:
        """Verify OCR blocks on a 300 DPI image (height=4399px) isolate header and ignore body."""
        detector = FolioDetector()
        ocr_blocks = [
            # Header block in top 3% (y0=120, y1=180 on 4399px -> relative_y = 0.027 < 0.08)
            OCRBlock(
                text="MINT | THURSDAY, 30 JULY 2026 | BENGALURU 13",
                bbox=(150.0, 120.0, 2800.0, 180.0),
                confidence=0.95,
            ),
            # Body advertisement with phone numbers and false folios (y0=2000, y1=2400)
            OCRBlock(
                text="For classifieds call 181-182-31. Special property deals Rs. 31 Lakhs. M.",
                bbox=(150.0, 2000.0, 1400.0, 2400.0),
                confidence=0.91,
            ),
            # Lower body story with numbers (y0=3000, y1=3400)
            OCRBlock(
                text="Financial analysis on 181 index stocks",
                bbox=(150.0, 3000.0, 1400.0, 3400.0),
                confidence=0.89,
            ),
        ]

        folio = detector.extract_folio(
            blocks=ocr_blocks,
            height_px=4399.0,
            width_px=3100.0,
            page_number=13,
        )
        assert folio == "13"

    def test_dpi_mismatch_auto_scale_sync(self) -> None:
        """Verify detector syncs when 72 DPI PDF blocks are passed with 300 DPI height."""
        detector = FolioDetector()
        # Blocks in 72 DPI points (0..842)
        digital_blocks = [
            DigitalTextBlock(
                block_id=0,
                text="DELHI | MONDAY, JULY 7, 2026 | PAGE 12",
                bbox=(50.0, 20.0, 550.0, 45.0),
            ),
            # Body block at y0=300, y1=400 (which would be < 0.08 of 3508 without auto-sync)
            DigitalTextBlock(
                block_id=1,
                text="Body article mentioning 181 and 182",
                bbox=(50.0, 300.0, 550.0, 400.0),
            ),
        ]

        # Height passed as 3508 (rasterized pixels) while blocks are at ~842 points
        folio = detector.extract_folio(
            blocks=digital_blocks,
            height_px=3508.0,
            width_px=2479.0,
            page_number=12,
        )
        assert folio == "12"

    def test_missing_bbox_coordinates_fallback_gracefully(self) -> None:
        """Verify blocks missing bbox coordinates fall back gracefully without scanning body."""
        detector = FolioDetector()
        # Blocks missing bounding box keys or attributes
        unstructured_blocks = [
            {"text": "Call 181-182-31 for customer support. Page 181."},
            {"text": "Random body text."},
        ]

        folio = detector.extract_folio(
            blocks=unstructured_blocks,
            height_px=1400.0,
            width_px=1000.0,
            page_number=5,
        )
        # Should not extract "181" from body text
        assert folio == "Unnumbered (PDF p.5)"

    def test_rejects_single_brand_chars_such_as_m_and_extrapolates(self) -> None:
        """Verify brand letter 'M' is rejected as Roman numeral and extrapolates cleanly."""
        detector = FolioDetector()
        ocr_blocks = [
            OCRBlock(
                text="M",
                bbox=(50.0, 20.0, 80.0, 50.0),
                confidence=0.98,
            ),
            OCRBlock(
                text="MINT.COM | THURSDAY, JULY 30 2026",
                bbox=(100.0, 20.0, 900.0, 50.0),
                confidence=0.95,
            ),
        ]
        # PDF page 19, with last known folio 18 on PDF page 18
        folio = detector.extract_printed_page_number(
            page_number=19,
            height_px=4399.0,
            width_px=2800.0,
            ocr_blocks=ocr_blocks,
            last_known_folio_num=18,
            last_known_pdf_page=18,
        )
        # Must extrapolate to 19 instead of extracting 'M'
        assert folio == "19"

    def test_section_boundary_safety_does_not_increment_non_integer(self) -> None:
        """Verify non-integer section folio does not blindly increment."""
        detector = FolioDetector()
        folio = detector.extract_printed_page_number(
            page_number=20,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=[],
            last_known_folio_num=None,  # Section folio e.g. "B-1"
            last_known_pdf_page=19,
        )
        assert folio == "Unnumbered (PDF p.20)"

    def test_masthead_date_30_31_july_not_extracted_as_page_number(self) -> None:
        """Verify date stamps like '30 JULY 2026' are not extracted as page 30."""
        detector = FolioDetector()
        ocr_blocks = [
            OCRBlock(
                text="MINT | THURSDAY, 30 JULY 2026",
                bbox=(50.0, 20.0, 950.0, 50.0),
                confidence=0.98,
            ),
            OCRBlock(
                text="Opinion column on artificial intelligence...",
                bbox=(50.0, 100.0, 450.0, 600.0),
                confidence=0.95,
            ),
        ]
        folio = detector.extract_printed_page_number(
            page_number=15,
            height_px=1400.0,
            width_px=1000.0,
            ocr_blocks=ocr_blocks,
            total_issue_pages=16,
        )
        assert folio != "30"
        assert folio == "Unnumbered (PDF p.15)"

    def test_total_pages_upper_bound_rejects_out_of_range_folios(self) -> None:
        """Verify out-of-range folios (e.g. 42 on 16-page issue) are rejected."""
        detector = FolioDetector()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="PAGE 42",
                bbox=(50.0, 20.0, 150.0, 50.0),
            )
        ]
        folio = detector.extract_printed_page_number(
            page_number=10,
            height_px=1400.0,
            width_px=1000.0,
            digital_blocks=blocks,
            total_issue_pages=16,
        )
        assert folio != "42"
        assert folio == "Unnumbered (PDF p.10)"
