"""Unit and integration tests for tasks.py, masthead detection, and pipeline execution."""

from __future__ import annotations

from datetime import date

from app.ingestion.detector import DigitalTextBlock
from app.ingestion.tasks import detect_masthead_and_date
from app.providers.base import OCRBlock


class TestMastheadAndDateDetection:
    """Test dynamic masthead and publication date extraction from Page 1 blocks."""

    def test_detect_mint_masthead_and_date_digital(self) -> None:
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="MINT | THURSDAY, 30 JULY 2026 | NEW DELHI",
                bbox=(50.0, 10.0, 950.0, 40.0),
            ),
            DigitalTextBlock(
                block_id=1,
                text="Cognizant beats IT peers, cuts outlook",
                bbox=(50.0, 100.0, 950.0, 150.0),
            ),
        ]
        brand, pub_date = detect_masthead_and_date(blocks, height_px=1400.0)
        assert brand == "Mint"
        assert pub_date == date(2026, 7, 30)

    def test_detect_business_standard_and_date_ocr(self) -> None:
        blocks = [
            OCRBlock(
                text="BUSINESS STANDARD | MUMBAI | AUGUST 21, 2026",
                bbox=(50.0, 10.0, 950.0, 40.0),
                confidence=0.98,
            ),
            OCRBlock(
                text="Market rallies to fresh lifetime highs",
                bbox=(50.0, 100.0, 950.0, 150.0),
                confidence=0.95,
            ),
        ]
        brand, pub_date = detect_masthead_and_date(blocks, height_px=1400.0)
        assert brand == "Business Standard"
        assert pub_date == date(2026, 8, 21)

    def test_ignore_blocks_lower_in_page(self) -> None:
        """Blocks far down the page should not override masthead."""
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="Some article mentioning 15 August 1947",
                bbox=(50.0, 600.0, 950.0, 650.0),
            )
        ]
        brand, pub_date = detect_masthead_and_date(blocks, height_px=1400.0)
        assert brand is None
        assert pub_date is None
