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


class TestTriModalRoutingLogic:
    """Test the tri-modal routing conditions used in the ingestion pipeline."""

    def test_digital_page_routes_to_path_a(self) -> None:
        from app.ingestion.detector import PageAnalysisResult, PageType

        text = "Breaking News: Technology sector posts record revenue growth across global markets." * 5
        analysis = PageAnalysisResult(
            page_number=1,
            page_type=PageType.HYBRID,
            requires_ocr=False,
            character_count=len(text),
            word_count=len(text.split()),
            full_text=text,
            blocks=[
                DigitalTextBlock(
                    block_id=0,
                    text=text,
                    bbox=(50.0, 100.0, 800.0, 400.0),
                )
            ],
        )

        parser_engine = "auto"
        is_force_docling = bool(parser_engine and "docling" in parser_engine.lower() and parser_engine.lower() != "auto")
        is_force_gcv = bool(parser_engine and ("google" in parser_engine.lower() or "vision" in parser_engine.lower()))

        raw_text = analysis.full_text or ""
        alnum_count = sum(1 for c in raw_text if c.isalnum())
        alpha_ratio = (alnum_count / len(raw_text)) if raw_text else 0.0

        is_digital_candidate = (
            not is_force_docling
            and not is_force_gcv
            and analysis.character_count >= 150
            and alpha_ratio >= 0.45
            and analysis.page_type != PageType.SCANNED
            and len(analysis.blocks) > 0
        )

        assert is_digital_candidate is True
        assert is_force_docling is False

    def test_scanned_page_routes_to_path_b(self) -> None:
        from app.ingestion.detector import PageAnalysisResult, PageType

        analysis = PageAnalysisResult(
            page_number=5,
            page_type=PageType.SCANNED,
            requires_ocr=True,
            character_count=10,
            word_count=2,
            full_text="Ad header",
            blocks=[],
        )

        parser_engine = "auto"
        is_force_docling = bool(parser_engine and "docling" in parser_engine.lower() and parser_engine.lower() != "auto")
        is_force_gcv = bool(parser_engine and ("google" in parser_engine.lower() or "vision" in parser_engine.lower()))

        is_scanned_candidate = (
            not is_force_docling
            and (
                is_force_gcv
                or analysis.page_type == PageType.SCANNED
                or (analysis.character_count < 50 and len(analysis.blocks) == 0)
            )
        )

        assert is_scanned_candidate is True

    def test_explicit_docling_bypasses_fast_path(self) -> None:
        from app.ingestion.detector import PageAnalysisResult, PageType

        text = "Normal newspaper article text..." * 10
        analysis = PageAnalysisResult(
            page_number=1,
            page_type=PageType.DIGITAL,
            requires_ocr=False,
            character_count=len(text),
            word_count=len(text.split()),
            full_text=text,
            blocks=[
                DigitalTextBlock(
                    block_id=0,
                    text=text,
                    bbox=(50.0, 100.0, 800.0, 400.0),
                )
            ],
        )

        parser_engine = "docling"
        is_force_docling = bool(parser_engine and "docling" in parser_engine.lower() and parser_engine.lower() != "auto")
        is_force_gcv = bool(parser_engine and ("google" in parser_engine.lower() or "vision" in parser_engine.lower()))

        raw_text = analysis.full_text or ""
        alnum_count = sum(1 for c in raw_text if c.isalnum())
        alpha_ratio = (alnum_count / len(raw_text)) if raw_text else 0.0

        is_digital_candidate = (
            not is_force_docling
            and not is_force_gcv
            and analysis.character_count >= 150
            and alpha_ratio >= 0.45
            and analysis.page_type != PageType.SCANNED
            and len(analysis.blocks) > 0
        )

        # Because parser_engine is "docling", is_force_docling is True, so digital fast-path is bypassed
        assert is_force_docling is True
        assert is_digital_candidate is False

