"""Unit tests for LayoutAnalyzer (VLM and rule-based modes)."""

from __future__ import annotations

import pytest

from app.ingestion.detector import DigitalTextBlock
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.reading_order import BlockType
from app.providers.base import ModelResponse, ProviderCapability


class MockVisionProvider:
    """Mock VisionModelProvider for testing VLM layout analysis."""

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(supports_vision=True, supports_structured_output=True)

    @property
    def provider_name(self) -> str:
        return "mock_vlm"

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        return ModelResponse(
            text="",
            parsed={
                "headlines": [
                    {
                        "text": "SENATE PASSES TARIFF REFORM",
                        "level": "banner",
                        "bbox": [50.0, 30.0, 950.0, 100.0],
                    }
                ],
                "columns": [
                    {
                        "column_index": 1,
                        "text": "Debate concluded late last night...",
                        "bbox": [50.0, 120.0, 480.0, 800.0],
                    },
                    {
                        "column_index": 2,
                        "text": "Opposition leaders argued strongly against...",
                        "bbox": [520.0, 120.0, 950.0, 800.0],
                    },
                ],
                "photos": [
                    {
                        "caption": "Senate floor during vote",
                        "bbox": [520.0, 120.0, 950.0, 350.0],
                    }
                ],
                "tables": [],
            },
            model="mock-vlm",
            provider="mock",
        )


class TestLayoutAnalyzer:
    """Test layout analysis using VLM and rule-based methods."""

    @pytest.mark.asyncio
    async def test_analyze_page_with_vlm(self) -> None:
        mock_vision = MockVisionProvider()
        analyzer = LayoutAnalyzer(vision_provider=mock_vision)

        res = await analyzer.analyze_page(
            page_number=1,
            width_px=1000,
            height_px=1500,
            image_bytes=b"dummy-image",
        )

        assert res.source == "vlm"
        assert len(res.elements) >= 3
        # Banner headline
        banner_elements = [e for e in res.elements if e.block_type == BlockType.BANNER_HEADLINE]
        assert len(banner_elements) == 1
        assert banner_elements[0].text == "SENATE PASSES TARIFF REFORM"
        assert len(res.reading_order) >= 3

    def test_analyze_from_digital_text_blocks_rule_based(self) -> None:
        analyzer = LayoutAnalyzer()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="MAJOR CITY DISPATCH",
                bbox=(50.0, 30.0, 950.0, 90.0),
                mean_font_size=24.0,
                is_heading_candidate=True,
            ),
            DigitalTextBlock(
                block_id=1,
                text="Column 1 story content...",
                bbox=(50.0, 100.0, 450.0, 600.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
            DigitalTextBlock(
                block_id=2,
                text="Column 2 story content...",
                bbox=(500.0, 100.0, 900.0, 600.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=1,
            width_px=1000,
            height_px=1500,
            digital_blocks=blocks,
        )

        assert len(res.reading_order) == 3
        # First reading block should be the wide headline
        assert res.reading_order[0].block_type == BlockType.BANNER_HEADLINE

    def test_analyze_from_ocr_blocks_distinguishes_headlines_and_body(self) -> None:
        """Verify dynamic line height correctly separates OCR headlines from body text."""
        from app.providers.base import OCRBlock

        analyzer = LayoutAnalyzer()
        ocr_blocks = [
            OCRBlock(
                text="TATA POWER POSTS RECORD PROFIT",
                bbox=(50.0, 50.0, 700.0, 110.0),
                confidence=0.95,
            ),
            OCRBlock(
                text="By Special Correspondent\nNew Delhi",
                bbox=(50.0, 120.0, 250.0, 160.0),
                confidence=0.92,
            ),
            OCRBlock(
                text=(
                    "Tata Power reported strong revenue growth in the first quarter.\n"
                    "Net profit rose by 15 percent year on year."
                ),
                bbox=(50.0, 180.0, 450.0, 280.0),
                confidence=0.90,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=2,
            width_px=1000,
            height_px=1400,
            ocr_blocks=ocr_blocks,
        )

        assert res.source == "spatial_rule_based"
        assert len(res.elements) == 2
        assert res.elements[0].block_type in (
            BlockType.BANNER_HEADLINE,
            BlockType.HEADLINE,
        )
        assert res.elements[1].block_type == BlockType.BODY_TEXT

    def test_spatial_consolidation_merges_adjacent_column_paragraphs(self) -> None:
        """Verify adjacent body paragraphs in the same column are merged."""
        analyzer = LayoutAnalyzer()
        blocks = [
            # Headline
            DigitalTextBlock(
                block_id=0,
                text="MARKET SUMMARY REPORT",
                bbox=(50.0, 30.0, 950.0, 70.0),
                mean_font_size=20.0,
                is_heading_candidate=True,
            ),
            # Paragraph 1 (same column)
            DigitalTextBlock(
                block_id=1,
                text="Equity benchmark indices opened in the green on positive Asian cues.",
                bbox=(50.0, 80.0, 450.0, 130.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
            # Paragraph 2 directly below (gap = 8px)
            DigitalTextBlock(
                block_id=2,
                text="Banking and IT stocks led the initial rally with broad-based buying.",
                bbox=(50.0, 138.0, 450.0, 185.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=1,
            width_px=1000,
            height_px=1400,
            digital_blocks=blocks,
        )

        # The 2 adjacent body paragraphs should be consolidated into 1 body element
        assert len(res.elements) == 2
        assert res.elements[0].block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
        assert res.elements[1].block_type == BlockType.BODY_TEXT
        assert "Equity benchmark" in res.elements[1].text
        assert "Banking and IT" in res.elements[1].text

    def test_spatial_consolidation_merges_multiline_headlines(self) -> None:
        """Verify multi-line headline fragments are merged into a single headline element."""
        analyzer = LayoutAnalyzer()
        blocks = [
            # Headline line 1
            DigitalTextBlock(
                block_id=0,
                text="GOVERNMENT APPROVES MAJOR",
                bbox=(50.0, 30.0, 950.0, 65.0),
                mean_font_size=22.0,
                is_heading_candidate=True,
            ),
            # Headline line 2
            DigitalTextBlock(
                block_id=1,
                text="RENEWABLE ENERGY INCENTIVE PACKAGE",
                bbox=(50.0, 70.0, 950.0, 105.0),
                mean_font_size=22.0,
                is_heading_candidate=True,
            ),
            # Body text
            DigitalTextBlock(
                block_id=2,
                text="The union cabinet approved the scheme late yesterday evening.",
                bbox=(50.0, 120.0, 450.0, 200.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=1,
            width_px=1000,
            height_px=1400,
            digital_blocks=blocks,
        )

        assert len(res.elements) == 2
        assert res.elements[0].block_type == BlockType.BANNER_HEADLINE
        expected_hl = "GOVERNMENT APPROVES MAJOR RENEWABLE ENERGY INCENTIVE PACKAGE"
        assert expected_hl in res.elements[0].text
        assert res.elements[1].block_type == BlockType.BODY_TEXT

    def test_masthead_purged_from_layout_elements(self) -> None:
        """Verify date stamps, folios, and mastheads in top 8% are purged."""
        analyzer = LayoutAnalyzer()
        blocks = [
            # Top running masthead (y0=10, y1=30, in top 8% of 1400px page)
            DigitalTextBlock(
                block_id=0,
                text="MINT | THURSDAY, 30 JULY 2026 | PAGE 14",
                bbox=(50.0, 10.0, 950.0, 30.0),
                mean_font_size=9.0,
                is_heading_candidate=False,
            ),
            # Actual article headline (y0=100)
            DigitalTextBlock(
                block_id=1,
                text="CENTRAL BANK MAINTAINS REPO RATE STABILITY",
                bbox=(50.0, 100.0, 950.0, 140.0),
                mean_font_size=20.0,
                is_heading_candidate=True,
            ),
            # Body text (y0=150)
            DigitalTextBlock(
                block_id=2,
                text="The monetary policy committee voted unanimously to hold rates steady.",
                bbox=(50.0, 150.0, 450.0, 250.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=14,
            width_px=1000,
            height_px=1400,
            digital_blocks=blocks,
        )

        # The masthead block must be completely filtered out
        assert len(res.elements) == 2
        assert not any("THURSDAY, 30 JULY 2026" in e.text for e in res.elements)
        assert res.elements[0].text == "CENTRAL BANK MAINTAINS REPO RATE STABILITY"
