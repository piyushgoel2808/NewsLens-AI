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

    def test_noise_and_boilerplate_blocks_purged(self) -> None:
        """Verify brand logos, sponsor boilerplate (JM Financial, ASBA), and dates are purged."""
        analyzer = LayoutAnalyzer()
        blocks = [
            # Top 5% isolated brand logo
            DigitalTextBlock(
                block_id=0,
                text="mint",
                bbox=(50.0, 10.0, 150.0, 40.0),
                mean_font_size=14.0,
                is_heading_candidate=False,
            ),
            # Financial sponsor box
            DigitalTextBlock(
                block_id=1,
                text="BOOK RUNNING LEAD MANAGERS: JM FINANCIAL | AXIS CAPITAL",
                bbox=(50.0, 800.0, 950.0, 840.0),
                mean_font_size=9.0,
                is_heading_candidate=False,
            ),
            # ASBA stamp
            DigitalTextBlock(
                block_id=2,
                text="ASBA: Applications Supported by Blocked Amount",
                bbox=(50.0, 850.0, 450.0, 880.0),
                mean_font_size=8.0,
                is_heading_candidate=False,
            ),
            # Actual article headline and body
            DigitalTextBlock(
                block_id=3,
                text="PHARMA EXPORTS GROW 12 PERCENT IN FIRST QUARTER",
                bbox=(50.0, 100.0, 950.0, 140.0),
                mean_font_size=20.0,
                is_heading_candidate=True,
            ),
            DigitalTextBlock(
                block_id=4,
                text="Indian generic pharmaceutical shipments to the US and Europe rose sharply.",
                bbox=(50.0, 150.0, 450.0, 250.0),
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
        element_texts = [e.text for e in res.elements]
        assert not any(t.lower() == "mint" for t in element_texts)
        assert not any("JM FINANCIAL" in t for t in element_texts)
        assert not any("ASBA" in t for t in element_texts)
        assert res.elements[0].text == "PHARMA EXPORTS GROW 12 PERCENT IN FIRST QUARTER"

    def test_horizontal_headline_stitching_across_columns(self) -> None:
        """Verify multi-column sliced headlines are merged horizontally into a banner headline."""
        analyzer = LayoutAnalyzer()
        blocks = [
            # Headline Slice 1 (Left column track: x=50..450, y=100..140)
            DigitalTextBlock(
                block_id=0,
                text="OpenAI says",
                bbox=(50.0, 100.0, 450.0, 140.0),
                mean_font_size=24.0,
                is_heading_candidate=True,
            ),
            # Headline Slice 2 (Right column track: x=480..950, y=100..140)
            DigitalTextBlock(
                block_id=1,
                text="rogue AI agent attack hit other companies",
                bbox=(480.0, 100.0, 950.0, 140.0),
                mean_font_size=24.0,
                is_heading_candidate=True,
            ),
            # Underlying body text
            DigitalTextBlock(
                block_id=2,
                text="A state-sponsored group attempted to exploit developer toolchains.",
                bbox=(50.0, 160.0, 450.0, 250.0),
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

        # The two headline slices must be merged horizontally into 1 banner headline
        assert len(res.elements) == 2
        assert res.elements[0].block_type == BlockType.BANNER_HEADLINE
        expected_full_hl = "OpenAI says rogue AI agent attack hit other companies"
        assert res.elements[0].text == expected_full_hl
        assert res.elements[0].bbox[0] == 50.0
        assert res.elements[0].bbox[2] == 950.0

    def test_side_by_side_independent_headlines_not_merged(self) -> None:
        """Verify independent complete headlines across column tracks are not merged."""
        analyzer = LayoutAnalyzer()
        blocks = [
            # Story 1 Headline (Left column track: x=50..450, y=100..140)
            DigitalTextBlock(
                block_id=0,
                text="ChrysCapital buys controlling stake in Novartis India",
                bbox=(50.0, 100.0, 450.0, 140.0),
                mean_font_size=22.0,
                is_heading_candidate=True,
            ),
            # Story 2 Headline (Right column track: x=480..950, y=100..140)
            DigitalTextBlock(
                block_id=1,
                text="E-bus makers may seek new localization waiver",
                bbox=(480.0, 100.0, 950.0, 140.0),
                mean_font_size=22.0,
                is_heading_candidate=True,
            ),
            # Story 1 Body (Left column: x=50..450, y=150..300)
            DigitalTextBlock(
                block_id=2,
                text="Private equity major ChrysCapital is set to acquire the pharmaceutical unit.",
                bbox=(50.0, 150.0, 450.0, 300.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
            # Story 2 Body (Right column: x=480..950, y=150..300)
            DigitalTextBlock(
                block_id=3,
                text="Electric bus manufacturers are preparing a joint petition to the ministry.",
                bbox=(480.0, 150.0, 950.0, 300.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=7,
            width_px=1000,
            height_px=1400,
            digital_blocks=blocks,
        )

        # Must remain 4 distinct elements (2 separate headlines, 2 separate body blocks)
        assert len(res.elements) == 4
        headlines = [
            e.text for e in res.elements
            if e.block_type in (BlockType.HEADLINE, BlockType.BANNER_HEADLINE)
        ]
        assert "ChrysCapital buys controlling stake in Novartis India" in headlines
        assert "E-bus makers may seek new localization waiver" in headlines

    def test_numeric_stat_box_classified_as_table_not_headline(self) -> None:
        """Verify numeric figures/currency stat boxes are tagged as TABLE, never HEADLINE."""
        analyzer = LayoutAnalyzer()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="75 cr 3,620.40 cr 4,167 cr $250 mn",
                bbox=(50.0, 100.0, 450.0, 140.0),
                mean_font_size=24.0,  # Large font but pure numbers/currency
                is_heading_candidate=True,
            ),
            DigitalTextBlock(
                block_id=1,
                text="The key financial metrics reported across public sector undertakings.",
                bbox=(50.0, 150.0, 450.0, 250.0),
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
        assert res.elements[0].block_type == BlockType.TABLE
        assert res.elements[1].block_type == BlockType.BODY_TEXT

    def test_vertical_multiline_headlines_stitched_before_horizontal_lookahead(self) -> None:
        """Verify side-by-side headlines stitch vertically into columns and do not collide."""
        analyzer = LayoutAnalyzer()
        blocks = [
            # Column 1 - Line 1 (ends with auxiliary verb 'could')
            DigitalTextBlock(
                block_id=0,
                text="How artificial intelligence could",
                bbox=(50.0, 100.0, 450.0, 130.0),
                mean_font_size=20.0,
                is_heading_candidate=True,
            ),
            # Column 2 - Line 1 (ends with 'as')
            DigitalTextBlock(
                block_id=1,
                text="Boeing's runway looks clear as",
                bbox=(480.0, 100.0, 950.0, 130.0),
                mean_font_size=20.0,
                is_heading_candidate=True,
            ),
            # Column 1 - Line 2
            DigitalTextBlock(
                block_id=2,
                text="reinforce the dollar's dominance",
                bbox=(50.0, 135.0, 450.0, 165.0),
                mean_font_size=20.0,
                is_heading_candidate=True,
            ),
            # Column 2 - Line 2
            DigitalTextBlock(
                block_id=3,
                text="makers of jet engines struggle",
                bbox=(480.0, 135.0, 950.0, 165.0),
                mean_font_size=20.0,
                is_heading_candidate=True,
            ),
            # Column 1 - Body
            DigitalTextBlock(
                block_id=4,
                text="Research notes explain the rising currency dynamic across markets.",
                bbox=(50.0, 175.0, 450.0, 400.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
            # Column 2 - Body
            DigitalTextBlock(
                block_id=5,
                text="Commercial aircraft deliveries are expected to surge as backlogs clear.",
                bbox=(480.0, 175.0, 950.0, 400.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=15,
            width_px=1000,
            height_px=1400,
            digital_blocks=blocks,
        )

        headlines = [
            e.text for e in res.elements
            if e.block_type in (BlockType.HEADLINE, BlockType.BANNER_HEADLINE)
        ]
        # Must have 2 clean, discrete vertical headlines
        assert len(headlines) == 2
        assert "How artificial intelligence could reinforce the dollar's dominance" in headlines
        assert "Boeing's runway looks clear as makers of jet engines struggle" in headlines
        # Must never have the cross-column corrupted collision
        assert not any("could Boeing's runway" in h for h in headlines)

    def test_heading_boundary_break_prevents_multi_article_swallowing(self) -> None:
        """Test that vertically stacked articles in same lane never swallow each other."""
        analyzer = LayoutAnalyzer()
        # Story 1: Headline + Body (Column 1)
        # Story 2: Headline + Body (Column 1, beneath Story 1)
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="BMW offers severance packages to employees to cut 8,000 jobs",
                bbox=(50.0, 100.0, 450.0, 140.0),
                mean_font_size=18.0,
                is_heading_candidate=True,
            ),
            DigitalTextBlock(
                block_id=1,
                text="German automaker BMW announced restructuring packages for corporate staff.",
                bbox=(50.0, 150.0, 450.0, 300.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
            DigitalTextBlock(
                block_id=2,
                text="France accuses Telegram CEO Pavel Durov in broad probe",
                bbox=(50.0, 330.0, 450.0, 370.0),
                mean_font_size=18.0,
                is_heading_candidate=True,
            ),
            DigitalTextBlock(
                block_id=3,
                text="French prosecutors opened formal judicial probe into cybercrime allegations.",
                bbox=(50.0, 380.0, 450.0, 520.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=10,
            width_px=1000,
            height_px=1400,
            digital_blocks=blocks,
        )

        headlines = [
            e.text for e in res.elements
            if e.block_type in (BlockType.HEADLINE, BlockType.BANNER_HEADLINE)
        ]
        assert len(headlines) == 2
        assert "BMW offers severance packages to employees to cut 8,000 jobs" in headlines
        assert "France accuses Telegram CEO Pavel Durov in broad probe" in headlines

        from app.ingestion.segmenter import ArticleSegmenter
        segmenter = ArticleSegmenter()
        articles = segmenter.segment_page(page_number=10, ordered_blocks=res.reading_order)
        assert len(articles) == 2
        art_headlines = [a.headline for a in articles]
        assert "BMW offers severance packages to employees to cut 8,000 jobs" in art_headlines
        assert "France accuses Telegram CEO Pavel Durov in broad probe" in art_headlines
        # Verify Story 2's body text is NOT swallowed into Story 1
        bmw_art = next(a for a in articles if "BMW" in a.headline)
        assert "German automaker BMW" in bmw_art.body_text
        assert "Telegram CEO" not in bmw_art.body_text

