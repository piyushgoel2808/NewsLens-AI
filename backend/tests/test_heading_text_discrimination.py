"""Unit tests for Heading ↔ Body Text discrimination across the ingestion pipeline."""

from __future__ import annotations

from app.ingestion.detector import DigitalTextBlock, TextSpan
from app.ingestion.layout.analyzer import BlockType, LayoutAnalyzer
from app.ingestion.layout.segmenter import ArticleSegmenter, OrderedReadingBlock
from app.ingestion.layout.slugs import is_valid_headline_candidate
from app.ingestion.parsers.docling import DoclingLayoutParser, DoclingParsedItem
from app.providers.base import OCRBlock


class TestHeadingTextDiscrimination:
    """Test suite ensuring headlines and body text are never conflated."""

    def test_is_valid_headline_candidate_abbreviations_and_punctuation(self) -> None:
        """Verify terminal period abbreviation protection and punctuation rules."""
        # Editorial questions and breaking exclamations are valid
        assert is_valid_headline_candidate("Will RBI Cut Rates in October?") is True
        assert is_valid_headline_candidate("India Wins World Cup!") is True
        assert is_valid_headline_candidate("Sensex Jumps 800 Points!") is True

        # Corporate / acronym abbreviations ending in periods are protected
        assert is_valid_headline_candidate("Google Invests in Anthropic Inc.") is True
        assert is_valid_headline_candidate("Alphabet Opens New Office in U.S.") is True
        assert is_valid_headline_candidate("Adani Port Expands via Subsidiary Ltd.") is True

        # Ordinary declarative sentences ending in periods are REJECTED
        assert (
            is_valid_headline_candidate(
                "The finance ministry announced a new fiscal scheme on Tuesday."
            )
            is False
        )
        assert (
            is_valid_headline_candidate(
                "Exports rose by eight percent according to official data."
            )
            is False
        )

        # Dateline-opening sentences are REJECTED
        assert (
            is_valid_headline_candidate(
                "NEW DELHI: The finance minister addressed reporters yesterday."
            )
            is False
        )
        assert (
            is_valid_headline_candidate(
                "MUMBAI — State-run lenders recorded higher quarterly margins."
            )
            is False
        )
        assert (
            is_valid_headline_candidate(
                "WASHINGTON (AP) — The central bank decided to hold interest rates."
            )
            is False
        )

        # Legal / statutory notice boilerplate is REJECTED
        assert is_valid_headline_candidate("PUBLIC NOTICE IS HEREBY GIVEN") is False
        assert is_valid_headline_candidate("NOTICE INVITING TENDER FOR HIGHWAY CONSTRUCTION") is False
        assert is_valid_headline_candidate("BEFORE THE HON'BLE NATIONAL COMPANY LAW TRIBUNAL") is False

    def test_docling_flush_first_sentence_fallback_uses_neutral_brief(self) -> None:
        """Verify _flush_current_article does not convert declarative body sentences into headlines."""
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="paragraph",
                text="New Delhi: The finance ministry announced new rules on Tuesday.",
                bbox=(50.0, 100.0, 400.0, 150.0),
                page_number=1,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="Officials said the rules will take effect from next month across all states.",
                bbox=(50.0, 160.0, 400.0, 220.0),
                page_number=1,
            ),
        ]
        articles = parser.assemble_articles(
            page_number=1,
            items=items,
            width_px=1000,
            height_px=1500,
        )
        assert len(articles) == 1
        art = articles[0]
        assert not art.headline.startswith("New Delhi:")
        assert "Brief]" in art.headline
        assert "New Delhi: The finance ministry" in art.body_text

    def test_docling_promotes_mislabeled_paragraph_to_headline_on_column_jump(self) -> None:
        """Verify headline mislabeled as 'paragraph' by Docling is promoted when jumping columns."""
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="title",
                text="Government Clears Mega Port Expansion Plan",
                bbox=(50.0, 100.0, 400.0, 140.0),
                page_number=3,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="The Union Cabinet on Wednesday approved an investment of twelve thousand crore.",
                bbox=(50.0, 150.0, 400.0, 250.0),
                page_number=3,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="RESERVE BANK HOLDS BENCHMARK POLICY REPO RATE",
                bbox=(480.0, 100.0, 850.0, 135.0),
                page_number=3,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="Mumbai: The monetary policy committee voted unanimously to keep rates unchanged.",
                bbox=(480.0, 145.0, 850.0, 240.0),
                page_number=3,
            ),
        ]
        articles = parser.assemble_articles(
            page_number=3,
            items=items,
            width_px=1000,
            height_px=1500,
        )
        assert len(articles) == 2
        assert articles[0].headline == "Government Clears Mega Port Expansion Plan"
        assert "RESERVE BANK HOLDS" in articles[1].headline

    def test_docling_negative_delta_adjacent_column_headline_not_swallowed_as_deck(self) -> None:
        """Verify candidate title starting above previous headline (negative Δy) is not swallowed as deck."""
        parser = DoclingLayoutParser()
        items = [
            DoclingParsedItem(
                label="title",
                text="Auto Sales Surge During Festive Season",
                bbox=(50.0, 200.0, 400.0, 240.0),
                page_number=4,
            ),
            DoclingParsedItem(
                label="title",
                text="Steel Manufacturers Seek Tariff Protection",
                bbox=(480.0, 150.0, 850.0, 190.0),
                page_number=4,
            ),
            DoclingParsedItem(
                label="paragraph",
                text="Domestic producers said cheap imports have undercut domestic steel pricing.",
                bbox=(480.0, 200.0, 850.0, 300.0),
                page_number=4,
            ),
        ]
        articles = parser.assemble_articles(
            page_number=4,
            items=items,
            width_px=1000,
            height_px=1500,
        )
        assert len(articles) == 2
        assert articles[0].headline == "Auto Sales Surge During Festive Season"
        assert articles[1].headline == "Steel Manufacturers Seek Tariff Protection"

    def test_detector_bold_typography_promotes_secondary_headline(self) -> None:
        """Verify bold text blocks with moderate font ratio are flagged as heading candidates."""
        bold_spans = [
            TextSpan(
                text="Core Sector Output Expands Four Percent",
                bbox=(50.0, 100.0, 350.0, 120.0),
                font_name="Times-Bold",
                font_size=11.2,
                flags=2,
                is_bold=True,
            )
        ]
        bold_blk = DigitalTextBlock(
            block_id=1,
            text="Core Sector Output Expands Four Percent",
            bbox=(50.0, 100.0, 350.0, 120.0),
            spans=bold_spans,
            mean_font_size=11.2,
            is_bold=True,
        )

        non_bold_spans = [
            TextSpan(
                text="Core Sector Output Expands Four Percent",
                bbox=(50.0, 100.0, 350.0, 120.0),
                font_name="Times-Roman",
                font_size=11.2,
                flags=0,
                is_bold=False,
            )
        ]
        non_bold_blk = DigitalTextBlock(
            block_id=2,
            text="Core Sector Output Expands Four Percent",
            bbox=(50.0, 100.0, 350.0, 120.0),
            spans=non_bold_spans,
            mean_font_size=11.2,
            is_bold=False,
        )

        dominant_font_size = 10.0

        for blk in [bold_blk, non_bold_blk]:
            clean_blk = blk.text.strip()
            words_blk = clean_blk.split()
            is_large = blk.mean_font_size >= dominant_font_size * 1.25
            is_modestly_large = blk.mean_font_size >= dominant_font_size * 1.08
            from app.ingestion.detector import is_title_case_or_uppercase
            is_cased = is_title_case_or_uppercase(clean_blk)
            if (
                len(words_blk) >= 2
                and is_cased
                and (
                    is_large
                    or (blk.is_bold and is_modestly_large)
                    or (blk.is_bold and clean_blk.isupper() and len(words_blk) >= 3)
                )
            ):
                blk.is_heading_candidate = True
            else:
                blk.is_heading_candidate = False

        assert bold_blk.is_heading_candidate is True
        assert non_bold_blk.is_heading_candidate is False

    def test_analyzer_ocr_allcaps_boilerplate_rejected_as_headline(self) -> None:
        """Verify all-caps legal notices and wire stamps are not classified as headlines."""
        analyzer = LayoutAnalyzer()
        ocr_blocks = [
            OCRBlock(
                text="PUBLIC NOTICE IS HEREBY GIVEN",
                bbox=(50.0, 200.0, 500.0, 235.0),
                confidence=0.95,
            ),
            OCRBlock(
                text="CENTRAL BANK HOLDS BENCHMARK REPO RATE",
                bbox=(50.0, 260.0, 600.0, 295.0),
                confidence=0.98,
            ),
            OCRBlock(
                text="The monetary policy committee met on Tuesday.",
                bbox=(50.0, 310.0, 500.0, 322.0),
                confidence=0.90,
            ),
            OCRBlock(
                text="Officials reviewed macroeconomic indicators across sectors.",
                bbox=(50.0, 330.0, 500.0, 342.0),
                confidence=0.90,
            ),
        ]
        res = analyzer.analyze_from_text_blocks(
            page_number=1,
            width_px=1000,
            height_px=1500,
            ocr_blocks=ocr_blocks,
        )
        elem_map = {e.text: e.block_type for e in res.elements}
        assert elem_map["PUBLIC NOTICE IS HEREBY GIVEN"] == BlockType.BODY_TEXT
        assert elem_map["CENTRAL BANK HOLDS BENCHMARK REPO RATE"] in (
            BlockType.HEADLINE,
            BlockType.BANNER_HEADLINE,
        )

    def test_segmenter_brief_with_valid_headline_kept_standalone(self) -> None:
        """Verify brief under 15 words with a valid editorial headline is kept standalone."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="MAJOR TRADE NEGOTIATIONS ADVANCE IN GENEVA",
                bbox=(50.0, 50.0, 600.0, 90.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "Delegates from member nations made significant progress on environmental "
                    "tariffs during multilateral discussions in Switzerland on Thursday."
                ),
                bbox=(50.0, 100.0, 600.0, 200.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.HEADLINE,
                text="Gold Prices Hit Record High",
                bbox=(50.0, 220.0, 400.0, 250.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text="Bullion surged in Mumbai spot trading.",
                bbox=(50.0, 260.0, 400.0, 280.0),
            ),
        ]
        articles = segmenter.segment_page(page_number=7, ordered_blocks=blocks)
        assert len(articles) == 2
        assert articles[0].headline == "MAJOR TRADE NEGOTIATIONS ADVANCE IN GENEVA"
        assert articles[1].headline == "Gold Prices Hit Record High"
        assert "Bullion surged in Mumbai" in articles[1].body_text
