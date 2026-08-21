"""Unit tests for Article Boundary Segmenter."""
from __future__ import annotations

from app.ingestion.reading_order import BlockType, OrderedReadingBlock
from app.ingestion.segmenter import ArticleSegmenter


class TestArticleSegmenter:
    """Test suite for ArticleSegmenter."""

    def test_empty_blocks_returns_empty(self) -> None:
        segmenter = ArticleSegmenter()
        articles = segmenter.segment_page(1, [])
        assert articles == []

    def test_segment_single_article(self) -> None:
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.BANNER_HEADLINE,
                text="MAJOR TECH REVOLUTION ANNOUNCED",
                bbox=(50.0, 50.0, 500.0, 80.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text="By John Doe",
                bbox=(50.0, 90.0, 200.0, 105.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "Scientists and engineers unveiled a groundbreaking system "
                    "today that changes everything."
                ),
                bbox=(50.0, 110.0, 250.0, 300.0),
            ),
        ]

        articles = segmenter.segment_page(1, blocks)
        assert len(articles) == 1
        art = articles[0]
        assert art.headline == "MAJOR TECH REVOLUTION ANNOUNCED"
        assert art.byline_author == "By John Doe"
        assert "groundbreaking system" in art.body_text
        assert art.word_count > 5
        assert len(art.bbox_list) == 3

    def test_segment_multiple_articles_and_jump_detection(self) -> None:
        segmenter = ArticleSegmenter()
        blocks = [
            # Article 1
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="MARKETS HIT RECORD HIGH",
                bbox=(30.0, 40.0, 250.0, 70.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "Stocks rallied across the globe as investor sentiment soared.\n"
                    "Continued on Page 4"
                ),
                bbox=(30.0, 75.0, 250.0, 200.0),
            ),
            # Article 2
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.HEADLINE,
                text="NEW INFRASTRUCTURE BILL PASSED",
                bbox=(270.0, 40.0, 500.0, 70.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text="Parliament approved the key funding bill late last evening.",
                bbox=(270.0, 75.0, 500.0, 200.0),
            ),
        ]

        articles = segmenter.segment_page(1, blocks)
        assert len(articles) == 2

        art1 = articles[0]
        assert art1.headline == "MARKETS HIT RECORD HIGH"
        assert art1.jump_to_page == 4

        art2 = articles[1]
        assert art2.headline == "NEW INFRASTRUCTURE BILL PASSED"
        assert art2.jump_to_page is None

    def test_ocr_page_without_headlines_generates_fallback_article(self) -> None:
        """Verify scanned/OCR page with no detected headlines creates fallback article."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "The economic survey reported steady industrial growth "
                    "across manufacturing sectors."
                ),
                bbox=(50.0, 50.0, 400.0, 150.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "Exports rose by 8 percent year-on-year according to "
                    "official ministry statistics."
                ),
                bbox=(50.0, 160.0, 400.0, 260.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=5, ordered_blocks=blocks)
        assert len(articles) == 1
        art = articles[0]
        assert "economic survey" in art.headline.lower()
        assert "industrial growth" in art.body_text
        assert "Exports rose" in art.body_text
        assert art.word_count > 15

    def test_single_block_article_has_non_empty_body_and_word_count(self) -> None:
        """Verify single block is never left with empty body_text or 0 word count."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.BODY_TEXT,
                text="Tata Power posts record quarterly net profit.",
                bbox=(50.0, 50.0, 400.0, 100.0),
            )
        ]

        articles = segmenter.segment_page(page_number=2, ordered_blocks=blocks)
        assert len(articles) == 1
        assert articles[0].headline == "Tata Power posts record quarterly net profit."
        assert articles[0].body_text == "Tata Power posts record quarterly net profit."
        assert articles[0].word_count == 7

