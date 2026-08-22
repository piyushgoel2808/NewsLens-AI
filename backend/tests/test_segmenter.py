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

    def test_stopword_not_treated_as_headline_and_fragment_merged(self) -> None:
        """Verify single stopwords ('of', 'and') do not create separate 1-word articles."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="CENTRAL BANK HOLDS BENCHMARK REPO RATE",
                bbox=(30.0, 40.0, 300.0, 70.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "The monetary policy committee voted unanimously to maintain "
                    "status quo on rates."
                ),
                bbox=(30.0, 75.0, 300.0, 150.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.HEADLINE,
                text="of",  # Noise stopword flagged as headline
                bbox=(30.0, 155.0, 50.0, 170.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text="Inflation expectations remain anchored within the target tolerance band.",
                bbox=(30.0, 175.0, 300.0, 240.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=7, ordered_blocks=blocks)
        # Should not create a separate article for 'of', but merge the page content
        assert len(articles) == 1
        assert articles[0].headline == "CENTRAL BANK HOLDS BENCHMARK REPO RATE"
        assert "Inflation expectations remain anchored" in articles[0].body_text

    def test_full_page_advertisement_groups_into_single_article(self) -> None:
        """Verify full-page ad with is_advertisement_page=True creates exactly 1 article."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.BANNER_HEADLINE,
                text="JUNIPER GREEN ENERGY LIMITED - INITIAL PUBLIC OFFERING",
                bbox=(50.0, 50.0, 950.0, 100.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.HEADLINE,
                text="EQUITY",
                bbox=(50.0, 110.0, 200.0, 140.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.HEADLINE,
                text="LIMITED",
                bbox=(220.0, 110.0, 350.0, 140.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "The company is proposing an initial public offer of equity shares of "
                    "face value Rs 10 each aggregating up to Rs 3,000 crores. Bid opens on Monday."
                ),
                bbox=(50.0, 150.0, 950.0, 400.0),
            ),
        ]

        articles = segmenter.segment_page(
            page_number=3,
            ordered_blocks=blocks,
            is_advertisement_page=True,
        )
        assert len(articles) == 1
        art = articles[0]
        assert art.headline.startswith("[Advertisement]")
        assert "JUNIPER GREEN ENERGY" in art.headline
        assert "face value Rs 10" in art.body_text
        assert art.word_count > 20

    def test_single_word_boilerplate_rejected_as_headline(self) -> None:
        """Verify single uppercase corporate boilerplate tokens do not shatter articles."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="GOVERNMENT ANNOUNCES NEW GREEN ENERGY POLICY",
                bbox=(50.0, 50.0, 600.0, 80.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "The Ministry of Power unveiled new solar subsidies to boost domestic "
                    "manufacturing across states. The initial financial outlay is estimated "
                    "at over fifty thousand crore rupees over the next five fiscal years."
                ),
                bbox=(50.0, 90.0, 600.0, 200.0),
            ),
            # Noise boilerplate tagged as headline
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.HEADLINE,
                text="LIMITED",
                bbox=(50.0, 210.0, 150.0, 230.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "Industry leaders welcomed the reforms and pledged additional investments "
                    "in renewable technology infrastructure."
                ),
                bbox=(50.0, 240.0, 600.0, 320.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=4, ordered_blocks=blocks)
        assert len(articles) == 1
        assert articles[0].headline == "GOVERNMENT ANNOUNCES NEW GREEN ENERGY POLICY"
        assert "Industry leaders welcomed" in articles[0].body_text

    def test_orphan_snippets_absorbed_into_adjacent_article(self) -> None:
        """Verify short snippets under 30 words are absorbed rather than standing alone."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="QUARTERLY EARNINGS BEAT ANALYST ESTIMATES",
                bbox=(50.0, 50.0, 600.0, 80.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "Major IT services companies recorded robust growth in North American "
                    "demand with operating margins improving significantly in the third quarter. "
                    "Revenue expanded by twelve percent compared to the preceding quarter."
                ),
                bbox=(50.0, 90.0, 600.0, 200.0),
            ),
            # Tiny sidebar snippet with only 8 words
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.HEADLINE,
                text="BRIEF UPDATE",
                bbox=(50.0, 220.0, 300.0, 240.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text="Stock index rose two hundred points today.",
                bbox=(50.0, 250.0, 300.0, 270.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=6, ordered_blocks=blocks)
        # The 8-word brief is absorbed into the main article
        assert len(articles) == 1
        assert articles[0].headline == "QUARTERLY EARNINGS BEAT ANALYST ESTIMATES"
        assert "Stock index rose two hundred points" in articles[0].body_text
