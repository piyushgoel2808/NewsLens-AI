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

    def test_debundle_mint_shorts_column_with_accurate_bbox_slicing(self) -> None:
        """Verify 'Mint Shorts' column debundles with non-overlapping bboxes."""
        segmenter = ArticleSegmenter()
        shorts_text = (
            "• ADANI PORTS: Adani Ports has completed the strategic acquisition of eighty percent "
            "equity stake in the terminal operator for over two thousand crore rupees.\n\n"
            "• GOVT INCENTIVE: The central government approved a new capital subsidy framework "
            "to accelerate green hydrogen manufacturing plants across industrial corridors.\n\n"
            "• SEBI NOTICE: The market regulator issued fresh disclosure guidelines for "
            "algorithmic trading firms operating in high frequency equity derivatives markets."
        )
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.BANNER_HEADLINE,
                text="MINT SHORTS",
                bbox=(50.0, 50.0, 300.0, 80.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=shorts_text,
                bbox=(50.0, 90.0, 300.0, 690.0),  # 600px tall column
            ),
        ]

        articles = segmenter.segment_page(page_number=2, ordered_blocks=blocks)
        assert len(articles) == 3

        art1, art2, art3 = articles[0], articles[1], articles[2]
        assert "ADANI PORTS" in art1.headline
        assert "GOVT INCENTIVE" in art2.headline
        assert "SEBI NOTICE" in art3.headline

        # Verify accurate non-overlapping bounding box vertical slicing
        b1 = art1.bbox_list[0]
        b2 = art2.bbox_list[0]
        b3 = art3.bbox_list[0]

        # Verify X coordinates match column
        assert b1[0] == 50.0 and b1[2] == 300.0
        assert b2[0] == 50.0 and b2[2] == 300.0
        assert b3[0] == 50.0 and b3[2] == 300.0

        # Verify strictly partitioned, non-overlapping Y bounds
        assert 90.0 <= b1[1] < b1[3] <= b2[1] < b2[3] <= b3[1] < b3[3] <= 690.0

    def test_kicker_extraction_and_clean_headline(self) -> None:
        """Verify kicker slugs are extracted to subheadline and headline is cleaned."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="OUR VIEW: Calm student anxiety by addressing job scarcity",
                bbox=(50.0, 50.0, 950.0, 90.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "The education ministry must reform testing procedures and focus "
                    "on employment generation across manufacturing sectors."
                ),
                bbox=(50.0, 100.0, 450.0, 250.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.HEADLINE,
                text="PLAIN FACTS: June macro softens on rising inflation",
                bbox=(500.0, 50.0, 950.0, 90.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "Macroeconomic indicators weakened slightly as consumer food prices "
                    "ticked up."
                ),
                bbox=(500.0, 100.0, 950.0, 250.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=14, ordered_blocks=blocks)
        assert len(articles) == 2

        assert articles[0].headline == "Calm student anxiety by addressing job scarcity"
        assert articles[0].subheadline == "OUR VIEW"
        assert "education ministry" in articles[0].body_text

        assert articles[1].headline == "June macro softens on rising inflation"
        assert articles[1].subheadline == "PLAIN FACTS"
        assert "Macroeconomic indicators" in articles[1].body_text

    def test_dense_page_preserves_discrete_titled_articles(self) -> None:
        """Verify concise titled articles on dense pages are preserved individually."""
        segmenter = ArticleSegmenter()
        blocks = [
            # Story 1
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="At UNSC, India condemns attacks on vessels in Hormuz",
                bbox=(50.0, 50.0, 450.0, 90.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "India on Thursday condemned recent drone and missile strikes "
                    "targeting merchant vessels in international shipping corridors."
                ),
                bbox=(50.0, 100.0, 450.0, 200.0),
            ),
            # Story 2
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.HEADLINE,
                text="Private labs may get to test power meters",
                bbox=(50.0, 220.0, 450.0, 260.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "The power ministry is considering accrediting private testing "
                    "laboratories to expedite roll-out of smart electric meters."
                ),
                bbox=(50.0, 270.0, 450.0, 360.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=2, ordered_blocks=blocks)
        assert len(articles) == 2
        assert articles[0].headline == "At UNSC, India condemns attacks on vessels in Hormuz"
        assert articles[1].headline == "Private labs may get to test power meters"

    def test_syndication_slug_rejected_as_headline_and_actual_headline_preserved(self) -> None:
        """Verify syndication slugs like 'THE WALL STREET JOURNAL' do not become headlines."""
        segmenter = ArticleSegmenter()
        hl_text = "Trump approves landmark nuclear deal with Saudi Arabia in big win for kingdom"
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.BYLINE,
                text="THE WALL STREET JOURNAL",
                bbox=(50.0, 50.0, 300.0, 70.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BANNER_HEADLINE,
                text=hl_text,
                bbox=(50.0, 80.0, 950.0, 130.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.BODY_TEXT,
                text="By Felicia Schwartz in Washington and Summer Said in Dubai",
                bbox=(50.0, 140.0, 500.0, 160.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "President Trump has approved a historic civil nuclear cooperation pact "
                    "with Saudi Arabia, marking a breakthrough in bilateral energy ties."
                ),
                bbox=(50.0, 170.0, 500.0, 350.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=8, ordered_blocks=blocks)
        assert len(articles) == 1
        art = articles[0]
        assert art.headline == hl_text
        assert "THE WALL STREET JOURNAL" in (art.byline_author or "")
        assert "historic civil nuclear cooperation" in art.body_text

    def test_mint_primer_grouped_into_single_article_with_all_questions(self) -> None:
        """Verify multi-part 'mint primer' with Q1..Q5 is consolidated into 1 unified article."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="mint primer",
                bbox=(50.0, 50.0, 200.0, 70.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BANNER_HEADLINE,
                text="Can India mop up $50 bn via NRI deposit scheme?",
                bbox=(50.0, 80.0, 950.0, 120.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.SUBHEAD,
                text="1 How does the RBI's swap facility work?",
                bbox=(50.0, 130.0, 450.0, 150.0),
            ),
            OrderedReadingBlock(
                reading_order_index=3,
                element_id=4,
                block_type=BlockType.BODY_TEXT,
                text="The Reserve Bank offers dollar-rupee buy-sell swap windows to banks.",
                bbox=(50.0, 160.0, 450.0, 220.0),
            ),
            OrderedReadingBlock(
                reading_order_index=4,
                element_id=5,
                block_type=BlockType.SUBHEAD,
                text="2 Why are banks so keen on the scheme?",
                bbox=(50.0, 230.0, 450.0, 250.0),
            ),
            OrderedReadingBlock(
                reading_order_index=5,
                element_id=6,
                block_type=BlockType.BODY_TEXT,
                text="Banks earn attractive spreads while boosting foreign currency liquidity.",
                bbox=(50.0, 260.0, 450.0, 320.0),
            ),
            OrderedReadingBlock(
                reading_order_index=6,
                element_id=7,
                block_type=BlockType.SUBHEAD,
                text="3 What prompted the RBI to act now?",
                bbox=(500.0, 130.0, 950.0, 150.0),
            ),
            OrderedReadingBlock(
                reading_order_index=7,
                element_id=8,
                block_type=BlockType.BODY_TEXT,
                text="Rising global interest rate differentials necessitated forex management.",
                bbox=(500.0, 160.0, 950.0, 220.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=3, ordered_blocks=blocks)
        assert len(articles) == 1
        art = articles[0]
        assert art.headline == "Can India mop up $50 bn via NRI deposit scheme?"
        assert art.subheadline == "mint primer"
        assert "How does the RBI's swap facility work?" in art.body_text
        assert "Why are banks so keen on the scheme?" in art.body_text
        assert "What prompted the RBI to act now?" in art.body_text
        assert art.word_count >= 40

    def test_plain_facts_lead_economy_story_headline_preserved(self) -> None:
        """Verify 'PLAIN FACTS' feature adopts the overarching lead story banner headline."""
        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.HEADLINE,
                text="PLAIN FACTS",
                bbox=(50.0, 50.0, 250.0, 70.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BANNER_HEADLINE,
                text="INDIA RETAINS EM LEAD IN JUNE AS RUPEE RISES",
                bbox=(50.0, 80.0, 950.0, 120.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "India's macroeconomic performance outpaced emerging market peers in June "
                    "driven by resilient services exports and appreciating currency."
                ),
                bbox=(50.0, 130.0, 950.0, 280.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=4, ordered_blocks=blocks)
        assert len(articles) == 1
        art = articles[0]
        assert art.headline == "INDIA RETAINS EM LEAD IN JUNE AS RUPEE RISES"
        assert art.subheadline == "PLAIN FACTS"
        assert "macroeconomic performance outpaced emerging market peers" in art.body_text

    def test_toc_index_block_isolated_and_severed(self) -> None:
        """Verify Table of Contents / Index teasers are isolated and severed from news articles."""
        from app.ingestion.layout_analyzer import is_toc_index_block
        from app.ingestion.segmenter import is_valid_headline_candidate

        toc_text = "Global | Trump approves tariff roadmap... >P14\nMoney | Markets rise... >P8"
        assert is_toc_index_block(toc_text) is True
        assert is_valid_headline_candidate(toc_text) is False

        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.TOC_INDEX,
                text=toc_text,
                bbox=(50.0, 50.0, 250.0, 150.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.BANNER_HEADLINE,
                text="CENTRAL BANK HOLDS POLICY RATES UNCHANGED",
                bbox=(50.0, 160.0, 950.0, 200.0),
            ),
            OrderedReadingBlock(
                reading_order_index=2,
                element_id=3,
                block_type=BlockType.BODY_TEXT,
                text=(
                    "The Monetary Policy Committee decided unanimously to keep the policy "
                    "repo rate unchanged at 6.50% while maintaining an active focus."
                ) * 3,
                bbox=(50.0, 210.0, 950.0, 380.0),
            ),
        ]

        articles = segmenter.segment_page(page_number=1, ordered_blocks=blocks)
        assert len(articles) == 1
        art = articles[0]
        assert art.headline == "CENTRAL BANK HOLDS POLICY RATES UNCHANGED"
        assert "Trump approves tariff roadmap" not in art.body_text
        assert "Global |" not in art.body_text

    def test_pullquote_author_attribution_rejected_as_headline(self) -> None:
        """Verify ALL CAPS pullquote author designations are rejected as article headlines."""
        from app.ingestion.layout_analyzer import is_pullquote_author_block
        from app.ingestion.segmenter import is_valid_headline_candidate

        attr_text = "PENNYWONG AUSTRALIANFOREIGN MINISTER"
        assert is_pullquote_author_block(attr_text) is True
        assert is_valid_headline_candidate(attr_text) is False

        segmenter = ArticleSegmenter()
        blocks = [
            OrderedReadingBlock(
                reading_order_index=0,
                element_id=1,
                block_type=BlockType.PULLQUOTE_AUTHOR,
                text=attr_text,
                bbox=(50.0, 50.0, 250.0, 80.0),
            ),
            OrderedReadingBlock(
                reading_order_index=1,
                element_id=2,
                block_type=BlockType.TOC_INDEX,
                text="Global | Trade pacts signed >P14",
                bbox=(50.0, 90.0, 250.0, 140.0),
            ),
        ]

        # Standalone attribution and ToC block should never form an article
        articles = segmenter.segment_page(page_number=1, ordered_blocks=blocks)
        assert len(articles) == 0
