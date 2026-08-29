"""Unit tests for Article Type Classifier and Prominence Scorer."""

from __future__ import annotations

from app.ingestion.classifier import ArticleClassifier
from app.ingestion.cross_page_assembler import AssembledArticle, PageBBoxMapping


class TestArticleClassifier:
    """Test suite for ArticleClassifier."""

    def test_classify_news_frontpage(self) -> None:
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="GOVERNMENT ANNOUNCES MAJOR REFORMS IN PARLIAMENT",
            full_text=(
                "The Prime Minister addressed lawmakers this morning detailing a "
                "comprehensive roadmap for reform."
            ),
            primary_page_number=1,
            word_count=550,
            pages_mapping=[PageBBoxMapping(page_number=1, bbox_list=[(50.0, 50.0, 400.0, 300.0)])],
        )

        res = classifier.classify_and_score(article, total_issue_pages=4)
        assert res.article_type == "news"
        assert res.prominence_score >= 0.70  # Page 1 + long text + strong headline

    def test_classify_editorial(self) -> None:
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="EDITORIAL: THE PATH FORWARD FOR ENERGY SECURITY",
            full_text=(
                "As nations transition away from fossil fuels, pragmatic policy "
                "must govern grid investments."
            ),
            primary_page_number=2,
            word_count=450,
            pages_mapping=[PageBBoxMapping(page_number=2, bbox_list=[(50.0, 50.0, 300.0, 400.0)])],
        )

        res = classifier.classify_and_score(article, total_issue_pages=4)
        assert res.article_type == "editorial"
        assert res.section == "Opinion & Editorial"

    def test_classify_advertisement(self) -> None:
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="SPECIAL OFFER: LUXURY APARTMENTS FOR SALE",
            full_text=(
                "Exclusive residential towers in prime location. Call now for "
                "inquiries and discount rates."
            ),
            primary_page_number=3,
            word_count=20,
            pages_mapping=[PageBBoxMapping(page_number=3, bbox_list=[(50.0, 50.0, 200.0, 200.0)])],
        )

        res = classifier.classify_and_score(article, total_issue_pages=4)
        assert res.article_type == "advertisement"
        assert res.prominence_score <= 0.40

    def test_classify_table_content(self) -> None:
        classifier = ArticleClassifier()
        table_rows = [
            f"Ticker {i} High 12{i}.50 Low 11{i}.20 Close 12{i}.00 Volume 500000" for i in range(10)
        ]
        table_text = "\n".join(table_rows)
        article = AssembledArticle(
            headline="DAILY MARKET SUMMARY AND STOCK CLOSING QUOTES",
            full_text=table_text,
            primary_page_number=4,
            word_count=150,
            pages_mapping=[PageBBoxMapping(page_number=4, bbox_list=[(50.0, 50.0, 500.0, 400.0)])],
        )

        res = classifier.classify_and_score(article, total_issue_pages=4)
        assert res.article_type == "table_content"
        assert res.section == "Markets & Data"

    def test_classify_inside_page_standard_news_is_not_sidebar(self) -> None:
        """Verify standard inside page news story defaults to 'news' (not 'sidebar')."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="SERVICES INDEX EXPANDS AT FASTEST PACE IN FIVE MONTHS",
            full_text=(
                "India's services sector growth accelerated sharply in May, driven by strong "
                "inflows of new business orders from international clients. Employment across "
                "service providers continued to expand steadily according to survey findings."
            ),
            primary_page_number=6,
            word_count=50,
            pages_mapping=[PageBBoxMapping(page_number=6, bbox_list=[(50.0, 50.0, 300.0, 400.0)])],
        )

        res = classifier.classify_and_score(article, total_issue_pages=12)
        assert res.article_type == "news"
        assert res.section in ("National", "Inside News", "International", "Corporate & Industry")

    def test_classify_explicit_sidebar_box_story(self) -> None:
        """Verify explicit sidebar or box story is classified as 'sidebar'."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="[BOX] KEY TAKEAWAYS FROM MONETARY POLICY STATEMENT",
            full_text=(
                "1. Repo rate unchanged at 6.50%.\n"
                "2. GDP growth projection retained at 7.2%.\n"
                "3. CPI inflation forecast steady at 4.5%."
            ),
            primary_page_number=3,
            word_count=35,
            pages_mapping=[PageBBoxMapping(page_number=3, bbox_list=[(50.0, 50.0, 300.0, 200.0)])],
        )

        res = classifier.classify_and_score(article, total_issue_pages=12)
        assert res.article_type == "sidebar"

    def test_classify_kicker_subheadline_sections(self) -> None:
        """Verify subheadline kickers route to appropriate sections."""
        classifier = ArticleClassifier()

        art_edit = AssembledArticle(
            headline="Calm student anxiety by addressing job scarcity",
            subheadline="OUR VIEW",
            full_text="Educational reforms and youth career paths are vital.",
            primary_page_number=14,
            word_count=60,
        )
        res_edit = classifier.classify_and_score(art_edit, total_issue_pages=16)
        assert res_edit.article_type == "editorial"
        assert res_edit.section == "Opinion & Editorial"

        art_mkt = AssembledArticle(
            headline="Why L&T faces a valuation test",
            subheadline="MARK TO MARKET",
            full_text="Infrastructure capital expenditure trends remain positive.",
            primary_page_number=6,
            word_count=80,
        )
        res_mkt = classifier.classify_and_score(art_mkt, total_issue_pages=16)
        assert res_mkt.section == "Markets & Data"

        art_tech = AssembledArticle(
            headline="Grant Thornton US to acquire rival accountant CBIZ in $5 billion deal",
            subheadline="DEALS, TECH & STARTUPS",
            full_text="The merger will create one of the largest advisory networks.",
            primary_page_number=3,
            word_count=90,
        )
        res_tech = classifier.classify_and_score(art_tech, total_issue_pages=16)
        assert res_tech.section in ("Deals, Tech & Startups", "Technology")

    def test_classify_genuine_sports_cricket(self) -> None:
        """Verify cricket / sports stories are accurately classified into Sports."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="India Turn the Screw",
            subheadline="SSC Test Sri Lanka's resistance fades after Sooriyabandara and Mendis fifties as India close in on victory and a 2-0 series win",
            full_text=(
                "India Turn the Screw\nAnand Vasu REGARDLESS\n\n"
                "Test cricket is the longest format of the game. Prasidh Krishna dropped one short "
                "only to be deposited onto the midwicket stands. Manav Suthar was compulsive in taking on "
                "the short ball as India's fielders and spinners bowled out the remaining wickets."
            ),
            primary_page_number=20,
            word_count=450,
            printed_section="Sports World Play",
        )
        res = classifier.classify_and_score(article, total_issue_pages=20)
        assert res.category == "Sports"
        assert res.section == "Sports World Play"
        assert res.category_confidence >= 0.80

    def test_metaphorical_financial_headline_disambiguation(self) -> None:
        """Verify market headlines with sports idioms ('hit for a six') stay in Business & Markets."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="Bulls hit market for a six as Sensex rallies 800 points",
            subheadline="Nifty crosses 25,000 mark on strong FPI equity inflows and quarterly corporate earnings",
            full_text=(
                "Dalal Street witnessed robust buying across banking and IT shares. Foreign portfolio "
                "investors poured ₹3,500 crore into blue-chip stocks as valuation multiples expanded."
            ),
            primary_page_number=6,
            word_count=400,
        )
        res = classifier.classify_and_score(article, total_issue_pages=16)
        assert res.category == "Business & Markets"
        assert res.category != "Sports"

    def test_metaphorical_political_headline_disambiguation(self) -> None:
        """Verify political headlines with gaming metaphors ('political chess') stay in Politics."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="BJP plays political chess ahead of state assembly election",
            subheadline="Cabinet reshuffle in Uttar Pradesh focuses on caste equations and voter mobilization",
            full_text=(
                "The Prime Minister and Home Minister held detailed discussions with the state leadership "
                "and MLAs to finalize candidate lists for the upcoming polls."
            ),
            primary_page_number=5,
            word_count=350,
        )
        res = classifier.classify_and_score(article, total_issue_pages=16)
        assert res.category == "Politics"
        assert res.category != "Sports"

    def test_metaphorical_health_pharma_headline(self) -> None:
        """Verify pharma headlines with conflict metaphors stay in Health."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="Pharma giants battle in blockbuster drug war over patent rights",
            subheadline="Global healthcare manufacturers clash in court over next-gen cancer vaccines and clinical trials",
            full_text=(
                "Leading pharmaceutical companies are scaling production of breakthrough mRNA vaccines "
                "as hospitals and doctors await FDA clearance for clinical distribution."
            ),
            primary_page_number=9,
            word_count=380,
        )
        res = classifier.classify_and_score(article, total_issue_pages=16)
        assert res.category == "Health"
        assert res.category != "World/International"

    def test_multi_topic_sports_business_acquisition(self) -> None:
        """Verify sports business story captures both primary and secondary domain topics."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="Rajasthan Royals Deal: CCI Seeks More Details on ₹4,000 Cr Acquisition",
            subheadline="The cricket franchise sale attracts global venture capital and private equity investors",
            full_text=(
                "The Competition Commission of India has sought additional disclosures regarding the multi-crore "
                "merger involving the IPL cricket team ownership and media rights valuation."
            ),
            primary_page_number=8,
            word_count=420,
        )
        res = classifier.classify_and_score(article, total_issue_pages=16)
        # Primary is Business & Markets or Sports, and secondary contains the other
        categories_detected = [res.category] + [c[0] for c in res.secondary_categories]
        assert "Business & Markets" in categories_detected
        assert "Sports" in categories_detected

    def test_classify_science_and_space(self) -> None:
        """Verify ISRO / space exploration story is classified into Science & Environment."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="ISRO successfully places solar observation satellite into lunar orbit",
            subheadline="The mission aims to study coronal mass ejections and deep space plasma physics",
            full_text=(
                "Scientists at the space agency confirmed all payload instruments are operating nominally "
                "as the rocket completed its multi-stage trajectory."
            ),
            primary_page_number=7,
            word_count=320,
        )
        res = classifier.classify_and_score(article, total_issue_pages=16)
        assert res.category == "Science & Environment"

    def test_classify_entertainment_cinema(self) -> None:
        """Verify movie / cinema story is classified into Entertainment."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="Bollywood blockbuster sets new worldwide box office record in opening weekend",
            subheadline="The star-studded cinema release earns ₹250 crore across global theatrical and OTT screens",
            full_text=(
                "Directed by an award-winning filmmaker, the film captivated audiences with cutting-edge "
                "visual effects and an acclaimed musical score."
            ),
            primary_page_number=12,
            word_count=300,
        )
        res = classifier.classify_and_score(article, total_issue_pages=16)
        assert res.category == "Entertainment"

    def test_editorial_preserves_article_type_and_classifies_domain(self) -> None:
        """Verify editorial op-ed preserves article_type='editorial' while identifying domain topic."""
        classifier = ArticleClassifier()
        article = AssembledArticle(
            headline="OPINION: REGULATING ARTIFICIAL INTELLIGENCE WITHOUT KILLING INNOVATION",
            subheadline="BY SWAMINATHAN AIYAR",
            full_text=(
                "Generative AI and large language models present both systemic opportunities and risks. "
                "Policymakers must balance algorithmic safety with venture capital investment in cloud chips."
            ),
            primary_page_number=10,
            word_count=600,
        )
        res = classifier.classify_and_score(article, total_issue_pages=16)
        assert res.article_type == "editorial"
        assert res.section == "Opinion & Editorial"
        assert res.category == "Technology"

