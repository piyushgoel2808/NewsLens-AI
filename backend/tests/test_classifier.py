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
        assert res.section == "Inside News"

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
