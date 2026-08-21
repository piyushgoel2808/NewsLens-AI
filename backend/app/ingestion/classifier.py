"""Article Type Classifier and Prominence Scorer.

Classifies articles into 8 standardized types:
- 'news'
- 'editorial'
- 'sidebar'
- 'advertisement'
- 'photo_caption'
- 'table_content'
- 'continuation'
- 'unknown'

Calculates normalized prominence score (0.0 to 1.0) based on page positioning,
headline bounding box scale, and word count.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.ingestion.cross_page_assembler import AssembledArticle

logger = get_logger(__name__)

EDITORIAL_KEYWORDS = re.compile(
    r"\b(?:editorial|opinion|commentary|op-ed|column|letters\s+to\s+the\s+editor|perspective|viewpoint)\b",
    re.IGNORECASE,
)

ADVERTISEMENT_KEYWORDS = re.compile(
    r"\b(?:advertisement|sponsored|classified|for\s+sale|special\s+offer|discount|call\s+now|inquiries)\b",
    re.IGNORECASE,
)

TABLE_PATTERNS = re.compile(
    r"(?:\d+\.\d{2}|\b(?:high|low|close|volume|yield|prev|change|indices|stock|commodity)\b)",
    re.IGNORECASE,
)


@dataclass
class ClassificationResult:
    """Type classification and prominence assessment."""

    article_type: str
    prominence_score: float
    section: str | None = None


class ArticleClassifier:
    """Classifies article types and computes prominence scores."""

    def classify_and_score(
        self,
        article: AssembledArticle,
        total_issue_pages: int = 1,
    ) -> ClassificationResult:
        """Classify article type and compute prominence score."""
        headline = (article.headline or "").strip()
        body = (article.full_text or "").strip()
        combined_text = f"{headline}\n{body}"

        # 1. Determine Article Type
        if EDITORIAL_KEYWORDS.search(headline) or (
            article.primary_page_number in (2, 4) and EDITORIAL_KEYWORDS.search(combined_text[:300])
        ):
            article_type = "editorial"
            section = "Opinion & Editorial"
        elif ADVERTISEMENT_KEYWORDS.search(combined_text[:400]) and article.word_count < 150:
            article_type = "advertisement"
            section = "Advertisements"
        elif len(TABLE_PATTERNS.findall(body)) > 15 and article.word_count < 400:
            article_type = "table_content"
            section = "Markets & Data"
        elif article.word_count < 40 and not headline:
            article_type = "photo_caption"
            section = "Graphics"
        elif article.word_count < 80 and article.primary_page_number > 1:
            article_type = "sidebar"
            section = "General"
        else:
            article_type = "news"
            section = "General News" if article.primary_page_number == 1 else "Inside News"

        # 2. Compute Prominence Score (0.0 to 1.0)
        # Factor A: Page Placement (Page 1 is most prominent)
        if article.primary_page_number == 1:
            page_score = 0.50
        elif article.primary_page_number in (2, 3):
            page_score = 0.30
        else:
            page_score = 0.15

        # Factor B: Word Count Score (up to 0.25)
        wc = article.word_count
        if wc > 800:
            wc_score = 0.25
        elif wc > 400:
            wc_score = 0.20
        elif wc > 200:
            wc_score = 0.15
        elif wc > 80:
            wc_score = 0.10
        else:
            wc_score = 0.05

        # Factor C: Headline Scale & Cross-Page Span (up to 0.25)
        span_score = 0.05 * len(article.pages_mapping)
        headline_score = 0.15 if len(headline) > 30 else 0.05
        headline_factor = min(0.25, headline_score + span_score)

        raw_prominence = page_score + wc_score + headline_factor
        if article_type in ("advertisement", "photo_caption"):
            raw_prominence *= 0.40
        elif article_type in ("sidebar", "table_content"):
            raw_prominence *= 0.70

        prominence_score = round(max(0.05, min(1.0, raw_prominence)), 2)

        return ClassificationResult(
            article_type=article_type,
            prominence_score=prominence_score,
            section=section,
        )
