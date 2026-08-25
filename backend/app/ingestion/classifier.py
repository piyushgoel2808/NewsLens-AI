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
from app.models.article import _ARTICLE_TYPES

logger = get_logger(__name__)

_VALID_TYPES = set(_ARTICLE_TYPES)

EDITORIAL_KEYWORDS = re.compile(
    r"\b(?:editorial|opinion|commentary|op-ed|column|letters\s+to\s+the\s+editor|perspective|viewpoint)\b",
    re.IGNORECASE,
)

ADVERTISEMENT_KEYWORDS = re.compile(
    r"\b(?:advertisement|advertorial|special\s+promotional\s+feature|sponsored|"
    r"initial\s+public\s+offering|\bipo\b|red\s+herring\s+prospectus|"
    r"book\s+running\s+lead\s+manager|price\s+band|public\s+notice|statutory\s+notice|"
    r"tender\s+notice|auction\s+sale|corrigendum|classified|for\s+sale|special\s+offer|"
    r"discount|call\s+now|inquiries)\b",
    re.IGNORECASE,
)

TABLE_PATTERNS = re.compile(
    r"(?:\d+\.\d{2}|\b(?:high|low|close|volume|yield|prev|change|indices|stock|commodity)\b)",
    re.IGNORECASE,
)


SIDEBAR_KEYWORDS = re.compile(
    r"(?i)\b(?:sidebar|box\s+story|highlights|key\s+takeaways|at\s+a\s+glance|in\s+a\s+nutshell|quick\s+facts)\b"
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
        vlm_section: str | None = None,
        vlm_article_type: str | None = None,
        vlm_prominence: str | None = None,
    ) -> ClassificationResult:
        """Classify article type and compute prominence score with VLM guidance."""
        headline = (article.headline or "").strip()
        body = (article.full_text or "").strip()
        combined_text = f"{headline}\n{body}"

        # 1. Determine Article Type
        if headline.upper().startswith("[ADVERTISEMENT]") or headline.upper().startswith(
            "[PUBLIC NOTICE]"
        ) or ADVERTISEMENT_KEYWORDS.search(combined_text[:400]):
            article_type = "advertisement"
            section = "Advertisements & Notices"
        elif vlm_article_type and vlm_article_type in _VALID_TYPES:
            article_type = vlm_article_type
            section = vlm_section or ("Opinion & Editorial" if article_type in ("editorial", "opinion") else "National")
        elif EDITORIAL_KEYWORDS.search(headline) or (
            article.primary_page_number in (2, 4)
            and EDITORIAL_KEYWORDS.search(combined_text[:300])
        ):
            article_type = "editorial"
            section = "Opinion & Editorial"
        elif len(TABLE_PATTERNS.findall(body)) > 15 and article.word_count < 400:
            article_type = "table_content"
            section = "Markets & Data"
        elif article.word_count < 40 and not headline:
            article_type = "photo_caption"
            section = "Life & Culture"
        elif (
            headline.upper().startswith("[SIDEBAR]")
            or headline.upper().startswith("[BOX]")
            or SIDEBAR_KEYWORDS.search(headline)
        ):
            article_type = "sidebar"
            section = vlm_section or ("Front Page" if article.primary_page_number == 1 else "National")
        elif headline.upper().startswith("[SHORTS]"):
            article_type = "news"
            section = "News Briefs"
        else:
            article_type = "news"
            if vlm_section:
                section = vlm_section
            else:
                sub = (article.subheadline or "").lower()
                editorial_kws = ("our view", "my view", "their view", "column", "curator", "quick edit")
                if any(k in sub for k in editorial_kws):
                    article_type = "editorial"
                    section = "Opinion & Editorial"
                elif any(k in sub for k in ("mark to market", "plain facts")):
                    section = "Markets & Data"
                elif "economy" in sub or "policy" in sub:
                    section = "Economy & Policy"
                elif "deal" in sub or "tech" in sub or "startup" in sub:
                    section = "Deals, Tech & Startups"
                elif "corporate" in sub or "company" in sub or "companies" in sub:
                    section = "Corporate & Industry"
                elif "global" in sub or "world" in sub:
                    section = "International"
                elif "money" in sub or "ask mint" in sub or "power point" in sub:
                    section = "Personal Finance"
                elif "life" in sub or "culture" in sub:
                    section = "Life & Culture"
                else:
                    section = "Front Page" if article.primary_page_number == 1 else "National"

        # 2. Compute Prominence Score (0.0 to 1.0)
        tier_map = {"lead": 0.90, "major": 0.70, "standard": 0.50, "minor": 0.30, "filler": 0.15}
        if vlm_prominence and vlm_prominence in tier_map:
            prominence_score = tier_map[vlm_prominence]
        else:
            # Heuristic calculation
            if article.primary_page_number == 1:
                page_score = 0.50
            elif article.primary_page_number in (2, 3):
                page_score = 0.30
            else:
                page_score = 0.15

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

            span_score = 0.05 * len(article.pages_mapping)
            headline_score = 0.15 if len(headline) > 30 else 0.05
            headline_factor = min(0.25, headline_score + span_score)

            raw_prominence = page_score + wc_score + headline_factor
            if article_type in ("advertisement", "photo_caption"):
                raw_prominence *= 0.40
            elif article_type in ("sidebar", "table_content", "table_data"):
                raw_prominence *= 0.70

            prominence_score = round(max(0.05, min(1.0, raw_prominence)), 2)

        return ClassificationResult(
            article_type=article_type,
            prominence_score=prominence_score,
            section=section,
        )
