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
from pathlib import Path

import yaml

from app.core.logging import get_logger
from app.ingestion.cross_page_assembler import AssembledArticle
from app.models.article import _ARTICLE_TYPES

logger = get_logger(__name__)

_VALID_TYPES = set(_ARTICLE_TYPES)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "category_aliases.yaml"
_CATEGORY_ALIASES: dict[str, str] = {}
_CANONICAL_CATEGORIES: list[str] = [
    "Politics",
    "Business & Markets",
    "Sports",
    "Entertainment",
    "World/International",
    "Technology",
    "Health",
    "Crime & Law",
    "Opinion/Editorial",
    "Lifestyle",
    "Science & Environment",
    "Local/Metro",
    "Other",
]

if _CONFIG_PATH.exists():
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            _CATEGORY_ALIASES = {
                k.lower().strip(): v for k, v in cfg.get("printed_section_aliases", {}).items()
            }
            if "canonical_categories" in cfg:
                _CANONICAL_CATEGORIES = cfg["canonical_categories"]
    except Exception as e:
        logger.warning("Failed to load category_aliases.yaml", extra={"error": str(e)})

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

SPORTS_KEYWORDS = re.compile(
    r"(?i)\b(?:match|cricket|football|wicket|innings|runs|goals|tournament|"
    r"olympics|championship|trophy|coach|league|fifa|icc|bcci|atp|wta|ipl|test\s+match)\b"
)
TECH_KEYWORDS = re.compile(
    r"(?i)\b(?:artificial\s+intelligence|\bai\b|software|cloud|semiconductor|cybersecurity|"
    r"hardware|app|startup|fintech|venture\s+capital|algorithm|chipmaker)\b"
)
POLITICS_KEYWORDS = re.compile(
    r"(?i)\b(?:parliament|minister|government|election|cabinet|bill|policy|"
    r"lok\s+sabha|rajya\s+sabha|bjp|congress|mla|mp|constituency|governor)\b"
)


@dataclass
class ClassificationResult:
    """Type classification, canonical category, and prominence assessment."""

    article_type: str
    prominence_score: float
    section: str | None = None
    category: str | None = None
    category_confidence: float = 0.85
    printed_section: str | None = None


class ArticleClassifier:
    """Classifies article types and computes prominence scores & canonical categories."""

    def map_section_to_category(self, section_text: str | None) -> tuple[str | None, float]:
        """Map printed section text to canonical category using alias table."""
        if not section_text or not section_text.strip():
            return None, 0.0
        sec_clean = section_text.strip().lower()
        if sec_clean in _CATEGORY_ALIASES:
            return _CATEGORY_ALIASES[sec_clean], 0.95
        # Substring search in alias keys
        for alias_key, canon in _CATEGORY_ALIASES.items():
            if alias_key in sec_clean or sec_clean in alias_key:
                return canon, 0.90
        return None, 0.0

    def infer_category_from_content(self, headline: str, body: str) -> tuple[str, float]:
        """Classify article into canonical category from content heuristics."""
        content = f"{headline}\n{body[:600]}".lower()
        if SPORTS_KEYWORDS.search(content):
            return "Sports", 0.85
        if TECH_KEYWORDS.search(content):
            return "Technology", 0.85
        if POLITICS_KEYWORDS.search(content):
            return "Politics", 0.85
        if EDITORIAL_KEYWORDS.search(headline):
            return "Opinion/Editorial", 0.90
        if len(TABLE_PATTERNS.findall(body)) > 10:
            return "Business & Markets", 0.85
        # Default fallback
        return "Business & Markets" if ("market" in content or "crore" in content or "rupee" in content or "bank" in content) else "Politics", 0.70

    def classify_and_score(
        self,
        article: AssembledArticle,
        total_issue_pages: int = 1,
        vlm_section: str | None = None,
        vlm_article_type: str | None = None,
        vlm_prominence: str | None = None,
        printed_section: str | None = None,
    ) -> ClassificationResult:
        """Classify article type, canonical category, and compute prominence score."""
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

        # 2. Determine Canonical Newsroom Category
        canonical_category: str | None = None
        category_conf = 0.0

        if article_type in ("advertisement", "photo_caption") and article.word_count < 40:
            canonical_category = None
            category_conf = 1.0
        else:
            # Signal 1: Printed Section match
            sec_to_test = printed_section or section
            mapped_cat, mapped_conf = self.map_section_to_category(sec_to_test)
            if mapped_cat:
                canonical_category = mapped_cat
                category_conf = mapped_conf
            else:
                # Signal 2: Content-based inference
                inf_cat, inf_conf = self.infer_category_from_content(headline, body)
                canonical_category = inf_cat
                category_conf = inf_conf

        # 3. Compute Prominence Score (0.0 to 1.0)
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
            category=canonical_category,
            category_confidence=category_conf,
            printed_section=printed_section,
        )
