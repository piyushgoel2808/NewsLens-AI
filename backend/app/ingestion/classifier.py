"""Article Type Classifier, Multi-Signal Taxonomy Classifier & Prominence Scorer.

Classifies articles into 8 standardized types:
- 'news'
- 'editorial'
- 'sidebar'
- 'advertisement'
- 'photo_caption'
- 'table_content'
- 'continuation'
- 'unknown'

Provides multi-signal probabilistic taxonomy scoring across 12 canonical domains:
Sports, Entertainment, Science & Environment, Technology, Business & Markets,
Economy & Policy, Politics, Health, Crime & Law, Opinion/Editorial,
World/International, Lifestyle.

Features:
- Headline (3.0x), Subheadline (2.0x), and Body (1.0x) weighted token scoring.
- Domain Context Anchor Dampening for metaphorical keyword collision disambiguation
  (e.g., "Bulls hit market for a six" -> Business & Markets, not Sports).
- Multi-topic secondary tagging for cross-domain stories.
- Physical layout section vs. semantic category decoupling.
- Strict preservation of editorial / op-ed article types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    "Sports",
    "Entertainment",
    "Science & Environment",
    "Technology",
    "Business & Markets",
    "Economy & Policy",
    "Politics",
    "Health",
    "Crime & Law",
    "Opinion/Editorial",
    "World/International",
    "Lifestyle",
    "Local/Metro",
    "Other",
]
_CATEGORY_KEYWORDS: dict[str, list[str]] = {}
_DOMAIN_ANCHORS: dict[str, list[str]] = {}
_USER_QUERY_SYNONYMS: dict[str, str] = {}


def _load_taxonomy_config() -> None:
    global _CATEGORY_ALIASES, _CANONICAL_CATEGORIES, _CATEGORY_KEYWORDS, _DOMAIN_ANCHORS, _USER_QUERY_SYNONYMS
    if not _CONFIG_PATH.exists():
        return
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            _CATEGORY_ALIASES = {
                k.lower().strip(): v for k, v in cfg.get("printed_section_aliases", {}).items()
            }
            if "canonical_categories" in cfg:
                _CANONICAL_CATEGORIES = cfg["canonical_categories"]
            _CATEGORY_KEYWORDS = cfg.get("category_keywords", {})
            _DOMAIN_ANCHORS = cfg.get("domain_anchors", {})
            _USER_QUERY_SYNONYMS = {
                k.lower().strip(): v for k, v in cfg.get("user_query_synonyms", {}).items()
            }
    except Exception as e:
        logger.warning("Failed to load category_aliases.yaml", extra={"error": str(e)})


_load_taxonomy_config()

EDITORIAL_KEYWORDS = re.compile(
    r"\b(?:editorial|opinion|commentary|op-ed|column|columnist|letters\s+to\s+the\s+editor|perspective|viewpoint|our\s+view|my\s+view|their\s+view|quick\s+edit)\b",
    re.IGNORECASE,
)

ADVERTISEMENT_KEYWORDS = re.compile(
    r"\b(?:advertisement|advertorial|special\s+promotional\s+feature|sponsored|"
    r"initial\s+public\s+offering|\bipo\b|red\s+herring\s+prospectus|"
    r"book\s+running\s+lead\s+manager|price\s+band|public\s+notice|statutory\s+notice|"
    r"tender\s+notice|auction\s+sale|corrigendum|classified|for\s+sale|special\s+offer|"
    r"discount|call\s+now|inquiries|toll\s+free)\b",
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
    """Type classification, canonical category, and prominence assessment."""

    article_type: str
    prominence_score: float
    section: str | None = None
    category: str | None = None
    category_confidence: float = 0.85
    printed_section: str | None = None
    secondary_categories: list[tuple[str, float]] = field(default_factory=list)


class ArticleClassifier:
    """Classifies article types, computes multi-signal taxonomy categories and prominence."""

    def __init__(self) -> None:
        _load_taxonomy_config()

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

    def infer_category_from_content(
        self,
        headline: str,
        subheadline: str | None = None,
        body: str = "",
        printed_section: str | None = None,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Multi-signal probabilistic category classification across 12 newsroom domains.

        Returns:
            (primary_category, confidence_score, secondary_categories_list)
        """
        hl = (headline or "").lower()
        sub = (subheadline or "").lower()
        b_sample = (body or "")[:1500].lower()

        # Token bags with weights: Headline 3.0x, Subheadline 2.0x, Body 1.0x
        category_scores: dict[str, float] = {cat: 0.0 for cat in _CANONICAL_CATEGORIES}

        # 1. Match Keywords per category
        for cat, kw_list in _CATEGORY_KEYWORDS.items():
            cat_score = 0.0
            for kw in kw_list:
                kw_lower = kw.lower()
                pattern = r"\b" + re.escape(kw_lower) + r"\b"
                # Headline matches (3x)
                if re.search(pattern, hl):
                    cat_score += 3.0
                # Subheadline matches (2x)
                if sub and re.search(pattern, sub):
                    cat_score += 2.0
                # Body matches (1x)
                if b_sample and re.search(pattern, b_sample):
                    cat_score += 1.0

            category_scores[cat] = cat_score

        # 2. Metaphorical Keyword Collision Disambiguation & Anchor Dampening
        # Financial Press Metaphors: "Bulls hit market for a six", "Rally hits boundary"
        biz_anchors = _DOMAIN_ANCHORS.get("business_anchors", [])
        pol_anchors = _DOMAIN_ANCHORS.get("politics_anchors", [])
        sports_anchors = _DOMAIN_ANCHORS.get("sports_anchors", [])

        has_biz_anchors = any(
            re.search(r"\b" + re.escape(a) + r"\b", f"{hl}\n{sub}\n{b_sample}")
            for a in biz_anchors
        )
        has_pol_anchors = any(
            re.search(r"\b" + re.escape(a) + r"\b", f"{hl}\n{sub}\n{b_sample}")
            for a in pol_anchors
        )
        has_sports_anchors = any(
            re.search(r"\b" + re.escape(a) + r"\b", f"{hl}\n{sub}\n{b_sample}")
            for a in sports_anchors
        )

        # Disambiguate: If business or political anchors are dominant and no genuine sports anchors exist:
        if (has_biz_anchors or has_pol_anchors) and not has_sports_anchors:
            category_scores["Sports"] *= 0.15
            category_scores["World/International"] *= 0.25  # Dampens "war" in "drug war" / "price war"

        if has_biz_anchors:
            category_scores["Business & Markets"] += 4.0

        if has_pol_anchors:
            category_scores["Politics"] += 4.0

        # 3. Printed Section Boost
        if printed_section:
            mapped_cat, mapped_conf = self.map_section_to_category(printed_section)
            if mapped_cat and mapped_cat in category_scores:
                category_scores[mapped_cat] += 6.0

        # Sort categories by score
        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = sorted_cats[0]

        if top_score <= 0.0:
            # Fallback based on generic content cues
            combined = f"{hl}\n{sub}\n{b_sample}"
            if any(k in combined for k in ("market", "crore", "rupee", "bank", "fund", "shares")):
                top_cat, top_score = "Business & Markets", 2.0
            elif any(k in combined for k in ("minister", "government", "police", "court", "state")):
                top_cat, top_score = "Politics", 2.0
            else:
                top_cat, top_score = "Politics", 1.0

        # Compute confidence (bounded 0.60 to 0.98)
        confidence = min(0.98, max(0.60, round(0.60 + (top_score / 15.0) * 0.38, 2)))

        # Find secondary categories (score >= 3.0 and within 60% of top score)
        secondary: list[tuple[str, float]] = []
        for cat, sc in sorted_cats[1:]:
            if sc >= 3.0 and sc >= (top_score * 0.40):
                norm_conf = min(0.90, max(0.50, round(0.50 + (sc / 15.0) * 0.40, 2)))
                secondary.append((cat, norm_conf))

        return top_cat, confidence, secondary

    def classify_and_score(
        self,
        article: AssembledArticle,
        total_issue_pages: int = 1,
        vlm_section: str | None = None,
        vlm_article_type: str | None = None,
        vlm_prominence: str | None = None,
        printed_section: str | None = None,
    ) -> ClassificationResult:
        """Classify article type, canonical category, secondary topics, and compute prominence."""
        headline = (article.headline or "").strip()
        subheadline = (article.subheadline or "").strip()
        body = (article.full_text or "").strip()
        combined_text = f"{headline}\n{subheadline}\n{body}"

        # Resolve genuine printed section from assembler or argument
        active_printed_sec = printed_section or article.printed_section

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
            sub_lower = subheadline.lower()
            editorial_kws = ("our view", "my view", "their view", "column", "curator", "quick edit")
            if any(k in sub_lower for k in editorial_kws):
                article_type = "editorial"
                section = "Opinion & Editorial"
            elif any(k in sub_lower for k in ("mark to market", "plain facts")):
                section = "Markets & Data"
            elif "deals, tech & startups" in sub_lower:
                section = "Deals, Tech & Startups"
            else:
                section = None

        # 2. Determine Canonical Newsroom Category & Secondary Topics
        canonical_category: str | None = None
        category_conf = 0.85
        secondary_cats: list[tuple[str, float]] = []

        if article_type in ("advertisement", "photo_caption") and article.word_count < 40:
            canonical_category = None
            category_conf = 1.0
        else:
            top_cat, conf, secondary = self.infer_category_from_content(
                headline=headline,
                subheadline=subheadline,
                body=body,
                printed_section=active_printed_sec,
            )
            canonical_category = top_cat
            category_conf = conf
            secondary_cats = secondary

        # 3. Harmonize Physical Section with Semantic Category and Page Position
        if not section:
            if article.primary_page_number == 1:
                section = "Front Page"
            elif active_printed_sec:
                section = active_printed_sec
            elif vlm_section:
                section = vlm_section
            elif canonical_category in (
                "Sports",
                "Technology",
                "Entertainment",
                "Science & Environment",
                "Health",
                "Crime & Law",
                "Lifestyle",
            ):
                section = canonical_category
            elif canonical_category == "Business & Markets":
                section = "Corporate & Industry"
            elif canonical_category == "Economy & Policy":
                section = "Economy & Policy"
            elif canonical_category == "Opinion/Editorial":
                section = "Opinion & Editorial"
            elif canonical_category == "World/International":
                section = "International"
            else:
                section = "National"

        # 4. Compute Prominence Score (0.05 to 1.0)
        tier_map = {"lead": 0.90, "major": 0.70, "standard": 0.50, "minor": 0.30, "filler": 0.15}
        if vlm_prominence and vlm_prominence in tier_map:
            prominence_score = tier_map[vlm_prominence]
        else:
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
            printed_section=active_printed_sec,
            secondary_categories=secondary_cats,
        )

