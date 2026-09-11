"""Re-export shim for segmenter backward compatibility.

Canonical implementation has moved to `app.ingestion.layout.segmenter`.
"""

from __future__ import annotations

from app.ingestion.layout.segmenter import (
    BYLINE_REGEX,
    JUMP_IN_REGEX,
    JUMP_OUT_REGEX,
    MIN_ARTICLE_WORD_COUNT,
    TEASER_REGEX,
    ArticleSegmenter,
    SegmentedArticle,
)
from app.ingestion.layout.slugs import (
    DATELINE_CITIES,
    SECTION_HEADER_BLACKLIST,
    SYNDICATION_SLUGS,
    WIRE_AGENCIES,
    extract_kicker_and_clean_headline,
    is_garbled_ocr_noise,
    is_numbered_feature_subhead,
    is_syndication_or_agency_slug,
    is_valid_headline_candidate,
)

__all__ = [
    "ArticleSegmenter",
    "BYLINE_REGEX",
    "DATELINE_CITIES",
    "JUMP_IN_REGEX",
    "JUMP_OUT_REGEX",
    "MIN_ARTICLE_WORD_COUNT",
    "SECTION_HEADER_BLACKLIST",
    "SYNDICATION_SLUGS",
    "SegmentedArticle",
    "TEASER_REGEX",
    "WIRE_AGENCIES",
    "extract_kicker_and_clean_headline",
    "is_garbled_ocr_noise",
    "is_numbered_feature_subhead",
    "is_syndication_or_agency_slug",
    "is_valid_headline_candidate",
]
