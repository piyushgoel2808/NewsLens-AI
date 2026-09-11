"""Newspaper Layout Analysis & Story Continuity Subpackage.

Unifies:
1. Spatial 2D column reading order resolution and multi-column bounding box extraction.
2. Rule-based & VLM-driven article boundary segmentation.
3. Cross-page jump stitching and multi-page article continuation assembly.
4. Centralized wire agency, syndication, and dateline token classification.
"""

from __future__ import annotations

from app.ingestion.layout.analyzer import (
    BlockType,
    LayoutAnalyzer,
    LayoutElement,
    OrderedReadingBlock,
    PageLayoutResult,
    ReadingOrderResolver,
)
from app.ingestion.layout.segmenter import (
    ArticleSegmenter,
    AssembledArticle,
    CrossPageAssembler,
    PageBBoxMapping,
    SegmentedArticle,
)
from app.ingestion.layout.slugs import (
    DATELINE_CITIES,
    SECTION_HEADER_BLACKLIST,
    SYNDICATION_SLUGS,
    WIRE_AGENCIES,
    clean_ocr_text_artifacts,
    extract_kicker_and_clean_headline,
    is_garbled_ocr_noise,
    is_numbered_feature_subhead,
    is_numeric_stat_box,
    is_pullquote_author_block,
    is_syndication_or_agency_slug,
    is_toc_index_block,
    is_valid_headline_candidate,
)

__all__ = [
    "ArticleSegmenter",
    "AssembledArticle",
    "BlockType",
    "CrossPageAssembler",
    "DATELINE_CITIES",
    "LayoutAnalyzer",
    "LayoutElement",
    "OrderedReadingBlock",
    "PageBBoxMapping",
    "PageLayoutResult",
    "ReadingOrderResolver",
    "SECTION_HEADER_BLACKLIST",
    "SYNDICATION_SLUGS",
    "SegmentedArticle",
    "WIRE_AGENCIES",
    "clean_ocr_text_artifacts",
    "extract_kicker_and_clean_headline",
    "is_garbled_ocr_noise",
    "is_numbered_feature_subhead",
    "is_numeric_stat_box",
    "is_pullquote_author_block",
    "is_syndication_or_agency_slug",
    "is_toc_index_block",
    "is_valid_headline_candidate",
]
