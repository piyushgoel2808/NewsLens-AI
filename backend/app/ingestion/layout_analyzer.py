"""Re-export shim for layout_analyzer backward compatibility.

Canonical implementation has moved to `app.ingestion.layout.analyzer`.
"""

from __future__ import annotations

from app.ingestion.layout.analyzer import (
    NUMBERED_QUESTION_REGEX,
    SYNDICATION_REGEX,
    SYNDICATION_SLUGS,
    BlockType,
    LayoutAnalyzer,
    LayoutElement,
    OrderedReadingBlock,
    PageLayoutResult,
    ReadingOrderResolver,
    clean_ocr_text_artifacts,
    is_numbered_feature_subhead,
    is_numeric_stat_box,
    is_pullquote_author_block,
    is_syndication_or_agency_slug,
    is_toc_index_block,
)

__all__ = [
    "BlockType",
    "LayoutAnalyzer",
    "LayoutElement",
    "NUMBERED_QUESTION_REGEX",
    "OrderedReadingBlock",
    "PageLayoutResult",
    "ReadingOrderResolver",
    "SYNDICATION_REGEX",
    "SYNDICATION_SLUGS",
    "clean_ocr_text_artifacts",
    "is_numbered_feature_subhead",
    "is_numeric_stat_box",
    "is_pullquote_author_block",
    "is_syndication_or_agency_slug",
    "is_toc_index_block",
]
