"""Re-export shim for extraction_schemas backward compatibility.

Canonical implementation has moved to `app.ingestion.parsers.schemas`.
"""

from __future__ import annotations

from app.ingestion.parsers.schemas import (
    ArticleEnrichment,
    ArticleGenre,
    ArticleSkeleton,
    ExtractedEntity,
    ExtractedTable,
    PageLayoutExtraction,
    ProminenceTier,
    SectionType,
)

__all__ = [
    "ArticleEnrichment",
    "ArticleGenre",
    "ArticleSkeleton",
    "ExtractedEntity",
    "ExtractedTable",
    "PageLayoutExtraction",
    "ProminenceTier",
    "SectionType",
]
