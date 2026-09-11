"""Re-export shim for docling_parser backward compatibility.

Canonical implementation has moved to `app.ingestion.parsers.docling`.
"""

from __future__ import annotations

from app.ingestion.parsers.docling import (
    _AUTHOR_NAME_PATTERN,
    _DATELINE_PATTERN,
    _PAGE_HEADER_KEYWORDS,
    CorruptedPdfTextLayerError,
    DoclingLayoutParser,
    DoclingParsedItem,
    ExtractedPhotoData,
)

__all__ = [
    "CorruptedPdfTextLayerError",
    "DoclingLayoutParser",
    "DoclingParsedItem",
    "ExtractedPhotoData",
    "_AUTHOR_NAME_PATTERN",
    "_DATELINE_PATTERN",
    "_PAGE_HEADER_KEYWORDS",
]
