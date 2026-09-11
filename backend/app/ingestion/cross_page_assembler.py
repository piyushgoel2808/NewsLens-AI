"""Re-export shim for cross_page_assembler backward compatibility.

Canonical implementation has moved to `app.ingestion.layout.segmenter`.
"""

from __future__ import annotations

from app.ingestion.layout.segmenter import (
    AssembledArticle,
    CrossPageAssembler,
    PageBBoxMapping,
    _has_continuation_marker,
    _headline_overlap_metrics,
)

__all__ = [
    "AssembledArticle",
    "CrossPageAssembler",
    "PageBBoxMapping",
    "_has_continuation_marker",
    "_headline_overlap_metrics",
]
