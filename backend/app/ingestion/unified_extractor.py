"""Re-export shim for unified_extractor backward compatibility.

Canonical implementation has moved to `app.ingestion.parsers.vlm`.
"""

from __future__ import annotations

from app.ingestion.parsers.vlm import (
    PHASE1_LAYOUT_PROMPT,
    PHASE2_ENRICH_PROMPT,
    UnifiedExtractor,
)

__all__ = [
    "PHASE1_LAYOUT_PROMPT",
    "PHASE2_ENRICH_PROMPT",
    "UnifiedExtractor",
]
