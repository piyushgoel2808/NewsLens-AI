"""Re-export shim for consensus_extractor backward compatibility.

Canonical implementation has moved to `app.ingestion.metadata`.
"""

from __future__ import annotations

from app.ingestion.metadata import (
    _DATE_PATTERNS,
    _KNOWN_MASTHEADS,
    _MONTH_MAP,
    _MONTH_PATTERN,
    ConsensusExtractor,
    _parse_extracted_date,
    extract_newspaper_and_date_consensus,
)

__all__ = [
    "ConsensusExtractor",
    "_DATE_PATTERNS",
    "_KNOWN_MASTHEADS",
    "_MONTH_MAP",
    "_MONTH_PATTERN",
    "_parse_extracted_date",
    "extract_newspaper_and_date_consensus",
]
