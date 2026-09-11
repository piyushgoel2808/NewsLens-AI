"""Re-export shim for masthead_verifier backward compatibility.

Canonical implementation has moved to `app.ingestion.metadata`.
"""

from __future__ import annotations

from app.ingestion.metadata import (
    _DATE_PATTERNS,
    _MASTHEAD_RULES,
    _MONTH_MAP,
    _MONTH_PATTERN,
    MastheadVerifier,
    _parse_date_groups,
    _parse_extracted_date,
)

__all__ = [
    "MastheadVerifier",
    "_DATE_PATTERNS",
    "_MASTHEAD_RULES",
    "_MONTH_MAP",
    "_MONTH_PATTERN",
    "_parse_date_groups",
    "_parse_extracted_date",
]
