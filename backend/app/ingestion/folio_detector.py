"""Re-export shim for folio_detector backward compatibility.

Canonical implementation has moved to `app.ingestion.metadata`.
"""

from __future__ import annotations

from app.ingestion.metadata import (
    _DATE_PATTERNS,
    _DAYS_OF_WEEK_PATTERN,
    _ISOLATED_FOLIO_REGEX,
    _LEADING_FOLIO_REGEX,
    _MONTHS_PATTERN,
    _ROMAN_FOLIO_PATTERN,
    _TRAILING_FOLIO_REGEX,
    DISALLOWED_BRAND_FOLIOS,
    FOLIO_CORNER_DIGIT_REGEX,
    FOLIO_HEADER_LINE_REGEX,
    FOLIO_PAGE_REGEX,
    SECTION_FOLIO_REGEX,
    FolioDetector,
    _validate_folio_candidate,
    strip_dates_and_metadata,
)

__all__ = [
    "DISALLOWED_BRAND_FOLIOS",
    "FOLIO_CORNER_DIGIT_REGEX",
    "FOLIO_HEADER_LINE_REGEX",
    "FOLIO_PAGE_REGEX",
    "SECTION_FOLIO_REGEX",
    "FolioDetector",
    "_DATE_PATTERNS",
    "_DAYS_OF_WEEK_PATTERN",
    "_ISOLATED_FOLIO_REGEX",
    "_LEADING_FOLIO_REGEX",
    "_MONTHS_PATTERN",
    "_ROMAN_FOLIO_PATTERN",
    "_TRAILING_FOLIO_REGEX",
    "_validate_folio_candidate",
    "strip_dates_and_metadata",
]
