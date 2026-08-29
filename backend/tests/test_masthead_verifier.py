"""Unit tests for Visual MastheadVerifier."""

from __future__ import annotations

from datetime import date

import pytest

from app.ingestion.masthead_verifier import (
    _MASTHEAD_RULES,
    _parse_date_groups,
    MastheadVerifier,
)


class TestMastheadVerifier:
    """Test suite for MastheadVerifier."""

    def test_verifier_initialization(self) -> None:
        verifier = MastheadVerifier()
        assert verifier is not None
        assert hasattr(verifier, "verify_from_pdf_bytes")

    def test_date_parsing_formats(self) -> None:
        # Day Month Year
        d1 = _parse_date_groups(("27", "august", "2026"))
        assert d1 == date(2026, 8, 27)

        # Month Day Year
        d2 = _parse_date_groups(("august", "27", "2026"))
        assert d2 == date(2026, 8, 27)

        # ISO format
        d3 = _parse_date_groups(("2026", "08", "27"))
        assert d3 == date(2026, 8, 27)

        # Invalid dates rejected
        assert _parse_date_groups(("35", "august", "2026")) is None
        assert _parse_date_groups(("27", "invalid_month", "2026")) is None

    def test_brand_matching_priority(self) -> None:
        text_full = "THE ECONOMIC TIMES WWW.ECONOMICTIMES.COM THURSDAY, 27 AUGUST 2026"
        matched_brand = None
        for pattern, brand in _MASTHEAD_RULES:
            if pattern.search(text_full):
                matched_brand = brand
                break
        assert matched_brand == "The Economic Times"
