"""Unit tests for Multi-Page Consensus Extraction Engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pymupdf

from app.ingestion.consensus_extractor import (
    _parse_extracted_date,
    extract_newspaper_and_date_consensus,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_extracted_date():
    """Verify robust date parsing across formats."""
    # Day Month Year
    d1 = _parse_extracted_date(("07", "july", "2026"))
    assert d1 == date(2026, 7, 7)

    # Month Day Year
    d2 = _parse_extracted_date(("july", "07", "2026"))
    assert d2 == date(2026, 7, 7)

    # ISO YYYY-MM-DD
    d3 = _parse_extracted_date(("2026", "07", "07"))
    assert d3 == date(2026, 7, 7)

    # DD/MM/YYYY
    d4 = _parse_extracted_date(("15", "08", "2026"))
    assert d4 == date(2026, 8, 15)

    # Invalid
    assert _parse_extracted_date(("99", "invalid", "2026")) is None


def test_extract_consensus_on_synthetic_pdf():
    """Test consensus extraction across a 3-page synthetic PDF with header/folios."""
    doc = pymupdf.open()
    for page_idx in range(3):
        p = doc.new_page(width=600, height=800)
        # Add running header with Business Standard and date on each page
        p.insert_text(
            pymupdf.Point(50, 40),
            f"BUSINESS STANDARD | MUMBAI • 07 JULY 2026 • PAGE {page_idx + 1}",
            fontsize=10,
        )
        p.insert_text(
            pymupdf.Point(50, 150),
            "Major economic rally on Dalal Street as markets surge.",
            fontsize=12,
        )

    pdf_bytes = doc.write()
    doc.close()

    brand, d, meta = extract_newspaper_and_date_consensus(
        pdf_bytes=pdf_bytes,
        max_pages=5,
        filename="BS_07_July_2026.pdf",
    )

    assert brand == "Business Standard"
    assert d == date(2026, 7, 7)
    assert meta["status"] == "success"
    assert meta["total_pages"] == 3


def test_extract_consensus_invalid_stream():
    """Verify invalid or corrupt stream handling."""
    brand, d, meta = extract_newspaper_and_date_consensus(b"NOT_A_PDF")
    assert brand is None
    assert d is None
    assert meta["status"] == "invalid_pdf"
