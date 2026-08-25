"""Multi-Page Consensus Extraction Engine.

Extracts publication brand and issue date by analyzing headers, folios,
and mastheads across multiple pages of an uploaded newspaper PDF, applying
majority-vote consensus to eliminate single-page misidentifications.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from typing import Any

import pymupdf

from app.core.logging import get_logger

logger = get_logger(__name__)

_KNOWN_MASTHEADS = [
    ("BUSINESS STANDARD", "Business Standard"),
    ("MINT", "Mint"),
    ("LIVEMINT", "Mint"),
    ("THE HINDU", "The Hindu"),
    ("ECONOMIC TIMES", "The Economic Times"),
    ("TIMES OF INDIA", "The Times of India"),
    ("FINANCIAL EXPRESS", "Financial Express"),
    ("INDIAN EXPRESS", "The Indian Express"),
    ("THE TRIBUNE", "The Tribune"),
    ("DECCAN HERALD", "Deccan Herald"),
    ("THE TELEGRAPH", "The Telegraph"),
    ("DAINIK BHASKAR", "Dainik Bhaskar"),
    ("AMAR UJALA", "Amar Ujala"),
    ("HINDUSTAN", "Hindustan"),
    ("NAVBHARAT TIMES", "Navbharat Times"),
]

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
)

_DATE_PATTERNS = [
    # 07 July 2026 / 7th July, 2026 / 07-Jul-2026
    re.compile(rf"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\.\,\-\/]+({_MONTH_PATTERN})[\s\.\,\-\/]+(\d{{4}})\b"),
    # July 07, 2026 / July 7 2026
    re.compile(rf"(?i)\b({_MONTH_PATTERN})[\s\.\,\-\/]+(\d{{1,2}})(?:st|nd|rd|th)?[\s\.\,\-\/]+(\d{{4}})\b"),
    # ISO: 2026-07-07 / 2026/07/07
    re.compile(r"\b(\d{4})[\s\.\,\-\/]+(\d{1,2})[\s\.\,\-\/]+(\d{1,2})\b"),
    # DD/MM/YYYY or MM/DD/YYYY: 07/07/2026
    re.compile(r"\b(\d{1,2})[\s\.\,\-\/]+(\d{1,2})[\s\.\,\-\/]+(\d{4})\b"),
]


def _parse_extracted_date(groups: tuple[str, ...]) -> date | None:
    """Parse regex match groups into a validated date object."""
    try:
        if len(groups) == 3:
            g1, g2, g3 = groups
            # Case 1: Day Month Year (e.g. ("07", "july", "2026"))
            if g1.isdigit() and g2.lower() in _MONTH_MAP and g3.isdigit():
                day = int(g1)
                mon = _MONTH_MAP[g2.lower()]
                yr = int(g3)
            # Case 2: Month Day Year (e.g. ("july", "07", "2026"))
            elif g1.lower() in _MONTH_MAP and g2.isdigit() and g3.isdigit():
                mon = _MONTH_MAP[g1.lower()]
                day = int(g2)
                yr = int(g3)
            # Case 3: ISO YYYY-MM-DD (e.g. ("2026", "07", "07"))
            elif g1.isdigit() and int(g1) > 1900 and g2.isdigit() and g3.isdigit():
                yr = int(g1)
                mon = int(g2)
                day = int(g3)
            # Case 4: DD/MM/YYYY
            elif g1.isdigit() and g2.isdigit() and g3.isdigit():
                day = int(g1)
                mon = int(g2)
                yr = int(g3)
            else:
                return None

            if 1 <= day <= 31 and 1 <= mon <= 12 and 1900 <= yr <= 2050:
                return date(yr, mon, day)
    except Exception:
        pass
    return None


def extract_newspaper_and_date_consensus(
    pdf_bytes: bytes,
    max_pages: int = 15,
    existing_newspaper_names: list[str] | None = None,
    filename: str | None = None,
) -> tuple[str | None, date | None, dict[str, Any]]:
    """Scan all/first N pages and extract publication brand and issue date by consensus.

    Args:
        pdf_bytes: Raw PDF bytes.
        max_pages: Maximum number of pages to inspect (default: 15).
        existing_newspaper_names: Optional list of known newspaper titles in DB.
        filename: Optional filename for supplementary date/brand hints.

    Returns:
        tuple of (consensus_newspaper_name, consensus_issue_date, telemetry_dict).
    """
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        return None, None, {"status": "invalid_pdf"}

    date_counter: Counter[date] = Counter()
    brand_counter: Counter[str] = Counter()
    page_reports: list[dict[str, Any]] = []

    # Prepare active brand dictionary
    candidate_brands: list[tuple[str, str]] = list(_KNOWN_MASTHEADS)
    if existing_newspaper_names:
        for ex_name in existing_newspaper_names:
            if ex_name and (ex_name.upper(), ex_name) not in candidate_brands:
                candidate_brands.insert(0, (ex_name.upper(), ex_name))

    doc: pymupdf.Document | None = None
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_to_check = min(total_pages, max_pages)

        for page_idx in range(pages_to_check):
            page = doc[page_idx]
            rect = page.rect
            page_h = rect.height

            # Extract header zone (top 18%) and footer zone (bottom 10%)
            header_rect = pymupdf.Rect(0, 0, rect.width, page_h * 0.18)
            footer_rect = pymupdf.Rect(0, page_h * 0.90, rect.width, page_h)

            header_text = page.get_text("text", clip=header_rect)
            footer_text = page.get_text("text", clip=footer_rect)
            full_text = page.get_text("text")

            combined_zone_text = f"{header_text}\n{footer_text}"
            page_dates: list[date] = []
            page_brands: list[str] = []

            # 1. Brand matching in header / page
            combined_upper = combined_zone_text.upper()
            full_upper = full_text.upper()

            for keyword, brand_label in candidate_brands:
                if keyword in combined_upper or (page_idx == 0 and keyword in full_upper[:1000]):
                    brand_counter[brand_label] += 1
                    page_brands.append(brand_label)
                    break

            # 2. Date extraction in header & footer zones first, then full page
            for target_text in (combined_zone_text, full_text):
                for pat in _DATE_PATTERNS:
                    for match in pat.finditer(target_text):
                        d = _parse_extracted_date(match.groups())
                        if d:
                            date_counter[d] += 1
                            page_dates.append(d)

            page_reports.append({
                "page_number": page_idx + 1,
                "detected_brands": page_brands,
                "detected_dates": [str(d) for d in set(page_dates)],
            })

        # 3. Inspect Filename for supplementary clues
        if filename:
            fn_upper = filename.upper()
            for keyword, brand_label in candidate_brands:
                if keyword in fn_upper:
                    brand_counter[brand_label] += 2  # Strong signal

            for pat in _DATE_PATTERNS:
                for match in pat.finditer(filename):
                    d = _parse_extracted_date(match.groups())
                    if d:
                        date_counter[d] += 3  # High confidence hint

        # Compute consensus winners
        consensus_date: date | None = None
        if date_counter:
            consensus_date = date_counter.most_common(1)[0][0]

        consensus_brand: str | None = None
        if brand_counter:
            consensus_brand = brand_counter.most_common(1)[0][0]

        telemetry = {
            "status": "success",
            "pages_inspected": pages_to_check,
            "total_pages": total_pages,
            "date_votes": {str(k): v for k, v in date_counter.most_common(5)},
            "brand_votes": dict(brand_counter.most_common(5)),
            "consensus_date": str(consensus_date) if consensus_date else None,
            "consensus_brand": consensus_brand,
            "page_details": page_reports[:5],
        }

        return consensus_brand, consensus_date, telemetry

    except Exception as exc:
        logger.warning(
            "Consensus extraction encountered an error",
            extra={"error": str(exc)},
        )
        return None, None, {"status": "error", "error": str(exc)}
    finally:
        if doc is not None:
            doc.close()


class ConsensusExtractor:
    """Class wrapper for multi-page consensus extraction."""

    def __init__(self, max_pages: int = 15) -> None:
        self.max_pages = max_pages

    def extract_consensus(
        self,
        pdf_bytes: bytes,
        filename: str | None = None,
        existing_newspaper_names: list[str] | None = None,
    ) -> tuple[str | None, date | None, dict[str, Any]]:
        return extract_newspaper_and_date_consensus(
            pdf_bytes=pdf_bytes,
            max_pages=self.max_pages,
            existing_newspaper_names=existing_newspaper_names,
            filename=filename,
        )
