"""Consolidated Newspaper Metadata, Masthead & Folio Extraction Subsystem.

Unifies header band parsing across printed broadsheets:
1. Printed folio extraction with DPI-synchronized spatial coordinate zone filtering.
2. Visual OCR-based masthead brand & publication date verification via RapidOCR.
3. Multi-page consensus extraction applying majority-vote and visual fallback.
4. Centralized date parsing regexes and metadata stripping routines.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import date
from typing import Any

import pymupdf
from rapidocr import RapidOCR

from app.core.logging import get_logger
from app.ingestion.detector import DigitalTextBlock
from app.providers.base import OCRBlock

logger = get_logger(__name__)

# =============================================================================
# Shared Date & Month Parsing Engine
# =============================================================================

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_MONTH_PATTERN: str = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
)
_MONTHS_PATTERN = _MONTH_PATTERN  # Alias for backward compatibility

_DAYS_OF_WEEK_PATTERN: str = (
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|wed|thu|fri|sat|sun)"
)

# Date extraction regex patterns (matching diverse broadsheet dateline variations)
_DATE_PATTERNS: list[re.Pattern[str]] = [
    # Day Month Year (e.g. "30 JULY 2026", "30th July, 2026", "27-Aug-2026")
    re.compile(
        rf"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\.\,\-\/]+({_MONTH_PATTERN})[\s\.\,\-\/]+(\d{{4}})\b"
    ),
    # Month Day Year (e.g. "JULY 30, 2026", "July 30th 2026")
    re.compile(
        rf"(?i)\b({_MONTH_PATTERN})[\s\.\,\-\/]+(\d{{1,2}})(?:st|nd|rd|th)?[\s\.\,\-\/]+(\d{{4}})\b"
    ),
    # ISO: 2026-08-27 / 2026/08/27
    re.compile(r"\b(\d{4})[\s\.\,\-\/]+(\d{1,2})[\s\.\,\-\/]+(\d{1,2})\b"),
    # DD/MM/YYYY or MM/DD/YYYY: 27/08/2026
    re.compile(r"\b(\d{1,2})[\s\.\,\-\/]+(\d{1,2})[\s\.\,\-\/]+(\d{4})\b"),
    # Compact DDMMYYYY: 27082026 (e.g. from normalized unicode superscripts)
    re.compile(r"\b(\d{2})(\d{2})(202\d)\b"),
]

# Supplementary stripping patterns for folio isolation
_METADATA_STRIP_PATTERNS: list[re.Pattern[str]] = [
    # Day Month Year without 4-digit year (e.g. "30 JULY", "THURSDAY. 30 JULY")
    re.compile(
        rf"(?i)\b\d{{1,2}}(?:st|nd|rd|th)?[\s\.\,\-\/]+{_MONTHS_PATTERN}\b(?:\s*[\.\,\-\/]*\s*\d{{2,4}})?"
    ),
    # Month Day without 4-digit year (e.g. "July 30")
    re.compile(
        rf"(?i)\b{_MONTHS_PATTERN}[\s\.\,\-\/]+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*[\.\,\-\/]*\s*\d{{2,4}})?\b"
    ),
    # Day-of-week with day number e.g. "THURSDAY. 30", "FRIDAY, 31"
    re.compile(rf"(?i)\b{_DAYS_OF_WEEK_PATTERN}[\s\.\,\-\/]+\d{{1,2}}\b"),
    # Standalone 4-digit years (e.g. 1920..2099)
    re.compile(r"\b(?:19|20)\d{2}\b"),
    # Day names
    re.compile(rf"(?i)\b{_DAYS_OF_WEEK_PATTERN}\b"),
    # Month names alone
    re.compile(rf"(?i)\b{_MONTHS_PATTERN}\b"),
    # Volume / Issue / Edition identifiers (e.g. "VOL. 18 NO. 145", "VOL. LXVIII NO. 22,415")
    re.compile(r"(?i)\b(?:VOL(?:UME)?\.?|NO\.?|ISSUE|EDITION)\s*[\w\d,\.-]+"),
    # Currency / Price tags (e.g. "Rs. 10", "Rs 10.00", "₹10")
    re.compile(r"(?i)\b(?:RS\.?|INR|₹)\s*\d+(?:\.\d{2})?"),
]


def _parse_extracted_date(groups: tuple[str, ...]) -> date | None:
    """Validate and convert regex match groups into a Python date."""
    try:
        if len(groups) == 3:
            g1, g2, g3 = groups
            # Case 1: Day Month Year (e.g. ("27", "august", "2026"))
            if g1.isdigit() and g2.lower() in _MONTH_MAP and g3.isdigit():
                day = int(g1)
                mon = _MONTH_MAP[g2.lower()]
                yr = int(g3)
            # Case 2: Month Day Year (e.g. ("august", "27", "2026"))
            elif g1.lower() in _MONTH_MAP and g2.isdigit() and g3.isdigit():
                mon = _MONTH_MAP[g1.lower()]
                day = int(g2)
                yr = int(g3)
            # Case 3: ISO YYYY-MM-DD (e.g. ("2026", "08", "27"))
            elif g1.isdigit() and int(g1) >= 2020 and g2.isdigit() and g3.isdigit():
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

            if 1 <= day <= 31 and 1 <= mon <= 12 and 2020 <= yr <= 2035:
                return date(yr, mon, day)
    except Exception:
        pass
    return None


_parse_date_groups = _parse_extracted_date  # Backward-compatibility alias


def strip_dates_and_metadata(text: str) -> str:
    """Remove date expressions, years, days of week, and volume metadata."""
    cleaned = text
    for pat in _DATE_PATTERNS + _METADATA_STRIP_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# =============================================================================
# Printed Page Folio Detection Subsystem
# =============================================================================

# Strict Roman numerals for printed folios (I through XX)
_ROMAN_FOLIO_PATTERN = r"(?:I{1,3}|IV|V|VI{1,3}|IX|X{1,3}|XI{1,3}|XIV|XV|XVI{1,3}|XIX|XX)"

# Primary regex for printed folio in header/footer with explicit keyword prefix
FOLIO_PAGE_REGEX = re.compile(
    rf"(?i)\b(?:PAGE|PG|P\.)\s*([A-Z]\s*[-–]\s*\d{{1,2}}|\d{{1,3}}|{_ROMAN_FOLIO_PATTERN})\b"
)
FOLIO_HEADER_LINE_REGEX = re.compile(
    rf"(?i)(?:DELHI|MUMBAI|KOLKATA|CHENNAI|BANGALORE|BENGALURU|HYDERABAD|AHMEDABAD|PUNE|"
    r"BUSINESS STANDARD|THE HINDU|TIMES|RECORD|CHRONICLE|TRIBUNE|EXPRESS|MINT|LIVE MINT|LIVEMINT)"
    rf"\s*[\|•·\-]?\s*.*?\s*[\|•·\-]?\s*(?:PAGE\s*)?([A-Z]\s*[-–]\s*\d{{1,2}}|\d{{1,3}}|{_ROMAN_FOLIO_PATTERN})\s*$"
)
FOLIO_CORNER_DIGIT_REGEX = re.compile(r"^\s*(\d{1,3})\s*$")
SECTION_FOLIO_REGEX = re.compile(r"\b([A-Z]\s*[-–]\s*\d{1,2})\b")

# Common brand initials / logos to reject from standalone folio matching
DISALLOWED_BRAND_FOLIOS = {
    "M", "BS", "ET", "TH", "HT", "TOI", "BL", "FE", "IE", "MINT", "LIVEMINT", "HINDU"
}

# Edge-based isolated folio regex (e.g. "13" or "B-3" at string boundaries)
_TRAILING_FOLIO_REGEX = re.compile(
    r"(?:^|[\s|•·\-,])([A-Z]\s*[-–]\s*\d{1,2}|\d{1,3})\s*$"
)
_LEADING_FOLIO_REGEX = re.compile(
    r"^\s*([A-Z]\s*[-–]\s*\d{1,2}|\d{1,3})(?:[\s|•·\-,]|$)"
)
_ISOLATED_FOLIO_REGEX = re.compile(
    r"(?:^|\s)([A-Z]\s*[-–]\s*\d{1,2}|\d{1,3})(?:\s|$)"
)


def _validate_folio_candidate(
    cand: str | None,
    total_issue_pages: int | None = None,
    current_pdf_page: int | None = None,
) -> str | None:
    """Validate that candidate string is a valid printed folio and within bounds."""
    if not cand:
        return None
    val = cand.strip().upper()
    if val in DISALLOWED_BRAND_FOLIOS:
        return None
    # Reject single alpha characters that are not digits or valid Roman 'I', 'V', 'X'
    if len(val) == 1 and not val.isdigit() and val not in {"I", "V", "X"}:
        return None
    # If digits, validate within physical document bounds
    if val.isdigit():
        num = int(val)
        max_allowed = 200
        if total_issue_pages is not None:
            max_allowed = total_issue_pages
        elif current_pdf_page is not None:
            max_allowed = current_pdf_page + 2
        if 1 <= num <= max_allowed:
            return str(num)
        return None
    # If section code e.g. B-3 or A-12
    if re.match(r"^[A-Z]\s*[-–]\s*\d{1,2}$", val):
        return re.sub(r"\s+", "", val)
    # If Roman numeral (I through XX)
    if re.match(rf"^{_ROMAN_FOLIO_PATTERN}$", val):
        return val
    return None


class FolioDetector:
    """Extracts printed newspaper page numbers (folios) using spatial coordinate zone parsing."""

    def _extract_bbox_and_text(
        self, block: Any, height_px: float
    ) -> tuple[tuple[float, float, float, float], str] | None:
        """Extract bounding box (x0, y0, x1, y1) and text from various block shapes."""
        raw_bbox = None
        text = ""

        if isinstance(block, dict):
            raw_bbox = block.get("bbox")
            text = str(block.get("text", ""))
        elif hasattr(block, "bbox") and hasattr(block, "text"):
            raw_bbox = block.bbox
            text = str(block.text)

        if raw_bbox is None or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
            return None

        clean_text = text.strip()
        if not clean_text:
            return None

        try:
            x0, y0, x1, y1 = (
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
            return ((x0, y0, x1, y1), clean_text)
        except (ValueError, TypeError):
            return None

    def _extract_folio_from_text(
        self,
        text: str,
        total_issue_pages: int | None = None,
        current_pdf_page: int | None = None,
    ) -> str | None:
        """Scan a candidate text string for an explicit or boundary-positioned folio."""
        # Step 1: Check for explicit PAGE 7 / PAGE B-2 format
        pg_match = FOLIO_PAGE_REGEX.search(text)
        if pg_match:
            cand = _validate_folio_candidate(
                pg_match.group(1),
                total_issue_pages=total_issue_pages,
                current_pdf_page=current_pdf_page,
            )
            if cand is not None:
                return cand

        # Step 2: Strip date strings, days of week, 4-digit years, and issue metadata
        cleaned = strip_dates_and_metadata(text)
        if not cleaned:
            return None

        # Step 3: Check header line with pipes/delimiters on cleaned text
        hl_match = FOLIO_HEADER_LINE_REGEX.search(cleaned)
        if hl_match:
            cand = _validate_folio_candidate(
                hl_match.group(1),
                total_issue_pages=total_issue_pages,
                current_pdf_page=current_pdf_page,
            )
            if cand is not None:
                return cand

        # Step 4: Check for explicit section folio: "B-4"
        sec_match = SECTION_FOLIO_REGEX.search(cleaned)
        if sec_match:
            cand = _validate_folio_candidate(
                sec_match.group(1),
                total_issue_pages=total_issue_pages,
                current_pdf_page=current_pdf_page,
            )
            if cand is not None:
                return cand

        # Step 5: Check for standalone corner digit (e.g. single number line "13")
        corner_match = FOLIO_CORNER_DIGIT_REGEX.match(cleaned)
        if corner_match:
            cand = _validate_folio_candidate(
                corner_match.group(1),
                total_issue_pages=total_issue_pages,
                current_pdf_page=current_pdf_page,
            )
            if cand is not None:
                return cand

        # Step 6: Positional Priority - Trailing edge (e.g. "BENGALURU 13" -> "13")
        trail_match = _TRAILING_FOLIO_REGEX.search(cleaned)
        if trail_match:
            cand = _validate_folio_candidate(
                trail_match.group(1),
                total_issue_pages=total_issue_pages,
                current_pdf_page=current_pdf_page,
            )
            if cand is not None:
                return cand

        # Step 7: Positional Priority - Leading edge (e.g. "13 BENGALURU" -> "13")
        lead_match = _LEADING_FOLIO_REGEX.search(cleaned)
        if lead_match:
            cand = _validate_folio_candidate(
                lead_match.group(1),
                total_issue_pages=total_issue_pages,
                current_pdf_page=current_pdf_page,
            )
            if cand is not None:
                return cand

        # Step 8: Isolated number or section code within cleaned string
        isolated_matches = _ISOLATED_FOLIO_REGEX.findall(cleaned)
        if isolated_matches:
            cand = _validate_folio_candidate(
                str(isolated_matches[-1]),
                total_issue_pages=total_issue_pages,
                current_pdf_page=current_pdf_page,
            )
            if cand is not None:
                return cand

        return None

    def extract_folio(
        self,
        blocks: Sequence[Any],
        height_px: float,
        width_px: float,
        page_number: int = 1,
        is_advertisement_page: bool = False,
        last_known_folio_num: int | None = None,
        last_known_pdf_page: int | None = None,
        total_issue_pages: int | None = None,
    ) -> str:
        """Extract printed folio using relative Y-axis coordinate zone filtering."""
        valid_blocks: list[tuple[tuple[float, float, float, float], str]] = []
        for block in blocks:
            extracted = self._extract_bbox_and_text(block, height_px)
            if extracted:
                valid_blocks.append(extracted)

        # Fallback gracefully if no blocks have valid bounding boxes
        if not valid_blocks:
            if is_advertisement_page:
                return "Cover/Ad Wrap"
            if (
                last_known_folio_num is not None
                and last_known_pdf_page is not None
                and page_number > last_known_pdf_page
            ):
                delta = page_number - last_known_pdf_page
                inferred = last_known_folio_num + delta
                if total_issue_pages is None or inferred <= total_issue_pages + 2:
                    return str(inferred)
            return f"Unnumbered (PDF p.{page_number})"

        # Auto-detect coordinate scale (DPI Sync)
        max_y = max(b[0][3] for b in valid_blocks)
        given_h = max(float(height_px), 1.0)
        given_w = max(float(width_px), 1.0)

        if max_y <= 1.05 and given_h > 10.0:
            # Normalized (0.0 .. 1.0) coordinate space
            effective_height = 1.0
        elif max_y <= 1200.0 and given_h >= 2800.0:
            # height_px was passed at 300 DPI raster while blocks are in 72 DPI PDF points (A4/tabloid)
            effective_height = given_h / (300.0 / 72.0)
        elif max_y <= 1200.0 and 1700.0 <= given_h < 2800.0:
            # height_px was passed at 150 DPI raster while blocks are in 72 DPI PDF points (A4/tabloid)
            effective_height = given_h / (150.0 / 72.0)
        elif max_y <= 1800.0 and given_h >= 5000.0:
            # Broadsheet at 300 DPI raster while blocks are in 72 DPI PDF points
            effective_height = given_h / (300.0 / 72.0)
        elif max_y <= 1800.0 and 3000.0 <= given_h < 5000.0 and given_w < 3000.0:
            # Broadsheet at 150 DPI raster while blocks are in 72 DPI PDF points
            effective_height = given_h / (150.0 / 72.0)
        else:
            effective_height = given_h

        header_blocks: list[tuple[tuple[float, float, float, float], str]] = []
        footer_blocks: list[tuple[tuple[float, float, float, float], str]] = []

        for (x0, y0, x1, y1), text in valid_blocks:
            rel_y0 = y0 / effective_height
            rel_y1 = y1 / effective_height
            height_span = rel_y1 - rel_y0

            # Reject full-page spanning blocks (e.g. height span > 10% of page)
            if height_span > 0.10:
                continue

            # Strict spatial check: Top 5% header strip (relative_y0 < 0.05 and rel_y1 <= 0.08)
            if rel_y0 < 0.05 and rel_y1 <= 0.08:
                header_blocks.append(((x0, y0, x1, y1), text))
            # Strict spatial check: Bottom 5% footer strip (relative_y0 >= 0.95 or rel_y1 >= 0.95)
            elif rel_y0 >= 0.95 or rel_y1 >= 0.95:
                footer_blocks.append(((x0, y0, x1, y1), text))

        # Sort header and footer blocks left-to-right (x0)
        header_blocks.sort(key=lambda item: item[0][0])
        footer_blocks.sort(key=lambda item: item[0][0])

        # Step 1: Scan isolated header zone
        if header_blocks:
            # First check individual header blocks
            for _, h_text in header_blocks:
                folio = self._extract_folio_from_text(
                    h_text,
                    total_issue_pages=total_issue_pages,
                    current_pdf_page=page_number,
                )
                if folio is not None:
                    logger.debug(
                        "Detected folio from header block",
                        extra={
                            "page_number": page_number,
                            "folio": folio,
                            "raw_text": h_text[:60],
                        },
                    )
                    return folio

            # Next check concatenated header string
            combined_header = " | ".join(t for _, t in header_blocks)
            folio = self._extract_folio_from_text(
                combined_header,
                total_issue_pages=total_issue_pages,
                current_pdf_page=page_number,
            )
            if folio is not None:
                logger.debug(
                    "Detected folio from concatenated header text",
                    extra={
                        "page_number": page_number,
                        "folio": folio,
                        "raw_text": combined_header[:60],
                    },
                )
                return folio

        # Step 2: Scan isolated footer zone
        if footer_blocks:
            for _, f_text in footer_blocks:
                folio = self._extract_folio_from_text(
                    f_text,
                    total_issue_pages=total_issue_pages,
                    current_pdf_page=page_number,
                )
                if folio is not None:
                    logger.debug(
                        "Detected folio from footer block",
                        extra={
                            "page_number": page_number,
                            "folio": folio,
                            "raw_text": f_text[:60],
                        },
                    )
                    return folio

            combined_footer = " | ".join(t for _, t in footer_blocks)
            folio = self._extract_folio_from_text(
                combined_footer,
                total_issue_pages=total_issue_pages,
                current_pdf_page=page_number,
            )
            if folio is not None:
                logger.debug(
                    "Detected folio from concatenated footer text",
                    extra={
                        "page_number": page_number,
                        "folio": folio,
                        "raw_text": combined_footer[:60],
                    },
                )
                return folio

        # Step 3: Robust Fallback handling
        if is_advertisement_page:
            return "Cover/Ad Wrap"

        # Section Boundary Safety: Sequential offset extrapolation only applies
        # if last_known_folio_num is a genuine integer and within a 5-page window.
        if (
            isinstance(last_known_folio_num, int)
            and last_known_pdf_page is not None
            and page_number > last_known_pdf_page
        ):
            delta = page_number - last_known_pdf_page
            if 1 <= delta <= 5:
                inferred = last_known_folio_num + delta
                if total_issue_pages is None or inferred <= total_issue_pages + 2:
                    logger.debug(
                        "Extrapolated folio from last known page",
                        extra={
                            "page_number": page_number,
                            "inferred_folio": inferred,
                            "last_known": last_known_folio_num,
                        },
                    )
                    return str(inferred)

        return f"Unnumbered (PDF p.{page_number})"

    def extract_printed_page_number(
        self,
        page_number: int,
        height_px: float,
        width_px: float,
        digital_blocks: Sequence[DigitalTextBlock] | None = None,
        ocr_blocks: Sequence[OCRBlock] | None = None,
        is_advertisement_page: bool = False,
        last_known_folio_num: int | None = None,
        last_known_pdf_page: int | None = None,
        blocks: Sequence[Any] | None = None,
        total_issue_pages: int | None = None,
    ) -> str:
        """Determine the printed folio string for a page with spatial coordinate filtering."""
        all_blocks: list[Any] = []
        if blocks is not None:
            all_blocks.extend(blocks)
        if digital_blocks is not None:
            all_blocks.extend(digital_blocks)
        if ocr_blocks is not None:
            all_blocks.extend(ocr_blocks)

        return self.extract_folio(
            blocks=all_blocks,
            height_px=height_px,
            width_px=width_px,
            page_number=page_number,
            is_advertisement_page=is_advertisement_page,
            last_known_folio_num=last_known_folio_num,
            last_known_pdf_page=last_known_pdf_page,
            total_issue_pages=total_issue_pages,
        )


# =============================================================================
# Visual Masthead Verification Subsystem
# =============================================================================

# Known broadsheet masthead signatures (checked in priority order, longest/most specific first)
_MASTHEAD_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:THE\s+)?NEW\s+YORK\s+TIMES\b|\bINTERNATIONAL\s+NEW\s+YORK\s+TIMES\b|\bNYTIMES(?:\.COM)?\b", re.I), "The New York Times"),
    (re.compile(r"\b(?:THE\s+)?WALL\s+STREET\s+JOURNAL\b|\bWSJ(?:\.COM)?\b", re.I), "The Wall Street Journal"),
    (re.compile(r"\bFINANCIAL\s+TIMES\b|\bFT(?:\.COM)?\b", re.I), "Financial Times"),
    (re.compile(r"\b(?:THE\s+)?WASHINGTON\s+POST\b|\bWAPO(?:\.COM)?\b", re.I), "The Washington Post"),
    (re.compile(r"\b(?:THE\s+)?GUARDIAN\b", re.I), "The Guardian"),
    (re.compile(r"\bUSA\s+TODAY\b", re.I), "USA Today"),
    (re.compile(r"\bLOS\s+ANGELES\s+TIMES\b|\bLA\s+TIMES\b", re.I), "Los Angeles Times"),
    (re.compile(r"\b(?:THE\s+)?ECONOMIC\s+TIMES\b|\bECONOMICTIMES(?:\.COM)?\b|\bET\s+DELHI\b|\bET\s+MUMBAI\b", re.I), "The Economic Times"),
    (re.compile(r"\b(?:THE\s+)?TIMES\s+OF\s+INDIA\b|\bTIMESOFINDIA\b", re.I), "The Times of India"),
    (re.compile(r"\bBUSINESS\s+STANDARD\b", re.I), "Business Standard"),
    (re.compile(r"\bFINANCIAL\s+EXPRESS\b", re.I), "Financial Express"),
    (re.compile(r"\b(?:THE\s+)?INDIAN\s+EXPRESS\b", re.I), "The Indian Express"),
    (re.compile(r"\b(?:THE\s+)?HINDU\b|\bTH\s+DELHI\b", re.I), "The Hindu"),
    (re.compile(r"\b(?:THE\s+)?TRIBUNE\b|\bDAILY\s+TRIBUNE\b", re.I), "The Tribune"),
    (re.compile(r"\bMINT\b|\bLIVEMINT\b", re.I), "Mint"),
    (re.compile(r"\bDAINIK\s+BHASKAR\b", re.I), "Dainik Bhaskar"),
    (re.compile(r"\bAMAR\s+UJALA\b", re.I), "Amar Ujala"),
    (re.compile(r"\bHINDUSTAN\s+TIMES\b|\bHT\s+DELHI\b", re.I), "Hindustan Times"),
    (re.compile(r"\bDECCAN\s+HERALD\b", re.I), "Deccan Herald"),
    (re.compile(r"\bTHE\s+TELEGRAPH\b", re.I), "The Telegraph"),
    (re.compile(r"\bNAVBHARAT\s+TIMES\b", re.I), "Navbharat Times"),
]


class MastheadVerifier:
    """Fast, visual OCR-based masthead and publication date verifier for broadsheets."""

    def __init__(self) -> None:
        self._ocr: RapidOCR | None = None

    def _get_ocr(self) -> RapidOCR:
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

    def verify_from_pdf_bytes(
        self,
        pdf_bytes: bytes,
        filename: str | None = None,
    ) -> tuple[str | None, date | None, float, dict[str, Any]]:
        """Verify masthead brand and publication date from PDF Page 1."""
        if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
            return None, None, 0.0, {"error": "Invalid PDF bytes"}

        doc: pymupdf.Document | None = None
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) == 0:
                return None, None, 0.0, {"error": "Empty PDF document"}

            page1 = doc[0]
            rect = page1.rect
            # Crop top 22% of Page 1 (standard masthead region on broadsheets)
            masthead_rect = pymupdf.Rect(0, 0, rect.width, rect.height * 0.22)
            pix = page1.get_pixmap(clip=masthead_rect, dpi=200)
            crop_png_bytes = pix.tobytes("png")

            return self.verify_from_image_bytes(
                image_bytes=crop_png_bytes,
                filename=filename,
            )
        except Exception as e:
            logger.warning("Masthead verification encountered error", extra={"error": str(e)})
            return None, None, 0.0, {"error": str(e)}
        finally:
            if doc is not None:
                doc.close()

    def verify_from_image_bytes(
        self,
        image_bytes: bytes,
        filename: str | None = None,
    ) -> tuple[str | None, date | None, float, dict[str, Any]]:
        """Verify masthead brand and publication date from masthead crop image."""
        ocr = self._get_ocr()
        ocr_out = ocr(image_bytes)

        txts: tuple[str, ...] = getattr(ocr_out, "txts", None) or ()
        scores: tuple[float, ...] = getattr(ocr_out, "scores", None) or ()

        all_lines: list[str] = list(txts)
        full_masthead_text = " ".join(all_lines)

        detected_brand: str | None = None
        brand_confidence: float = 0.0
        detected_date: date | None = None
        date_confidence: float = 0.0

        # 1. Match Brand from OCR text lines
        for pat, brand_name in _MASTHEAD_RULES:
            for idx, line in enumerate(all_lines):
                if pat.search(line):
                    detected_brand = brand_name
                    score = float(scores[idx]) if idx < len(scores) else 0.95
                    brand_confidence = max(brand_confidence, score)
                    break
            if detected_brand:
                break

        # If brand not matched in lines, check combined text
        if not detected_brand:
            for pat, brand_name in _MASTHEAD_RULES:
                if pat.search(full_masthead_text):
                    detected_brand = brand_name
                    brand_confidence = 0.90
                    break

        # 2. Match Publication Date from OCR text lines
        for idx, line in enumerate(all_lines):
            norm_line = unicodedata.normalize("NFKD", line)
            for pat in _DATE_PATTERNS:
                for match in pat.finditer(norm_line):
                    d = _parse_extracted_date(match.groups())
                    if d:
                        detected_date = d
                        score = float(scores[idx]) if idx < len(scores) else 0.95
                        date_confidence = max(date_confidence, score)
                        break
                if detected_date:
                    break
            if detected_date:
                break

        # 3. Filename supplementary hints
        if filename:
            norm_fn = unicodedata.normalize("NFKD", filename)
            # Brand hint in filename
            if not detected_brand:
                fn_upper = norm_fn.upper()
                if re.search(r"\bNYT\b|NEW\s*YORK\s*TIMES", fn_upper):
                    detected_brand = "The New York Times"
                    brand_confidence = 0.90
                elif re.search(r"\bWSJ\b|WALL\s*STREET\s*JOURNAL", fn_upper):
                    detected_brand = "The Wall Street Journal"
                    brand_confidence = 0.90
                elif re.search(r"\bFT\b|FINANCIAL\s*TIMES", fn_upper):
                    detected_brand = "Financial Times"
                    brand_confidence = 0.90
                elif re.search(r"\bWAPO\b|WASHINGTON\s*POST", fn_upper):
                    detected_brand = "The Washington Post"
                    brand_confidence = 0.90
                elif re.search(r"\bET\b|ECONOMIC\s*TIMES", fn_upper):
                    detected_brand = "The Economic Times"
                    brand_confidence = 0.85
                elif re.search(r"\bTOI\b|TIMES\s*OF\s*INDIA", fn_upper):
                    detected_brand = "The Times of India"
                    brand_confidence = 0.85
                elif re.search(r"\bHT\b|HINDUSTAN\s*TIMES", fn_upper):
                    detected_brand = "Hindustan Times"
                    brand_confidence = 0.85
                elif re.search(r"\bMINT\b", fn_upper):
                    detected_brand = "Mint"
                    brand_confidence = 0.85
                elif re.search(r"\bBS\b|BUSINESS\s*STANDARD", fn_upper):
                    detected_brand = "Business Standard"
                    brand_confidence = 0.85
                elif re.search(r"\bIE\b|INDIAN\s*EXPRESS", fn_upper):
                    detected_brand = "The Indian Express"
                    brand_confidence = 0.85

            # Date hint in filename
            if not detected_date:
                for pat in _DATE_PATTERNS:
                    for match in pat.finditer(norm_fn):
                        d = _parse_extracted_date(match.groups())
                        if d:
                            detected_date = d
                            date_confidence = 0.85
                            break
                    if detected_date:
                        break

        overall_conf = round(
            (brand_confidence + date_confidence) / 2.0
            if (detected_brand and detected_date)
            else (brand_confidence or date_confidence),
            3,
        )

        telemetry = {
            "detected_brand": detected_brand,
            "detected_date": str(detected_date) if detected_date else None,
            "brand_confidence": brand_confidence,
            "date_confidence": date_confidence,
            "overall_confidence": overall_conf,
            "ocr_lines_sample": all_lines[:10],
        }

        return detected_brand, detected_date, overall_conf, telemetry

    async def verify_async(
        self,
        pdf_bytes: bytes,
        filename: str | None = None,
    ) -> tuple[str | None, date | None, float, dict[str, Any]]:
        """Asynchronously verify masthead on executor thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.verify_from_pdf_bytes,
            pdf_bytes,
            filename,
        )


# =============================================================================
# Multi-Page Consensus Extraction Subsystem
# =============================================================================

_KNOWN_MASTHEADS: list[tuple[str, str]] = [
    ("THE NEW YORK TIMES", "The New York Times"),
    ("NEW YORK TIMES", "The New York Times"),
    ("INTERNATIONAL NEW YORK TIMES", "The New York Times"),
    ("THE WALL STREET JOURNAL", "The Wall Street Journal"),
    ("WALL STREET JOURNAL", "The Wall Street Journal"),
    ("THE WASHINGTON POST", "The Washington Post"),
    ("WASHINGTON POST", "The Washington Post"),
    ("THE GUARDIAN", "The Guardian"),
    ("FINANCIAL TIMES", "Financial Times"),
    ("USA TODAY", "USA Today"),
    ("LOS ANGELES TIMES", "Los Angeles Times"),
    ("THE ECONOMIC TIMES", "The Economic Times"),
    ("ECONOMIC TIMES", "The Economic Times"),
    ("WWW.ECONOMICTIMES.COM", "The Economic Times"),
    ("THE TIMES OF INDIA", "The Times of India"),
    ("TIMES OF INDIA", "The Times of India"),
    ("BUSINESS STANDARD", "Business Standard"),
    ("THE INDIAN EXPRESS", "The Indian Express"),
    ("INDIAN EXPRESS", "The Indian Express"),
    ("FINANCIAL EXPRESS", "Financial Express"),
    ("THE HINDU", "The Hindu"),
    ("HINDUSTAN TIMES", "Hindustan Times"),
    ("THE TRIBUNE", "The Tribune"),
    ("DAILY TRIBUNE", "The Tribune"),
    ("THE TELEGRAPH", "The Telegraph"),
    ("DECCAN HERALD", "Deccan Herald"),
    ("MINT", "Mint"),
    ("LIVEMINT", "Mint"),
    ("THE GOAN", "The Goan"),
    ("THE GOAN EVERYDAY", "The Goan"),
    ("GOAN EVERYDAY", "The Goan"),
    ("THE DAILY CHRONICLE", "The Daily Chronicle"),
    ("DAILY BROADSHEET", "Daily Broadsheet"),
    ("DAINIK BHASKAR", "Dainik Bhaskar"),
    ("AMAR UJALA", "Amar Ujala"),
    ("HINDUSTAN", "Hindustan"),
    ("NAVBHARAT TIMES", "Navbharat Times"),
]


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

    # Build active candidate brands list, sorting by length descending
    brand_map: dict[str, str] = {kw: brand for kw, brand in _KNOWN_MASTHEADS}
    if existing_newspaper_names:
        for ex_name in existing_newspaper_names:
            if ex_name and ex_name.upper() not in brand_map:
                brand_map[ex_name.upper()] = ex_name

    candidate_brands: list[tuple[str, str]] = sorted(
        brand_map.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )

    doc: pymupdf.Document | None = None
    total_digital_chars = 0
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
            total_digital_chars += len(full_text.strip())

            combined_zone_text = f"{header_text}\n{footer_text}"
            page_dates: list[date] = []
            page_brands: list[str] = []

            # 1. Brand matching in header / page
            header_upper = header_text.upper()
            combined_upper = combined_zone_text.upper()
            full_upper = full_text.upper()

            for keyword, brand_label in candidate_brands:
                if keyword in header_upper:
                    brand_counter[brand_label] += 3  # Header match is high confidence
                    page_brands.append(brand_label)
                    break
                elif keyword in combined_upper or (page_idx == 0 and keyword in full_upper[:1000]):
                    brand_counter[brand_label] += 1
                    page_brands.append(brand_label)
                    break

            # 2. Date extraction: weight header zone 5x, footer zone 2x, full page 1x
            for pat in _DATE_PATTERNS:
                for match in pat.finditer(header_text):
                    d = _parse_extracted_date(match.groups())
                    if d:
                        date_counter[d] += 5
                        page_dates.append(d)

                for match in pat.finditer(footer_text):
                    d = _parse_extracted_date(match.groups())
                    if d:
                        date_counter[d] += 2
                        page_dates.append(d)

                for match in pat.finditer(full_text):
                    d = _parse_extracted_date(match.groups())
                    if d and d not in page_dates:
                        date_counter[d] += 1
                        page_dates.append(d)

            page_reports.append({
                "page_number": page_idx + 1,
                "detected_brands": page_brands,
                "detected_dates": [str(d) for d in set(page_dates)],
            })

        # 3. Inspect Filename for supplementary clues
        if filename:
            norm_fn = unicodedata.normalize("NFKD", filename)
            fn_upper = norm_fn.upper()

            # Check shorthand broadsheet acronyms
            if re.search(r"\bNYT\b|NEW\s*YORK\s*TIMES", fn_upper):
                brand_counter["The New York Times"] += 5
            elif re.search(r"\bWSJ\b|WALL\s*STREET\s*JOURNAL", fn_upper):
                brand_counter["The Wall Street Journal"] += 5
            elif re.search(r"\bFT\b|FINANCIAL\s*TIMES", fn_upper):
                brand_counter["Financial Times"] += 5
            elif re.search(r"\bWAPO\b|WASHINGTON\s*POST", fn_upper):
                brand_counter["The Washington Post"] += 5
            elif re.search(r"\bET\b|ECONOMIC\s*TIMES", fn_upper):
                brand_counter["The Economic Times"] += 4
            elif re.search(r"\bTOI\b|TIMES\s*OF\s*INDIA", fn_upper):
                brand_counter["The Times of India"] += 4
            elif re.search(r"\bHT\b|HINDUSTAN\s*TIMES", fn_upper):
                brand_counter["Hindustan Times"] += 4
            elif re.search(r"\bMINT\b", fn_upper):
                brand_counter["Mint"] += 4
            elif re.search(r"\bBS\b|BUSINESS\s*STANDARD", fn_upper):
                brand_counter["Business Standard"] += 4
            elif re.search(r"\bIE\b|INDIAN\s*EXPRESS", fn_upper):
                brand_counter["The Indian Express"] += 4

            for keyword, brand_label in candidate_brands:
                if keyword in fn_upper:
                    brand_counter[brand_label] += 3

            for pat in _DATE_PATTERNS:
                for match in pat.finditer(norm_fn):
                    d = _parse_extracted_date(match.groups())
                    if d:
                        date_counter[d] += 5  # High confidence hint

        # Compute consensus winners
        consensus_date: date | None = None
        if date_counter:
            consensus_date = date_counter.most_common(1)[0][0]

        consensus_brand: str | None = None
        if brand_counter:
            consensus_brand = brand_counter.most_common(1)[0][0]

        # 4. MastheadVerifier: If digital text was sparse / scanned or brand/date missing
        vlm_verifier_used = False
        if not consensus_brand or not consensus_date or total_digital_chars < 300:
            logger.info("Digital text sparse or incomplete; running visual MastheadVerifier")
            try:
                verifier = MastheadVerifier()
                v_brand, v_date, v_conf, _v_telem = verifier.verify_from_pdf_bytes(
                    pdf_bytes=pdf_bytes,
                    filename=filename,
                )
                if v_brand and (not consensus_brand or v_conf >= 0.8):
                    consensus_brand = v_brand
                if v_date and (not consensus_date or v_conf >= 0.8):
                    consensus_date = v_date
                vlm_verifier_used = True
            except Exception as verifier_err:
                logger.warning("MastheadVerifier execution failed", extra={"error": str(verifier_err)})

        telemetry = {
            "status": "success",
            "pages_inspected": pages_to_check,
            "total_pages": total_pages,
            "total_digital_chars": total_digital_chars,
            "vlm_verifier_used": vlm_verifier_used,
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


__all__ = [
    "ConsensusExtractor",
    "FolioDetector",
    "MastheadVerifier",
    "extract_newspaper_and_date_consensus",
    "strip_dates_and_metadata",
]
