from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from app.core.logging import get_logger
from app.ingestion.detector import DigitalTextBlock
from app.providers.base import OCRBlock

logger = get_logger(__name__)

# Strict Roman numerals for printed folios (I through XX)
# Rejects single brand characters like 'M', 'C', 'D'
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

# Regexes for dates, years, and metadata removal
_MONTHS_PATTERN = (
    r"(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|"
    r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)"
)
_DAYS_OF_WEEK_PATTERN = (
    r"(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY|"
    r"MON|TUE|WED|THU|FRI|SAT|SUN)"
)

# Date and metadata stripping patterns
_DATE_PATTERNS: list[re.Pattern[str]] = [
    # Day Month Year (e.g. "30 JULY 2026", "30th July, 2026", "30 JULY")
    re.compile(
        rf"(?i)\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS_PATTERN}\b(?:\s*,?\s*\d{{2,4}})?",
    ),
    # Month Day Year (e.g. "JULY 30, 2026", "July 30th 2026", "July 30")
    re.compile(
        rf"(?i)\b{_MONTHS_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{2,4}})?\b",
    ),
    # Full numeric dates (e.g. "30/07/2026", "30-07-2026", "30.07.2026")
    re.compile(r"\b\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}\b"),
    # Standalone 4-digit years (e.g. 1920..2099)
    re.compile(r"\b(?:19|20)\d{2}\b"),
    # Day names
    re.compile(rf"(?i)\b{_DAYS_OF_WEEK_PATTERN}\b"),
    # Volume / Issue / Edition identifiers (e.g. "VOL. 18 NO. 145", "VOL. LXVIII NO. 22,415")
    re.compile(r"(?i)\b(?:VOL(?:UME)?\.?|NO\.?|ISSUE|EDITION)\s*[\w\d,\.-]+"),
    # Currency / Price tags (e.g. "Rs. 10", "Rs 10.00", "₹10")
    re.compile(r"(?i)\b(?:RS\.?|INR|₹)\s*\d+(?:\.\d{2})?"),
]

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


def _validate_folio_candidate(cand: str | None) -> str | None:
    """Validate that candidate string is a genuine printed folio and not brand text."""
    if not cand:
        return None
    val = cand.strip().upper()
    if val in DISALLOWED_BRAND_FOLIOS:
        return None
    # Reject single alpha characters that are not digits or valid Roman 'I', 'V', 'X'
    if len(val) == 1 and not val.isdigit() and val not in {"I", "V", "X"}:
        return None
    # If digits, validate reasonable newspaper page range
    if val.isdigit():
        num = int(val)
        if 1 <= num <= 200:
            return str(num)
        return None
    # If section code e.g. B-3 or A-12
    if re.match(r"^[A-Z]\s*[-–]\s*\d{1,2}$", val):
        return re.sub(r"\s+", "", val)
    # If Roman numeral (I through XX)
    if re.match(rf"^{_ROMAN_FOLIO_PATTERN}$", val):
        return val
    return None


def strip_dates_and_metadata(text: str) -> str:
    """Remove date expressions, years, days of week, and volume metadata."""
    cleaned = text
    for pat in _DATE_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


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

    def _extract_folio_from_text(self, text: str) -> str | None:
        """Scan a candidate text string for an explicit or boundary-positioned folio."""
        # Step 1: Check for explicit PAGE 7 / PAGE B-2 format
        pg_match = FOLIO_PAGE_REGEX.search(text)
        if pg_match:
            cand = _validate_folio_candidate(pg_match.group(1))
            if cand is not None:
                return cand

        # Step 2: Strip date strings, days of week, 4-digit years, and issue metadata
        cleaned = strip_dates_and_metadata(text)
        if not cleaned:
            return None

        # Step 3: Check header line with pipes/delimiters on cleaned text
        hl_match = FOLIO_HEADER_LINE_REGEX.search(cleaned)
        if hl_match:
            cand = _validate_folio_candidate(hl_match.group(1))
            if cand is not None:
                return cand

        # Step 4: Check for explicit section folio: "B-4"
        sec_match = SECTION_FOLIO_REGEX.search(cleaned)
        if sec_match:
            cand = _validate_folio_candidate(sec_match.group(1))
            if cand is not None:
                return cand

        # Step 5: Check for standalone corner digit (e.g. single number line "13")
        corner_match = FOLIO_CORNER_DIGIT_REGEX.match(cleaned)
        if corner_match:
            cand = _validate_folio_candidate(corner_match.group(1))
            if cand is not None:
                return cand

        # Step 6: Positional Priority - Trailing edge (e.g. "BENGALURU 13" -> "13")
        trail_match = _TRAILING_FOLIO_REGEX.search(cleaned)
        if trail_match:
            cand = _validate_folio_candidate(trail_match.group(1))
            if cand is not None:
                return cand

        # Step 7: Positional Priority - Leading edge (e.g. "13 BENGALURU" -> "13")
        lead_match = _LEADING_FOLIO_REGEX.search(cleaned)
        if lead_match:
            cand = _validate_folio_candidate(lead_match.group(1))
            if cand is not None:
                return cand

        # Step 8: Isolated number or section code within cleaned string
        isolated_matches = _ISOLATED_FOLIO_REGEX.findall(cleaned)
        if isolated_matches:
            cand = _validate_folio_candidate(str(isolated_matches[-1]))
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
    ) -> str:
        """Extract printed folio using relative Y-axis coordinate zone filtering.

        Calculates relative_y = y0 / effective_page_height to ensure strict DPI sync
        across 300 DPI OCR images and 72 DPI PDF documents.

        Only blocks residing strictly in the top 8% (relative_y < 0.08) or bottom 5%
        (relative_y > 0.95) are inspected. Body text, advertisements, and blocks missing
        bounding boxes are discarded before regex runs.
        """
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
                return str(last_known_folio_num + delta)
            return f"Unnumbered (PDF p.{page_number})"

        # Auto-detect coordinate scale (DPI Sync)
        max_y = max(b[0][3] for b in valid_blocks)
        given_h = max(float(height_px), 1.0)

        if max_y <= 1.05 and given_h > 10.0:
            # Normalized (0.0 .. 1.0) coordinate space
            effective_height = 1.0
        elif given_h >= 2000.0 and max_y <= 1200.0:
            # height_px was passed at 300 DPI raster while blocks are in 72 DPI PDF points
            effective_height = given_h / (300.0 / 72.0)
        else:
            effective_height = given_h

        header_blocks: list[tuple[tuple[float, float, float, float], str]] = []
        footer_blocks: list[tuple[tuple[float, float, float, float], str]] = []

        for (x0, y0, x1, y1), text in valid_blocks:
            rel_y0 = y0 / effective_height
            rel_y1 = y1 / effective_height
            height_span = rel_y1 - rel_y0

            # Reject full-page spanning blocks (e.g. height span > 15% of page)
            if height_span > 0.15:
                continue

            # Strict spatial check: Top 8% header strip (relative_y0 < 0.08)
            if rel_y0 < 0.08 and rel_y1 <= 0.12:
                header_blocks.append(((x0, y0, x1, y1), text))
            # Strict spatial check: Bottom 5% footer strip (relative_y0 > 0.95)
            elif rel_y0 > 0.95 or rel_y1 >= 0.95:
                footer_blocks.append(((x0, y0, x1, y1), text))

        # Sort header and footer blocks left-to-right (x0)
        header_blocks.sort(key=lambda item: item[0][0])
        footer_blocks.sort(key=lambda item: item[0][0])

        # Step 1: Scan isolated header zone
        if header_blocks:
            # First check individual header blocks
            for _, h_text in header_blocks:
                folio = self._extract_folio_from_text(h_text)
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
            folio = self._extract_folio_from_text(combined_header)
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
                folio = self._extract_folio_from_text(f_text)
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
            folio = self._extract_folio_from_text(combined_footer)
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
        # Supplements crossing section boundaries (e.g. Page B-1) do not increment numerically.
        if (
            isinstance(last_known_folio_num, int)
            and last_known_pdf_page is not None
            and page_number > last_known_pdf_page
        ):
            delta = page_number - last_known_pdf_page
            if 1 <= delta <= 5:
                inferred = last_known_folio_num + delta
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
        )

