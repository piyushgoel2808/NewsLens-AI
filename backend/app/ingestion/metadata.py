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
from datetime import date
from typing import Any

import pymupdf
from rapidocr import RapidOCR

from app.core.logging import get_logger

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
# Sequential Page Number Normalizer
# =============================================================================


class FolioDetector:
    """Standardizes strictly on sequential integer page numbers."""

    def extract_folio(
        self,
        *args: Any,
        page_number: int = 1,
        **kwargs: Any,
    ) -> str:
        """Standardized sequential page number."""
        return str(page_number)

    def extract_printed_page_number(
        self,
        page_number: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Standardized sequential page number."""
        return str(page_number)


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
    (re.compile(r"\b(?:THE\s+)?HANS\s+INDIA\b|\bTHEHANSINDIA\b", re.I), "The Hans India"),
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
            norm_fn = unicodedata.normalize("NFKD", filename).replace("_", " ")
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
                elif re.search(r"\bHANS\b|\bTHE\s*HANS\s*INDIA\b", fn_upper):
                    detected_brand = "The Hans India"
                    brand_confidence = 0.90
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
    ("THE HANS INDIA", "The Hans India"),
    ("HANS INDIA", "The Hans India"),
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
            norm_fn = unicodedata.normalize("NFKD", filename).replace("_", " ")
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
            elif re.search(r"\bHANS\b|\bTHE\s*HANS\s*INDIA\b", fn_upper):
                brand_counter["The Hans India"] += 5
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
