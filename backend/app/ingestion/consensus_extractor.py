"""Multi-Page Consensus Extraction Engine.

Extracts publication brand and issue date by analyzing headers, folios,
and mastheads across multiple pages of an uploaded newspaper PDF, applying
majority-vote consensus and visual MastheadVerifier fallback to eliminate
misidentifications.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import date
from typing import Any

import pymupdf

from app.core.logging import get_logger
from app.ingestion.masthead_verifier import MastheadVerifier

logger = get_logger(__name__)

_KNOWN_MASTHEADS = [
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
    # 07 July 2026 / 7th July, 2026 / 07-Jul-2026 / 27 AUGUST 2026
    re.compile(rf"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\.\,\-\/]+({_MONTH_PATTERN})[\s\.\,\-\/]+(\d{{4}})\b"),
    # July 07, 2026 / July 7 2026 / August 27, 2026
    re.compile(rf"(?i)\b({_MONTH_PATTERN})[\s\.\,\-\/]+(\d{{1,2}})(?:st|nd|rd|th)?[\s\.\,\-\/]+(\d{{4}})\b"),
    # ISO: 2026-07-07 / 2026/07/07
    re.compile(r"\b(\d{4})[\s\.\,\-\/]+(\d{1,2})[\s\.\,\-\/]+(\d{1,2})\b"),
    # DD/MM/YYYY or MM/DD/YYYY: 07/07/2026
    re.compile(r"\b(\d{1,2})[\s\.\,\-\/]+(\d{1,2})[\s\.\,\-\/]+(\d{4})\b"),
    # Compact DDMMYYYY: 27082026
    re.compile(r"\b(\d{2})(\d{2})(202\d)\b"),
]


def _parse_extracted_date(groups: tuple[str, ...]) -> date | None:
    """Parse regex match groups into a validated date object."""
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

    # Build active candidate brands list, sorting by length descending to prioritize specific brands
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
                v_brand, v_date, v_conf, v_telem = verifier.verify_from_pdf_bytes(
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
