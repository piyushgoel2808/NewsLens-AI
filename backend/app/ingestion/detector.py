"""PDF text layer detection and extraction service for NewsLens-AI.

Classifies each PDF page as:
- 'digital': Native selectable text layer present.
- 'scanned': Pure image bitmap without selectable text (requires OCR in Phase 2).
- 'hybrid': Native text with major embedded raster images/figures.

Extracts structured text blocks with spatial bounding boxes, font metadata,
and reading order candidates for downstream article assembly.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

import pymupdf

from app.core.logging import get_logger

logger = get_logger(__name__)

# Minimum characters on a page to qualify as having a usable digital text layer
MIN_DIGITAL_CHARS_THRESHOLD = 80

COMMON_ENGLISH_WORDS = frozenset(
    {
        "the",
        "and",
        "in",
        "to",
        "of",
        "for",
        "is",
        "on",
        "that",
        "by",
        "with",
        "as",
        "said",
        "from",
        "at",
        "it",
        "be",
        "an",
        "have",
        "has",
        "was",
        "were",
        "not",
        "market",
        "company",
        "crore",
        "bank",
        "india",
        "delhi",
        "year",
        "per",
        "cent",
        "power",
        "business",
        "growth",
        "share",
        "new",
        "government",
        "policy",
        "report",
    }
)


def is_text_gibberish(text: str, threshold: float = 0.10) -> bool:
    """Check if extracted text is corrupt/gibberish due to missing/broken ToUnicode font CMap.

    Heuristics:
    1. Positive Check: If text contains sufficient recognizable common English words,
       it is valid digital text (not gibberish).
    2. Count replacement characters (\\ufffd, \\ufeff) and unprintable control / private-use codes.
    3. Check for single character dominance (e.g. font mapping bug where all glyphs become 'b').
    4. Check for repeated character runs relative to document size.
    5. Check word validity ratio.
    """
    if not text or not text.strip():
        return False
    cleaned = re.sub(r"[\s\n\r\t]+", "", text)
    if len(cleaned) < 50:
        return False

    words_list = re.findall(r"\b[a-z]{2,}\b", text.lower())
    common_matches = len(set(words_list).intersection(COMMON_ENGLISH_WORDS))

    # 1. Count replacement characters and unprintable control / private-use codes
    bad_chars = sum(
        1
        for c in cleaned
        if c in ("\ufffd", "\ufeff") or unicodedata.category(c) in ("Cc", "Cs", "Co")
    )

    if (bad_chars / len(cleaned)) >= threshold:
        return True

    # 2. Check for single character dominance (e.g. font mapping bug where all glyphs become 'b')
    counts = Counter(cleaned.lower())
    most_common_char, most_common_count = counts.most_common(1)[0]
    if (
        most_common_char not in ("-", "_", ".", "=", "*", "/", " ")
        and (most_common_count / len(cleaned)) >= 0.25
        and common_matches < 5
    ):
        return True

    # 3. If page contains high number of common dictionary words, it is valid digital text
    if common_matches >= 8 and len(words_list) >= 20:
        return False

    # 4. Check for repeated character runs relative to document size (e.g. 'bbbbbbbb')
    repeated_matches = re.findall(r"([a-z])\1{4,}", text.lower())
    if len(repeated_matches) >= 3 and common_matches < 5:
        return True

    # 5. Check word validity ratio
    words = [w for w in text.split() if w.strip()]
    if words and len(words) >= 10:
        gibberish_words = sum(
            1 for w in words if len(w) > 35 or (len(set(w.lower())) == 1 and len(w) >= 5)
        )
        if (gibberish_words / len(words)) >= 0.25 and common_matches < 5:
            return True

    return False


class PageType(StrEnum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"
    ADVERTISEMENT = "advertisement"


AD_KEYWORDS_REGEX = re.compile(
    r"(?i)\b(?:advertisement|advertorial|special\s+promotional\s+feature|sponsored\s+feature|"
    r"promotional\s+feature|t&c\s+apply|terms\s+(?:and|&)\s+conditions\s+apply|"
    r"call\s+now|visit\s+us\s+at|toll\s+free|showroom|mrp\s*rs|flat\s+\d+%\s+off|"
    r"exclusive\s+offer|limited\s+period\s+offer|book\s+now\s+at|for\s+bookings\s+call|"
    # Financial IPO Notices
    r"initial\s+public\s+offering|red\s+herring\s+prospectus|draft\s+red\s+herring\s+prospectus|"
    r"book\s+running\s+lead\s+manager|registrar\s+to\s+the\s+issue|bid/issue\s+opens\s+on|"
    r"price\s+band|floor\s+price|promoters\s+of\s+our\s+company|equity\s+shares\s+of\s+face\s+value|"
    # Statutory & Legal Notices
    r"public\s+notice|statutory\s+notice|notice\s+is\s+hereby\s+given|in\s+the\s+matter\s+of|"
    r"national\s+company\s+law\s+tribunal|\bnclt\b|insolvency\s+and\s+bankruptcy\s+code|"
    r"auction\s+sale\s+notice|tender\s+notice|e-auction|corrigendum|possession\s+notice)\b"
)


@dataclass
class TextSpan:
    """A granular span of text sharing the same font style and size."""

    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in points
    font_name: str
    font_size: float
    flags: int
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class DigitalTextBlock:
    """A coherent block of text with spatial bounds and font metrics."""

    block_id: int
    text: str
    bbox: tuple[float, float, float, float]
    lines: list[str] = field(default_factory=list)
    spans: list[TextSpan] = field(default_factory=list)
    mean_font_size: float = 10.0
    is_heading_candidate: bool = False


@dataclass
class PageAnalysisResult:
    """Complete analysis and classification for a single PDF page."""

    page_number: int
    page_type: PageType
    requires_ocr: bool
    character_count: int
    word_count: int
    full_text: str
    blocks: list[DigitalTextBlock] = field(default_factory=list)
    dominant_font_size: float = 10.0
    image_count: int = 0
    is_advertisement: bool = False
    page_width: float = 0.0
    page_height: float = 0.0


class PDFPageDetector:
    """Analyzes PDF pages for digital text vs scanned image classification."""

    def analyze_page(
        self,
        doc: pymupdf.Document,
        page_index: int,
    ) -> PageAnalysisResult:
        """Analyze and extract structured text from a 0-indexed page in an open PyMuPDF document."""
        page = doc.load_page(page_index)
        page_num = page_index + 1
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

        # Extract structured text dictionary
        text_page = page.get_text("dict")
        raw_text = page.get_text("text").strip()
        images = page.get_images()
        image_count = len(images)

        char_count = len(raw_text.replace(" ", "").replace("\n", ""))
        words = raw_text.split()
        word_count = len(words)

        blocks: list[DigitalTextBlock] = []
        font_sizes: list[float] = []

        block_counter = 0
        for b in text_page.get("blocks", []):
            if b.get("type") == 0:  # Text block
                block_lines: list[str] = []
                block_spans: list[TextSpan] = []
                block_font_sizes: list[float] = []

                for line in b.get("lines", []):
                    line_text_parts: list[str] = []
                    for span in line.get("spans", []):
                        stext = span.get("text", "")
                        if not stext.strip():
                            continue
                        fsize = float(span.get("size", 10.0))
                        fname = str(span.get("font", "unknown"))
                        flags = int(span.get("flags", 0))
                        is_bold = bool(flags & 2 or "bold" in fname.lower())
                        is_italic = bool(
                            flags & 1 or "italic" in fname.lower() or "oblique" in fname.lower()
                        )

                        span_bbox = (
                            float(span["bbox"][0]),
                            float(span["bbox"][1]),
                            float(span["bbox"][2]),
                            float(span["bbox"][3]),
                        )
                        block_spans.append(
                            TextSpan(
                                text=stext,
                                bbox=span_bbox,
                                font_name=fname,
                                font_size=fsize,
                                flags=flags,
                                is_bold=is_bold,
                                is_italic=is_italic,
                            )
                        )
                        line_text_parts.append(stext)
                        block_font_sizes.append(fsize)
                        font_sizes.append(fsize)

                    if line_text_parts:
                        block_lines.append(" ".join(line_text_parts))

                block_full_text = "\n".join(block_lines)
                if block_full_text.strip():
                    mean_fsize = (
                        sum(block_font_sizes) / len(block_font_sizes) if block_font_sizes else 10.0
                    )
                    block_bbox = (
                        float(b["bbox"][0]),
                        float(b["bbox"][1]),
                        float(b["bbox"][2]),
                        float(b["bbox"][3]),
                    )
                    blocks.append(
                        DigitalTextBlock(
                            block_id=block_counter,
                            text=block_full_text,
                            bbox=block_bbox,
                            lines=block_lines,
                            spans=block_spans,
                            mean_font_size=mean_fsize,
                        )
                    )
                    block_counter += 1

        # Calculate dominant font size (body text font size)
        dominant_font_size = 10.0
        if font_sizes:
            # Rounded mode of font sizes
            rounded_sizes = [round(s, 1) for s in font_sizes]
            from collections import Counter

            dominant_font_size = Counter(rounded_sizes).most_common(1)[0][0]

        # Boilerplate tokens that must never stand alone as article headings
        boilerplate_stopwords = {
            "limited", "ltd", "corp", "corporation", "pvt", "private", "equity", "issue",
            "issue,", "shares", "company", "notice", "promoters", "price", "band", "page",
            "continued", "from", "and", "or", "of", "in", "on", "at", "to", "for", "with",
        }

        # Flag heading candidates (font size > 1.35 * dominant body size or bold font)
        for blk in blocks:
            clean_blk = blk.text.strip()
            words_blk = clean_blk.split()
            # Reject single-word corporate boilerplate or very short fragments
            if len(words_blk) == 1 and words_blk[0].lower().rstrip(",.:;") in boilerplate_stopwords:
                blk.is_heading_candidate = False
                continue
            if len(words_blk) < 2 and len(clean_blk) < 12:
                blk.is_heading_candidate = False
                continue

            is_large = blk.mean_font_size >= dominant_font_size * 1.35
            is_bold_prominent = any(
                s.is_bold and s.font_size > dominant_font_size for s in blk.spans
            )
            if is_large or is_bold_prominent:
                blk.is_heading_candidate = True

        # Advertisement & Legal Notice detection heuristics
        upper_text = raw_text.strip().upper()
        ad_matches = len(AD_KEYWORDS_REGEX.findall(raw_text))
        is_ad = (
            upper_text.startswith("ADVERTISEMENT")
            or upper_text.startswith("ADVERTORIAL")
            or upper_text.startswith("SPECIAL PROMOTIONAL FEATURE")
            or upper_text.startswith("SPONSORED FEATURE")
            or upper_text.startswith("PUBLIC NOTICE")
            or upper_text.startswith("INITIAL PUBLIC OFFERING")
            or (ad_matches >= 2)
            or (ad_matches >= 1 and (word_count < 250 or image_count >= 1))
        )

        # Classification heuristics
        if is_ad:
            page_type = PageType.ADVERTISEMENT
            requires_ocr = False
        elif char_count < MIN_DIGITAL_CHARS_THRESHOLD:
            page_type = PageType.SCANNED
            requires_ocr = True
        elif is_text_gibberish(raw_text):
            page_type = PageType.SCANNED
            requires_ocr = True
            blocks = []
            logger.warning(
                "Page text flagged as corrupted font gibberish; routing to OCR fallback",
                extra={"page_number": page_num, "char_count": char_count},
            )
        elif image_count > 0 and char_count >= MIN_DIGITAL_CHARS_THRESHOLD:
            page_type = PageType.HYBRID
            requires_ocr = False
        else:
            page_type = PageType.DIGITAL
            requires_ocr = False

        logger.info(
            "Page structure analyzed",
            extra={
                "page_number": page_num,
                "type": page_type.value,
                "is_advertisement": is_ad,
                "char_count": char_count,
                "blocks": len(blocks),
                "dominant_font_size": dominant_font_size,
            },
        )

        return PageAnalysisResult(
            page_number=page_num,
            page_type=page_type,
            requires_ocr=requires_ocr,
            character_count=char_count,
            word_count=word_count,
            full_text=raw_text,
            blocks=blocks,
            dominant_font_size=dominant_font_size,
            image_count=image_count,
            is_advertisement=is_ad,
            page_width=page_width,
            page_height=page_height,
        )

    def analyze_document_bytes(self, pdf_bytes: bytes) -> list[PageAnalysisResult]:
        """Analyze all pages of a PDF from raw byte buffer."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        results: list[PageAnalysisResult] = []
        for i in range(len(doc)):
            results.append(self.analyze_page(doc, i))
        doc.close()
        return results
