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


UUID_REGEX = re.compile(
    r"\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b"
)

SOCIAL_PROMO_REGEX = re.compile(
    r"(?i)(?:Join\s+FREE\s+(?:Whatsapp|Telegram)\s+Channel.*|"
    r"https?://(?:t\.me|chat\.whatsapp\.com|wa\.me|bit\.ly|tinyurl\.com)/\S*|"
    r"\bt\.me/\S*|\bwhatsapp\s+channel\b|\btelegram\s+group\b)",
    re.IGNORECASE,
)

PRINTER_MARKS_REGEX = re.compile(
    r"(?i)(?:A\s*ND-NDE\s*C\s*M\s*Y\s*K|\bC\s*M\s*Y\s*K\b|\bcyan\s+magenta\s+yellow\s+black\b|"
    r"epaper\s*[\.\-]\s*livemint|pdf\s*version\s*generated|epaper\s*edition\s*generated)",
    re.IGNORECASE,
)


def is_noise_or_promo_text(text: str) -> bool:
    """Detect if text is a UUID hash, social media/WhatsApp promo, or printer mark."""
    t = text.strip()
    if not t:
        return True
    if UUID_REGEX.search(t):
        return True
    if SOCIAL_PROMO_REGEX.search(t):
        return True
    return bool(PRINTER_MARKS_REGEX.search(t))


def sanitize_block_text(text: str) -> str:
    """Sanitize block text by removing UUIDs, promo links, and printer registration marks."""
    if not text:
        return ""
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paragraphs: list[str] = []
    for para in paragraphs:
        lines = para.split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            l_str = line.strip()
            if not l_str or is_noise_or_promo_text(l_str):
                continue
            l_clean = UUID_REGEX.sub("", l_str)
            l_clean = SOCIAL_PROMO_REGEX.sub("", l_clean)
            l_clean = PRINTER_MARKS_REGEX.sub("", l_clean).strip()
            if l_clean:
                cleaned_lines.append(l_clean)
        if cleaned_lines:
            cleaned_paragraphs.append("\n".join(cleaned_lines))
    return "\n\n".join(cleaned_paragraphs)


def is_title_case_or_uppercase(text: str) -> bool:
    """Check if text is in Title Case or UPPERCASE."""
    clean = re.sub(r"[^\w\s]", "", text).strip()
    words = clean.split()
    if not words:
        return False
    if clean.isupper() and len(words) >= 2:
        return True

    stopwords = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
        "by", "with", "as", "is", "its", "from", "into", "over", "after",
    }
    content_words = [w for w in words if w.lower() not in stopwords]
    if not content_words:
        content_words = words

    cap_count = sum(1 for w in content_words if w and w[0].isupper())
    return (cap_count / len(content_words)) >= 0.55


def reattach_drop_caps(blocks: list[DigitalTextBlock]) -> list[DigitalTextBlock]:
    """Reattach single-letter uppercase drop caps to subsequent body paragraphs."""
    if len(blocks) <= 1:
        return blocks

    merged_blocks: list[DigitalTextBlock] = []
    skip_indices: set[int] = set()

    for idx, blk in enumerate(blocks):
        if idx in skip_indices:
            continue

        raw_t = blk.text.strip()
        is_single_letter_drop_cap = (
            len(raw_t) == 1
            and raw_t.isalpha()
            and raw_t.isupper()
        )

        if is_single_letter_drop_cap and idx + 1 < len(blocks):
            target_next: DigitalTextBlock | None = None
            target_idx: int | None = None

            for next_idx in range(idx + 1, min(idx + 5, len(blocks))):
                if next_idx in skip_indices:
                    continue
                cand = blocks[next_idx]
                cand_t = cand.text.strip()
                if not cand_t:
                    continue

                is_near_y = (blk.bbox[1] - 8.0 <= cand.bbox[1] <= blk.bbox[3] + 25.0)
                h_overlap = min(blk.bbox[2], cand.bbox[2]) - max(blk.bbox[0], cand.bbox[0])
                is_col_aligned = h_overlap > -40.0 and cand.bbox[0] >= blk.bbox[0] - 10.0

                if is_near_y and is_col_aligned:
                    target_next = cand
                    target_idx = next_idx
                    break

            if target_next and target_idx is not None:
                drop_char = raw_t
                next_text = target_next.text
                if next_text.startswith(" ") or next_text.startswith("\n"):
                    new_text = drop_char + next_text.lstrip()
                elif next_text and next_text[0].islower():
                    new_text = drop_char + next_text
                else:
                    new_text = f"{drop_char} {next_text}".strip()

                new_bbox = (
                    min(blk.bbox[0], target_next.bbox[0]),
                    min(blk.bbox[1], target_next.bbox[1]),
                    max(blk.bbox[2], target_next.bbox[2]),
                    max(blk.bbox[3], target_next.bbox[3]),
                )
                new_spans = blk.spans + target_next.spans
                new_lines = (
                    [drop_char + target_next.lines[0]] + target_next.lines[1:]
                    if target_next.lines
                    else [new_text]
                )

                target_next.text = new_text
                target_next.bbox = new_bbox
                target_next.spans = new_spans
                target_next.lines = new_lines
                target_next.mean_font_size = (
                    (blk.mean_font_size + target_next.mean_font_size * 4) / 5.0
                )
                skip_indices.add(idx)
                continue

        merged_blocks.append(blk)

    return merged_blocks


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

    # 1. Explicit replacement characters (\ufffd, \ufeff) signify decoded font corruption
    replacement_chars = sum(1 for c in cleaned if c in ("\ufffd", "\ufeff"))
    if (replacement_chars / len(cleaned)) >= 0.05:
        return True

    words_list = re.findall(r"\b[a-z]{2,}\b", text.lower())
    common_matches = len(set(words_list).intersection(COMMON_ENGLISH_WORDS))

    # 2. Positive Check: If page contains dictionary words, it is valid digital text
    if common_matches >= 6 and len(words_list) >= 15:
        return False

    # 3. Count unprintable control codes
    bad_chars = sum(
        1
        for c in cleaned
        if unicodedata.category(c) in ("Cc", "Cs", "Co")
    )

    if (bad_chars / len(cleaned)) >= threshold and common_matches < 4:
        return True

    # 3. Check for single character dominance (e.g. font mapping bug where all glyphs become 'b')
    counts = Counter(cleaned.lower())
    most_common_char, most_common_count = counts.most_common(1)[0]
    if (
        most_common_char not in ("-", "_", ".", "=", "*", "/", " ")
        and (most_common_count / len(cleaned)) >= 0.25
        and common_matches < 4
    ):
        return True

    # 4. Check for repeated character runs relative to document size (e.g. 'bbbbbbbb')
    repeated_matches = re.findall(r"([a-z])\1{4,}", text.lower())
    if len(repeated_matches) >= 3 and common_matches < 4:
        return True

    # 5. Check word validity ratio
    words = [w for w in text.split() if w.strip()]
    if words and len(words) >= 10:
        gibberish_words = sum(
            1 for w in words if len(w) > 35 or (len(set(w.lower())) == 1 and len(w) >= 5)
        )
        if (gibberish_words / len(words)) >= 0.25 and common_matches < 4:
            return True

    return False


class PageType(StrEnum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"
    ADVERTISEMENT = "advertisement"


GUTTER_MIN_WIDTH = 0.012  # ~33px on 2800px broadsheet
VERTICAL_PARA_GAP_MAX = 0.015  # ~65px on 4399px broadsheet
MASTHEAD_TOP_ZONE = 0.06  # top 6% strip
FOOTER_BOTTOM_ZONE = 0.94  # bottom 6% strip

# =============================================================================
# Multi-Category Universal Advertisement & Statutory Notice Detection Engine
# =============================================================================

# 1. Modular Functional Regex Patterns across commercial and statutory verticals
EXPLICIT_AD_HEADER_REGEX = re.compile(
    r"(?i)\b(?:advertisement|advertorial|sponsored\s*feature|promotional\s*feature|"
    r"brand\s*connect|special\s*marketing\s*feature|consumer\s*connect|media\s*marketing|"
    r"special\s*promotional\s*feature)\b"
)

CTA_REGEX = re.compile(
    r"(?i)\b(?:book\s*now|order\s*now|pre-order|pre\s*order|pre-book|pre\s*book|buy\s*now|"
    r"call\s*toll\s*free|visit\s*us\s*at|apply\s*now|register\s*now|scan\s*to\s*know\s*more|"
    r"scan\s*(?:the\s*)?qr|download\s*(?:the\s*)?app|toll\s*free\s*(?:no|number)?)\b"
)

PRICING_FINANCE_REGEX = re.compile(
    r"(?i)\b(?:starting\s*at|starts\s*at|special\s*offer|limited\s*period\s*offer|"
    r"inaugural\s*offer|flat\s*\d+%\s*off|save\s*up\s*to|price\s*inclusive\s*of|"
    r"down\s*payment|no\s*cost\s*emi|easy\s*emi|zero\s*processing\s*fee|exchange\s*bonus|"
    r"exchange\s*value|cashback\s*up\s*to|t&c\s*apply|terms\s*(?:and|&)\s*conditions\s*apply)\b"
)

REAL_ESTATE_AUTO_REGEX = re.compile(
    r"(?i)\b(?:ready\s*to\s*move|possession\s*soon|\d+\s*bhk|rera\s*reg|rera\s*registration|"
    r"ex-showroom\s*price|test\s*drive\s*today|authorized\s*dealership|super\s*built-up\s*area)\b"
)

STATUTORY_TENDERS_REGEX = re.compile(
    r"(?i)\b(?:public\s*notice|statutory\s*notice|notice\s*is\s*hereby\s*given|"
    r"before\s*the\s*hon'ble|national\s*company\s*law\s*tribunal|\bnclt\b|"
    r"auction\s*sale\s*notice|possession\s*notice|tender\s*notice|notice\s*inviting\s*tender|"
    r"corrigendum|addendum)\b"
)

IPO_FINANCIAL_REGEX = re.compile(
    r"(?i)\b(?:initial\s*public\s*offering|price\s*band|equity\s*shares\s*of\s*face\s*value|"
    r"bid/issue\s*opens|bid/issue\s*period|retail\s*individual\s*bidders|"
    r"qualified\s*institutional\s*buyers|book\s*running\s*lead\s*managers?|"
    r"registrars?\s*to\s*the\s*issue|red\s*herring\s*prospectus|\basba\b)\b"
)

# 2. Digital Discovery & Contact Footprint Scoring
CONTACT_FOOTPRINT_REGEX = re.compile(
    r"(?i)(?:\b(?:1800|1860)[-\s]?\d{3}[-\s]?\d{3,4}\b|"
    r"https?://\S+|www\.[a-z0-9\-]+\.[a-z]{2,}|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})"
)

# 3. Editorial Contrast & Safety Safeguards
EDITORIAL_MARKERS_REGEX = re.compile(
    r"(?i)\b(?:bureau|correspondent|special\s*correspondent|express\s*news\s*service|"
    r"edited\s*by|opinion|editorial|columns?|continued\s*on\s*page\s*\d+|from\s*page\s*\d+)\b"
)

# Backward-compatibility alias
AD_KEYWORDS_REGEX = re.compile(
    r"(?i)(?:"
    + "|".join([
        EXPLICIT_AD_HEADER_REGEX.pattern[6:-2],
        CTA_REGEX.pattern[6:-2],
        PRICING_FINANCE_REGEX.pattern[6:-2],
        REAL_ESTATE_AUTO_REGEX.pattern[6:-2],
        STATUTORY_TENDERS_REGEX.pattern[6:-2],
        IPO_FINANCIAL_REGEX.pattern[6:-2],
    ])
    + ")"
)


TC_LEGAL_PHRASES_PATTERNS = [
    r"t&c\s*apply",
    r"terms\s*(?:and|&)\s*conditions\s*apply",
    r"inclusive\s*of\s*(?:all\s*)?taxes",
    r"sole\s*discretion",
    r"no\s*cost\s*emi",
    r"easy\s*emi",
    r"cashback",
    r"damage\s*protection",
    r"buyback",
    r"images?\s*simulated",
    r"screen\s*simulated",
    r"optional\s*accessories",
    r"emi\s*options?",
    r"exchange\s*bonus",
    r"down\s*payment",
    r"zero\s*processing\s*fee",
    r"annual\s*percentage\s*rate",
    r"prices?\s*subject\s*to\s*change",
]
TC_LEGAL_PHRASES_REGEX = re.compile(
    r"(?i)\b(?:" + "|".join(TC_LEGAL_PHRASES_PATTERNS) + r")\b"
)


def count_distinct_tc_phrases(text: str) -> int:
    """Count how many distinct T&C / legal / commercial phrases appear in the text."""
    norm = text.lower()
    matches = 0
    for pattern in TC_LEGAL_PHRASES_PATTERNS:
        if re.search(r"(?i)\b" + pattern + r"\b", norm):
            matches += 1
    return matches


def calculate_commercial_lexicon_density(text: str) -> float:
    """Calculate the ratio of commercial/legal tokens to total words."""
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    all_commercial_matches = len(AD_KEYWORDS_REGEX.findall(text)) + len(
        TC_LEGAL_PHRASES_REGEX.findall(text)
    )
    return all_commercial_matches / len(words)


def calculate_ad_score(
    text: str,
    page_number: int | None = None,
    total_pages: int | None = None,
) -> tuple[float, int, bool, int, float]:
    """Calculate multi-category ad score, editorial hits, header presence, TC count, and density."""
    if not text or not text.strip():
        return 0.0, 0, False, 0, 0.0

    norm_text = re.sub(r"\s+", " ", text).strip()

    # 1. Immediate match on explicit ad header in top masthead area (first 400 chars)
    first_chunk = norm_text[:400]
    has_explicit_header = bool(EXPLICIT_AD_HEADER_REGEX.search(first_chunk))
    if not has_explicit_header:
        has_explicit_header = bool(
            re.search(
                r"(?i)^[^\w]*(?:advertisement|advertorial|special\s*promotional\s*feature|"
                r"sponsored\s*feature|public\s*notice|statutory\s*notice)\b",
                norm_text,
            )
        )

    distinct_tc_count = count_distinct_tc_phrases(norm_text)
    commercial_density = calculate_commercial_lexicon_density(norm_text)

    ad_score = 0.0

    # 2. Distinct functional commercial category hits (+2.0 per category)
    if CTA_REGEX.search(norm_text):
        ad_score += 2.0
    if PRICING_FINANCE_REGEX.search(norm_text):
        ad_score += 2.0
    if REAL_ESTATE_AUTO_REGEX.search(norm_text):
        ad_score += 2.0
    if STATUTORY_TENDERS_REGEX.search(norm_text):
        ad_score += 2.0
    if IPO_FINANCIAL_REGEX.search(norm_text):
        ad_score += 2.0

    # 3. Contact / URL footprint (+1.5)
    if CONTACT_FOOTPRINT_REGEX.search(norm_text):
        ad_score += 1.5

    # 4. Positional weighting: Page 1 and final back page (+1.0)
    if (
        page_number is not None
        and total_pages is not None
        and (page_number == 1 or page_number == total_pages)
    ):
        ad_score += 1.0

    # 5. Editorial markers count
    editorial_hits = len(EDITORIAL_MARKERS_REGEX.findall(norm_text))

    return ad_score, editorial_hits, has_explicit_header, distinct_tc_count, commercial_density


def check_is_advertisement_text(text: str, word_count: int | None = None) -> bool:
    """Evaluate whether extracted digital or OCR text represents an advertisement/notice."""
    if not text or not text.strip():
        return False

    norm_text = re.sub(r"\s+", " ", text).strip()
    w_count = word_count if word_count is not None else len(norm_text.split())

    ad_score, editorial_hits, has_explicit_header, distinct_tc_count, commercial_density = (
        calculate_ad_score(norm_text)
    )

    # 1. Immediate match on explicit ad header
    if has_explicit_header:
        return True

    # 2. The T&C Trapdoor Rule: >= 3 distinct legal/commercial phrases triggers True
    if distinct_tc_count >= 3:
        return True

    # 3. High commercial density trigger
    if commercial_density >= 0.04 and ad_score >= 3.5:
        return True

    # 4. Editorial Contrast & Safety Guardrail (>= 3 editorial markers & >= 350 words)
    if editorial_hits >= 3 and w_count >= 350:
        return ad_score >= 6.0

    # 5. Short marketing / retail pages (< 250 words)
    if w_count < 250 and ad_score >= 3.0:
        return True

    # 6. Standard commercial confidence threshold
    return ad_score >= 4.5


def is_page_advertisement(
    page_blocks_text: str,
    page_number: int,
    total_pages: int,
    word_count: int | None = None,
    image_area_ratio: float = 0.0,
) -> bool:
    """Multi-signal evaluation for full-page advertisements, jacket wraps, and marketing spreads."""
    if not page_blocks_text or not page_blocks_text.strip():
        return False

    norm_text = re.sub(r"\s+", " ", page_blocks_text).strip()
    w_count = word_count if word_count is not None else len(norm_text.split())

    ad_score, editorial_hits, has_explicit_header, distinct_tc_count, commercial_density = (
        calculate_ad_score(norm_text, page_number=page_number, total_pages=total_pages)
    )

    # 1. Immediate match on explicit ad header in top masthead area
    if has_explicit_header:
        return True

    # 2. The T&C Trapdoor Rule: >= 3 distinct legal/commercial phrases triggers True
    # regardless of word count (e.g. dense retail wraps with full specs & T&Cs)
    if distinct_tc_count >= 3:
        return True

    # 3. High commercial density trigger
    if commercial_density >= 0.04 and ad_score >= 3.5:
        return True

    # 4. Editorial Contrast & Safety Guardrail (>= 3 editorial markers & >= 350 words)
    if editorial_hits >= 3 and w_count >= 350:
        return ad_score >= 6.0

    # 5. Short marketing / retail pages (< 250 words)
    if w_count < 250 and ad_score >= 3.0:
        return True

    # 6. Standard commercial confidence threshold
    return ad_score >= 4.5


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
    image_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


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
        try:
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
        except Exception:
            page_width = 595.0
            page_height = 842.0

        # Extract structured text dictionary
        text_page = page.get_text("dict")
        raw_text = page.get_text("text").strip()
        images = page.get_images()
        image_count = len(images)

        # Harvest exact image bounding boxes (photos, figures, infographics)
        # Scaled to 300 DPI raster pixel coordinates to match rendered page images
        image_boxes: list[tuple[float, float, float, float]] = []

        # 1. Harvest from get_image_info (fastest & most accurate in PyMuPDF)
        try:
            img_infos = page.get_image_info(xrefs=True)
            for info in img_infos:
                bbox = info.get("bbox")
                if bbox and len(bbox) == 4:
                    rx0 = float(bbox[0]) * (300.0 / 72.0)
                    ry0 = float(bbox[1]) * (300.0 / 72.0)
                    rx1 = float(bbox[2]) * (300.0 / 72.0)
                    ry1 = float(bbox[3]) * (300.0 / 72.0)
                    if (rx1 - rx0 >= 40.0) and (ry1 - ry0 >= 40.0):
                        image_boxes.append((rx0, ry0, rx1, ry1))
        except Exception:
            pass

        # 2. Harvest from page.get_images() and get_image_rects()
        for img_info in images:
            xref = img_info[0] if isinstance(img_info, (list, tuple)) and len(img_info) > 0 else None
            if xref:
                try:
                    rects = page.get_image_rects(xref)
                    for r in rects:
                        rx0 = float(r.x0) * (300.0 / 72.0)
                        ry0 = float(r.y0) * (300.0 / 72.0)
                        rx1 = float(r.x1) * (300.0 / 72.0)
                        ry1 = float(r.y1) * (300.0 / 72.0)
                        if (rx1 - rx0 >= 40.0) and (ry1 - ry0 >= 40.0) and not any(
                            abs(rx0 - ex[0]) < 10 and abs(ry0 - ex[1]) < 10
                            for ex in image_boxes
                        ):
                            image_boxes.append((rx0, ry0, rx1, ry1))
                except Exception:
                    pass

        # 3. Also inspect image blocks directly from text_page dict
        for b in text_page.get("blocks", []):
            if b.get("type") == 1:  # Image block in PyMuPDF
                bbox = b.get("bbox")
                if bbox and len(bbox) == 4:
                    bx0 = float(bbox[0]) * (300.0 / 72.0)
                    by0 = float(bbox[1]) * (300.0 / 72.0)
                    bx1 = float(bbox[2]) * (300.0 / 72.0)
                    by1 = float(bbox[3]) * (300.0 / 72.0)
                    if (bx1 - bx0 >= 40.0) and (by1 - by0 >= 40.0) and not any(
                        abs(bx0 - ex[0]) < 10 and abs(by0 - ex[1]) < 10
                        for ex in image_boxes
                    ):
                        image_boxes.append((bx0, by0, bx1, by1))

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
                        line_str = " ".join(line_text_parts)
                        if not is_noise_or_promo_text(line_str):
                            block_lines.append(line_str)

                block_full_text = "\n".join(block_lines)
                sanitized_text = sanitize_block_text(block_full_text)
                if sanitized_text.strip():
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
                            text=sanitized_text,
                            bbox=block_bbox,
                            lines=block_lines,
                            spans=block_spans,
                            mean_font_size=mean_fsize,
                        )
                    )
                    block_counter += 1

        # Reattach single-letter uppercase drop caps to subsequent body paragraphs
        blocks = reattach_drop_caps(blocks)

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

        # Flag heading candidates using Font-Heuristic:
        # Font size >= 1.25 * dominant_font_size AND Title Case or UPPERCASE
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
            if is_noise_or_promo_text(clean_blk):
                blk.is_heading_candidate = False
                continue
            # Multi-sentence paragraphs ending in period/semicolon are never headlines
            if len(words_blk) > 12 and clean_blk.rstrip().endswith((".", ";")):
                blk.is_heading_candidate = False
                continue

            is_large = blk.mean_font_size >= dominant_font_size * 1.25
            is_cased = is_title_case_or_uppercase(clean_blk)
            if is_large and is_cased and len(words_blk) >= 2:
                blk.is_heading_candidate = True
            else:
                blk.is_heading_candidate = False

        # Multi-Category Universal Advertisement & Statutory Notice detection
        is_ad = is_page_advertisement(
            page_blocks_text=raw_text,
            page_number=page_num,
            total_pages=len(doc),
            word_count=word_count,
            image_area_ratio=float(image_count > 0),
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
                "image_boxes": len(image_boxes),
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
            image_boxes=image_boxes,
        )

    def analyze_document_bytes(self, pdf_bytes: bytes) -> list[PageAnalysisResult]:
        """Analyze all pages of a PDF from raw byte buffer."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        results: list[PageAnalysisResult] = []
        for i in range(len(doc)):
            results.append(self.analyze_page(doc, i))
        doc.close()
        return results
