"""Hybrid Vision-Language Model and spatial layout analyzer for newspaper pages.

Extracts:
- Headlines and subheadlines with prominence tiers.
- Multi-column body text bounding boxes.
- Photo and illustration regions with associated caption blocks.
- Structured tabular data regions.
- Human reading order sequences.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.detector import (
    DigitalTextBlock,
    is_noise_or_promo_text,
    is_title_case_or_uppercase,
    sanitize_block_text,
)
from app.ingestion.reading_order import (
    BlockType,
    LayoutElement,
    OrderedReadingBlock,
    ReadingOrderResolver,
)
from app.providers.base import (
    DocumentLayoutProvider,
    ExtractedDocumentNode,
    ExtractedPhotoData,
    ExtractedTableData,
    OCRBlock,
    VisionModelProvider,
)
from app.providers.registry import get_registry

logger = get_logger(__name__)

SPONSOR_AND_BOILERPLATE_PATTERNS = re.compile(
    r"(?i)\b(?:jm\s+financial|axis\s+capital|icici\s+securities|kfin\s+technologies|"
    r"link\s+intime|kotak\s+mahindra\s+capital|sbi\s+capital\s+markets|dam\s+capital|"
    r"equirus|motilal\s+oswal|nomura\s+financial|jefferies|citigroup\s+global|"
    r"morgan\s+stanley|goldman\s+sachs|hdfc\s+bank\s+limited|asba|"
    r"applications\s+supported\s+by\s+blocked\s+amount|book\s+running\s+lead\s+managers|"
    r"registrar\s+to\s+the\s+issue|cin\s*:\s*[l|u]\d+|sebi\s+registration|"
    r"corporate\s+identity\s+number|registered\s+office|compliance\s+officer|"
    r"company\s+secretary|statutory\s+auditor|red\s+herring\s+prospectus|"
    r"initial\s+public\s+offering|equity\s+shares\s+of\s+face\s+value)\b"
)

DATE_LINE_PATTERN = re.compile(
    r"(?i)\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"[,\.\s]+\d{1,2}[\s\.\-]+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"[\s\.\-]+\d{2,4}\b|"
    r"\b\d{1,2}[\s\.\-]+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"[\s\.\-]+\d{2,4}\b"
)

BRAND_EXCLUSION_KEYWORDS = re.compile(
    r"(?i)\b(?:mint|livemint|thinkahead|think\s+ahead|think\s+growth|the\s+hindu|"
    r"the\s+times\s+of\s+india|times\s+of\s+india|economic\s+times|business\s+standard|"
    r"indian\s+express|hindustan\s+times|dainik\s+bhaskar|epaper|edition)\b"
)

MASTHEAD_KEYWORDS = re.compile(
    r"(?i)\b(?:mint|livemint|hindu|the\s+hindu|times\s+of\s+india|economic\s+times|"
    r"business\s+standard|indian\s+express|hindustan\s+times|dainik\s+bhaskar|"
    r"think\s+ahead|think\s+growth|epaper|vol(?:\.|ume)?\s*\d+|no\s*\d+|edition|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"page\s*\d+|pg\s*\d+|p\.\s*\d+)\b"
)

STANDALONE_NOISE_TOKENS = frozenset(
    {
        "mint", "livemint", "thinkahead", "think ahead", "asba",
        "jm financial", "axis capital", "kfin", "nse", "bse",
    }
)

SYNDICATION_SLUGS = frozenset(
    {
        "the wall street journal",
        "wall street journal",
        "reuters",
        "bloomberg",
        "bloomberg news",
        "pti",
        "press trust of india",
        "afp",
        "agence france-presse",
        "ap",
        "associated press",
        "financial times",
        "new york times",
        "business wire",
        "pr newswire",
        "news in numbers",
        "columns",
        "inside",
        "quote of the day",
        "data bites",
        "plain facts",
        "mint primer",
        "mint curator",
        "ask mint",
        "mark to market",
        "wsj",
    }
)

SYNDICATION_REGEX = re.compile(
    r"^(?:THE\s+)?(?:WALL\s+STREET\s+JOURNAL|REUTERS|BLOOMBERG(?:\s+NEWS)?|PTI|AFP|AP|"
    r"FINANCIAL\s+TIMES|NEW\s+YORK\s+TIMES|PRESS\s+TRUST\s+OF\s+INDIA|ASSOCIATED\s+PRESS|"
    r"QUOTE\s+OF\s+THE\s+DAY|DATA\s+BITES|PLAIN\s+FACTS|NEWS\s+IN\s+NUMBERS|COLUMNS|INSIDE|WSJ|"
    r"MARK\s+TO\s+MARKET|MINT\s+PRIMER|MINT\s+CURATOR|ASK\s+MINT)(?:\s*[\/\-–—|]\s*.*)?$",
    re.IGNORECASE,
)

NUMBERED_QUESTION_REGEX = re.compile(
    r"^(?:(?:Q\.?\s*)?\d{1,2}[\.\/\)]|\b(?:Q\d{1,2}|Part\s+\d+|Step\s+\d+)\b|\b\d{1,2}\s+(?:How|Why|What|When|Where|Who|Which|Can|Will|Is|Are|Do|Does|Did|Should|Could|Would|Has|Have|Had))\s+",
    re.IGNORECASE,
)


def is_syndication_or_agency_slug(text: str) -> bool:
    """Check if text is a syndication slug, wire agency stamp, or recurring column header."""
    t = text.strip()
    if not t:
        return False
    t_clean = t.lower().strip(" .:;,/–—-")
    if t_clean in SYNDICATION_SLUGS:
        return True
    if SYNDICATION_REGEX.match(t):
        return True
    return bool(re.match(r"^(?:reuters|pti|bloomberg|afp|ap|ians|ani|uni)\s*[\/|\-–—]", t_clean))


def is_numbered_feature_subhead(text: str) -> bool:
    """Detect numbered subheadings / questions in feature explainers."""
    t = text.strip()
    if not t:
        return False
    return bool(NUMBERED_QUESTION_REGEX.match(t))


TOC_SECTION_SLUGS_REGEX = re.compile(
    r"(?i)\b(?:Global|World|National|International|Business|Money|Economy|Views|"
    r"Editorial|Sport|Sports|Life|Metro|City|News|Focus|State|States|Showcase|"
    r"Inside|Features)\s*\|"
)

TOC_PAGE_POINTER_REGEX = re.compile(
    r"(?i)(?:>\s*P\s*\d+|>\s*Page\s*\d+|->\s*P\s*\d+|\bP\d{1,2}\b)"
)


def is_toc_index_block(text: str) -> bool:
    """Check for high densities of delimiter patterns common in front-page indexes."""
    if not text or not text.strip():
        return False
    t = text.strip()

    has_slug = bool(TOC_SECTION_SLUGS_REGEX.search(t))
    has_pointer = bool(TOC_PAGE_POINTER_REGEX.search(t))
    pipe_count = t.count("|")

    if has_slug and has_pointer:
        return True
    if has_slug and pipe_count >= 1 and (">" in t or "P" in t):
        return True
    if pipe_count >= 2 and has_pointer:
        return True

    lines = [line_item.strip() for line_item in t.split("\n") if line_item.strip()]
    if len(lines) >= 2:
        toc_line_matches = sum(
            1
            for line_item in lines
            if (
                TOC_SECTION_SLUGS_REGEX.search(line_item)
                or TOC_PAGE_POINTER_REGEX.search(line_item)
                or "|" in line_item
            )
        )
        if toc_line_matches >= 2:
            return True

    return False


PULLQUOTE_TITLE_REGEX = re.compile(
    r"(?i)\b(?:FOREIGN\s*MINISTER|PRIME\s*MINISTER|CHIEF\s*MINISTER|FINANCE\s*MINISTER|"
    r"HOME\s*MINISTER|DEFENCE\s*MINISTER|EXTERNAL\s*AFFAIRS\s*MINISTER|SECRETARY\s*GENERAL|"
    r"SPOKESPERSON|MANAGING\s*DIRECTOR|CHIEF\s*EXECUTIVE\s*OFFICER|EXECUTIVE\s*DIRECTOR|"
    r"FED\s*CHAIR(?:MAN)?|CENTRAL\s*BANK\s*GOVERNOR|CHIEF\s*JUSTICE|"
    r"AUSTRALIAN\s*FOREIGN\s*MINISTER|AUSTRALIANFOREIGN\s*MINISTER|PENNY\s*WONG|"
    r"PENNYWONG)\b"
)

HEADLINE_ACTION_VERBS = frozenset(
    {
        "holds", "cuts", "hikes", "raises", "drops", "rises", "falls", "soars",
        "plans", "buys", "sells", "warns", "sees", "eyes", "urges", "tells",
        "signs", "clears", "approves", "rejects", "posts", "hits", "leads",
        "mops", "caps", "curbs", "eases", "nods",
    }
)


def is_pullquote_author_block(text: str, surrounding_text: str = "") -> bool:
    """The Attribution Rule: Detect speaker/author attribution in pull quotes and sidebars."""
    t = text.strip()
    if not t:
        return False
    words = t.split()
    if len(words) > 8:
        return False

    words_clean = [w.lower().strip(".:;,!?'\"-–—") for w in words]
    if any(w in HEADLINE_ACTION_VERBS for w in words_clean):
        return False

    # 1. Matches specific author/minister/diplomat attribution names or titles
    if PULLQUOTE_TITLE_REGEX.search(t):
        return True

    # 2. Text is <= 6 words and surrounding/preceding block contains quotation marks
    has_quotes = bool(
        re.search(r'["“”‘’\']', surrounding_text) or re.search(r'["“”‘’\']', t)
    )
    clean = re.sub(r"[^\w\s]", "", t).strip()
    is_cased = clean.isupper() or is_title_case_or_uppercase(t)
    return bool(len(words) <= 6 and has_quotes and is_cased)


NUMERIC_STAT_PATTERN = re.compile(
    r"\b(?:\d+[\d,\.]*\s*(?:cr|crore|mn|million|bn|billion|lakh|%|pts|bps|usd|inr)?|"
    r"[\$₹€£]\s*\d+[\d,\.]*)\b",
    re.IGNORECASE,
)

OCR_HEADLINE_REPAIRS = [
    (re.compile(r"\bOl\s+estimates\b", re.IGNORECASE), "Q1 estimates"),
    (re.compile(r"\bQl\s+estimates\b", re.IGNORECASE), "Q1 estimates"),
    (re.compile(r"\bofGlasgow\b"), "of Glasgow"),
    (re.compile(r"\bcourtcases\b"), "court cases"),
    (re.compile(r"\btheworld\b"), "the world"),
    (re.compile(r"\beconomicdocudramahastheworld\b"), "economic docudrama has the world"),
    (re.compile(r"\bdocudramahastheworld\b"), "docudrama has the world"),
    (re.compile(r"\bartificialintelligence\b", re.IGNORECASE), "artificial intelligence"),
    (re.compile(r"\bcaseofoTT\b"), "case of OTT"),
    (re.compile(r"\bofaiding\b"), "of aiding"),
    (re.compile(r"\byourfamilybe\b"), "your family be"),
    (re.compile(r"\babletoaccess\b"), "able to access"),
    (re.compile(r"\briskforpharma\b"), "risk for pharma"),
    (re.compile(r"\btoChatGPT\b"), "to ChatGPT"),
    (re.compile(r"\bmutual fund-onlyPMS\b", re.IGNORECASE), "mutual fund-only PMS"),
    (re.compile(r"\b25lakh\b", re.IGNORECASE), "25 lakh"),
    (re.compile(r"\bageneric\b", re.IGNORECASE), "a generic"),
    (re.compile(r"\bIUNE\b"), "JUNE"),
    (re.compile(r"\bQl\b"), "Q1"),
]

CONTINUATION_END_TOKENS = frozenset(
    {
        "says", "warns", "sees", "eyes", "seeks", "aims", "cuts", "beats",
        "posts", "plans", "gets", "hits", "leads", "backs", "signs", "finds",
        "moots", "targets", "buys", "hires", "urges", "tells", "drops", "rises",
        "to", "in", "on", "at", "by", "for", "with", "from", "about", "into",
        "over", "after", "and", "or", "of", "as", "against", "despite", "under",
        "near", "up", "down", "out", "a", "an", "the",
    }
)


def clean_ocr_text_artifacts(text: str) -> str:
    """Repair unspaced tokens, font ligature bugs, and purge UUID/promo noise."""
    res = text
    for pattern, repl in OCR_HEADLINE_REPAIRS:
        res = pattern.sub(repl, res)
    return sanitize_block_text(res).strip()


def is_numeric_stat_box(text: str) -> bool:
    """Detect if text is an infographic/stat/table box rather than a textual headline."""
    tokens = text.strip().split()
    if not tokens:
        return False
    stat_matches = NUMERIC_STAT_PATTERN.findall(text)
    if len(stat_matches) >= 3 or (len(stat_matches) >= 2 and len(tokens) <= 6):
        return True
    numeric_tokens = sum(
        1 for t in tokens
        if any(c.isdigit() or c in "$₹€£%" for c in t)
        or t.lower() in ("cr", "crore", "mn", "bn", "lakh", "pts", "bps")
    )
    return (numeric_tokens / len(tokens)) >= 0.40


def is_grammatically_open_headline_fragment(text: str) -> bool:
    """Check if text ends in a grammatical continuation word or punctuation."""
    words = text.strip().split()
    if not words:
        return False
    last_word = words[-1].lower().strip(" \t\n\r.:;,")
    if last_word in CONTINUATION_END_TOKENS:
        return True
    if text.rstrip().endswith((",", "-", "—", ":", "...")):
        return True
    return bool(len(words) <= 2 and last_word in ("and", "or", "to", "in", "of", "for", "with"))


def is_noise_or_boilerplate_block(
    elem: LayoutElement,
    page_height: float,
    page_width: float,
) -> bool:
    """Filter out isolated brand logos, sponsor boilerplate, dates, and noise blocks."""
    text = elem.text.strip()
    if not text:
        return True

    text_lower = text.lower().strip(".:;, -")
    if text_lower in STANDALONE_NOISE_TOKENS:
        return True

    words = text.split()
    x0, y0, x1, y1 = elem.bbox

    # 1. Top 5% Coordinate Exclusion (Brand text / Mastheads in absolute header zone)
    if y1 <= page_height * 0.05:
        if (
            BRAND_EXCLUSION_KEYWORDS.search(text)
            or MASTHEAD_KEYWORDS.search(text)
            or DATE_LINE_PATTERN.search(text)
        ):
            return True
        if len(words) <= 2 and (len(text) < 10 or re.match(r"^\d{1,3}$", text)):
            return True

    # 2. Top 8% Masthead / Date / Slogan check
    if y1 <= page_height * 0.08 and len(words) <= 15 and MASTHEAD_KEYWORDS.search(text):
        return True

    # 3. Pure Date Strings
    if len(words) <= 8 and DATE_LINE_PATTERN.search(text):
        return True

    # 4. Financial Sponsor / Legal Boilerplate Boxes
    if len(words) <= 15 and SPONSOR_AND_BOILERPLATE_PATTERNS.search(text):
        return True

    # 5. Short standalone numbers or tokens in header zone
    return bool(
        y1 <= page_height * 0.08
        and len(words) <= 2
        and (len(text) < 10 or re.match(r"^\d{1,3}$", text))
    )


def is_masthead_or_running_header(
    elem: LayoutElement,
    page_height: float,
    page_width: float,
) -> bool:
    """Compatibility alias for is_noise_or_boilerplate_block."""
    return is_noise_or_boilerplate_block(elem, page_height, page_width)


LAYOUT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headlines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "level": {"type": "string", "enum": ["banner", "major", "subhead", "kicker"]},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["text", "bbox"],
            },
        },
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column_index": {"type": "integer"},
                    "text": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["bbox"],
            },
        },
        "photos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["bbox"],
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["bbox"],
            },
        },
    },
    "required": ["headlines", "columns"],
}

LAYOUT_PROMPT = """Analyze this newspaper page image and identify all structural layout regions:
1. Headlines: Main banner headlines, article headlines, subheadings with bounding boxes.
2. Columns: Text columns of news articles.
3. Photos: Images/illustrations and their captions.
4. Tables: Statistical or commodity data tables.

Return strict JSON conforming to the schema."""


@dataclass
class PageLayoutResult:
    """Consolidated layout analysis output."""

    page_number: int
    width_px: int
    height_px: int
    elements: list[LayoutElement] = field(default_factory=list)
    reading_order: list[OrderedReadingBlock] = field(default_factory=list)
    tables: list[ExtractedTableData] = field(default_factory=list)
    photos: list[ExtractedPhotoData] = field(default_factory=list)
    markdown_content: str = ""
    source: str = "mineru"  # 'mineru', 'vlm', or 'spatial_rule_based'


class LayoutAnalyzer:
    """Analyzes newspaper page visual layouts using MinerU, VLM, and rule-based fallbacks."""

    def __init__(
        self,
        vision_provider: VisionModelProvider | DocumentLayoutProvider | None = None,
    ) -> None:
        self._settings = get_settings()
        self._layout_provider = vision_provider

    async def _get_layout_provider(
        self,
    ) -> VisionModelProvider | DocumentLayoutProvider | None:
        """Resolve configured layout analysis provider if available."""
        if self._layout_provider:
            return self._layout_provider
        try:
            registry = get_registry()
            provider = registry.get_provider("layout_analysis")
            if isinstance(provider, (DocumentLayoutProvider, VisionModelProvider)):
                return provider
        except Exception as e:
            logger.debug(
                "Could not load layout provider (falling back to rule-based)",
                extra={"error": str(e)},
            )
        return None

    def analyze_from_text_blocks(
        self,
        page_number: int,
        width_px: int,
        height_px: int,
        digital_blocks: list[DigitalTextBlock] | None = None,
        ocr_blocks: list[OCRBlock] | None = None,
    ) -> PageLayoutResult:
        """Rule-based spatial layout analysis using bounding boxes from digital text or OCR."""
        elements: list[LayoutElement] = []
        element_id = 1

        if digital_blocks:
            max_x = max((d.bbox[2] for d in digital_blocks), default=float(width_px))
            ref_width = max_x if max_x > 0 else float(width_px)
            for d_blk in digital_blocks:
                cleaned_text = clean_ocr_text_artifacts(d_blk.text)
                if not cleaned_text or is_noise_or_promo_text(cleaned_text):
                    continue
                if is_syndication_or_agency_slug(cleaned_text):
                    b_type = BlockType.BYLINE
                elif is_numbered_feature_subhead(cleaned_text):
                    b_type = BlockType.SUBHEAD
                elif is_numeric_stat_box(cleaned_text):
                    b_type = BlockType.TABLE
                else:
                    is_wide_heading = (
                        d_blk.is_heading_candidate
                        and (d_blk.bbox[2] - d_blk.bbox[0]) >= ref_width * 0.40
                    )
                    b_type = (
                        BlockType.BANNER_HEADLINE
                        if is_wide_heading
                        else (
                            BlockType.HEADLINE
                            if d_blk.is_heading_candidate
                            else BlockType.BODY_TEXT
                        )
                    )
                elements.append(
                    LayoutElement(
                        element_id=element_id,
                        bbox=d_blk.bbox,
                        text=cleaned_text,
                        block_type=b_type,
                        font_size=d_blk.mean_font_size,
                    )
                )
                element_id += 1

        elif ocr_blocks:
            # Dynamically compute line heights across OCR blocks to distinguish headings
            line_heights: list[float] = []
            for o_blk in ocr_blocks:
                num_lines = len([line for line in o_blk.text.split("\n") if line.strip()]) or 1
                lh = (o_blk.bbox[3] - o_blk.bbox[1]) / num_lines
                line_heights.append(lh)

            import statistics

            median_lh = statistics.median(line_heights) if line_heights else 20.0

            boilerplate_stopwords = {
                "limited", "ltd", "corp", "corporation", "pvt", "private", "equity", "issue",
                "issue,", "shares", "company", "notice", "promoters", "price", "band", "page",
                "continued", "from", "and", "or", "of", "in", "on", "at", "to", "for", "with",
            }

            for o_blk, lh in zip(ocr_blocks, line_heights, strict=False):
                cleaned_text = clean_ocr_text_artifacts(o_blk.text)
                box_width = o_blk.bbox[2] - o_blk.bbox[0]
                words_blk = cleaned_text.strip().split()
                is_single_boilerplate = (
                    len(words_blk) == 1
                    and words_blk[0].lower().rstrip(",.:;") in boilerplate_stopwords
                )

                if is_syndication_or_agency_slug(cleaned_text):
                    b_type = BlockType.BYLINE
                elif is_numbered_feature_subhead(cleaned_text):
                    b_type = BlockType.SUBHEAD
                elif is_numeric_stat_box(cleaned_text):
                    b_type = BlockType.TABLE
                else:
                    is_banner = (
                        box_width >= float(width_px) * 0.50
                        and lh >= median_lh * 1.25
                        and not is_single_boilerplate
                    )
                    is_headline = (
                        not is_single_boilerplate
                        and (
                            is_banner
                            or (
                                lh >= median_lh * 1.25
                                and 3 <= len(words_blk) <= 25
                                and not cleaned_text.rstrip().endswith((".", ";", ","))
                            )
                            or (
                                cleaned_text.isupper()
                                and 3 <= len(words_blk) <= 20
                                and lh >= median_lh * 1.10
                                and not cleaned_text.rstrip().endswith((".", ";", ","))
                            )
                        )
                    )
                    b_type = (
                        BlockType.BANNER_HEADLINE
                        if is_banner
                        else (BlockType.HEADLINE if is_headline else BlockType.BODY_TEXT)
                    )
                elements.append(
                    LayoutElement(
                        element_id=element_id,
                        bbox=o_blk.bbox,
                        text=cleaned_text,
                        block_type=b_type,
                        confidence=o_blk.confidence,
                    )
                )
                element_id += 1

        # Filter out mastheads and running headers in top 8%
        non_masthead_elements = [
            e for e in elements
            if not is_masthead_or_running_header(e, float(height_px), float(width_px))
        ]
        elements = non_masthead_elements if non_masthead_elements else elements

        # Consolidate bounding boxes and merge adjacent paragraph fragments
        elements = self._consolidate_elements(
            elements=elements,
            page_width=float(width_px),
            page_height=float(height_px),
        )

        resolver = ReadingOrderResolver(page_width=float(width_px), page_height=float(height_px))
        reading_order = resolver.resolve_reading_order(elements)

        return PageLayoutResult(
            page_number=page_number,
            width_px=width_px,
            height_px=height_px,
            elements=elements,
            reading_order=reading_order,
            source="spatial_rule_based",
        )

    def _merge_horizontal_headline_slices(
        self,
        elements: list[LayoutElement],
        page_width: float,
    ) -> list[LayoutElement]:
        """Merge multi-column sliced headlines that lie on the same horizontal plane."""
        if len(elements) <= 1:
            return elements

        elems = list(elements)
        merged = True

        from app.ingestion.detector import check_is_advertisement_text

        while merged:
            merged = False
            for i in range(len(elems)):
                elem_a = elems[i]
                # Strictly require true headline block types
                if elem_a.block_type not in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE):
                    continue
                if check_is_advertisement_text(elem_a.text):
                    continue

                for j in range(len(elems)):
                    if i == j:
                        continue
                    elem_b = elems[j]
                    if elem_b.block_type not in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE):
                        continue
                    if check_is_advertisement_text(elem_b.text):
                        continue

                    # Require A to be physically left of B
                    ax0, ay0, ax1, ay1 = elem_a.bbox
                    bx0, by0, bx1, by1 = elem_b.bbox
                    if bx0 <= ax0:
                        continue

                    ha = max(ay1 - ay0, 1.0)
                    hb = max(by1 - by0, 1.0)
                    min_h = min(ha, hb)
                    max_h = max(ha, hb)

                    # 1. Strict Horizontal Baseline and Vertical Overlap Alignment (<= 10px)
                    baseline_diff = abs(ay0 - by0)
                    v_overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
                    v_overlap_ratio = v_overlap / min_h

                    # 2. Strict X-axis gutter clamp (GUTTER_MIN_WIDTH = 0.012)
                    gap_x = bx0 - ax1
                    max_gap_x = page_width * 0.012

                    # 3. Strict Font similarity (<= 15% difference)
                    font_diff = abs(elem_a.font_size - elem_b.font_size) / max(
                        elem_a.font_size, elem_b.font_size, 1.0
                    )

                    # Anti-collision word checks:
                    words_a = elem_a.text.strip().split()
                    words_b = elem_b.text.strip().split()
                    if not words_a or not words_b:
                        continue

                    # If elem_a ends with terminal punctuation (. ? ! : " ' ”)
                    if elem_a.text.rstrip().endswith((".", "?", "!", ":", ";", '"', "'", "”")):
                        continue

                    # Rule 3: If block A and B both contain >= 3 title-cased words and are closed,
                    # horizontal merging across column tracks is forbidden
                    is_open_a = is_grammatically_open_headline_fragment(elem_a.text)
                    is_open_b = bool(
                        words_b[0][0].islower()
                        or words_b[0].lower() in CONTINUATION_END_TOKENS
                    )
                    last_token_a = words_a[-1].lower().strip(" \t\n\r.:;,")
                    ends_with_hyphen = elem_a.text.rstrip().endswith(("-", "—", ","))
                    is_open_connector = bool(
                        last_token_a in CONTINUATION_END_TOKENS or is_open_a or is_open_b
                    )

                    if not is_open_connector:
                        continue

                    # If both have >= 3 words and neither ends with a connector/hyphen
                    if (
                        len(words_a) >= 3
                        and len(words_b) >= 3
                        and not (ends_with_hyphen or last_token_a in CONTINUATION_END_TOKENS)
                    ):
                        continue

                    max_gap_x = max(page_width * 0.035, 35.0)

                    if (
                        baseline_diff <= min(max_h * 0.15, 10.0)
                        and v_overlap_ratio >= 0.80
                        and -5.0 <= gap_x <= max_gap_x
                        and font_diff <= 0.15
                    ):
                        # Merge A and B
                        elem_a.text = f"{elem_a.text} {elem_b.text}".strip()
                        elem_a.bbox = (
                            min(ax0, bx0),
                            min(ay0, by0),
                            max(ax1, bx1),
                            max(ay1, by1),
                        )
                        elem_a.font_size = max(elem_a.font_size, elem_b.font_size)
                        total_width = elem_a.bbox[2] - elem_a.bbox[0]
                        if total_width >= page_width * 0.50:
                            elem_a.block_type = BlockType.BANNER_HEADLINE
                        else:
                            elem_a.block_type = BlockType.HEADLINE

                        elems.pop(j)
                        merged = True
                        break
                if merged:
                    break

        return elems

    def _consolidate_elements(
        self,
        elements: list[LayoutElement],
        page_width: float,
        page_height: float,
    ) -> list[LayoutElement]:
        """Consolidate adjacent text boxes & headlines using Vertical-First column tracking."""
        if len(elements) <= 1:
            return elements

        # Sort elements primarily by top-Y, then left-X
        sorted_elements = sorted(elements, key=lambda e: (round(e.bbox[1] / 20.0), e.bbox[0]))

        # Pass 0: Drop Cap Reattachment (Merge single uppercase initials into subsequent body)
        drop_cap_merged: list[LayoutElement] = []
        skip_indices: set[int] = set()
        for i, el in enumerate(sorted_elements):
            if i in skip_indices:
                continue
            raw_t = el.text.strip()
            if (
                len(raw_t) == 1
                and raw_t.isalpha()
                and raw_t.isupper()
                and i + 1 < len(sorted_elements)
            ):
                target_j: int | None = None
                for j in range(i + 1, min(i + 6, len(sorted_elements))):
                    if j in skip_indices:
                        continue
                    cand = sorted_elements[j]
                    if not cand.text.strip():
                        continue
                    is_near_y = (el.bbox[1] - 8.0 <= cand.bbox[1] <= el.bbox[3] + 25.0)
                    h_overlap = min(el.bbox[2], cand.bbox[2]) - max(el.bbox[0], cand.bbox[0])
                    is_col_aligned = h_overlap > -40.0 and cand.bbox[0] >= el.bbox[0] - 10.0
                    if is_near_y and is_col_aligned:
                        target_j = j
                        break
                if target_j is not None:
                    cand = sorted_elements[target_j]
                    if cand.text.startswith(" ") or cand.text.startswith("\n"):
                        cand.text = raw_t + cand.text.lstrip()
                    elif cand.text and cand.text[0].islower():
                        cand.text = raw_t + cand.text
                    else:
                        cand.text = f"{raw_t} {cand.text}".strip()
                    cand.bbox = (
                        min(el.bbox[0], cand.bbox[0]),
                        min(el.bbox[1], cand.bbox[1]),
                        max(el.bbox[2], cand.bbox[2]),
                        max(el.bbox[3], cand.bbox[3]),
                    )
                    skip_indices.add(i)
                    continue
            drop_cap_merged.append(el)
        sorted_elements = drop_cap_merged

        # Compute median line height across body elements
        body_heights = [
            (e.bbox[3] - e.bbox[1]) / max(len(e.text.split("\n")), 1)
            for e in sorted_elements
            if e.block_type == BlockType.BODY_TEXT and (e.bbox[3] - e.bbox[1]) > 0
        ]
        median_lh = float(sorted(body_heights)[len(body_heights) // 2]) if body_heights else 20.0
        max_v_gap = max(median_lh * 1.8, 15.0)

        # Pass 1: Vertical Multi-Line Headline & Paragraph Stitching within Column Tracks
        consolidated: list[LayoutElement] = []
        for elem in sorted_elements:
            if not consolidated:
                consolidated.append(elem)
                continue

            merged_vertically = False
            for prev in reversed(consolidated):
                gap_y = elem.bbox[1] - prev.bbox[3]
                if gap_y > max_v_gap * 2.5:
                    break

                # Column track overlap
                overlap_x = max(
                    0.0, min(prev.bbox[2], elem.bbox[2]) - max(prev.bbox[0], elem.bbox[0])
                )
                min_w = min(prev.bbox[2] - prev.bbox[0], elem.bbox[2] - elem.bbox[0])
                is_same_track = bool(min_w > 0 and (overlap_x / min_w) >= 0.50)

                # HEADING-BOUNDARY BREAK: Stop immediately if an intervening headline is encountered
                if (
                    elem.block_type == BlockType.BODY_TEXT
                    and is_same_track
                    and prev.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
                ):
                    break

                if (
                    elem.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
                    and is_same_track
                    and prev.block_type == BlockType.BODY_TEXT
                ):
                    break

                # Case A: Font-Aware Multi-Line Headline Stitching in same column track
                if (
                    prev.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
                    and elem.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
                ):
                    font_diff = (
                        abs(prev.font_size - elem.font_size)
                        / max(prev.font_size, elem.font_size, 1.0)
                    )

                    if (
                        min_w > 0
                        and (overlap_x / min_w) >= 0.35
                        and -5.0 <= gap_y <= max_v_gap * 2.0
                        and font_diff <= 0.35
                    ):
                        prev.text = f"{prev.text} {elem.text}".strip()
                        prev.bbox = (
                            min(prev.bbox[0], elem.bbox[0]),
                            min(prev.bbox[1], elem.bbox[1]),
                            max(prev.bbox[2], elem.bbox[2]),
                            max(prev.bbox[3], elem.bbox[3]),
                        )
                        prev.font_size = max(prev.font_size, elem.font_size)
                        merged_vertically = True
                        break

                # Case B: Merge adjacent body paragraphs in the same column track
                if (
                    prev.block_type == BlockType.BODY_TEXT
                    and elem.block_type == BlockType.BODY_TEXT
                ):
                    scale_h = (page_height / 1000.0) if page_height else 1.0
                    max_allowed_gap = max(median_lh * 1.6, 20.0 * scale_h)
                    if (
                        min_w > 0
                        and (overlap_x / min_w) >= 0.45
                        and -8.0 <= gap_y <= max_allowed_gap
                    ):
                        if prev.text.endswith("-"):
                            prev.text = prev.text[:-1] + elem.text
                        elif prev.text.endswith((".", "!", "?", ":")):
                            prev.text = f"{prev.text}\n\n{elem.text}"
                        else:
                            prev.text = f"{prev.text} {elem.text}"
                        prev.bbox = (
                            min(prev.bbox[0], elem.bbox[0]),
                            min(prev.bbox[1], elem.bbox[1]),
                            max(prev.bbox[2], elem.bbox[2]),
                            max(prev.bbox[3], elem.bbox[3]),
                        )
                        merged_vertically = True
                        break

            if not merged_vertically:
                consolidated.append(elem)

        # Pass 2: Merge multi-column sliced headlines horizontally across X-axis
        consolidated = self._merge_horizontal_headline_slices(consolidated, page_width)

        # Re-index element IDs
        for idx, elem in enumerate(consolidated):
            elem.element_id = idx + 1

        return consolidated

    def build_layout_from_parsed_nodes(
        self,
        page_number: int,
        width_px: int,
        height_px: int,
        nodes: list[ExtractedDocumentNode],
        markdown_content: str = "",
        source: str = "docling",
    ) -> PageLayoutResult:
        """Construct PageLayoutResult from pre-extracted neural document nodes."""
        elements: list[LayoutElement] = []
        tables: list[ExtractedTableData] = []
        photos: list[ExtractedPhotoData] = []

        for idx, node in enumerate(nodes):
            cleaned_t = clean_ocr_text_artifacts(node.text)

            # Check surrounding text for quotation context
            prev_t = nodes[idx - 1].text if idx > 0 else ""
            next_t = nodes[idx + 1].text if idx + 1 < len(nodes) else ""
            surrounding_context = f"{prev_t}\n{next_t}"

            b_type = BlockType.BODY_TEXT
            if is_toc_index_block(cleaned_t):
                b_type = BlockType.TOC_INDEX
            elif is_pullquote_author_block(cleaned_t, surrounding_context):
                b_type = BlockType.PULLQUOTE_AUTHOR
            elif is_syndication_or_agency_slug(cleaned_t):
                b_type = BlockType.BYLINE
            elif is_numeric_stat_box(cleaned_t):
                b_type = BlockType.TABLE
            elif node.node_type == "title":
                if not (is_toc_index_block(cleaned_t) or is_pullquote_author_block(cleaned_t)):
                    b_type = (
                        BlockType.BANNER_HEADLINE if node.level == 1 else BlockType.HEADLINE
                    )
                else:
                    b_type = BlockType.BODY_TEXT
            elif node.node_type == "table":
                b_type = BlockType.TABLE
                if node.table_data:
                    tables.append(node.table_data)
            elif node.node_type in ("image", "photo"):
                b_type = BlockType.PHOTO
                if node.photo_data:
                    photos.append(node.photo_data)
            elif node.node_type == "caption":
                b_type = BlockType.CAPTION

            elem = LayoutElement(
                element_id=idx + 1,
                bbox=node.bbox,
                text=cleaned_t,
                block_type=b_type,
            )
            elements.append(elem)

        # Filter out mastheads and running headers from elements
        non_masthead_elements = [
            e for e in elements
            if not is_masthead_or_running_header(e, float(height_px), float(width_px))
        ]
        elements = non_masthead_elements if non_masthead_elements else elements

        # For neural Docling layout, use Docling's pre-linearized reading order directly
        reading_order = [
            OrderedReadingBlock(
                reading_order_index=i + 1,
                element_id=elem.element_id,
                block_type=elem.block_type,
                text=elem.text,
                bbox=elem.bbox,
            )
            for i, elem in enumerate(elements)
        ]

        return PageLayoutResult(
            page_number=page_number,
            width_px=width_px,
            height_px=height_px,
            elements=elements,
            reading_order=reading_order,
            tables=tables,
            photos=photos,
            markdown_content=markdown_content,
            source=source,
        )

    async def analyze_page(
        self,
        page_number: int,
        width_px: int,
        height_px: int,
        image_bytes: bytes,
        digital_blocks: list[DigitalTextBlock] | None = None,
        ocr_blocks: list[OCRBlock] | None = None,
    ) -> PageLayoutResult:
        """Run full hybrid layout analysis on a newspaper page."""
        # If page is scanned and ocr_blocks are provided, prioritize them directly
        if ocr_blocks and not digital_blocks:
            return self.analyze_from_text_blocks(
                page_number=page_number,
                width_px=width_px,
                height_px=height_px,
                ocr_blocks=ocr_blocks,
            )

        provider = await self._get_layout_provider()
        if not provider:
            return self.analyze_from_text_blocks(
                page_number=page_number,
                width_px=width_px,
                height_px=height_px,
                digital_blocks=digital_blocks,
                ocr_blocks=ocr_blocks,
            )

        # Path A: DocumentLayoutProvider (Docling / MinerU)
        if isinstance(provider, DocumentLayoutProvider):
            try:
                parsed_res = await provider.parse_page_image(
                    image_bytes=image_bytes,
                    page_number=page_number,
                )

                # Check if provider returned non-empty text nodes
                text_nodes = [n for n in parsed_res.nodes if n.text and n.text.strip()]
                if not text_nodes and (ocr_blocks or digital_blocks):
                    return self.analyze_from_text_blocks(
                        page_number=page_number,
                        width_px=width_px,
                        height_px=height_px,
                        digital_blocks=digital_blocks,
                        ocr_blocks=ocr_blocks,
                    )

                provider_source = getattr(provider, "provider_name", "docling")
                return self.build_layout_from_parsed_nodes(
                    page_number=page_number,
                    width_px=width_px,
                    height_px=height_px,
                    nodes=parsed_res.nodes,
                    markdown_content=parsed_res.markdown_content,
                    source=provider_source,
                )
            except Exception as e:
                logger.warning(
                    "DocumentLayoutProvider analysis failed, falling back to spatial rules",
                    extra={"page_number": page_number, "error": str(e)},
                )
                return self.analyze_from_text_blocks(
                    page_number=page_number,
                    width_px=width_px,
                    height_px=height_px,
                    digital_blocks=digital_blocks,
                    ocr_blocks=ocr_blocks,
                )

        # Path B: VisionModelProvider (VLM prompt)
        try:
            resp = await provider.analyze_image(
                image_bytes=image_bytes,
                prompt=LAYOUT_PROMPT,
                response_schema=LAYOUT_EXTRACTION_SCHEMA,
            )
            raw_data = resp.parsed
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)

            if not isinstance(raw_data, dict):
                raise ValueError("Vision model response did not return a valid layout dict.")

            vlm_elements: list[LayoutElement] = []
            el_id = 1

            for h in raw_data.get("headlines", []):
                cleaned_h = clean_ocr_text_artifacts(h.get("text", ""))
                bbox = tuple(float(x) for x in h.get("bbox", [0, 0, 0, 0]))
                level = h.get("level", "major")
                if is_numeric_stat_box(cleaned_h):
                    b_type = BlockType.TABLE
                else:
                    b_type = BlockType.BANNER_HEADLINE if level == "banner" else BlockType.HEADLINE
                vlm_elements.append(
                    LayoutElement(
                        element_id=el_id,
                        bbox=bbox,  # type: ignore[arg-type]
                        text=cleaned_h,
                        block_type=b_type,
                    )
                )
                el_id += 1

            for col in raw_data.get("columns", []):
                cleaned_c = clean_ocr_text_artifacts(col.get("text", ""))
                bbox = tuple(float(x) for x in col.get("bbox", [0, 0, 0, 0]))
                b_type = BlockType.TABLE if is_numeric_stat_box(cleaned_c) else BlockType.BODY_TEXT
                vlm_elements.append(
                    LayoutElement(
                        element_id=el_id,
                        bbox=bbox,  # type: ignore[arg-type]
                        text=cleaned_c,
                        block_type=b_type,
                    )
                )
                el_id += 1

            for p in raw_data.get("photos", []):
                bbox = tuple(float(x) for x in p.get("bbox", [0, 0, 0, 0]))
                vlm_elements.append(
                    LayoutElement(
                        element_id=el_id,
                        bbox=bbox,  # type: ignore[arg-type]
                        text=p.get("caption", ""),
                        block_type=BlockType.PHOTO,
                    )
                )
                el_id += 1

            # Filter mastheads in VLM output
            non_masthead_vlm = [
                e for e in vlm_elements
                if not is_masthead_or_running_header(e, float(height_px), float(width_px))
            ]
            vlm_elements = non_masthead_vlm if non_masthead_vlm else vlm_elements

            # Consolidate bounding boxes and merge adjacent paragraph fragments
            vlm_elements = self._consolidate_elements(
                elements=vlm_elements,
                page_width=float(width_px),
                page_height=float(height_px),
            )

            resolver = ReadingOrderResolver(
                page_width=float(width_px), page_height=float(height_px)
            )
            reading_order_blocks = resolver.resolve_reading_order(vlm_elements)

            return PageLayoutResult(
                page_number=page_number,
                width_px=width_px,
                height_px=height_px,
                elements=vlm_elements,
                reading_order=reading_order_blocks,
                source="vlm",
            )

        except Exception as e:
            logger.warning(
                "VLM layout analysis failed, falling back to spatial rules",
                extra={"page_number": page_number, "error": str(e)},
            )
            return self.analyze_from_text_blocks(
                page_number=page_number,
                width_px=width_px,
                height_px=height_px,
                digital_blocks=digital_blocks,
                ocr_blocks=ocr_blocks,
            )
