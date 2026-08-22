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
from app.ingestion.detector import DigitalTextBlock
from app.ingestion.reading_order import (
    BlockType,
    LayoutElement,
    OrderedReadingBlock,
    ReadingOrderResolver,
)
from app.providers.base import (
    DocumentLayoutProvider,
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
    """Repair unspaced tokens and font ligature bugs in OCR text."""
    res = text
    for pattern, repl in OCR_HEADLINE_REPAIRS:
        res = pattern.sub(repl, res)
    return res


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
                if is_numeric_stat_box(cleaned_text):
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

                if is_numeric_stat_box(cleaned_text):
                    b_type = BlockType.TABLE
                else:
                    is_banner = (
                        box_width >= float(width_px) * 0.60
                        and lh >= median_lh * 1.30
                        and not is_single_boilerplate
                    )
                    is_headline = (
                        not is_single_boilerplate
                        and (
                            is_banner
                            or (lh >= median_lh * 1.35 and len(words_blk) >= 2)
                            or (
                                cleaned_text.isupper()
                                and 2 <= len(words_blk) < 15
                                and lh >= median_lh * 1.10
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

        # Path A: MinerU DocumentLayoutProvider
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

                elements: list[LayoutElement] = []
                reading_order: list[OrderedReadingBlock] = []
                tables: list[ExtractedTableData] = []
                photos: list[ExtractedPhotoData] = []

                for idx, node in enumerate(parsed_res.nodes):
                    b_type = BlockType.BODY_TEXT
                    if node.node_type == "title":
                        b_type = (
                            BlockType.BANNER_HEADLINE if node.level == 1 else BlockType.HEADLINE
                        )
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

                    cleaned_t = clean_ocr_text_artifacts(node.text)
                    if is_numeric_stat_box(cleaned_t):
                        b_type = BlockType.TABLE

                    elem = LayoutElement(
                        element_id=idx + 1,
                        bbox=node.bbox,
                        text=cleaned_t,
                        block_type=b_type,
                    )
                    elements.append(elem)

                # Filter out mastheads and running headers from MinerU elements
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

                resolver = ReadingOrderResolver(
                    page_width=float(width_px), page_height=float(height_px)
                )
                reading_order = resolver.resolve_reading_order(elements)

                return PageLayoutResult(
                    page_number=page_number,
                    width_px=width_px,
                    height_px=height_px,
                    elements=elements,
                    reading_order=reading_order,
                    tables=tables,
                    photos=photos,
                    markdown_content=parsed_res.markdown_content,
                    source="mineru",
                )
            except Exception as e:
                logger.warning(
                    "MinerU layout analysis failed, falling back to spatial rules",
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
