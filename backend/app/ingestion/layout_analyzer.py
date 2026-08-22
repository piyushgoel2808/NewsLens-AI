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
                is_wide_heading = (
                    d_blk.is_heading_candidate
                    and (d_blk.bbox[2] - d_blk.bbox[0]) >= ref_width * 0.40
                )
                b_type = (
                    BlockType.BANNER_HEADLINE
                    if is_wide_heading
                    else (BlockType.HEADLINE if d_blk.is_heading_candidate else BlockType.BODY_TEXT)
                )
                elements.append(
                    LayoutElement(
                        element_id=element_id,
                        bbox=d_blk.bbox,
                        text=d_blk.text,
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
                box_width = o_blk.bbox[2] - o_blk.bbox[0]
                words_blk = o_blk.text.strip().split()
                is_single_boilerplate = (
                    len(words_blk) == 1
                    and words_blk[0].lower().rstrip(",.:;") in boilerplate_stopwords
                )

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
                            o_blk.text.isupper()
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
                        text=o_blk.text,
                        block_type=b_type,
                        confidence=o_blk.confidence,
                    )
                )
                element_id += 1

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

    def _consolidate_elements(
        self,
        elements: list[LayoutElement],
        page_width: float,
        page_height: float,
    ) -> list[LayoutElement]:
        """Consolidate spatially adjacent text boxes and multi-line headlines."""
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

        consolidated: list[LayoutElement] = []
        for elem in sorted_elements:
            if not consolidated:
                consolidated.append(elem)
                continue

            prev = consolidated[-1]

            # Case A: Merge multi-line headlines
            if (
                prev.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
                and elem.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
            ):
                overlap_x = max(
                    0.0, min(prev.bbox[2], elem.bbox[2]) - max(prev.bbox[0], elem.bbox[0])
                )
                min_w = min(prev.bbox[2] - prev.bbox[0], elem.bbox[2] - elem.bbox[0])
                gap_y = elem.bbox[1] - prev.bbox[3]
                if min_w > 0 and (overlap_x / min_w) >= 0.50 and 0.0 <= gap_y <= max_v_gap * 1.5:
                    prev.text = f"{prev.text} {elem.text}".strip()
                    prev.bbox = (
                        min(prev.bbox[0], elem.bbox[0]),
                        min(prev.bbox[1], elem.bbox[1]),
                        max(prev.bbox[2], elem.bbox[2]),
                        max(prev.bbox[3], elem.bbox[3]),
                    )
                    continue

            # Case B: Merge adjacent body paragraphs in the same column
            if prev.block_type == BlockType.BODY_TEXT and elem.block_type == BlockType.BODY_TEXT:
                overlap_x = max(
                    0.0, min(prev.bbox[2], elem.bbox[2]) - max(prev.bbox[0], elem.bbox[0])
                )
                min_w = min(prev.bbox[2] - prev.bbox[0], elem.bbox[2] - elem.bbox[0])
                gap_y = elem.bbox[1] - prev.bbox[3]
                if min_w > 0 and (overlap_x / min_w) >= 0.65 and 0.0 <= gap_y <= max_v_gap:
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
                    continue

            consolidated.append(elem)

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

                    elem = LayoutElement(
                        element_id=idx + 1,
                        bbox=node.bbox,
                        text=node.text,
                        block_type=b_type,
                    )
                    elements.append(elem)

                    reading_order.append(
                        OrderedReadingBlock(
                            reading_order_index=idx,
                            element_id=idx + 1,
                            block_type=b_type,
                            text=node.text,
                            bbox=node.bbox,
                        )
                    )

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
                bbox = tuple(float(x) for x in h.get("bbox", [0, 0, 0, 0]))
                level = h.get("level", "major")
                b_type = BlockType.BANNER_HEADLINE if level == "banner" else BlockType.HEADLINE
                vlm_elements.append(
                    LayoutElement(
                        element_id=el_id,
                        bbox=bbox,  # type: ignore[arg-type]
                        text=h.get("text", ""),
                        block_type=b_type,
                    )
                )
                el_id += 1

            for col in raw_data.get("columns", []):
                bbox = tuple(float(x) for x in col.get("bbox", [0, 0, 0, 0]))
                vlm_elements.append(
                    LayoutElement(
                        element_id=el_id,
                        bbox=bbox,  # type: ignore[arg-type]
                        text=col.get("text", ""),
                        block_type=BlockType.BODY_TEXT,
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
