"""Reading order resolution and column flow algorithms for newspaper pages.

Standard newspapers feature complex multi-column grids where:
- Banner headlines span 3-6 columns above independent article streams.
- Columns flow vertically top-to-bottom within column lanes.
- Column lanes sequence left-to-right across the page.
- Sidebars and boxed features are self-contained regions.

This module provides spatial topological sorting to linearize 2D layout blocks
into human reading order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BlockType(StrEnum):
    BANNER_HEADLINE = "banner_headline"
    HEADLINE = "headline"
    BODY_TEXT = "body_text"
    PHOTO = "photo"
    CAPTION = "caption"
    TABLE = "table"
    SIDEBAR = "sidebar"
    UNKNOWN = "unknown"


@dataclass
class LayoutElement:
    """A spatial bounding box element on a page."""

    element_id: str | int
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    text: str = ""
    block_type: BlockType = BlockType.BODY_TEXT
    font_size: float = 10.0
    confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderedReadingBlock:
    """An element placed in deterministic human reading order."""

    reading_order_index: int
    element_id: str | int
    block_type: BlockType
    text: str
    bbox: tuple[float, float, float, float]
    column_band: int = 0


class ReadingOrderResolver:
    """Resolves spatial 2D newspaper blocks into 1D human reading sequences."""

    def __init__(self, page_width: float = 2480.0, page_height: float = 3508.0) -> None:
        self.page_width = page_width
        self.page_height = page_height

    def resolve_reading_order(
        self,
        elements: list[LayoutElement],
    ) -> list[OrderedReadingBlock]:
        """Order layout elements following standard newspaper reading rules."""
        if not elements:
            return []

        # Step 1: Separate elements into top-banners vs columnar content
        # An element is a top banner if it's wide (> 40% page width) and in the upper half
        banners: list[LayoutElement] = []
        column_elements: list[LayoutElement] = []

        max_x = max((el.bbox[2] for el in elements), default=self.page_width)
        ref_width = max_x if max_x > 0 else self.page_width

        for el in elements:
            x0, y0, x1, y1 = el.bbox
            width = x1 - x0
            is_wide = width >= (ref_width * 0.40)
            is_header_type = el.block_type in (
                BlockType.BANNER_HEADLINE,
                BlockType.HEADLINE,
            ) or el.font_size > 14.0

            if is_wide and is_header_type:
                banners.append(el)
            else:
                column_elements.append(el)

        # Sort banners strictly top-to-bottom by y0
        banners.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        # Step 2: Cluster column elements into vertical columns based on x0
        ordered_blocks: list[OrderedReadingBlock] = []
        current_index = 1

        for b in banners:
            ordered_blocks.append(
                OrderedReadingBlock(
                    reading_order_index=current_index,
                    element_id=b.element_id,
                    block_type=b.block_type,
                    text=b.text,
                    bbox=b.bbox,
                    column_band=0,
                )
            )
            current_index += 1

        if not column_elements:
            return ordered_blocks

        # Cluster remaining elements into column bands
        # Sort candidate elements by left position x0
        sorted_by_x = sorted(column_elements, key=lambda e: e.bbox[0])

        columns: list[list[LayoutElement]] = []
        col_x_threshold = self.page_width * 0.08  # 8% width tolerance for column alignment

        for el in sorted_by_x:
            assigned = False
            el_x0 = el.bbox[0]
            for col in columns:
                # Average x0 of the column
                avg_col_x0 = sum(item.bbox[0] for item in col) / len(col)
                if abs(el_x0 - avg_col_x0) <= col_x_threshold:
                    col.append(el)
                    assigned = True
                    break
            if not assigned:
                columns.append([el])

        # Sort columns left-to-right
        columns.sort(key=lambda col: sum(item.bbox[0] for item in col) / len(col))

        # Within each column, sort top-to-bottom by y0
        for col_idx, col in enumerate(columns, start=1):
            col.sort(key=lambda item: item.bbox[1])
            for item in col:
                ordered_blocks.append(
                    OrderedReadingBlock(
                        reading_order_index=current_index,
                        element_id=item.element_id,
                        block_type=item.block_type,
                        text=item.text,
                        bbox=item.bbox,
                        column_band=col_idx,
                    )
                )
                current_index += 1

        return ordered_blocks
