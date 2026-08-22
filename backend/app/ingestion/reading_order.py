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
    SUBHEAD = "subhead"
    BODY_TEXT = "body_text"
    BYLINE = "byline"
    METADATA = "metadata"
    PHOTO = "photo"
    CAPTION = "caption"
    TABLE = "table"
    SIDEBAR = "sidebar"
    TEASER = "teaser"
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
        self.page_width = max(page_width, 100.0)
        self.page_height = max(page_height, 100.0)

    def resolve_reading_order(
        self,
        elements: list[LayoutElement],
    ) -> list[OrderedReadingBlock]:
        """Order layout elements following 2D newspaper column-binding rules."""
        if not elements:
            return []

        # Step 1: Identify all headline / banner elements
        headlines: list[LayoutElement] = [
            el for el in elements
            if el.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)
        ]

        # Sort headlines primarily top-to-bottom by y0, then left-to-right by x0
        quant_y = max(self.page_height * 0.01, 15.0)
        headlines.sort(key=lambda h: (round(h.bbox[1] / quant_y), h.bbox[0]))

        assigned_ids: set[str | int] = set()
        ordered_blocks: list[OrderedReadingBlock] = []
        current_index = 1

        # Step 2: For each headline, bind underlying multi-column body blocks
        for h in headlines:
            if h.element_id in assigned_ids:
                continue

            assigned_ids.add(h.element_id)
            ordered_blocks.append(
                OrderedReadingBlock(
                    reading_order_index=current_index,
                    element_id=h.element_id,
                    block_type=h.block_type,
                    text=h.text,
                    bbox=h.bbox,
                    column_band=0,
                )
            )
            current_index += 1

            h_x0, h_y0, h_x1, h_y1 = h.bbox
            span_tol_x = max(self.page_width * 0.015, 20.0)
            span_x0 = max(0.0, h_x0 - span_tol_x)
            span_x1 = min(self.page_width, h_x1 + span_tol_x)

            # Find lower vertical bound: next headline below intersecting this horizontal span
            y_limit = self.page_height * 0.94
            for other_h in headlines:
                if other_h.element_id == h.element_id:
                    continue
                oh_x0, oh_y0, oh_x1, _ = other_h.bbox
                h_overlap = min(oh_x1, span_x1) - max(oh_x0, span_x0)
                if (
                    oh_y0 > h_y1
                    and h_overlap > 10.0
                    and oh_y0 < y_limit
                ):
                    y_limit = oh_y0

            # Collect candidate body blocks whose centroid or top falls inside 2D container
            candidate_body: list[LayoutElement] = []
            for el in elements:
                if el.element_id in assigned_ids:
                    continue
                if el.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE):
                    continue

                el_x0, el_y0, el_x1, el_y1 = el.bbox
                el_mid_x = (el_x0 + el_x1) / 2.0
                el_mid_y = (el_y0 + el_y1) / 2.0

                b_overlap = min(el_x1, span_x1) - max(el_x0, span_x0)
                min_el_w = max(el_x1 - el_x0, 1.0)
                overlap_ratio = b_overlap / min_el_w if min_el_w > 0 else 0.0

                is_inside_container = (
                    (span_x0 <= el_mid_x <= span_x1 or overlap_ratio >= 0.40)
                    and (
                        (h_y1 - (self.page_height * 0.008) <= el_y0 < y_limit)
                        or (h_y1 <= el_mid_y <= y_limit)
                    )
                )

                if is_inside_container:
                    candidate_body.append(el)

            if not candidate_body:
                continue

            # Cluster candidate body blocks into column lanes (left-to-right)
            sorted_candidates = sorted(candidate_body, key=lambda e: e.bbox[0])
            column_lanes: list[list[LayoutElement]] = []
            lane_tolerance = max(self.page_width * 0.045, 30.0)

            for cand in sorted_candidates:
                assigned_to_lane = False
                for lane in column_lanes:
                    avg_lane_x = sum(item.bbox[0] for item in lane) / len(lane)
                    if abs(cand.bbox[0] - avg_lane_x) <= lane_tolerance:
                        lane.append(cand)
                        assigned_to_lane = True
                        break
                if not assigned_to_lane:
                    column_lanes.append([cand])

            # Sort column lanes left-to-right
            column_lanes.sort(key=lambda lane: sum(item.bbox[0] for item in lane) / len(lane))

            # Within each column lane, sort top-to-bottom and add to reading sequence
            for lane_idx, lane in enumerate(column_lanes, start=1):
                lane.sort(key=lambda item: item.bbox[1])
                for item in lane:
                    assigned_ids.add(item.element_id)
                    ordered_blocks.append(
                        OrderedReadingBlock(
                            reading_order_index=current_index,
                            element_id=item.element_id,
                            block_type=item.block_type,
                            text=item.text,
                            bbox=item.bbox,
                            column_band=lane_idx,
                        )
                    )
                    current_index += 1

        # Step 3: Zero-Drop Guarantee — Append all remaining unassigned elements
        remaining = [el for el in elements if el.element_id not in assigned_ids]
        if remaining:
            remaining_sorted = sorted(remaining, key=lambda e: e.bbox[0])
            rem_lanes: list[list[LayoutElement]] = []
            lane_tol = max(self.page_width * 0.05, 35.0)

            for rem in remaining_sorted:
                assigned_to_rem = False
                for r_lane in rem_lanes:
                    avg_x = sum(item.bbox[0] for item in r_lane) / len(r_lane)
                    if abs(rem.bbox[0] - avg_x) <= lane_tol:
                        r_lane.append(rem)
                        assigned_to_rem = True
                        break
                if not assigned_to_rem:
                    rem_lanes.append([rem])

            rem_lanes.sort(key=lambda lane: sum(item.bbox[0] for item in lane) / len(lane))
            for r_lane_idx, r_lane in enumerate(rem_lanes, start=1):
                r_lane.sort(key=lambda item: item.bbox[1])
                for item in r_lane:
                    ordered_blocks.append(
                        OrderedReadingBlock(
                            reading_order_index=current_index,
                            element_id=item.element_id,
                            block_type=item.block_type,
                            text=item.text,
                            bbox=item.bbox,
                            column_band=r_lane_idx,
                        )
                    )
                    current_index += 1

        return ordered_blocks
