"""Unit tests for spatial reading order resolver."""
from __future__ import annotations

from app.ingestion.reading_order import (
    BlockType,
    LayoutElement,
    ReadingOrderResolver,
)


class TestReadingOrderResolver:
    """Test multi-column and banner headline reading order linearization."""

    def test_empty_elements_returns_empty_list(self) -> None:
        resolver = ReadingOrderResolver(page_width=1000, page_height=1500)
        assert resolver.resolve_reading_order([]) == []

    def test_banner_precedes_columns(self) -> None:
        resolver = ReadingOrderResolver(page_width=1000, page_height=1500)

        # Banner spanning 800px (80% width)
        banner = LayoutElement(
            element_id="banner_1",
            bbox=(100, 50, 900, 150),
            text="PRESIDENT SIGNS HISTORIC ACCORD",
            block_type=BlockType.BANNER_HEADLINE,
            font_size=28.0,
        )

        # Column 1 (Left column, x=100..350)
        col1_p1 = LayoutElement(
            element_id="c1_p1",
            bbox=(100, 200, 350, 400),
            text="Column 1 paragraph 1",
            block_type=BlockType.BODY_TEXT,
        )
        col1_p2 = LayoutElement(
            element_id="c1_p2",
            bbox=(100, 420, 350, 600),
            text="Column 1 paragraph 2",
            block_type=BlockType.BODY_TEXT,
        )

        # Column 2 (Right column, x=400..650)
        col2_p1 = LayoutElement(
            element_id="c2_p1",
            bbox=(400, 200, 650, 400),
            text="Column 2 paragraph 1",
            block_type=BlockType.BODY_TEXT,
        )

        elements = [col2_p1, col1_p2, banner, col1_p1]  # Intentionally scrambled
        ordered = resolver.resolve_reading_order(elements)

        assert len(ordered) == 4
        # 1. Banner must come first
        assert ordered[0].element_id == "banner_1"
        # 2. Column 1 (left) top paragraph
        assert ordered[1].element_id == "c1_p1"
        # 3. Column 1 (left) bottom paragraph
        assert ordered[2].element_id == "c1_p2"
        # 4. Column 2 (right) top paragraph
        assert ordered[3].element_id == "c2_p1"

    def test_multi_column_ordering_sequence(self) -> None:
        resolver = ReadingOrderResolver(page_width=1200, page_height=1800)

        # 3 Columns
        c1 = LayoutElement("c1", (50, 100, 350, 500), "Col 1", BlockType.BODY_TEXT)
        c2 = LayoutElement("c2", (400, 100, 700, 500), "Col 2", BlockType.BODY_TEXT)
        c3 = LayoutElement("c3", (750, 100, 1050, 500), "Col 3", BlockType.BODY_TEXT)

        ordered = resolver.resolve_reading_order([c3, c1, c2])
        assert [b.element_id for b in ordered] == ["c1", "c2", "c3"]
