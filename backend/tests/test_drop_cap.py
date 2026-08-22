"""Unit tests for Drop Cap Reattachment on DigitalTextBlock and LayoutElement."""

from __future__ import annotations

from app.ingestion.detector import DigitalTextBlock, reattach_drop_caps
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.reading_order import BlockType, LayoutElement


class TestDropCapReattachment:
    """Test reattachment of single-letter initial drop caps to body paragraphs."""

    def test_reattach_digital_text_block_lowercase_continuation(self) -> None:
        # Initial 'I' at font size 36, followed by 'ndia is planning...'
        drop_blk = DigitalTextBlock(
            block_id=0,
            text="I",
            bbox=(50.0, 100.0, 75.0, 140.0),
            lines=["I"],
            mean_font_size=36.0,
        )
        body_blk = DigitalTextBlock(
            block_id=1,
            text="ndia is planning a major expansion of its semiconductor manufacturing.",
            bbox=(80.0, 105.0, 300.0, 180.0),
            lines=["ndia is planning a major expansion of its semiconductor manufacturing."],
            mean_font_size=10.0,
        )

        blocks = [drop_blk, body_blk]
        result = reattach_drop_caps(blocks)

        assert len(result) == 1
        assert result[0].text.startswith("India is planning")
        assert result[0].bbox[0] == 50.0  # expanded to include drop cap x0

    def test_reattach_digital_text_block_capitalized_continuation(self) -> None:
        # Initial 'W' followed by 'ith the new fiscal policy...'
        drop_blk = DigitalTextBlock(
            block_id=0,
            text="W",
            bbox=(50.0, 100.0, 80.0, 140.0),
            lines=["W"],
            mean_font_size=36.0,
        )
        body_blk = DigitalTextBlock(
            block_id=1,
            text="ith the new fiscal policy, inflation is expected to moderate.",
            bbox=(85.0, 105.0, 300.0, 180.0),
            lines=["ith the new fiscal policy, inflation is expected to moderate."],
            mean_font_size=10.0,
        )

        blocks = [drop_blk, body_blk]
        result = reattach_drop_caps(blocks)

        assert len(result) == 1
        assert result[0].text.startswith("With the new fiscal policy")

    def test_layout_analyzer_drop_cap_consolidation(self) -> None:
        analyzer = LayoutAnalyzer()
        drop_elem = LayoutElement(
            element_id=1,
            bbox=(50.0, 200.0, 75.0, 240.0),
            text="S",
            block_type=BlockType.BODY_TEXT,
            font_size=32.0,
        )
        body_elem = LayoutElement(
            element_id=2,
            bbox=(80.0, 205.0, 350.0, 320.0),
            text="tock markets rallied today as foreign institutional investors turned net buyers.",
            block_type=BlockType.BODY_TEXT,
            font_size=10.0,
        )

        consolidated = analyzer._consolidate_elements(
            elements=[drop_elem, body_elem],
            page_width=1000.0,
            page_height=1500.0,
        )

        assert len(consolidated) == 1
        assert consolidated[0].text.startswith("Stock markets rallied today")
