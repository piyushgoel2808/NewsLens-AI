"""Unit tests for LayoutAnalyzer (VLM and rule-based modes)."""
from __future__ import annotations

import pytest

from app.ingestion.detector import DigitalTextBlock
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.reading_order import BlockType
from app.providers.base import ModelResponse, ProviderCapability


class MockVisionProvider:
    """Mock VisionModelProvider for testing VLM layout analysis."""

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(supports_vision=True, supports_structured_output=True)

    @property
    def provider_name(self) -> str:
        return "mock_vlm"

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        return ModelResponse(
            text="",
            parsed={
                "headlines": [
                    {
                        "text": "SENATE PASSES TARIFF REFORM",
                        "level": "banner",
                        "bbox": [50.0, 30.0, 950.0, 100.0],
                    }
                ],
                "columns": [
                    {
                        "column_index": 1,
                        "text": "Debate concluded late last night...",
                        "bbox": [50.0, 120.0, 480.0, 800.0],
                    },
                    {
                        "column_index": 2,
                        "text": "Opposition leaders argued strongly against...",
                        "bbox": [520.0, 120.0, 950.0, 800.0],
                    },
                ],
                "photos": [
                    {
                        "caption": "Senate floor during vote",
                        "bbox": [520.0, 120.0, 950.0, 350.0],
                    }
                ],
                "tables": [],
            },
            model="mock-vlm",
            provider="mock",
        )


class TestLayoutAnalyzer:
    """Test layout analysis using VLM and rule-based methods."""

    @pytest.mark.asyncio
    async def test_analyze_page_with_vlm(self) -> None:
        mock_vision = MockVisionProvider()
        analyzer = LayoutAnalyzer(vision_provider=mock_vision)

        res = await analyzer.analyze_page(
            page_number=1,
            width_px=1000,
            height_px=1500,
            image_bytes=b"dummy-image",
        )

        assert res.source == "vlm"
        assert len(res.elements) >= 3
        # Banner headline
        banner_elements = [e for e in res.elements if e.block_type == BlockType.BANNER_HEADLINE]
        assert len(banner_elements) == 1
        assert banner_elements[0].text == "SENATE PASSES TARIFF REFORM"
        assert len(res.reading_order) >= 3

    def test_analyze_from_digital_text_blocks_rule_based(self) -> None:
        analyzer = LayoutAnalyzer()
        blocks = [
            DigitalTextBlock(
                block_id=0,
                text="MAJOR CITY DISPATCH",
                bbox=(50.0, 30.0, 950.0, 90.0),
                mean_font_size=24.0,
                is_heading_candidate=True,
            ),
            DigitalTextBlock(
                block_id=1,
                text="Column 1 story content...",
                bbox=(50.0, 100.0, 450.0, 600.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
            DigitalTextBlock(
                block_id=2,
                text="Column 2 story content...",
                bbox=(500.0, 100.0, 900.0, 600.0),
                mean_font_size=10.0,
                is_heading_candidate=False,
            ),
        ]

        res = analyzer.analyze_from_text_blocks(
            page_number=1,
            width_px=1000,
            height_px=1500,
            digital_blocks=blocks,
        )

        assert res.source == "spatial_rule_based"
        assert len(res.elements) == 3
        assert len(res.reading_order) == 3
        # First reading block should be the wide headline
        assert res.reading_order[0].block_type == BlockType.BANNER_HEADLINE
