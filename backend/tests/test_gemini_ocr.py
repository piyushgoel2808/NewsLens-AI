"""Unit tests for Google Gemini OCR and DocumentLayoutProvider integration."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from app.providers.base import (
    DocumentLayoutProvider,
    MinerUParseResult,
    OCREngine,
    OCRResult,
)
from app.providers.gemini_provider import GeminiProvider, _normalize_box


def create_dummy_png_bytes(width: int = 1000, height: int = 2000) -> bytes:
    """Create in-memory PNG bytes for testing."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestGeminiOCRAndLayout:
    """Tests for GeminiProvider OCR and Document Layout capabilities."""

    def test_protocols_conformance(self) -> None:
        provider = GeminiProvider(model="gemini-3.7-flash", api_key="test-gemini-key")
        assert isinstance(provider, OCREngine)
        assert isinstance(provider, DocumentLayoutProvider)

    def test_normalize_box_scaling(self) -> None:
        # Test 0-1000 scale
        box_1000 = [100, 200, 300, 400]  # ymin, xmin, ymax, xmax
        norm_box = _normalize_box(box_1000, width_px=1000, height_px=2000)
        assert norm_box == (200.0, 200.0, 400.0, 600.0)

        # Test 0.0-1.0 relative scale
        box_rel = [0.1, 0.2, 0.3, 0.4]
        norm_rel = _normalize_box(box_rel, width_px=1000, height_px=2000)
        assert norm_rel == (200.0, 200.0, 400.0, 600.0)

    @pytest.mark.asyncio
    async def test_gemini_ocr_extraction(self) -> None:
        provider = GeminiProvider(model="gemini-3.7-flash", api_key="test-gemini-key")
        img_bytes = create_dummy_png_bytes(1000, 2000)

        mock_payload = {
            "blocks": [
                {
                    "text": "TATA POWER INVESTS IN ODISHA",
                    "box_2d": [50, 100, 100, 900],
                    "confidence": 0.99,
                    "language": "en",
                },
                {
                    "text": "Bhubaneswar: Tata Power today announced major plans...",
                    "box_2d": [120, 100, 400, 500],
                    "confidence": 0.95,
                    "language": "en",
                },
            ],
            "full_text": (
                "TATA POWER INVESTS IN ODISHA\n"
                "Bhubaneswar: Tata Power today announced major plans..."
            ),
            "mean_confidence": 0.97,
        }

        with patch.object(provider, "analyze_image", new_callable=AsyncMock) as mock_analyze:
            from app.providers.base import ModelResponse

            mock_analyze.return_value = ModelResponse(
                text="",
                parsed=mock_payload,
                model="gemini-3.7-flash",
                provider="gemini",
            )

            res = await provider.ocr(image_bytes=img_bytes, lang_hint="en")

            assert isinstance(res, OCRResult)
            assert len(res.blocks) == 2
            assert res.blocks[0].text == "TATA POWER INVESTS IN ODISHA"
            assert res.blocks[0].bbox == (100.0, 100.0, 900.0, 200.0)
            assert res.blocks[1].text.startswith("Bhubaneswar:")
            assert round(res.mean_confidence, 2) == 0.97

    @pytest.mark.asyncio
    async def test_gemini_parse_page_image_layout(self) -> None:
        provider = GeminiProvider(model="gemini-3.7-flash", api_key="test-gemini-key")
        img_bytes = create_dummy_png_bytes(1000, 2000)

        mock_layout = {
            "elements": [
                {
                    "node_type": "title",
                    "text": "MARKETS RALLY AS GDP RISES",
                    "box_2d": [50, 50, 100, 950],
                    "level": 1,
                },
                {
                    "node_type": "text",
                    "text": "Mumbai: Indian equity benchmark indices gained 1.5%...",
                    "box_2d": [110, 50, 400, 500],
                },
                {
                    "node_type": "table",
                    "text": "Top Gainers Table",
                    "box_2d": [420, 50, 600, 500],
                    "table_data": {
                        "headers": ["Stock", "Gain %"],
                        "rows": [["TCS", "+2.5%"], ["INFY", "+1.8%"]],
                        "markdown": (
                            "| Stock | Gain % |\n"
                            "|---|---|\n"
                            "| TCS | +2.5% |\n"
                            "| INFY | +1.8% |"
                        ),
                    },
                },
                {
                    "node_type": "image",
                    "text": "Stock market bell ceremony",
                    "box_2d": [110, 520, 300, 950],
                    "caption": "BSE CEO rings the opening bell.",
                },
            ],
            "full_markdown": (
                "# MARKETS RALLY AS GDP RISES\n\n"
                "Mumbai: Indian equity benchmark indices gained 1.5%..."
            ),
        }

        with patch.object(provider, "analyze_image", new_callable=AsyncMock) as mock_analyze:
            from app.providers.base import ModelResponse

            mock_analyze.return_value = ModelResponse(
                text="",
                parsed=mock_layout,
                model="gemini-3.7-flash",
                provider="gemini",
            )

            result = await provider.parse_page_image(image_bytes=img_bytes, page_number=1)

            assert isinstance(result, MinerUParseResult)
            assert result.page_number == 1
            assert len(result.nodes) == 4

            # Check title node
            assert result.nodes[0].node_type == "title"
            assert result.nodes[0].text == "MARKETS RALLY AS GDP RISES"
            assert result.nodes[0].level == 1

            # Check table node
            assert result.nodes[2].node_type == "table"
            assert result.nodes[2].table_data is not None
            assert result.nodes[2].table_data.headers == ["Stock", "Gain %"]
            assert len(result.nodes[2].table_data.rows) == 2

            # Check photo node
            assert result.nodes[3].node_type == "image"
            assert result.nodes[3].photo_data is not None
            assert result.nodes[3].photo_data.caption == "BSE CEO rings the opening bell."

    def test_clean_schema_for_gemini(self) -> None:
        """Verify _clean_schema_for_gemini dereferences $defs and removes invalid keywords."""
        from app.ingestion.extraction_schemas import PageLayoutExtraction
        from app.providers.gemini_provider import _clean_schema_for_gemini

        raw_schema = PageLayoutExtraction.model_json_schema()
        assert "$defs" in raw_schema

        cleaned = _clean_schema_for_gemini(raw_schema)
        assert "$defs" not in cleaned
        assert "$ref" not in str(cleaned)
        assert "properties" in cleaned
        assert "articles" in cleaned["properties"]
        assert "items" in cleaned["properties"]["articles"]
        assert "properties" in cleaned["properties"]["articles"]["items"]
        assert "headline" in cleaned["properties"]["articles"]["items"]["properties"]

