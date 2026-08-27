"""Unit tests for Google Cloud Vision Pure OCR Provider."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from PIL import Image

from app.providers.base import OCREngine, OCRResult, ProviderError
from app.providers.google_vision_provider import GoogleCloudVisionOCR


def create_dummy_png_bytes(width: int = 1000, height: int = 2000) -> bytes:
    """Create in-memory PNG bytes for testing."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestGoogleCloudVisionOCR:
    """Tests for GoogleCloudVisionOCR engine."""

    def test_protocol_conformance(self) -> None:
        ocr = GoogleCloudVisionOCR(api_key="test-api-key")
        assert isinstance(ocr, OCREngine)
        assert ocr.provider_name == "google_cloud_vision"
        assert ocr.capability.supports_vision is True

    def test_raises_without_credentials(self) -> None:
        with (
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(ProviderError, match="requires a GCP Service Account key or API key"),
        ):
            GoogleCloudVisionOCR()

    @pytest.mark.asyncio
    async def test_google_cloud_vision_ocr_parsing(self) -> None:
        ocr = GoogleCloudVisionOCR(api_key="test-key")
        img_bytes = create_dummy_png_bytes(1000, 2000)

        mock_response_json = {
            "responses": [
                {
                    "fullTextAnnotation": {
                        "text": "RBI Keeps Repo Rate Steady\nInflation under control",
                        "pages": [
                            {
                                "blocks": [
                                    {
                                        "boundingBox": {
                                            "vertices": [
                                                {"x": 10, "y": 20},
                                                {"x": 400, "y": 20},
                                                {"x": 400, "y": 100},
                                                {"x": 10, "y": 100},
                                            ]
                                        },
                                        "confidence": 0.98,
                                        "paragraphs": [
                                            {
                                                "words": [
                                                    {"symbols": [{"text": "RBI"}, {"text": " "}]},
                                                    {"symbols": [{"text": "Keeps"}, {"text": " "}]},
                                                    {"symbols": [{"text": "Repo"}, {"text": " "}]},
                                                    {"symbols": [{"text": "Rate"}, {"text": " "}]},
                                                    {"symbols": [{"text": "Steady"}]},
                                                ]
                                            }
                                        ],
                                    }
                                ]
                            }
                        ],
                    }
                }
            ]
        }

        from unittest.mock import MagicMock
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response_json
            mock_post.return_value = mock_resp

            result = await ocr.ocr(img_bytes, lang_hint="en")

            assert isinstance(result, OCRResult)
            assert "RBI Keeps Repo Rate Steady" in result.full_text
            assert len(result.blocks) == 1
            assert result.blocks[0].bbox == (10.0, 20.0, 400.0, 100.0)
            assert result.blocks[0].confidence == 0.98

    @pytest.mark.asyncio
    async def test_google_cloud_vision_analyze_image(self) -> None:
        """Test analyze_image returns structured PageLayoutExtraction without external LLM."""
        from unittest.mock import MagicMock

        from app.providers.base import VisionModelProvider

        ocr = GoogleCloudVisionOCR(api_key="test-key")
        assert isinstance(ocr, VisionModelProvider)

        img_bytes = create_dummy_png_bytes(1000, 2000)

        mock_response_json = {
            "responses": [
                {
                    "fullTextAnnotation": {
                        "text": "RBI Keeps Repo Rate Steady\nInflation under control and growth robust.\n\nBy Rajesh Sharma\nMumbai",
                        "pages": [
                            {
                                "blocks": [
                                    {
                                        "boundingBox": {
                                            "vertices": [
                                                {"x": 50, "y": 50},
                                                {"x": 900, "y": 50},
                                                {"x": 900, "y": 150},
                                                {"x": 50, "y": 150},
                                            ]
                                        },
                                        "confidence": 0.99,
                                        "paragraphs": [
                                            {
                                                "words": [
                                                    {"symbols": [{"text": "RBI"}, {"text": " "}]},
                                                    {"symbols": [{"text": "Keeps"}, {"text": " "}]},
                                                    {"symbols": [{"text": "Repo"}, {"text": " "}]},
                                                    {"symbols": [{"text": "Rate"}, {"text": " "}]},
                                                    {"symbols": [{"text": "Steady"}]},
                                                ]
                                            }
                                        ],
                                    },
                                    {
                                        "boundingBox": {
                                            "vertices": [
                                                {"x": 50, "y": 160},
                                                {"x": 900, "y": 160},
                                                {"x": 900, "y": 400},
                                                {"x": 50, "y": 400},
                                            ]
                                        },
                                        "confidence": 0.95,
                                        "paragraphs": [
                                            {
                                                "words": [
                                                    {"symbols": [{"text": "Inflation"}, {"text": " "}]},
                                                    {"symbols": [{"text": "under"}, {"text": " "}]},
                                                    {"symbols": [{"text": "control"}]},
                                                ]
                                            }
                                        ],
                                    },
                                ]
                            }
                        ],
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response_json
            mock_post.return_value = mock_resp

            resp = await ocr.analyze_image(
                image_bytes=img_bytes,
                prompt="You are a high-precision broadsheet newspaper layout analyzer. Page Number: 1.",
            )

            assert resp.provider == "google_cloud_vision"
            assert resp.parsed is not None
            assert resp.parsed.get("page_number") == 1
            assert "articles" in resp.parsed
            assert len(resp.parsed["articles"]) >= 1
            assert resp.parsed["articles"][0]["headline"] != ""

