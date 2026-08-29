"""Tests for Visual Data Extractor and 3-stage visual intelligence pipeline."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.ingestion.chunker import NewspaperChunker
from app.ingestion.visual_extractor import (
    VisualClassification,
    VisualDataExtractor,
    VisualExtractionResult,
)
from app.providers.base import ModelResponse, VisionModelProvider


def create_test_image_bytes(width: int = 200, height: int = 200, color: str = "white") -> bytes:
    """Create test PNG image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_visual_extractor_heuristics_filter_small_icons() -> None:
    """Verify that tiny icons or extreme aspect ratio lines are filtered by PIL heuristics."""
    extractor = VisualDataExtractor()
    small_bytes = create_test_image_bytes(width=30, height=30)
    assert extractor.is_candidate_data_image(small_bytes, min_dim=80) is False

    thin_line_bytes = create_test_image_bytes(width=600, height=10)
    assert extractor.is_candidate_data_image(thin_line_bytes) is False

    chart_box_bytes = create_test_image_bytes(width=400, height=300)
    assert extractor.is_candidate_data_image(chart_box_bytes) is True


@pytest.mark.asyncio
async def test_visual_extractor_triage_classification() -> None:
    """Verify that Stage 1 Triage classifies visual assets using VisionModelProvider."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    mock_provider.analyze_image = AsyncMock(
        return_value=ModelResponse(
            text='{"visual_type": "data_chart", "contains_data": true, "confidence": 0.95}',
            input_tokens=100,
            output_tokens=30,
        )
    )

    extractor = VisualDataExtractor(vision_provider=mock_provider)
    test_bytes = create_test_image_bytes(width=400, height=300)

    classification = await extractor.classify_visual_asset(test_bytes)
    assert classification.visual_type == "data_chart"
    assert classification.contains_data is True
    assert classification.confidence == 0.95


@pytest.mark.asyncio
async def test_visual_extractor_structured_extraction() -> None:
    """Verify that Stage 2 extracts markdown table and key metrics."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    mock_response_json = """{
        "summary": "Tata Power revenue grew 18.3% YoY in Q1 FY26.",
        "markdown_table": "| Quarter | Revenue (Cr) | Net Profit (Cr) |\\n|---|---|---|\\n| Q1 FY26 | 12450 | 1140 |",
        "key_metrics": ["Revenue: ₹12,450 Cr (+18.3% YoY)", "Net Profit: ₹1,140 Cr"],
        "confidence": 0.92
    }"""
    mock_provider.analyze_image = AsyncMock(
        return_value=ModelResponse(
            text=mock_response_json,
            input_tokens=300,
            output_tokens=120,
        )
    )

    extractor = VisualDataExtractor(vision_provider=mock_provider)
    test_bytes = create_test_image_bytes(width=500, height=400)

    result = await extractor.extract_structured_data(test_bytes, visual_type="data_chart")
    assert "Tata Power revenue" in result.summary
    assert "| Q1 FY26 | 12450 | 1140 |" in result.markdown_table
    assert len(result.key_metrics) == 2
    assert result.confidence == 0.92


def test_visual_extractor_ocr_cross_validation() -> None:
    """Verify Stage 3 cross-validation calculates confidence based on OCR token overlap."""
    extractor = VisualDataExtractor()
    extraction = VisualExtractionResult(
        summary="Quarterly financial metrics.",
        markdown_table="| Metric | Value |\\n| Revenue | 12450 |\\n| Profit | 1140 |\\n| Growth | 18.3% |",
        key_metrics=["12450", "1140", "18.3%"],
        confidence=0.9,
    )

    # Perfect match scenario
    ocr_text = "Revenue was 12450 cr with profit of 1140 cr and growth of 18.3%."
    score = extractor.cross_validate_with_ocr(extraction, ocr_text)
    assert score >= 0.85

    # Partial / mismatch scenario
    mismatch_ocr = "Stock closed at 540 on Tuesday."
    low_score = extractor.cross_validate_with_ocr(extraction, mismatch_ocr)
    assert low_score < 0.60


def test_chunker_create_visual_chunk() -> None:
    """Verify NewspaperChunker creates dedicated, unfragmented visual chunks with correct header."""
    chunker = NewspaperChunker()
    markdown = "| Quarter | Revenue |\\n| Q1 | 12450 |"
    chunk = chunker.create_visual_chunk(
        visual_markdown=markdown,
        visual_type="data_chart",
        summary="Quarterly revenue breakdown",
        newspaper_name="Mint",
        issue_date="2026-08-28",
        headline="Tata Power Q1 Results",
        section="Corporate & Industry",
        pages=[1, 4],
        printed_pages=["1", "4"],
        chunk_index=3,
    )

    assert chunk.chunk_index == 3
    assert "[Newspaper: Mint" in chunk.text
    assert "[Visual Data Asset: Data Chart]" in chunk.text
    assert "[Visual Summary: Quarterly revenue breakdown]" in chunk.text
    assert "| Quarter | Revenue |" in chunk.text
