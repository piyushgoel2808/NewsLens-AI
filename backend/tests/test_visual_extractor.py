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


@pytest.mark.asyncio
async def test_visual_extractor_vlm_fallback_to_spatial_ocr(monkeypatch) -> None:
    """Verify that when VLM fails or returns empty, extract_structured_data falls back to spatial OCR."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    # VLM returns empty response
    mock_provider.analyze_image = AsyncMock(
        return_value=ModelResponse(text="", input_tokens=50, output_tokens=0)
    )

    extractor = VisualDataExtractor(vision_provider=mock_provider)
    test_bytes = create_test_image_bytes(width=400, height=300)

    # Mock pytesseract.image_to_data to simulate table structure
    fake_data = {
        "text": ["Measured", "Approach", "2023", "2024", "<1x", "15.0", "0.0", ">5x", "70.0", "68.0", "Source: Prime"],
        "conf": ["90", "90", "90", "90", "90", "90", "90", "90", "90", "90", "90"],
        "left": [20, 100, 150, 250, 20, 150, 250, 20, 150, 250, 20],
        "top": [20, 20, 60, 60, 100, 100, 100, 140, 140, 140, 220],
        "width": [60, 60, 40, 40, 30, 30, 30, 30, 30, 30, 80],
        "height": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 15],
    }
    monkeypatch.setattr("pytesseract.image_to_data", lambda *args, **kwargs: fake_data)

    result = await extractor.extract_structured_data(test_bytes, visual_type="table")
    assert result.confidence >= 0.80
    assert "Measured Approach" in result.summary
    assert "| 2023 | 2024 |" in result.markdown_table
    assert len(result.key_metrics) >= 1


@pytest.mark.asyncio
async def test_visual_extractor_classify_fallback_ocr_density(monkeypatch) -> None:
    """Verify that when VLM fails, triage classification uses OCR density heuristic."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    mock_provider.analyze_image = AsyncMock(side_effect=RuntimeError("VLM connection error"))

    extractor = VisualDataExtractor(vision_provider=mock_provider)
    test_bytes = create_test_image_bytes(width=400, height=300)

    # Mock pytesseract.image_to_string with numbers and lines
    ocr_table_text = "Metric 2023 2024 2025\n<1x 15.0 0.0 10.2\n>5x 70.0 68.0 63.3"
    monkeypatch.setattr("pytesseract.image_to_string", lambda *args, **kwargs: ocr_table_text)

    classification = await extractor.classify_visual_asset(test_bytes)
    assert classification.visual_type == "table"
    assert classification.contains_data is True


def test_repair_and_parse_json_utilities() -> None:
    """Verify repair_and_parse_json handles thought tags, code fences, and truncated JSON."""
    from app.ingestion.visual_extractor import (
        extract_markdown_table_from_raw_text,
        repair_and_parse_json,
    )

    # 1. Thought tags + markdown fences
    raw_thought = """<thought>
    I should extract the table from this image.
    </thought>
    ```json
    {
        "visual_type": "table",
        "contains_data": true,
        "confidence": 0.95
    }
    ```"""
    parsed = repair_and_parse_json(raw_thought)
    assert parsed is not None
    assert parsed["visual_type"] == "table"
    assert parsed["contains_data"] is True

    # 2. Truncated JSON without closing brackets
    truncated_json = '{"summary": "IPO metrics", "key_metrics": ["Retail: 38%", "QIB: 63%'
    parsed_trunc = repair_and_parse_json(truncated_json)
    assert parsed_trunc is not None
    assert parsed_trunc.get("summary") == "IPO metrics"

    # 3. Conversational Markdown table recovery
    conversational_text = """Here is the extracted table based on your request:

| Issue | 2024 | 2025 | 2026 |
|---|---|---|---|
| Mainboard | 50 | 42 | 42 |
| Retail Sub >5x | 68% | 63% | 38% |

This shows a decline in retail oversubscription."""
    table_md = extract_markdown_table_from_raw_text(conversational_text)
    assert table_md is not None
    assert "| Mainboard | 50 | 42 | 42 |" in table_md


@pytest.mark.asyncio
async def test_visual_extractor_recovers_conversational_markdown_table() -> None:
    """Verify that when VLM returns conversational text with a markdown table, it is parsed."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    conversational_vlm_out = """Certainly! Here is the table found in the infographic:

| Metric | FY25 | FY26 |
|---|---|---|
| Revenue | 1200 Cr | 1450 Cr |
| Profit | 150 Cr | 210 Cr |
"""
    mock_provider.analyze_image = AsyncMock(
        return_value=ModelResponse(text=conversational_vlm_out, input_tokens=100, output_tokens=80)
    )

    extractor = VisualDataExtractor(vision_provider=mock_provider)
    test_bytes = create_test_image_bytes(width=400, height=300)

    result = await extractor.extract_structured_data(test_bytes, visual_type="table")
    assert result.confidence >= 0.8
    assert "| Revenue | 1200 Cr | 1450 Cr |" in result.markdown_table


@pytest.mark.asyncio
async def test_describe_photo_scene_vlm_analysis() -> None:
    """Verify that describe_photo_scene generates detailed editorial scene descriptions."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    scene_json = """{
        "summary": "A mechanic is welding the chassis of a vintage sedan outside an automotive garage in Mexico.",
        "key_metrics": ["Subject: Mechanic welding chassis", "Vehicle: Classic yellow sedan", "Location: Ojinaga, Mexico"],
        "confidence": 0.95
    }"""
    mock_provider.analyze_image = AsyncMock(
        return_value=ModelResponse(text=scene_json, input_tokens=150, output_tokens=90)
    )

    extractor = VisualDataExtractor(vision_provider=mock_provider)
    test_bytes = create_test_image_bytes(width=500, height=400)

    result = await extractor.describe_photo_scene(test_bytes, caption="Welder working on car chassis in Mexico")
    assert result.visual_type == "photo"
    assert "welding the chassis" in result.summary
    assert len(result.key_metrics) == 3
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_process_image_crop_handles_editorial_photos() -> None:
    """Verify process_image_crop extracts rich VLM scene descriptions for editorial photos."""
    mock_provider = MagicMock(spec=VisionModelProvider)

    # 1. Triage returns photo
    triage_resp = ModelResponse(
        text='{"visual_type": "photo", "contains_data": false, "confidence": 0.9}',
        input_tokens=50,
        output_tokens=20,
    )
    # 2. Scene description returns details
    scene_resp = ModelResponse(
        text='{"summary": "Vehicles lined up along the border scrap yard.", "key_metrics": ["Vehicles: Scrapped cars", "Location: Southern California"]}',
        input_tokens=100,
        output_tokens=60,
    )
    mock_provider.analyze_image = AsyncMock(side_effect=[triage_resp, scene_resp])

    extractor = VisualDataExtractor(vision_provider=mock_provider)
    test_bytes = create_test_image_bytes(width=500, height=400)

    classification, extraction = await extractor.process_image_crop(
        test_bytes, ocr_text="Damaged cars filled a Southern California scrapyard"
    )
    assert classification.visual_type == "photo"
    assert extraction is not None
    assert "Vehicles lined up" in extraction.summary
    assert len(extraction.key_metrics) == 2



