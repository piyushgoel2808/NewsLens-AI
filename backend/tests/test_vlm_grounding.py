"""Unit tests for VLM Grounding Sweep fallback and coordinate translation in MediaExtractor."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.ingestion.media_extractor import MediaExtractor, parse_grounded_boxes
from app.models.article import Photo
from app.providers.base import ModelResponse


def test_parse_grounded_boxes_coordinate_translation() -> None:
    """Verify conversion from 1000x1000 normalized grid to absolute pixel space."""
    data = {
        "boxes": [
            {
                "label": "medicine capsules",
                "box_1000": [65, 498, 137, 722],  # [ymin, xmin, ymax, xmax]
            },
            {
                "label": "airplane",
                "box_1000": [65, 742, 137, 966],
            },
        ]
    }

    width_px, height_px = 2000, 3000
    results = parse_grounded_boxes(data, width_px=width_px, height_px=height_px)
    assert len(results) == 2

    box1, label1 = results[0]
    assert label1 == "medicine capsules"
    # x0 = (498 / 1000) * 2000 = 996.0
    # y0 = (65 / 1000) * 3000 = 195.0
    # x1 = (722 / 1000) * 2000 = 1444.0
    # y1 = (137 / 1000) * 3000 = 411.0
    assert box1[0] == pytest.approx(996.0, abs=1.0)
    assert box1[1] == pytest.approx(195.0, abs=1.0)
    assert box1[2] == pytest.approx(1444.0, abs=1.0)
    assert box1[3] == pytest.approx(411.0, abs=1.0)

    box2, label2 = results[1]
    assert label2 == "airplane"
    assert box2[0] == pytest.approx(1484.0, abs=1.0)
    assert box2[1] == pytest.approx(195.0, abs=1.0)
    assert box2[2] == pytest.approx(1932.0, abs=1.0)
    assert box2[3] == pytest.approx(411.0, abs=1.0)


def test_parse_grounded_boxes_area_filters() -> None:
    """Verify that speckles (<0.5% area) and monolithic backgrounds (>60% area) are filtered out."""
    data = {
        "boxes": [
            {
                "label": "tiny icon",
                "box_1000": [10, 10, 20, 20],  # 10x10 = 100 area units (0.01% of 1,000,000) -> should be filtered
            },
            {
                "label": "giant background canvas",
                "box_1000": [50, 50, 950, 950],  # 900x900 = 810,000 area units (81%) -> should be filtered
            },
            {
                "label": "valid photo",
                "box_1000": [100, 100, 300, 400],  # 200x300 = 60,000 area units (6%) -> valid
            },
        ]
    }

    results = parse_grounded_boxes(data, width_px=1000, height_px=1000)
    assert len(results) == 1
    assert results[0][1] == "valid photo"


@pytest.mark.asyncio
async def test_detect_subphotos_via_vlm_grounding_mock() -> None:
    """Test detect_subphotos_via_vlm_grounding with mocked VisionModelProvider."""
    mock_db = MagicMock()
    mock_minio = MagicMock()
    mock_vision_provider = MagicMock()

    json_response = """
    ```json
    {
      "boxes": [
        {"label": "portrait of CEO", "box_1000": [200, 100, 450, 400]},
        {"label": "quarterly revenue chart", "box_1000": [500, 550, 800, 900]}
      ]
    }
    ```
    """
    mock_vision_provider.analyze_image = AsyncMock(
        return_value=ModelResponse(text=json_response, model="qwen3-vl")
    )

    mock_visual_extractor = MagicMock()
    mock_visual_extractor._get_provider.return_value = mock_vision_provider

    extractor = MediaExtractor(
        db=mock_db,
        minio=mock_minio,
        visual_extractor=mock_visual_extractor,
    )

    # Generate small test image
    img = Image.new("RGB", (1000, 1000), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    grounded = await extractor.detect_subphotos_via_vlm_grounding(
        page_image_bytes=img_bytes,
        width_px=1000,
        height_px=1000,
    )

    assert len(grounded) == 2
    assert grounded[0][1] == "portrait of CEO"
    assert grounded[1][1] == "quarterly revenue chart"
    # Check bounding box values [xmin, ymin, xmax, ymax]
    assert grounded[0][0] == (200.0, 100.0, 450.0, 400.0)
    assert grounded[1][0] == (500.0, 550.0, 800.0, 900.0)


def test_extract_grounded_boxes_from_thinking() -> None:
    """Verify parsing itemized bounding boxes from native Qwen-VL reasoning traces."""
    from app.ingestion.media_extractor import extract_grounded_boxes_from_thinking

    thinking_text = """
    Let's identify the photo elements on this broadsheet page:
    1. Pills image (top right): [500, 45, 730, 155]
    2. IndiGo plane image: [750, 70, 970, 135]
    - The orange car: [230, 820, 610, 945]
    - Eiffel Tower with Olympic rings: [200, 620, 320, 745]
    - Wind turbine: [230, 360, 320, 435]
    """

    boxes = extract_grounded_boxes_from_thinking(thinking_text, width_px=2000, height_px=3000)
    assert len(boxes) == 5

    labels = [b[1] for b in boxes]
    assert "Pills image" in labels
    assert "IndiGo plane image" in labels
    assert "orange car" in labels
    assert "Eiffel Tower with Olympic rings" in labels
    assert "Wind turbine" in labels

    # Check pills box scaling
    pills_box = next(b[0] for b in boxes if b[1] == "Pills image")
    # x0 = 500/1000 * 2000 = 1000.0
    # y0 = 45/1000 * 3000 = 135.0
    # x1 = 730/1000 * 2000 = 1460.0
    # y1 = 155/1000 * 3000 = 465.0
    assert pills_box[0] == pytest.approx(1000.0, abs=1.0)
    assert pills_box[1] == pytest.approx(135.0, abs=1.0)
    assert pills_box[2] == pytest.approx(1460.0, abs=1.0)
    assert pills_box[3] == pytest.approx(465.0, abs=1.0)


@pytest.mark.asyncio
async def test_extract_subphotos_vlm_fallback_orchestration() -> None:
    """Verify extract_subphotos_vlm_fallback calls grounding, binds to article, and stores photo."""
    mock_db = MagicMock()
    mock_minio = MagicMock()
    mock_minio.put = AsyncMock()

    extractor = MediaExtractor(db=mock_db, minio=mock_minio)

    # Mock detect_subphotos_via_vlm_grounding
    mock_boxes = [
        ((1355.0, 280.0, 1965.0, 590.0), "medicine capsules"),
        ((2019.0, 280.0, 2629.0, 590.0), "airplane"),
    ]
    extractor.detect_subphotos_via_vlm_grounding = AsyncMock(return_value=mock_boxes)

    # Mock article envelopes
    article_envelopes = [
        (42, (1360.0, 577.0, 1980.0, 680.0), "Bulk drug exporters fret as China tightens screws"),
    ]

    # Mock extract_and_store_photo
    dummy_photo = MagicMock(spec=Photo)
    dummy_photo.id = 99
    extractor.extract_and_store_photo = AsyncMock(return_value=dummy_photo)

    img = Image.new("RGB", (2722, 4307), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    persisted = await extractor.extract_subphotos_vlm_fallback(
        page_image_bytes=img_bytes,
        page_id=10,
        article_envelopes=article_envelopes,
        width_px=2722,
        height_px=4307,
        start_photo_index=1,
    )

    assert len(persisted) == 2
    # Verify binding to article 42
    assert persisted[0][0] == 42
    assert persisted[0][1] == dummy_photo
    assert extractor.extract_and_store_photo.call_count == 2
