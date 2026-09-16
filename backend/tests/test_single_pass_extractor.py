"""Unit tests for SinglePassVisualExtractor (single-pass Gemini visual intelligence)."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.ingestion.single_pass_extractor import (
    PageVisualAnalysis,
    RegionAnalysis,
    SinglePassVisualExtractor,
    VisualRegion,
    VisualRegionResult,
)
from app.providers.base import ModelResponse, VisionModelProvider


def create_blank_image_bytes(width: int = 1500, height: int = 2100, color: str = "white") -> bytes:
    """Create test PNG image bytes matching 150 DPI broadsheet canvas."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pre-Filtering Tests (Zero API Cost)
# ---------------------------------------------------------------------------


def test_pre_filter_regions_drops_small_and_thin_elements() -> None:
    """Verify that icons, bullets, and extreme aspect ratio rules are dropped locally."""
    extractor = SinglePassVisualExtractor()

    page_w = 1500
    page_h = 2100

    regions = [
        # 1. Tiny icon: 80x80px -> dropped (< 120px)
        VisualRegion(region_id="icon_1", bbox=(100, 100, 180, 180)),
        # 2. Thin divider line: 800x10px -> dropped (aspect = 80 > 12)
        VisualRegion(region_id="rule_1", bbox=(50, 500, 850, 510)),
        # 3. Full-page background canvas: 1450x2050px -> dropped (area >= 75%)
        VisualRegion(region_id="bg_1", bbox=(20, 20, 1470, 2070)),
        # 4. Tiny noise: 125x125px on 1500x2100 -> dropped (< 0.5% area ratio: 15625 / 3150000 = 0.0049)
        VisualRegion(region_id="tiny_noise", bbox=(10, 10, 135, 135)),
        # 5. Valid editorial photo: 500x400px -> kept (area = 200,000, 6.3% of page)
        VisualRegion(region_id="photo_1", bbox=(100, 200, 600, 600), caption_hint="Prime Minister addressing parliament"),
        # 6. Valid infographic/table: 700x500px -> kept (area = 350,000, 11.1% of page)
        VisualRegion(region_id="chart_1", bbox=(700, 800, 1400, 1300), caption_hint="GDP growth trends FY26"),
    ]

    filtered = extractor.pre_filter_regions(
        regions=regions,
        page_width_px=page_w,
        page_height_px=page_h,
    )

    assert len(filtered) == 2
    surviving_ids = [r.region_id for r in filtered]
    assert "photo_1" in surviving_ids
    assert "chart_1" in surviving_ids
    assert "icon_1" not in surviving_ids
    assert "rule_1" not in surviving_ids
    assert "bg_1" not in surviving_ids
    assert "tiny_noise" not in surviving_ids


# ---------------------------------------------------------------------------
# Manifest Construction Tests
# ---------------------------------------------------------------------------


def test_build_manifest_normalizes_coordinates() -> None:
    """Verify that bounding boxes are normalized to 0.0-1.0 scale with caption hints attached."""
    extractor = SinglePassVisualExtractor()
    page_w = 1500
    page_h = 2000

    regions = [
        VisualRegion(
            region_id="media_0",
            bbox=(150, 200, 750, 1000),
            caption_hint="Indigo aircraft on runway",
        ),
    ]

    manifest = extractor.build_manifest(
        regions=regions,
        page_width_px=page_w,
        page_height_px=page_h,
    )

    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["id"] == "media_0"
    assert entry["bbox_normalized"] == [0.1, 0.1, 0.5, 0.5]
    assert entry["caption_hint"] == "Indigo aircraft on runway"


# ---------------------------------------------------------------------------
# End-to-End Extraction with Mocked Gemini Provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_all_regions_single_call() -> None:
    """Verify single-pass extraction analyzes multiple regions in exactly 1 API call."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    mock_response_json = """{
        "regions": [
            {
                "id": "media_0",
                "visual_type": "photo",
                "description": "An IndiGo Airbus aircraft parked on the airport tarmac with ground crew attending to luggage.",
                "key_metrics": [],
                "confidence": 0.98
            },
            {
                "id": "media_1",
                "visual_type": "table",
                "description": "| Sector | Q1 Growth | Q2 Target |\\n|---|---|---|\\n| Aviation | 14.2% | 16.0% |",
                "key_metrics": ["Aviation Growth: 14.2%", "Q2 Target: 16.0%"],
                "confidence": 0.94
            }
        ]
    }"""
    mock_provider.analyze_image = AsyncMock(
        return_value=ModelResponse(
            text=mock_response_json,
            parsed={
                "regions": [
                    {
                        "id": "media_0",
                        "visual_type": "photo",
                        "description": "An IndiGo Airbus aircraft parked on the airport tarmac with ground crew attending to luggage.",
                        "key_metrics": [],
                        "confidence": 0.98,
                    },
                    {
                        "id": "media_1",
                        "visual_type": "table",
                        "description": "| Sector | Q1 Growth | Q2 Target |\\n|---|---|---|\\n| Aviation | 14.2% | 16.0% |",
                        "key_metrics": ["Aviation Growth: 14.2%", "Q2 Target: 16.0%"],
                        "confidence": 0.94,
                    },
                ]
            },
            input_tokens=2500,
            output_tokens=220,
        )
    )

    extractor = SinglePassVisualExtractor(provider=mock_provider)
    page_bytes = create_blank_image_bytes(width=1500, height=2100)

    regions = [
        VisualRegion(region_id="media_0", bbox=(100, 200, 600, 700), caption_hint="IndiGo flight at Delhi airport"),
        VisualRegion(region_id="media_1", bbox=(700, 800, 1400, 1400), caption_hint="Aviation sector statistics"),
    ]

    results = await extractor.extract_all_regions(
        page_image_bytes=page_bytes,
        regions=regions,
        page_width_px=1500,
        page_height_px=2100,
        page_number=1,
    )

    # Exactly 1 provider call was made for both regions
    assert mock_provider.analyze_image.await_count == 1
    assert len(results) == 2

    # Check photo region
    res0 = next(r for r in results if r.region_id == "media_0")
    assert res0.visual_type == "photo"
    assert "IndiGo Airbus" in res0.description
    assert res0.confidence == 0.98

    # Check table region
    res1 = next(r for r in results if r.region_id == "media_1")
    assert res1.visual_type == "table"
    assert "| Sector |" in res1.description
    assert "Aviation Growth: 14.2%" in res1.key_metrics


@pytest.mark.asyncio
async def test_extract_all_regions_skips_call_when_no_valid_regions() -> None:
    """Verify that if all regions are filtered out locally, 0 API calls are made."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    mock_provider.analyze_image = AsyncMock()

    extractor = SinglePassVisualExtractor(provider=mock_provider)
    page_bytes = create_blank_image_bytes()

    # Only small icons and dividers
    regions = [
        VisualRegion(region_id="icon", bbox=(10, 10, 50, 50)),
        VisualRegion(region_id="rule", bbox=(0, 100, 1000, 105)),
    ]

    results = await extractor.extract_all_regions(
        page_image_bytes=page_bytes,
        regions=regions,
        page_width_px=1500,
        page_height_px=2100,
    )

    assert results == []
    assert mock_provider.analyze_image.await_count == 0


@pytest.mark.asyncio
async def test_extract_all_regions_fallback_on_provider_error() -> None:
    """Verify that when Gemini call fails, extractor gracefully falls back to deterministic extraction."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    mock_provider.analyze_image = AsyncMock(side_effect=RuntimeError("Gemini API connection error"))

    extractor = SinglePassVisualExtractor(provider=mock_provider)
    page_bytes = create_blank_image_bytes(width=1500, height=2100)

    regions = [
        VisualRegion(region_id="media_0", bbox=(100, 200, 600, 700), caption_hint="Historic speech at Red Fort"),
    ]

    results = await extractor.extract_all_regions(
        page_image_bytes=page_bytes,
        regions=regions,
        page_width_px=1500,
        page_height_px=2100,
        page_number=1,
    )

    # The pipeline does NOT throw, it returns a fallback VisualRegionResult
    assert len(results) == 1
    assert results[0].region_id == "media_0"
    assert "Historic speech" in results[0].description


@pytest.mark.asyncio
async def test_extract_all_regions_handles_unbracketed_comma_separated_json() -> None:
    """Verify that comma-separated objects (Qwen3-VL pattern) parse cleanly without fallback."""
    mock_provider = MagicMock(spec=VisionModelProvider)
    # Output has no outer array brackets
    raw_unbracketed = """
    {
        "id": "media_0",
        "visual_type": "photo",
        "description": "IndiGo plane taxiing on runway.",
        "key_metrics": [],
        "confidence": 0.95
    },
    {
        "id": "media_1",
        "visual_type": "photo",
        "description": "Scientist analyzing test tubes in laboratory.",
        "key_metrics": [],
        "confidence": 0.92
    }
    """
    mock_provider.analyze_image = AsyncMock(
        return_value=ModelResponse(
            text=raw_unbracketed,
            parsed=None,  # Simulates provider failure before repair
            input_tokens=1500,
            output_tokens=180,
        )
    )

    extractor = SinglePassVisualExtractor(provider=mock_provider)
    page_bytes = create_blank_image_bytes(width=1500, height=2100)

    regions = [
        VisualRegion(region_id="media_0", bbox=(100, 200, 600, 700), caption_hint="IndiGo plane"),
        VisualRegion(region_id="media_1", bbox=(700, 800, 1400, 1400), caption_hint="Lab researcher"),
    ]

    results = await extractor.extract_all_regions(
        page_image_bytes=page_bytes,
        regions=regions,
        page_width_px=1500,
        page_height_px=2100,
        page_number=2,
    )

    assert len(results) == 2
    res0 = next(r for r in results if r.region_id == "media_0")
    res1 = next(r for r in results if r.region_id == "media_1")
    assert "IndiGo plane taxiing" in res0.description
    assert "Scientist analyzing test tubes" in res1.description


def test_resolve_photo_article_binding_prevents_giant_envelope_theft() -> None:
    """Verify that a giant envelope covering >40% page does not steal unrelated photos."""
    from app.ingestion.media_extractor import MediaExtractor

    extractor = MediaExtractor(db=MagicMock())

    # Giant article envelope (covers x=20..1100, y=250..1700, 75% of page)
    # But constituent blocks are only at y=250..400 and y=1400..1700
    giant_envelope = (
        43560,
        (20.0, 250.0, 1100.0, 1700.0),
        "In Gujarat, a parallel genomics push: Help elite athletes, tribals",
        [(20.0, 250.0, 500.0, 400.0), (600.0, 1400.0, 1100.0, 1700.0)],
    )

    # Local UPI payment article at x=50..450, y=700..1000
    upi_envelope = (
        43561,
        (50.0, 700.0, 450.0, 1000.0),
        "UPI digital payments hit new record across merchant stores",
        [(50.0, 700.0, 450.0, 1000.0)],
    )

    envelopes = [giant_envelope, upi_envelope]

    # 1. UPI Photo: located inside the UPI article's column space
    upi_photo_bbox = (80.0, 720.0, 380.0, 920.0)
    upi_caption = "Customer scans QR code to pay via UPI on mobile phone"

    bound_upi = extractor.resolve_photo_article_binding(
        photo_bbox=upi_photo_bbox,
        article_envelopes=envelopes,
        caption=upi_caption,
        page_width_px=1200.0,
        page_height_px=1800.0,
    )
    # Must bind to UPI article (43561), NOT stolen by giant article (43560)
    assert bound_upi == 43561

    # 2. Genomics Photo: located at y=1420..1650 near the genomics lab block
    genomics_photo_bbox = (650.0, 1420.0, 1000.0, 1620.0)
    genomics_caption = "Scientist in Gujarat genomics research laboratory analyzing DNA samples"

    bound_genomics = extractor.resolve_photo_article_binding(
        photo_bbox=genomics_photo_bbox,
        article_envelopes=envelopes,
        caption=genomics_caption,
        page_width_px=1200.0,
        page_height_px=1800.0,
    )
    # Must bind to genomics article (43560)
    assert bound_genomics == 43560


def test_resolve_photo_article_binding_generic_token_matching() -> None:
    """Verify that photo binding uses generic 4-character significant token matching without hardcoded company lists."""
    from unittest.mock import MagicMock
    from app.ingestion.media_extractor import MediaExtractor

    extractor = MediaExtractor(db=MagicMock())

    # Two distinct articles with non-hardcoded terms
    art1 = (
        1001,
        (50.0, 50.0, 500.0, 600.0),
        "Semiconductor Fabrication Facility Breaks Ground in Dholera",
        [(50.0, 50.0, 500.0, 600.0)],
    )
    art2 = (
        1002,
        (550.0, 50.0, 1000.0, 600.0),
        "Renewable Battery Storage Plant Inaugurated Near Khavda",
        [(550.0, 50.0, 1000.0, 600.0)],
    )
    envelopes = [art1, art2]

    # Photo positioned centrally between the two columns
    central_bbox = (480.0, 200.0, 580.0, 400.0)

    # Caption matching article 1 via significant tokens ("semiconductor", "fabrication")
    caption1 = "Engineers inspect semiconductor fabrication machinery at the site"
    bound1 = extractor.resolve_photo_article_binding(
        photo_bbox=central_bbox,
        article_envelopes=envelopes,
        caption=caption1,
        page_width_px=1200.0,
        page_height_px=1800.0,
    )
    assert bound1 == 1001

    # Caption matching article 2 via significant tokens ("renewable", "battery", "storage")
    caption2 = "Technicians test the renewable battery storage cells"
    bound2 = extractor.resolve_photo_article_binding(
        photo_bbox=central_bbox,
        article_envelopes=envelopes,
        caption=caption2,
        page_width_px=1200.0,
        page_height_px=1800.0,
    )
    assert bound2 == 1002

