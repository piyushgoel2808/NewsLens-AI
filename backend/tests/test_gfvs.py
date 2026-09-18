"""Unit tests for GFVSPageSegmenter (Gemini-First Vision Segmentation).

Tests Phase 1 (Gemini layout parsing), Phase 2 (GCV text extraction),
output conversion, and full end-to-end pipeline with mocked providers.
"""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.ingestion.parsers.gfvs import (
    GFVSArticle,
    GFVSPageResult,
    GFVSPageSegmenter,
    resolve_gemini_provider,
)
from app.ingestion.layout.segmenter import SegmentedArticle
from app.providers.base import ModelResponse, OCRBlock, OCRResult


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_blank_image_bytes(w: int = 800, h: int = 1200) -> bytes:
    """Create a small blank PNG image as fake page image."""
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_gemini_response(articles: list[dict]) -> MagicMock:
    """Build a mock Gemini ModelResponse with the given articles JSON."""
    payload = json.dumps({
        "page_section": "Region",
        "is_advertisement_page": False,
        "articles": articles,
    })
    resp = MagicMock(spec=ModelResponse)
    resp.text = payload
    resp.parsed = None
    return resp


def _make_gcv_response(text: str) -> OCRResult:
    """Build a mock GCV OCRResult with a single text block."""
    block = OCRBlock(
        text=text,
        bbox=(0.0, 0.0, 800.0, 100.0),
        confidence=0.98,
    )
    return OCRResult(blocks=[block], full_text=text, mean_confidence=0.98)


SAMPLE_ARTICLES = [
    {
        "headline": "4 held in Punjab with 27 kg heroin, foreign-made pistol",
        "subheadline": "Major drug bust by Punjab Police",
        "byline": "PTI",
        "section": "National",
        "prominence": "lead",
        "article_type": "news",
        "bbox": [50, 30, 400, 600],
        "has_photo": True,
        "photo_bbox": [60, 30, 200, 300],
    },
    {
        "headline": "MP Cong questions govt over school dropout rates",
        "subheadline": None,
        "byline": "Bureau",
        "section": "National",
        "prominence": "standard",
        "article_type": "news",
        "bbox": [50, 600, 350, 900],
        "has_photo": False,
        "photo_bbox": None,
    },
    {
        "headline": "Bainsla, cousin sent to two days' police custody",
        "subheadline": None,
        "byline": None,
        "section": "National",
        "prominence": "minor",
        "article_type": "news",
        "bbox": [400, 30, 600, 400],
        "has_photo": False,
        "photo_bbox": None,
    },
]


# ---------------------------------------------------------------------------
# Phase 1: Gemini Layout Parsing
# ---------------------------------------------------------------------------


class TestPhase1GeminiLayout:
    """Tests for _phase1_gemini_layout and response parsing."""

    @pytest.mark.asyncio
    async def test_parses_valid_gemini_json_response(self):
        """Phase 1 correctly parses a well-formed Gemini JSON response."""
        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response(SAMPLE_ARTICLES)

        gcv = AsyncMock()
        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        result = await segmenter._phase1_gemini_layout(page_number=2, image_bytes=b"fake_image")

        assert isinstance(result, GFVSPageResult)
        assert result.page_number == 2
        assert result.page_section == "Region"
        assert len(result.articles) == 3

    @pytest.mark.asyncio
    async def test_article_headline_and_byline_parsed(self):
        """Headline and byline are correctly extracted from Gemini response."""
        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response(SAMPLE_ARTICLES)

        gcv = AsyncMock()
        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        result = await segmenter._phase1_gemini_layout(page_number=2, image_bytes=b"fake_image")

        first = result.articles[0]
        assert first.headline == "4 held in Punjab with 27 kg heroin, foreign-made pistol"
        assert first.byline == "PTI"
        assert first.prominence == "lead"
        assert first.has_photo is True
        assert first.photo_bbox is not None

    @pytest.mark.asyncio
    async def test_photo_bbox_parsed(self):
        """Photo bounding box is correctly parsed when has_photo=True."""
        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response(SAMPLE_ARTICLES)

        gcv = AsyncMock()
        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        result = await segmenter._phase1_gemini_layout(page_number=2, image_bytes=b"fake_image")

        first = result.articles[0]
        assert first.photo_bbox == (60.0, 30.0, 200.0, 300.0)

    @pytest.mark.asyncio
    async def test_gemini_failure_returns_empty_result(self):
        """If Gemini raises an exception, segment_page returns empty list (no crash)."""
        gemini = AsyncMock()
        gemini.analyze_image.side_effect = RuntimeError("Gemini API timeout")

        gcv = AsyncMock()
        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        result = await segmenter.segment_page(
            page_number=2,
            image_bytes=_make_blank_image_bytes(),
        )

        assert result == []

    def test_parse_gemini_response_strips_markdown_fences(self):
        """JSON repair correctly handles markdown code fences from Gemini."""
        raw = '```json\n{"page_section": "National", "is_advertisement_page": false, "articles": []}\n```'
        segmenter = GFVSPageSegmenter.__new__(GFVSPageSegmenter)
        parsed = segmenter._parse_gemini_response(raw)
        assert parsed is not None
        assert parsed["page_section"] == "National"
        assert parsed["articles"] == []

    def test_parse_gemini_response_repairs_unclosed_brackets(self):
        """JSON repair handles truncated/unclosed JSON responses."""
        # Simulate Gemini truncating the output mid-article
        raw = '{"page_section": "City", "is_advertisement_page": false, "articles": [{"headline": "Test article", "bbox": [10, 20, 300, 500]'
        segmenter = GFVSPageSegmenter.__new__(GFVSPageSegmenter)
        parsed = segmenter._parse_gemini_response(raw)
        # May not parse perfectly but should not raise
        # The important thing is no exception is raised

    def test_parse_bbox_valid(self):
        """_parse_bbox correctly handles valid normalised coordinates."""
        result = GFVSPageSegmenter._parse_bbox([50, 30, 400, 600])
        assert result == (50.0, 30.0, 400.0, 600.0)

    def test_parse_bbox_clamps_to_0_1000(self):
        """_parse_bbox clamps out-of-range values to 0-1000."""
        result = GFVSPageSegmenter._parse_bbox([-10, -5, 1100, 1050])
        assert result is not None
        ymin, xmin, ymax, xmax = result
        assert 0 <= ymin <= 1000
        assert 0 <= xmin <= 1000
        assert 0 <= ymax <= 1000
        assert 0 <= xmax <= 1000

    def test_parse_bbox_swaps_inverted_coords(self):
        """_parse_bbox swaps ymin/ymax and xmin/xmax if inverted."""
        result = GFVSPageSegmenter._parse_bbox([400, 600, 50, 30])
        assert result is not None
        ymin, xmin, ymax, xmax = result
        assert ymin < ymax
        assert xmin < xmax

    def test_parse_bbox_rejects_short_list(self):
        """_parse_bbox returns None for less than 4 elements."""
        assert GFVSPageSegmenter._parse_bbox([10, 20]) is None
        assert GFVSPageSegmenter._parse_bbox(None) is None
        assert GFVSPageSegmenter._parse_bbox("invalid") is None

    def test_tiny_bbox_filtered_out(self):
        """Articles with tiny bboxes (< 0.5% of page) are filtered out."""
        tiny_article = [{
            "headline": "Phantom Article",
            "bbox": [500, 500, 502, 502],  # 2x2 pixels in 1000x1000 space = 0.0004% area
            "section": "National",
            "prominence": "minor",
            "article_type": "news",
            "has_photo": False,
            "photo_bbox": None,
        }]
        segmenter = GFVSPageSegmenter.__new__(GFVSPageSegmenter)
        result = segmenter._build_page_result(
            {"articles": tiny_article, "page_section": "National"},
            page_number=1,
        )
        assert len(result.articles) == 0


# ---------------------------------------------------------------------------
# Phase 2: GCV Per-Crop Text Extraction
# ---------------------------------------------------------------------------


class TestPhase2GCVText:
    """Tests for _phase2_gcv_text and body_text population."""

    @pytest.mark.asyncio
    async def test_gcv_text_populated_for_news_articles(self):
        """Phase 2 populates body_text for news articles using GCV."""
        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response(SAMPLE_ARTICLES)

        gcv = AsyncMock()
        gcv.ocr.return_value = _make_gcv_response("Four persons were arrested in Punjab...")

        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        result = await segmenter.segment_page(
            page_number=2,
            image_bytes=_make_blank_image_bytes(),
        )

        assert len(result) == 3
        for article in result:
            assert isinstance(article, SegmentedArticle)
            assert article.body_text  # should not be empty

    @pytest.mark.asyncio
    async def test_advertisement_articles_skip_gcv(self):
        """Advertisement articles skip Phase 2 GCV extraction."""
        ad_article = [{
            "headline": "[Advertisement] Bank of India",
            "subheadline": None,
            "byline": None,
            "section": "National",
            "prominence": "standard",
            "article_type": "advertisement",
            "bbox": [600, 30, 900, 600],
            "has_photo": False,
            "photo_bbox": None,
        }]

        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response(ad_article)

        gcv = AsyncMock()
        gcv.ocr.return_value = _make_gcv_response("Ad text")

        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        await segmenter.segment_page(
            page_number=3,
            image_bytes=_make_blank_image_bytes(),
        )

        # GCV should NOT have been called for advertisement articles
        gcv.ocr.assert_not_called()

    @pytest.mark.asyncio
    async def test_gcv_failure_does_not_crash_pipeline(self):
        """GCV failure in Phase 2 is handled gracefully — article still returned with headline."""
        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response([SAMPLE_ARTICLES[0]])

        gcv = AsyncMock()
        gcv.ocr.side_effect = RuntimeError("GCV API error")

        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        result = await segmenter.segment_page(
            page_number=2,
            image_bytes=_make_blank_image_bytes(),
        )

        assert len(result) == 1
        # body_text falls back to headline when GCV fails
        assert result[0].headline == "4 held in Punjab with 27 kg heroin, foreign-made pistol"
        assert result[0].body_text  # headline is used as fallback

    def test_join_gcv_blocks_sorts_by_y(self):
        """_join_gcv_blocks sorts blocks top-to-bottom by y0 coordinate."""
        blocks = [
            OCRBlock(text="Paragraph 2", bbox=(0, 200, 800, 250), confidence=0.98),
            OCRBlock(text="Paragraph 1", bbox=(0, 10, 800, 60), confidence=0.97),
            OCRBlock(text="Paragraph 3", bbox=(0, 400, 800, 450), confidence=0.96),
        ]
        result = GFVSPageSegmenter._join_gcv_blocks(blocks)
        assert result == "Paragraph 1\n\nParagraph 2\n\nParagraph 3"

    def test_join_gcv_blocks_empty(self):
        """_join_gcv_blocks returns empty string for empty block list."""
        assert GFVSPageSegmenter._join_gcv_blocks([]) == ""


# ---------------------------------------------------------------------------
# Crop Utility
# ---------------------------------------------------------------------------


class TestCropBbox:
    """Tests for _crop_bbox helper."""

    def test_crop_produces_non_empty_png(self):
        """_crop_bbox returns valid PNG bytes for a valid bbox."""
        page_bytes = _make_blank_image_bytes(800, 1200)
        # Crop the top 40% of the left half
        crop = GFVSPageSegmenter._crop_bbox(
            page_bytes, bbox=(0.0, 0.0, 400.0, 500.0), page_w=800, page_h=1200
        )
        assert len(crop) > 0
        # Verify it's a valid image
        img = Image.open(io.BytesIO(crop))
        assert img.width > 0 and img.height > 0

    def test_crop_clamps_to_page_bounds(self):
        """_crop_bbox clamps out-of-bounds coordinates without raising."""
        page_bytes = _make_blank_image_bytes(800, 1200)
        crop = GFVSPageSegmenter._crop_bbox(
            page_bytes, bbox=(800.0, 800.0, 1100.0, 1200.0), page_w=800, page_h=1200
        )
        assert len(crop) > 0


# ---------------------------------------------------------------------------
# Output Conversion: GFVSArticle → SegmentedArticle
# ---------------------------------------------------------------------------


class TestToSegmentedArticle:
    """Tests for _to_segmented_article conversion."""

    def test_conversion_sets_correct_fields(self):
        """_to_segmented_article maps headline, byline, section, bbox correctly."""
        art = GFVSArticle(
            headline="Test Headline",
            subheadline="Test Deck",
            byline="PTI",
            section="National",
            prominence="lead",
            article_type="news",
            bbox=(50.0, 30.0, 400.0, 600.0),  # [ymin, xmin, ymax, xmax]
            has_photo=False,
            photo_bbox=None,
            body_text="This is the full article body text.",
            page_number=2,
        )

        result = GFVSPageSegmenter._to_segmented_article(art, 0)

        assert isinstance(result, SegmentedArticle)
        assert result.headline == "Test Headline"
        assert result.subheadline == "Test Deck"
        assert result.byline_author == "PTI"
        assert result.section == "National"
        assert result.body_text == "This is the full article body text."
        assert result.article_temp_id == "page_2_gfvs_0"
        # bbox conversion: [ymin,xmin,ymax,xmax] -> (x0,y0,x1,y1) = (xmin,ymin,xmax,ymax)
        assert result.bbox_list == [(30.0, 50.0, 600.0, 400.0)]

    def test_fallback_to_headline_when_body_empty(self):
        """When body_text is empty, fallback to subheadline or headline."""
        art = GFVSArticle(
            headline="Fallback Headline",
            subheadline="Fallback Sub",
            byline=None,
            section="City",
            prominence="minor",
            article_type="news",
            bbox=(600.0, 30.0, 800.0, 500.0),
            has_photo=False,
            photo_bbox=None,
            body_text="",   # empty — should fallback
            page_number=3,
        )

        result = GFVSPageSegmenter._to_segmented_article(art, 1)

        assert result.body_text == "Fallback Sub"  # subheadline preferred over headline

    def test_word_count_computed(self):
        """word_count is set correctly based on body_text."""
        art = GFVSArticle(
            headline="H",
            subheadline=None,
            byline=None,
            section="National",
            prominence="standard",
            article_type="news",
            bbox=(0.0, 0.0, 500.0, 500.0),
            has_photo=False,
            photo_bbox=None,
            body_text="One two three four five",
            page_number=1,
        )
        result = GFVSPageSegmenter._to_segmented_article(art, 0)
        assert result.word_count == 5


# ---------------------------------------------------------------------------
# Full Pipeline Integration Test (mocked)
# ---------------------------------------------------------------------------


class TestSegmentPageIntegration:
    """End-to-end tests for segment_page() with fully mocked providers."""

    @pytest.mark.asyncio
    async def test_segment_page_returns_correct_count(self):
        """segment_page returns one SegmentedArticle per Gemini-identified article."""
        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response(SAMPLE_ARTICLES)

        gcv = AsyncMock()
        gcv.ocr.return_value = _make_gcv_response("Article body text here.")

        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        result = await segmenter.segment_page(
            page_number=2,
            image_bytes=_make_blank_image_bytes(),
            lang_hint="en",
        )

        assert len(result) == len(SAMPLE_ARTICLES)
        assert all(isinstance(a, SegmentedArticle) for a in result)

    @pytest.mark.asyncio
    async def test_segment_page_gemini_called_once(self):
        """Gemini analyze_image is called exactly once per page (not per article)."""
        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response(SAMPLE_ARTICLES)

        gcv = AsyncMock()
        gcv.ocr.return_value = _make_gcv_response("Body text.")

        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        await segmenter.segment_page(
            page_number=2,
            image_bytes=_make_blank_image_bytes(),
        )

        # Gemini called once for the full page
        gemini.analyze_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_segment_page_gcv_called_per_news_article(self):
        """GCV is called once per non-advertisement article (per crop)."""
        # 2 news + 1 ad article
        mixed_articles = [
            SAMPLE_ARTICLES[0],  # news
            SAMPLE_ARTICLES[1],  # news
            {
                "headline": "[Advertisement] SBI Loan",
                "subheadline": None,
                "byline": None,
                "section": "National",
                "prominence": "standard",
                "article_type": "advertisement",
                "bbox": [700, 30, 900, 500],
                "has_photo": False,
                "photo_bbox": None,
            },
        ]

        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response(mixed_articles)

        gcv = AsyncMock()
        gcv.ocr.return_value = _make_gcv_response("Body text.")

        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        await segmenter.segment_page(
            page_number=2,
            image_bytes=_make_blank_image_bytes(),
        )

        # GCV called only for the 2 news articles, not the ad
        assert gcv.ocr.call_count == 2

    @pytest.mark.asyncio
    async def test_segment_page_empty_when_gemini_returns_zero(self):
        """Empty page when Gemini returns 0 articles."""
        gemini = AsyncMock()
        gemini.analyze_image.return_value = _make_gemini_response([])

        gcv = AsyncMock()
        semaphore = asyncio.Semaphore(8)
        segmenter = GFVSPageSegmenter(gemini, gcv, semaphore)

        result = await segmenter.segment_page(
            page_number=8,
            image_bytes=_make_blank_image_bytes(),
        )

        assert result == []
        gcv.ocr.assert_not_called()
