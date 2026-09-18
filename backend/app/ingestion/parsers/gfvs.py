"""Gemini-First Vision Segmentation (GFVS) for scanned newspaper pages.

Two-phase pipeline that solves article segmentation on scanned broadsheets:

  Phase 1 — Gemini Flash Vision (~2-4s per page):
    Sends the FULL page image to Gemini Flash.
    Gemini identifies all article boundaries using visual understanding:
    headline font weight, column layout, section headers, photo regions.
    Returns structured JSON with article bboxes + metadata.

  Phase 2 — Google Cloud Vision OCR per-crop (~0.3-0.5s per article):
    Crops each article region from the page image.
    Sends each crop to GCV for high-accuracy verbatim OCR.
    Per-crop OCR is accurate because GCV only sees a narrow 2-3 column
    strip where reading order is trivially correct.

Why GCV alone fails: GCV returns ~200 raw paragraph blocks on a full
broadsheet page with no understanding of article boundaries or column
ownership. The LayoutAnalyzer's statistical heuristics (1.25x median
line-height for headline detection) collapse on scanned images due to
scan noise and ink variation.

Why Gemini alone is not enough: Gemini's verbatim OCR (~92-95%) is less
accurate than GCV's (~98%) for long body text. Gemini for layout
understanding + GCV for text extraction gives us the best of both.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.core.logging import get_logger
from app.ingestion.layout.segmenter import SegmentedArticle
from app.providers.base import OCREngine, VisionModelProvider

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Gemini Phase 1 Prompt — Newspaper Layout Segmentation
# ---------------------------------------------------------------------------

GFVS_LAYOUT_PROMPT = """You are an expert newspaper page layout analyst. This is a scanned broadsheet newspaper page image.

YOUR TASK: Identify EVERY discrete news article, advertisement, and brief on this page.

CRITICAL RULES:
1. A newspaper page contains MULTIPLE separate articles. Each headline you can see begins a separate article. Do NOT merge separate articles.
2. Each article has a bounding box covering its headline + all body-text columns that belong to it. Draw the box tightly around that article only.
3. Multi-column articles: one article body text may span 2-4 columns below and beside its headline. Include all those columns in the bbox.
4. Small briefs, capsules, and "News Briefs" items (1-3 paragraphs each) are SEPARATE articles — list each one individually.
5. Photo-only regions (no body text) should be recorded as photo_bbox inside the nearest article. Do NOT list a photo alone as its own article.
6. Advertisements and sponsored content are separate items with article_type "advertisement".

For each article extract:
- headline: verbatim text exactly as printed (copy every word, preserve capitalisation)
- subheadline: deck or kicker text below the headline if present, otherwise null
- byline: reporter name or agency (PTI, ANI, Reuters, Bureau, Our Correspondent) if visible, otherwise null
- section: the newspaper section for this page (e.g. "National", "City", "Region", "Sports", "Business", "Opinion")
- prominence: "lead" (main story), "major" (multi-column), "standard" (normal), "minor" (brief), "filler" (1-2 sentence)
- article_type: "news", "advertisement", "opinion", "editorial", "notice", or "sidebar"
- bbox: bounding box [ymin, xmin, ymax, xmax] normalised 0-1000, where (0,0) is top-left and (1000,1000) is bottom-right
- has_photo: true if a photo or image appears inside or directly adjacent to this article
- photo_bbox: bounding box of JUST the photo as [ymin, xmin, ymax, xmax] 0-1000, or null if has_photo is false

Return ONLY valid JSON, no explanation:
{
  "page_section": "string or null",
  "is_advertisement_page": false,
  "articles": [
    {
      "headline": "string",
      "subheadline": "string or null",
      "byline": "string or null",
      "section": "string",
      "prominence": "lead|major|standard|minor|filler",
      "article_type": "news|advertisement|opinion|editorial|notice|sidebar",
      "bbox": [ymin, xmin, ymax, xmax],
      "has_photo": false,
      "photo_bbox": null
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Internal Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GFVSArticle:
    """Single article identified by the GFVS pipeline (before body text extraction)."""

    headline: str
    subheadline: str | None
    byline: str | None
    section: str
    prominence: str          # lead / major / standard / minor / filler
    article_type: str        # news / advertisement / opinion / editorial / notice / sidebar
    bbox: tuple[float, float, float, float]   # [ymin, xmin, ymax, xmax] normalised 0-1000
    has_photo: bool
    photo_bbox: tuple[float, float, float, float] | None
    body_text: str = ""      # populated in Phase 2 by GCV
    page_number: int = 1


@dataclass
class GFVSPageResult:
    """Complete result of running GFVS Phase 1 on one page."""

    page_number: int
    articles: list[GFVSArticle] = field(default_factory=list)
    page_section: str | None = None
    is_advertisement_page: bool = False


# ---------------------------------------------------------------------------
# Main Segmenter Class
# ---------------------------------------------------------------------------


class GFVSPageSegmenter:
    """Gemini-First Vision Segmenter for scanned newspaper pages.

    Usage::

        segmenter = GFVSPageSegmenter(
            gemini_provider=gemini_prov,
            gcv_provider=gcv_prov,
            gcv_semaphore=asyncio.Semaphore(8),
        )
        articles = await segmenter.segment_page(
            page_number=2,
            image_bytes=rendered.image_bytes,
            lang_hint="en",
        )
        # articles is list[SegmentedArticle] — same schema as Paths A/B/C
    """

    # Minimum article area (fraction of full 1000x1000 space) to keep.
    # Filters phantom 1-pixel bboxes sometimes returned by the model.
    _MIN_AREA_FRACTION = 0.005  # 0.5% of page

    def __init__(
        self,
        gemini_provider: VisionModelProvider,
        gcv_provider: OCREngine,
        gcv_semaphore: asyncio.Semaphore,
    ) -> None:
        self._gemini = gemini_provider
        self._gcv = gcv_provider
        self._semaphore = gcv_semaphore

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def segment_page(
        self,
        page_number: int,
        image_bytes: bytes,
        lang_hint: str = "en",
    ) -> list[SegmentedArticle]:
        """Run the full two-phase GFVS pipeline on a scanned page image.

        Returns:
            List of SegmentedArticle objects compatible with the rest of
            the ingestion pipeline (same type used by Paths A, B, C).
        """
        # Phase 1: Gemini layout analysis
        try:
            page_result = await self._phase1_gemini_layout(page_number, image_bytes)
        except Exception as phase1_err:
            logger.warning(
                "GFVS Phase 1 (Gemini layout) failed on page %d: %s",
                page_number,
                phase1_err,
            )
            return []

        if not page_result.articles:
            logger.warning(
                "GFVS Phase 1 returned 0 articles on page %d",
                page_number,
            )
            return []

        logger.info(
            "GFVS Phase 1 complete: %d articles identified by Gemini on page %d",
            len(page_result.articles),
            page_number,
        )

        # Phase 2: GCV per-article text extraction (parallel, bounded by semaphore)
        try:
            await self._phase2_gcv_text(page_result.articles, image_bytes, lang_hint)
        except Exception as phase2_err:
            logger.warning(
                "GFVS Phase 2 (GCV text extraction) encountered error on page %d: %s",
                page_number,
                phase2_err,
            )
            # Continue: articles still have Gemini-extracted headlines/metadata

        # Convert to SegmentedArticle output schema
        segmented = [
            self._to_segmented_article(art, idx)
            for idx, art in enumerate(page_result.articles)
            if art.headline.strip()
        ]

        logger.info(
            "GFVS complete on page %d: %d segmented articles produced",
            page_number,
            len(segmented),
        )
        return segmented

    # -----------------------------------------------------------------------
    # Phase 1: Gemini Flash Vision — Semantic Layout Understanding
    # -----------------------------------------------------------------------

    async def _phase1_gemini_layout(
        self,
        page_number: int,
        image_bytes: bytes,
    ) -> GFVSPageResult:
        """Send full page image to Gemini Flash and parse article boundaries."""
        prompt = GFVS_LAYOUT_PROMPT + f"\n\nThis is Page {page_number}."

        resp = await self._gemini.analyze_image(
            image_bytes=image_bytes,
            prompt=prompt,
            max_tokens=8192,
        )

        raw_text = (resp.text or "").strip()
        parsed: dict[str, Any] | None = resp.parsed if isinstance(resp.parsed, dict) else None

        if not parsed and raw_text:
            parsed = self._parse_gemini_response(raw_text)

        if not parsed:
            logger.warning(
                "GFVS Phase 1: Could not parse Gemini response on page %d",
                page_number,
                extra={"snippet": raw_text[:300]},
            )
            return GFVSPageResult(page_number=page_number)

        return self._build_page_result(parsed, page_number)

    def _parse_gemini_response(self, raw: str) -> dict[str, Any] | None:
        """Robust JSON parser with multiple fallback strategies for Gemini output."""
        # Strip reasoning tokens
        text = re.sub(r"<(thought|think)>.*?</\1>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Strategy 1: direct parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                return {"articles": data}
        except Exception:
            pass

        # Strategy 2: extract outermost JSON object
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        # Strategy 3: close unclosed brackets
        repaired = text
        if repaired.count('"') % 2 != 0:
            repaired += '"'
        repaired = re.sub(r",\s*([\}\]])", r"\1", repaired)
        open_b = repaired.count("{") - repaired.count("}")
        open_sq = repaired.count("[") - repaired.count("]")
        if open_sq > 0:
            repaired += "]" * open_sq
        if open_b > 0:
            repaired += "}" * open_b
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        return None

    def _build_page_result(self, parsed: dict[str, Any], page_number: int) -> GFVSPageResult:
        """Validate and normalise the Gemini JSON response into GFVSPageResult."""
        result = GFVSPageResult(
            page_number=page_number,
            page_section=parsed.get("page_section"),
            is_advertisement_page=bool(parsed.get("is_advertisement_page", False)),
        )

        raw_articles = parsed.get("articles", [])
        if not isinstance(raw_articles, list):
            return result

        for art in raw_articles:
            if not isinstance(art, dict):
                continue

            headline = str(art.get("headline") or "").strip()
            if not headline:
                headline = "News Item"

            # Parse and validate bbox [ymin, xmin, ymax, xmax] 0-1000
            bbox = self._parse_bbox(art.get("bbox"))
            if bbox is None:
                logger.debug("GFVS: skipping article with invalid bbox: %s", art.get("bbox"))
                continue

            # Filter phantom tiny bboxes
            ymin, xmin, ymax, xmax = bbox
            area = (ymax - ymin) * (xmax - xmin) / (1000.0 * 1000.0)
            if area < self._MIN_AREA_FRACTION:
                logger.debug(
                    "GFVS: skipping article '%s' — area too small (%.4f)", headline, area
                )
                continue

            photo_bbox: tuple[float, float, float, float] | None = None
            if art.get("has_photo"):
                photo_bbox = self._parse_bbox(art.get("photo_bbox"))

            prominence = str(art.get("prominence") or "standard").lower()
            if prominence not in ("lead", "major", "standard", "minor", "filler"):
                prominence = "standard"

            article_type = str(art.get("article_type") or "news").lower()
            if article_type not in ("news", "advertisement", "opinion", "editorial", "notice", "sidebar"):
                article_type = "news"

            result.articles.append(
                GFVSArticle(
                    headline=headline,
                    subheadline=str(art["subheadline"]).strip() if art.get("subheadline") else None,
                    byline=str(art["byline"]).strip() if art.get("byline") else None,
                    section=str(art.get("section") or result.page_section or "National"),
                    prominence=prominence,
                    article_type=article_type,
                    bbox=bbox,
                    has_photo=bool(art.get("has_photo", False)),
                    photo_bbox=photo_bbox,
                    page_number=page_number,
                )
            )

        return result

    @staticmethod
    def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
        """Parse and clamp a 4-element bbox list to a valid 0-1000 normalised tuple."""
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            return None
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in raw[:4])
        except (TypeError, ValueError):
            return None

        # Clamp to valid range
        ymin = max(0.0, min(1000.0, ymin))
        xmin = max(0.0, min(1000.0, xmin))
        ymax = max(0.0, min(1000.0, ymax))
        xmax = max(0.0, min(1000.0, xmax))

        # Swap if inverted
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        return (ymin, xmin, ymax, xmax)

    # -----------------------------------------------------------------------
    # Phase 2: Google Cloud Vision — Per-Article Text Extraction
    # -----------------------------------------------------------------------

    async def _phase2_gcv_text(
        self,
        articles: list[GFVSArticle],
        page_image: bytes,
        lang_hint: str,
    ) -> None:
        """Extract body text for all articles via GCV per-crop. Mutates articles in-place."""
        img = Image.open(io.BytesIO(page_image))
        page_w, page_h = img.size
        img.close()

        tasks = [
            self._extract_one_article_text(art, page_image, page_w, page_h, lang_hint)
            for art in articles
            # Skip advertisement articles (GCV on ad crops adds noise)
            if art.article_type != "advertisement"
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _extract_one_article_text(
        self,
        article: GFVSArticle,
        page_image: bytes,
        page_w: int,
        page_h: int,
        lang_hint: str,
    ) -> None:
        """Crop the article region and extract verbatim text via GCV. Updates article.body_text."""
        try:
            crop_bytes = self._crop_bbox(page_image, article.bbox, page_w, page_h)
            async with self._semaphore:
                ocr_result = await self._gcv.ocr(
                    image_bytes=crop_bytes,
                    lang_hint=lang_hint,
                )

            if ocr_result.blocks:
                article.body_text = self._join_gcv_blocks(ocr_result.blocks)
            else:
                logger.debug(
                    "GFVS Phase 2: GCV returned 0 blocks for article '%s'",
                    article.headline[:60],
                )

        except Exception as err:
            logger.warning(
                "GFVS Phase 2: GCV extraction failed for article '%s': %s",
                article.headline[:60],
                err,
            )
            # Leave body_text empty — article is stored with headline + metadata

    @staticmethod
    def _crop_bbox(
        image_bytes: bytes,
        bbox: tuple[float, float, float, float],
        page_w: int,
        page_h: int,
    ) -> bytes:
        """Crop a page image to the article region defined by a 0-1000 normalised bbox.

        Args:
            image_bytes: Full page image bytes (PNG or JPEG).
            bbox: (ymin, xmin, ymax, xmax) normalised 0-1000.
            page_w: Page image width in pixels.
            page_h: Page image height in pixels.

        Returns:
            PNG bytes of the cropped article region.
        """
        ymin, xmin, ymax, xmax = bbox
        x0 = max(0, int((xmin / 1000.0) * page_w))
        y0 = max(0, int((ymin / 1000.0) * page_h))
        x1 = min(page_w, int((xmax / 1000.0) * page_w))
        y1 = min(page_h, int((ymax / 1000.0) * page_h))

        # Ensure minimum crop size (10px) to avoid degenerate inputs to GCV
        if x1 - x0 < 10:
            x1 = min(page_w, x0 + 10)
        if y1 - y0 < 10:
            y1 = min(page_h, y0 + 10)

        img = Image.open(io.BytesIO(image_bytes))
        cropped = img.crop((x0, y0, x1, y1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _join_gcv_blocks(blocks: list[Any]) -> str:
        """Join GCV OCR blocks into clean readable text in top-to-bottom order.

        GCV on a narrow per-article crop returns blocks in roughly correct
        reading order. We sort by y-coordinate of each block's bbox top edge
        and join with double newlines to preserve paragraph structure.
        """
        if not blocks:
            return ""

        def block_y(b: Any) -> float:
            bbox = getattr(b, "bbox", (0, 0, 0, 0))
            # bbox is (x0, y0, x1, y1) — sort by y0
            return float(bbox[1]) if len(bbox) >= 2 else 0.0

        sorted_blocks = sorted(blocks, key=block_y)
        paragraphs = [getattr(b, "text", "").strip() for b in sorted_blocks]
        return "\n\n".join(p for p in paragraphs if p)

    # -----------------------------------------------------------------------
    # Output Conversion
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_segmented_article(art: GFVSArticle, idx: int) -> SegmentedArticle:
        """Convert GFVSArticle to the canonical SegmentedArticle used everywhere.

        GFVSArticle bbox: [ymin, xmin, ymax, xmax] normalised 0-1000.
        SegmentedArticle bbox_list: list of (x0, y0, x1, y1) tuples in pixels or 0-1000.

        We convert: x0=xmin, y0=ymin, x1=xmax, y1=ymax (standard [x0,y0,x1,y1]).
        """
        ymin, xmin, ymax, xmax = art.bbox
        # Convert [ymin,xmin,ymax,xmax] -> (x0, y0, x1, y1)
        box_tuple = (xmin, ymin, xmax, ymax)

        body = art.body_text.strip()
        if not body:
            # Fallback: use subheadline or headline if GCV returned nothing (e.g. ad pages)
            body = art.subheadline or art.headline

        return SegmentedArticle(
            article_temp_id=f"page_{art.page_number}_gfvs_{idx}",
            headline=art.headline,
            subheadline=art.subheadline,
            byline_author=art.byline,
            body_text=body,
            word_count=len(body.split()),
            bbox_list=[box_tuple],
            section=art.section,
            printed_section=art.section,
        )


# ---------------------------------------------------------------------------
# Provider Resolution Helper
# ---------------------------------------------------------------------------


def resolve_gemini_provider(settings: Any) -> VisionModelProvider:
    """Resolve the Gemini Flash vision provider from the registry.

    Tries the ``scanned_page_segmentation`` task binding first (new),
    then ``layout_analysis`` (which maps to gemini_flash in prod),
    then falls back to explicit Gemini provider IDs.

    Raises:
        RuntimeError: If no Gemini-capable vision provider is found.
    """
    from app.providers.registry import get_registry

    reg = get_registry()

    # Try task bindings first (preferred — user-configurable)
    for task_key in ("scanned_page_segmentation", "layout_analysis"):
        try:
            prov = reg.get_provider(task_key)
            if isinstance(prov, VisionModelProvider):
                logger.debug("GFVS: resolved Gemini provider via task binding '%s'", task_key)
                return prov
        except Exception:
            continue

    # Fall back to explicit provider IDs
    for provider_id in ("gemini_vision", "gemini_flash", "gemini_pro", "gemini_flash_lite"):
        try:
            prov = reg.get_provider_by_id(provider_id)
            if isinstance(prov, VisionModelProvider):
                logger.debug("GFVS: resolved Gemini provider via id '%s'", provider_id)
                return prov
        except Exception:
            continue

    raise RuntimeError(
        "GFVS: No Gemini-capable VisionModelProvider found in registry. "
        "Ensure gemini_vision or gemini_flash is configured in model_config.yaml "
        "and that GEMINI_API_KEY is set."
    )


__all__ = [
    "GFVS_LAYOUT_PROMPT",
    "GFVSArticle",
    "GFVSPageResult",
    "GFVSPageSegmenter",
    "resolve_gemini_provider",
]
