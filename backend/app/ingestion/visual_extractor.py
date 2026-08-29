"""Visual Data Extractor: 3-Stage Visual Intelligence Pipeline for Newspaper Ingestion.

Stage 1: Fast Visual Triage Gate (aspect ratio / variance heuristics + lightweight VLM classifier).
Stage 2: Structured VLM Extraction (Qwen3-VL / Gemini Vision with schema constraints).
Stage 3: Numerical Cross-Validation (verifies extracted metrics against OCR ground truth).
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.providers.base import Message, VisionModelProvider
from app.providers.registry import get_registry

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Visual Extraction
# ---------------------------------------------------------------------------

class VisualClassification(BaseModel):
    """Stage 1 Triage Classification output."""

    visual_type: Literal[
        "data_chart", "table", "infographic", "photo", "logo", "decorative"
    ] = Field(
        default="photo",
        description="Type of visual element on the broadsheet page.",
    )
    confidence: float = Field(
        default=0.9,
        description="Confidence score between 0.0 and 1.0.",
    )
    contains_data: bool = Field(
        default=False,
        description="True if image contains numerical data, charts, statistics, or structured tables.",
    )


class VisualExtractionResult(BaseModel):
    """Stage 2 Structured Extraction output."""

    summary: str = Field(
        default="",
        description="2-sentence executive summary of the chart/table/infographic findings.",
    )
    markdown_table: str = Field(
        default="",
        description="Clean GitHub-flavored Markdown table containing all extracted numerical data.",
    )
    key_metrics: list[str] = Field(
        default_factory=list,
        description="List of key metrics, figures, percentages, or data points extracted.",
    )
    confidence: float = Field(
        default=1.0,
        description="Extraction confidence score.",
    )
    visual_type: str = Field(
        default="data_chart",
        description="Specific classification type.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TRIAGE_CLASSIFICATION_PROMPT = """You are a high-speed document layout classifier for newspaper broadsheets.
Look at this cropped image asset from a newspaper and classify its visual category.

Categories:
- "data_chart": Bar chart, line graph, pie chart, stock trend, candlestick, financial diagram.
- "table": Tabular grid with rows, columns, balance sheets, quarterly results, economic data.
- "infographic": Explainer diagram, process flow, map with statistics, illustrated data timeline.
- "photo": Editorial news photograph (people, buildings, events, nature, portraits, sports).
- "logo": Company logo, masthead icon, decorative insignia, branding snippet.
- "decorative": Divider rule, border, spacer, cartoon, background texture.

Output strictly valid JSON matching this schema:
{
  "visual_type": "data_chart" | "table" | "infographic" | "photo" | "logo" | "decorative",
  "contains_data": true | false,
  "confidence": 0.0 to 1.0
}
"""

STRUCTURED_EXTRACTION_PROMPT = """You are an elite financial and statistical data transcription specialist for NewsLens-AI.
Analyze this newspaper visual asset ({visual_type}) and extract ALL numerical and factual data.

CRITICAL EXTRACTION GUIDELINES:
1. SUMMARY: Provide a precise 2-sentence executive summary explaining what the chart/table demonstrates and its main conclusion.
2. MARKDOWN TABLE: Transcribe the data into a clean, complete GitHub-flavored Markdown table.
   - Include exact headers (e.g. Metric, FY25, FY26, YoY Growth %).
   - Ensure every row has accurate values, currencies (₹, $, €), and units (Cr, Mn, %, bps).
3. KEY METRICS: List 3 to 6 bullet points of key metrics (e.g. "Revenue: ₹12,450 Cr (+18.3% YoY)").
4. ANTI-HALLUCINATION: Do NOT guess or interpolate missing values. If a number is blurry or unreadable, write "[unclear]".

Output strictly valid JSON matching this schema:
{
  "summary": "2-sentence executive summary of what the data shows.",
  "markdown_table": "| Header 1 | Header 2 |\\n|---|---|\\n| Val 1 | Val 2 |",
  "key_metrics": ["Metric 1: Value", "Metric 2: Value"],
  "confidence": 0.0 to 1.0
}
"""


# ---------------------------------------------------------------------------
# Visual Data Extractor
# ---------------------------------------------------------------------------

class VisualDataExtractor:
    """3-Stage Visual Intelligence Pipeline for newspaper infographics, charts, and tables."""

    def __init__(self, vision_provider: VisionModelProvider | None = None) -> None:
        self._provider = vision_provider

    def _get_provider(self) -> VisionModelProvider:
        """Resolve vision model provider from registry."""
        if self._provider:
            return self._provider
        registry = get_registry()
        try:
            provider = registry.get_provider("visual_extraction")
            if isinstance(provider, VisionModelProvider):
                return provider
        except Exception:
            pass

        # Fallback to layout_analysis or vision provider
        try:
            fallback = registry.get_provider("layout_analysis")
            if isinstance(fallback, VisionModelProvider):
                return fallback
        except Exception:
            pass

        raise RuntimeError("No VisionModelProvider configured for visual_extraction")

    # -----------------------------------------------------------------------
    # Stage 1: Fast Visual Triage Gate
    # -----------------------------------------------------------------------

    def is_candidate_data_image(
        self,
        image_bytes: bytes,
        min_dim: int = 80,
    ) -> bool:
        """Lightweight PIL heuristic to immediately filter tiny icons or blank spacers."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            if w < min_dim or h < min_dim:
                return False
            # Check aspect ratio — extreme thin strips (lines/separators) are decorative
            aspect = max(w / h, h / w)
            if aspect > 12.0:
                return False
            return True
        except Exception:
            return False

    async def classify_visual_asset(
        self,
        image_bytes: bytes,
    ) -> VisualClassification:
        """Classify image type using lightweight VLM inference."""
        if not self.is_candidate_data_image(image_bytes):
            return VisualClassification(
                visual_type="decorative",
                confidence=1.0,
                contains_data=False,
            )

        provider = self._get_provider()
        try:
            response = await provider.analyze_image(
                image_bytes=image_bytes,
                prompt=TRIAGE_CLASSIFICATION_PROMPT,
                response_schema=VisualClassification.model_json_schema(),
                max_tokens=512,
            )
            raw_text = response.text.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw_text)
            classification = VisualClassification(**parsed)
            # Re-confirm contains_data flag
            if classification.visual_type in {"data_chart", "table", "infographic"}:
                classification.contains_data = True
            return classification
        except Exception as e:
            logger.warning(
                "Visual triage classification fallback to photo",
                extra={"error": str(e)},
            )
            return VisualClassification(
                visual_type="photo",
                confidence=0.5,
                contains_data=False,
            )

    # -----------------------------------------------------------------------
    # Stage 2: Structured VLM Extraction
    # -----------------------------------------------------------------------

    async def extract_structured_data(
        self,
        image_bytes: bytes,
        visual_type: str = "data_chart",
    ) -> VisualExtractionResult:
        """Extract structured markdown table and metrics from a data-bearing visual asset."""
        provider = self._get_provider()
        prompt = STRUCTURED_EXTRACTION_PROMPT.replace("{visual_type}", visual_type)

        try:
            response = await provider.analyze_image(
                image_bytes=image_bytes,
                prompt=prompt,
                response_schema=VisualExtractionResult.model_json_schema(),
                max_tokens=2048,
            )
            raw_text = response.text.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw_text)
            return VisualExtractionResult(
                summary=parsed.get("summary", ""),
                markdown_table=parsed.get("markdown_table", ""),
                key_metrics=parsed.get("key_metrics", []),
                confidence=float(parsed.get("confidence", 0.9)),
                visual_type=visual_type,
            )
        except Exception as e:
            logger.warning(
                "Structured visual extraction failed",
                extra={"error": str(e), "visual_type": visual_type},
            )
            return VisualExtractionResult(
                summary=f"Visual asset: {visual_type} from broadsheet.",
                markdown_table="",
                key_metrics=[],
                confidence=0.3,
                visual_type=visual_type,
            )

    # -----------------------------------------------------------------------
    # Stage 3: Numerical Cross-Validation with OCR
    # -----------------------------------------------------------------------

    def cross_validate_with_ocr(
        self,
        extraction: VisualExtractionResult,
        ocr_text: str,
    ) -> float:
        """Verify VLM-extracted numerical figures against OCR tokens from the same region.
        
        Returns a confidence score between 0.0 and 1.0.
        """
        if not ocr_text or not extraction.markdown_table:
            return extraction.confidence

        # Extract numeric tokens (integers, floats, percentages, currency numbers)
        ocr_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", ocr_text))
        vlm_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", extraction.markdown_table))

        if not vlm_numbers:
            return extraction.confidence

        # Overlap ratio of numbers found in OCR
        overlap = vlm_numbers.intersection(ocr_numbers)
        if not ocr_numbers:
            return extraction.confidence

        match_ratio = len(overlap) / len(vlm_numbers)
        # Adjust confidence: blend extraction confidence with OCR match ratio
        adjusted = (extraction.confidence * 0.4) + (match_ratio * 0.6)
        logger.info(
            "Visual OCR cross-validation completed",
            extra={
                "vlm_numbers_count": len(vlm_numbers),
                "ocr_numbers_count": len(ocr_numbers),
                "matched_count": len(overlap),
                "match_ratio": round(match_ratio, 2),
                "adjusted_confidence": round(adjusted, 2),
            },
        )
        return round(min(1.0, max(0.2, adjusted)), 2)

    # -----------------------------------------------------------------------
    # End-to-End Visual Processing Pipeline
    # -----------------------------------------------------------------------

    async def process_image_crop(
        self,
        image_bytes: bytes,
        ocr_text: str = "",
    ) -> tuple[VisualClassification, VisualExtractionResult | None]:
        """Execute full 3-stage visual intelligence pipeline on a cropped image."""
        # Stage 1: Triage
        classification = await self.classify_visual_asset(image_bytes)
        if not classification.contains_data or classification.visual_type in {
            "photo",
            "logo",
            "decorative",
        }:
            logger.debug(
                "Skipping non-data visual asset",
                extra={"visual_type": classification.visual_type},
            )
            return classification, None

        # Stage 2: Extraction
        extraction = await self.extract_structured_data(
            image_bytes, visual_type=classification.visual_type
        )

        # Stage 3: Cross-validation
        extraction.confidence = self.cross_validate_with_ocr(extraction, ocr_text)

        logger.info(
            "Successfully extracted visual infographic data",
            extra={
                "visual_type": classification.visual_type,
                "metrics_count": len(extraction.key_metrics),
                "confidence": extraction.confidence,
            },
        )
        return classification, extraction
