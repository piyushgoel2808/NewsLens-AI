"""Visual Data Extractor: 3-Stage Visual Intelligence Pipeline for Newspaper Ingestion.

Stage 1: Fast Visual Triage Gate (aspect ratio / variance heuristics + lightweight VLM classifier).
Stage 2: Structured VLM Extraction (Qwen3-VL / Gemini Vision with schema constraints).
Stage 3: Numerical Cross-Validation (verifies extracted metrics against OCR ground truth).
"""

from __future__ import annotations

import asyncio
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
# Robust JSON Parsing & Markdown Recovery Utilities
# ---------------------------------------------------------------------------

def repair_and_parse_json(raw_text: str) -> dict[str, Any] | None:
    """Robust multi-layer JSON parser with regex recovery for truncated/malformed VLM responses."""
    if not raw_text or not raw_text.strip():
        return None

    # 1. Strip reasoning / thinking tokens (<thought>...</thought>, <think>...</think>)
    cleaned = re.sub(r"<(thought|think)>.*?</\1>", "", raw_text, flags=re.DOTALL).strip()
    # Also strip unclosed thought tags at the start of text
    cleaned = re.sub(r"^<(thought|think)>.*?</\1>", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw_text.strip()

    # 2. Strip markdown code fences (```json ... ``` or ``` ...)
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # 3. Direct JSON load attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Regex substring search for outermost JSON object {...}
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 5. Recovery for truncated JSON (e.g. hitting max_tokens mid-string/object)
    start_idx = cleaned.find("{")
    if start_idx != -1:
        truncated = cleaned[start_idx:]
        # Remove trailing commas
        truncated = re.sub(r",\s*$", "", truncated)
        # Close open string literals if quote count is odd
        if truncated.count('"') % 2 != 0:
            truncated += '"'
        # Balance open braces and brackets
        open_braces = truncated.count("{") - truncated.count("}")
        open_brackets = truncated.count("[") - truncated.count("]")
        if open_brackets > 0:
            truncated += "]" * open_brackets
        if open_braces > 0:
            truncated += "}" * open_braces

        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    return None


def extract_markdown_table_from_raw_text(raw_text: str) -> str | None:
    """Regex recovery for raw GitHub Markdown tables embedded in conversational VLM responses."""
    if not raw_text:
        return None
    # Look for table format: header line | ... | followed by separator line |---|...| and rows
    match = re.search(r"(\|[^\n\r]+\|\r?\n\|[-:\s|]+\|\r?\n(?:\|[^\n\r]+\|\r?\n?)+)", raw_text)
    if match:
        return match.group(1).strip()
    return None


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
- "data_chart": Bar chart, line graph, pie chart, stock trend, candlestick, financial diagram, column graph.
- "table": Tabular grid with rows, columns, balance sheets, quarterly results, economic data.
- "infographic": Explainer diagram, process flow, circular/donut map with statistics, illustrated data timeline.
- "photo": Editorial news photograph (people, portraits, events, outdoor scenes).
- "logo": Company logo, masthead icon, decorative insignia, branding snippet.
- "decorative": Divider rule, border, spacer, cartoon, background texture.

IMPORTANT: Do NOT perform chain-of-thought. Output strictly valid JSON matching this schema:
{
  "visual_type": "data_chart" | "table" | "infographic" | "photo" | "logo" | "decorative",
  "contains_data": true | false,
  "confidence": 0.0 to 1.0
}
"""

STRUCTURED_EXTRACTION_PROMPT = """You are an elite financial and statistical data transcription specialist for NewsLens-AI.
Analyze this newspaper visual asset ({visual_type}) and extract ALL numerical and factual data.

CRITICAL EXTRACTION GUIDELINES:
1. DIRECT TRANSCRIPTION: Do NOT perform lengthy chain-of-thought, manual calculation steps, or step-by-step arithmetic. Directly transcribe the data into the JSON object.
2. SUMMARY: Provide a precise 2-sentence executive summary explaining what the chart/table/infographic demonstrates and its main conclusion.
3. MARKDOWN TABLE: Transcribe all categories, sectors, bars, or periods into a clean, complete GitHub-flavored Markdown table.
4. KEY METRICS: List 3 to 6 bullet points of key metrics or items extracted from the visual asset.
5. ANTI-HALLUCINATION: Do NOT guess or interpolate missing values. If a number is blurry or unreadable, write "[unclear]".

Output strictly valid JSON matching this schema:
{
  "summary": "2-sentence executive summary of what the data shows.",
  "markdown_table": "| Category / Period | Value / Details |\\n|---|---|\\n| Val 1 | Val 2 |",
  "key_metrics": ["Metric 1: Value", "Metric 2: Value"],
  "confidence": 0.0 to 1.0
}
"""

PHOTO_SCENE_ANALYSIS_PROMPT = """You are a photojournalism visual intelligence and document analysis specialist for NewsLens-AI.
Analyze this editorial newspaper photograph and describe the scene, subjects, setting, and context in detail.

{caption_context}

CRITICAL VISUAL ANALYSIS GUIDELINES:
1. SUMMARY: Provide a vivid, precise 2 to 3-sentence editorial scene description detailing who/what is depicted, the setting/location, visible actions, and overall context.
2. KEY ELEMENTS: Extract 2 to 4 specific bullet points describing key visible subjects (e.g. people, uniforms, vintage cars, workshop tools, buildings, signage).
3. FACTUAL OBJECTIVITY: Describe only what is visually observable. Directly produce the JSON output without long chain-of-thought.

Output strictly valid JSON matching this schema:
{
  "summary": "2 to 3-sentence editorial description of the photograph.",
  "key_metrics": ["Subject/Person: description", "Setting/Vehicles: description", "Action/Context: description"],
  "confidence": 0.95
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
        """Enhanced PIL heuristic to immediately filter tiny icons, divider rules, and blank spacers."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            if w < min_dim or h < min_dim:
                return False
            # Check aspect ratio — extreme thin strips (lines/separators) are decorative
            aspect = max(w / h, h / w)
            if aspect > 12.0:
                return False

            # Check solid color variance for small icons/spacers (<120px)
            if w < 120 and h < 120:
                grayscale = img if img.mode == "L" else img.convert("L")
                stat = grayscale.getextrema()
                if stat and (stat[1] - stat[0] < 5):
                    return False

            return True
        except Exception:
            return False

    async def classify_visual_asset(
        self,
        image_bytes: bytes,
    ) -> VisualClassification:
        """Classify image type using lightweight VLM inference with strict timeout and JSON repair."""
        if not self.is_candidate_data_image(image_bytes):
            return VisualClassification(
                visual_type="decorative",
                confidence=1.0,
                contains_data=False,
            )

        provider = self._get_provider()
        try:
            response = await asyncio.wait_for(
                provider.analyze_image(
                    image_bytes=image_bytes,
                    prompt=TRIAGE_CLASSIFICATION_PROMPT,
                    response_schema=None,
                    max_tokens=1024,
                ),
                timeout=20.0,
            )
            raw_text = response.text.strip()
            parsed = repair_and_parse_json(raw_text)
            if parsed:
                classification = VisualClassification(**parsed)
                if classification.visual_type in {"data_chart", "table", "infographic"}:
                    classification.contains_data = True
                return classification
        except Exception as e:
            logger.warning(
                "Visual triage classification fallback, checking OCR density",
                extra={"error": str(e)},
            )

        # Tier 0/1 deterministic fallback via OCR density check
        try:
            import pytesseract

            img = Image.open(io.BytesIO(image_bytes))
            ocr_str = pytesseract.image_to_string(img)
            num_count = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", ocr_str))
            line_count = len([line for line in ocr_str.split("\n") if line.strip()])
            if num_count >= 3 and line_count >= 3:
                return VisualClassification(
                    visual_type="table",
                    confidence=0.8,
                    contains_data=True,
                )
            if num_count == 0 and len(ocr_str.split()) < 5:
                return VisualClassification(
                    visual_type="logo" if img.size[0] < 250 and img.size[1] < 250 else "photo",
                    confidence=0.7,
                    contains_data=False,
                )
        except Exception:
            pass

        return VisualClassification(
            visual_type="photo",
            confidence=0.5,
            contains_data=False,
        )

    # -----------------------------------------------------------------------
    # Stage 2: Structured VLM Extraction & Deterministic Spatial OCR Matrix
    # -----------------------------------------------------------------------

    def extract_table_via_spatial_ocr(
        self,
        image_bytes: bytes,
        visual_type: str = "table",
    ) -> VisualExtractionResult:
        """Deterministic OCR spatial matrix reconstruction engine.

        Extracts tabular grid rows and columns using spatial token coordinates,
        constructs clean GitHub-flavored Markdown tables, and derives statistical metrics.
        """
        import pytesseract

        try:
            img = Image.open(io.BytesIO(image_bytes))
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning(
                "OCR spatial matrix extraction failed to open image",
                extra={"error": str(e)},
            )
            return VisualExtractionResult(
                summary=f"Visual asset: {visual_type} from broadsheet.",
                markdown_table="",
                key_metrics=[],
                confidence=0.3,
                visual_type=visual_type,
            )

        n_boxes = len(data.get("text", []))
        blocks: list[dict[str, Any]] = []
        for i in range(n_boxes):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i])
            if not txt or conf < 15:
                continue
            blocks.append({
                "text": txt,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "right": data["left"][i] + data["width"][i],
                "center_x": data["left"][i] + data["width"][i] / 2.0,
                "conf": conf,
            })

        if not blocks:
            return VisualExtractionResult(
                summary=f"Visual asset: {visual_type} from broadsheet.",
                markdown_table="",
                key_metrics=[],
                confidence=0.3,
                visual_type=visual_type,
            )

        img_w, img_h = img.size
        heights = sorted([b["height"] for b in blocks])
        med_h = heights[len(heights) // 2] if heights else 16

        # Group words into spatial rows (Y-proximity)
        blocks.sort(key=lambda b: (b["top"], b["left"]))
        raw_rows: list[list[dict[str, Any]]] = []
        for b in blocks:
            placed = False
            for r in raw_rows:
                avg_top = sum(item["top"] for item in r) / len(r)
                if abs(b["top"] - avg_top) <= max(10, med_h * 0.45):
                    r.append(b)
                    placed = True
                    break
            if not placed:
                raw_rows.append([b])

        raw_rows.sort(key=lambda r: min(b["top"] for b in r))
        for r in raw_rows:
            r.sort(key=lambda b: b["left"])

        title_lines: list[str] = []
        footnote_lines: list[str] = []
        table_rows: list[list[str]] = []

        for idx, r in enumerate(raw_rows):
            row_text = " ".join(b["text"] for b in r).strip()
            row_y = min(b["top"] for b in r)

            # Detect Footnote / Source lines
            if re.search(r"(?i)^(?:[\*•\-\_~]|Source:|Note:|\*Not|All data)", row_text) or row_y > img_h * 0.88:
                footnote_lines.append(row_text)
                continue

            # Detect Title / Subtitle lines in header section
            if idx == 0 and not any(char.isdigit() for char in row_text) and row_y < img_h * 0.25:
                title_lines.append(row_text)
                continue
            if idx == 1 and len(title_lines) == 1 and not any(re.search(r"\b202\d\b", b["text"]) for b in r) and not any(char.isdigit() for char in row_text):
                title_lines.append(row_text)
                continue

            # Clean and split cells in tabular row
            clean_tokens = []
            for b in r:
                t = b["text"]
                t_clean = re.sub(r"^[|\[\]]+|[|\[\]]+$", "", t).strip()
                if t_clean and t_clean not in {"|", "—", "-", "MPA"}:
                    clean_tokens.append(t_clean)

            if clean_tokens:
                table_rows.append(clean_tokens)

        if not table_rows:
            full_text = "\n".join(" ".join(b["text"] for b in r) for r in raw_rows)
            return VisualExtractionResult(
                summary=f"Visual asset: {title_lines[0] if title_lines else visual_type}",
                markdown_table=full_text,
                key_metrics=[],
                confidence=0.5,
                visual_type=visual_type,
            )

        max_cols = max(len(r) for r in table_rows) if table_rows else 1
        header_row = table_rows[0] if table_rows else ["Data"]
        if len(header_row) < max_cols:
            if not any(re.search(r"\d", c) for c in header_row):
                header_row = ["Metric / Category"] + header_row
            while len(header_row) < max_cols:
                header_row.append(f"Col {len(header_row)+1}")

        md_lines: list[str] = []
        md_lines.append("| " + " | ".join(header_row) + " |")
        md_lines.append("| " + " | ".join([":---"] + [":---:" for _ in range(len(header_row) - 1)]) + " |")

        for r in table_rows[1:]:
            cells = r[:]
            while len(cells) < len(header_row):
                cells.append("—")
            if len(cells) > len(header_row):
                cells = cells[:len(header_row)]
            md_lines.append("| " + " | ".join(cells) + " |")

        if footnote_lines:
            md_lines.append("")
            for fn in footnote_lines:
                md_lines.append(f"*{fn}*")

        markdown_table = "\n".join(md_lines)

        # Extract Key Metrics
        key_metrics: list[str] = []
        chart_title = " - ".join(title_lines) if title_lines else "Infographic Data Table"
        key_metrics.append(f"Title: {chart_title}")

        for r in table_rows[1:]:
            row_label = r[0] if r else "Metric"
            num_tokens = [c for c in r[1:] if re.search(r"\d", c)]
            if num_tokens:
                key_metrics.append(f"{row_label}: {', '.join(num_tokens)}")

        if footnote_lines:
            source_match = re.search(r"(?i)Source:\s*([^|*\n]+)", " ".join(footnote_lines))
            if source_match:
                key_metrics.append(f"Source: {source_match.group(1).strip()}")

        summary = f"Data matrix showing {chart_title}. Transcribed {len(table_rows)} rows across {len(header_row)} columns."

        return VisualExtractionResult(
            summary=summary,
            markdown_table=markdown_table,
            key_metrics=key_metrics[:6],
            confidence=0.85,
            visual_type=visual_type,
        )

    async def extract_structured_data(
        self,
        image_bytes: bytes,
        visual_type: str = "data_chart",
    ) -> VisualExtractionResult:
        """Extract structured markdown table and metrics from a data-bearing visual asset."""
        prompt = STRUCTURED_EXTRACTION_PROMPT.replace("{visual_type}", visual_type)

        try:
            provider = self._get_provider()
            response = await asyncio.wait_for(
                provider.analyze_image(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    response_schema=None,
                    max_tokens=4096,
                ),
                timeout=120.0,
            )
            raw_text = response.text.strip()
            parsed = repair_and_parse_json(raw_text)
            if parsed and (parsed.get("markdown_table") or parsed.get("summary") or parsed.get("key_metrics")):
                return VisualExtractionResult(
                    summary=parsed.get("summary", ""),
                    markdown_table=parsed.get("markdown_table", ""),
                    key_metrics=parsed.get("key_metrics", []),
                    confidence=float(parsed.get("confidence", 0.9)),
                    visual_type=visual_type,
                )

            # Fallback 1: Attempt direct regex extraction of Markdown table from raw text
            extracted_md = extract_markdown_table_from_raw_text(raw_text)
            if extracted_md:
                logger.info(
                    "Recovered Markdown table directly from conversational VLM response via regex",
                    extra={"visual_type": visual_type},
                )
                return VisualExtractionResult(
                    summary=f"Visual {visual_type} data transcribed from broadsheet.",
                    markdown_table=extracted_md,
                    key_metrics=[],
                    confidence=0.8,
                    visual_type=visual_type,
                )
        except Exception as e:
            logger.warning(
                "VLM structured visual extraction failed or empty, engaging deterministic spatial OCR fallback",
                extra={"error": str(e), "visual_type": visual_type},
            )

        # Resilient Deterministic Fallback via Spatial OCR Matrix Engine
        logger.info(
            "Executing deterministic OCR spatial matrix reconstruction",
            extra={"visual_type": visual_type},
        )
        return self.extract_table_via_spatial_ocr(image_bytes, visual_type=visual_type)

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

    async def describe_photo_scene(
        self,
        image_bytes: bytes,
        caption: str = "",
    ) -> VisualExtractionResult:
        """Analyze editorial photograph to extract rich visual scene description and key subjects."""
        caption_context = f"Published Newspaper Caption: \"{caption}\"" if caption else ""
        prompt = PHOTO_SCENE_ANALYSIS_PROMPT.replace("{caption_context}", caption_context)

        try:
            provider = self._get_provider()
            response = await asyncio.wait_for(
                provider.analyze_image(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    response_schema=None,
                    max_tokens=2048,
                ),
                timeout=35.0,
            )
            raw_text = response.text.strip()
            parsed = repair_and_parse_json(raw_text)
            if parsed and (parsed.get("summary") or parsed.get("markdown_table")):
                return VisualExtractionResult(
                    summary=parsed.get("summary", ""),
                    markdown_table=parsed.get("markdown_table", ""),
                    key_metrics=parsed.get("key_metrics", []),
                    confidence=float(parsed.get("confidence", 0.9)),
                    visual_type="photo",
                )
        except Exception as e:
            logger.warning(
                "VLM photo scene analysis failed or timed out",
                extra={"error": str(e)},
            )

        # Fallback default description for editorial photo
        summary = f"Editorial news photograph. {caption}".strip() if caption else "Editorial news photograph."
        return VisualExtractionResult(
            summary=summary,
            markdown_table="",
            key_metrics=[f"Caption: {caption}"] if caption else [],
            confidence=0.6,
            visual_type="photo",
        )

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
        if classification.visual_type in {"logo", "decorative"}:
            logger.debug(
                "Skipping non-editorial decorative/logo visual asset",
                extra={"visual_type": classification.visual_type},
            )
            return classification, None

        if classification.visual_type == "photo":
            # Deep Photo Scene Analysis for Editorial Photographs
            extraction = await self.describe_photo_scene(image_bytes, caption=ocr_text)
            return classification, extraction

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
