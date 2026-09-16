"""Single-Pass Visual Intelligence Extractor for NewsLens-AI.

Sends one master broadsheet page image to Gemini (gemini-3.8-flash) accompanied by
a JSON manifest of pre-filtered target regions with normalized coordinates [x0, y0, x1, y1],
returning structured editorial descriptions and markdown tables for all regions in a single call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.ingestion.visual_extractor import (
    VisualDataExtractor,
    clean_vlm_text,
    repair_and_parse_json,
)
from app.providers.base import ProviderError, VisionModelProvider
from app.providers.registry import get_registry

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Input and Output Data Structures
# ---------------------------------------------------------------------------


@dataclass
class VisualRegion:
    """Pre-detected layout region candidate for visual analysis."""

    region_id: str
    bbox: tuple[float, float, float, float]  # [x0, y0, x1, y1] absolute pixels
    caption_hint: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class VisualRegionResult:
    """Extracted visual intelligence for a single region."""

    region_id: str
    visual_type: str  # "photo" | "infographic" | "table" | "data_chart" | "decorative" | "logo"
    description: str
    key_metrics: list[str] = field(default_factory=list)
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Strict Pydantic Output Schemas for Gemini Structured Output
# ---------------------------------------------------------------------------


class RegionAnalysis(BaseModel):
    """Structured extraction for a single layout region."""

    id: str = Field(
        description="Region identifier matching the input manifest (e.g. 'media_1')."
    )
    visual_type: Literal[
        "photo", "infographic", "table", "data_chart", "decorative", "logo"
    ] = Field(
        default="photo",
        description="Classification: photo, infographic, table, data_chart, decorative, or logo.",
    )
    description: str = Field(
        default="",
        description=(
            "For photos: a 2-3 sentence editorial scene description detailing visible subjects, "
            "actions, and context. For tables/infographics/data_charts: all data transcribed into "
            "a clean GitHub Markdown table. For decorative/logo: brief label or empty string."
        ),
    )
    key_metrics: list[str] = Field(
        default_factory=list,
        description="Key statistics, figures, or bullet points for charts/tables. Empty for photos.",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class PageVisualAnalysis(BaseModel):
    """Complete single-pass visual extraction output for a broadsheet page."""

    regions: list[RegionAnalysis] = Field(
        default_factory=list,
        description="List of analysis results for each region in the manifest.",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SINGLE_PASS_PROMPT = """You are an elite newspaper visual intelligence and data transcription specialist for NewsLens-AI.

Analyze this newspaper broadsheet page image. Below is a JSON manifest of pre-detected layout regions with their normalized coordinates [x0, y0, x1, y1] on a 0.0 to 1.0 scale (where [0, 0] is top-left and [1, 1] is bottom-right). Nearby caption text is provided where available.

Target Regions:
{manifest_json}

CRITICAL EXTRACTION GUIDELINES:
1. For each region in the manifest, determine its visual_type:
   - "photo": Editorial news photograph (people, portraits, events, outdoor scenes).
   - "infographic": Explainer diagram, process flow, circular/donut map with statistics.
   - "table": Tabular grid with rows, columns, balance sheets, quarterly results.
   - "data_chart": Bar chart, line graph, pie chart, stock trend, candlestick.
   - "logo": Company logo, masthead icon, decorative insignia.
   - "decorative": Divider rule, border, spacer, cartoon, background texture.
2. For "photo": Provide a vivid, precise 2 to 3-sentence editorial scene description in `description`. Detail visible people, setting, actions, and news context. Keep `key_metrics` empty.
3. For "infographic", "table", or "data_chart": Transcribe all categories, sectors, bars, or periods into a clean GitHub-flavored Markdown table in `description`. Extract 2 to 5 key metrics into `key_metrics`.
4. For "decorative" or "logo": Set `description` to a concise tag and leave `key_metrics` empty.

CRITICAL FORMAT RULES:
- You MUST output strictly valid JSON matching this schema:
  {{"regions": [{{"id": "media_0", "visual_type": "photo", "description": "...", "key_metrics": [], "confidence": 0.95}}]}}
- Do NOT output ANY preamble, conversational introduction (such as "Got it", "Sure", "Let's analyze"), reasoning trace, or markdown code fences.
- Start your response immediately with the opening curly brace '{{' and end with '}}'."""


# ---------------------------------------------------------------------------
# SinglePassVisualExtractor Implementation
# ---------------------------------------------------------------------------


class SinglePassVisualExtractor:
    """Extracts visual intelligence for all page regions in a single VLM call."""

    def __init__(
        self,
        provider: VisionModelProvider | None = None,
        fallback_extractor: VisualDataExtractor | None = None,
    ) -> None:
        self._provider = provider
        self._fallback_extractor = fallback_extractor or VisualDataExtractor()

    def _get_provider(self) -> VisionModelProvider:
        """Resolve VisionModelProvider from registry dynamically via model_config.yaml."""
        if self._provider:
            return self._provider

        registry = get_registry()
        try:
            prov = registry.get_provider("visual_extraction")
            if isinstance(prov, VisionModelProvider):
                return prov
        except Exception as ex:
            logger.warning(
                "Failed to resolve visual_extraction task binding from registry",
                extra={"error": str(ex)},
            )

        # Candidate fallback bindings in order of priority
        for candidate_id in ("gemini_vision", "gemini_flash", "ollama_qwen3vl"):
            try:
                prov = registry.get_provider_by_id(candidate_id)
                if isinstance(prov, VisionModelProvider):
                    return prov
            except Exception:
                continue

        raise ProviderError("No VisionModelProvider available for SinglePassVisualExtractor")

    # -----------------------------------------------------------------------
    # Local Pre-Filtering Gate (Zero API Cost)
    # -----------------------------------------------------------------------

    def pre_filter_regions(
        self,
        regions: list[VisualRegion],
        page_width_px: int,
        page_height_px: int,
        min_dim: int = 120,
        max_aspect: float = 12.0,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.75,
    ) -> list[VisualRegion]:
        """Programmatically drop trivial, noisy, or background elements before calling VLM."""
        page_area = float(max(1, page_width_px * page_height_px))
        surviving: list[VisualRegion] = []

        for reg in regions:
            w = reg.width
            h = reg.height

            # 1. Size gate: drop elements smaller than min_dim (e.g. 120px)
            if w < min_dim or h < min_dim:
                continue

            # 2. Aspect ratio gate: drop extreme thin horizontal/vertical rules
            aspect = max(w / max(1.0, h), h / max(1.0, w))
            if aspect > max_aspect:
                continue

            # 3. Canvas area ratio gate
            area_ratio = (w * h) / page_area
            if area_ratio < min_area_ratio:
                continue
            if area_ratio >= max_area_ratio:
                # Deduplicate full-page background/jacket canvas
                continue

            surviving.append(reg)

        return surviving

    # -----------------------------------------------------------------------
    # Manifest Construction
    # -----------------------------------------------------------------------

    def build_manifest(
        self,
        regions: list[VisualRegion],
        page_width_px: int,
        page_height_px: int,
    ) -> list[dict[str, Any]]:
        """Normalize bounding boxes to 0.0–1.0 scale and attach caption hints."""
        pw = float(max(1, page_width_px))
        ph = float(max(1, page_height_px))
        manifest: list[dict[str, Any]] = []

        for reg in regions:
            norm_x0 = round(max(0.0, min(1.0, reg.bbox[0] / pw)), 4)
            norm_y0 = round(max(0.0, min(1.0, reg.bbox[1] / ph)), 4)
            norm_x1 = round(max(0.0, min(1.0, reg.bbox[2] / pw)), 4)
            norm_y1 = round(max(0.0, min(1.0, reg.bbox[3] / ph)), 4)

            entry: dict[str, Any] = {
                "id": reg.region_id,
                "bbox_normalized": [norm_x0, norm_y0, norm_x1, norm_y1],
            }
            if reg.caption_hint and reg.caption_hint.strip():
                entry["caption_hint"] = reg.caption_hint.strip()[:200]

            manifest.append(entry)

        return manifest

    # -----------------------------------------------------------------------
    # Main Extraction Entrypoint
    # -----------------------------------------------------------------------

    async def extract_all_regions(
        self,
        page_image_bytes: bytes,
        regions: list[VisualRegion],
        page_width_px: int,
        page_height_px: int,
        page_number: int = 1,
    ) -> list[VisualRegionResult]:
        """Analyze all regions on a page in a single Gemini 3.8 Flash call."""
        # 1. Local Pre-Filtering Gate
        filtered_regions = self.pre_filter_regions(
            regions=regions,
            page_width_px=page_width_px,
            page_height_px=page_height_px,
        )

        if not filtered_regions:
            logger.info(
                "Single-pass visual extraction: 0 valid regions after pre-filtering",
                extra={"page_number": page_number, "raw_regions": len(regions)},
            )
            return []

        # 2. Build JSON manifest with normalized coordinates
        manifest = self.build_manifest(
            regions=filtered_regions,
            page_width_px=page_width_px,
            page_height_px=page_height_px,
        )
        manifest_json = json.dumps(manifest, indent=2)

        prompt = SINGLE_PASS_PROMPT.format(manifest_json=manifest_json)
        schema = PageVisualAnalysis.model_json_schema()

        # 3. Call Vision Model Provider (1 call per page)
        provider = self._get_provider()
        results: list[VisualRegionResult] = []
        parsed_regions_map: dict[str, RegionAnalysis] = {}

        provider_model = (
            getattr(provider, "model_name", None)
            or getattr(provider, "_model", None)
            or "gemini-3.8-flash"
        )

        is_local_provider = (
            "ollama" in str(provider_model).lower()
            or "qwen" in str(provider_model).lower()
            or "local" in str(provider_model).lower()
            or provider.__class__.__name__ == "OllamaProvider"
        )

        # For local vision models (Ollama/Qwen-VL), execute per-crop VLM extraction
        # to ensure sharp spatial resolution and eliminate token-budget monologue stalls.
        if is_local_provider:
            logger.info(
                "Executing per-crop VLM extraction for local vision model",
                extra={
                    "page_number": page_number,
                    "regions_count": len(filtered_regions),
                    "model": provider_model,
                },
            )
            import asyncio
            sem = asyncio.Semaphore(2)

            async def _extract_single(reg: VisualRegion) -> VisualRegionResult:
                async with sem:
                    return await self._fallback_extract_region(
                        page_image_bytes=page_image_bytes,
                        region=reg,
                    )

            crop_results = await asyncio.gather(*[_extract_single(r) for r in filtered_regions])
            return list(crop_results)

        # Token budgeting: ~350 tokens per region, capped at 3072 to avoid runaway local VLM loops
        token_budget = min(3072, max(1024, len(filtered_regions) * 350))

        try:
            logger.info(
                "Executing single-pass visual extraction",
                extra={
                    "page_number": page_number,
                    "regions_count": len(filtered_regions),
                    "model": provider_model,
                },
            )

            resp = await provider.analyze_image(
                image_bytes=page_image_bytes,
                prompt=prompt,
                response_schema=schema,
                max_tokens=token_budget,
            )

            # Parse structured response
            parsed_data: dict[str, Any] | None = None
            if resp.parsed and isinstance(resp.parsed, dict):
                parsed_data = resp.parsed
            elif resp.text:
                parsed_data = repair_and_parse_json(resp.text)

            # Resilient structure normalization: support direct list or alternate keys
            if isinstance(parsed_data, list):
                parsed_data = {"regions": parsed_data}
            elif isinstance(parsed_data, dict):
                for alt_key in ("items", "elements", "data", "results"):
                    if "regions" not in parsed_data and alt_key in parsed_data and isinstance(parsed_data[alt_key], list):
                        parsed_data["regions"] = parsed_data[alt_key]
                        break

            if parsed_data:
                try:
                    analysis = PageVisualAnalysis.model_validate(parsed_data)
                    for item in analysis.regions:
                        parsed_regions_map[item.id] = item
                except Exception as val_err:
                    logger.warning(
                        "Validation error on PageVisualAnalysis, attempting loose parsing",
                        extra={"page_number": page_number, "error": str(val_err)},
                    )
                    raw_items = parsed_data.get("regions", [])
                    if isinstance(raw_items, list):
                        for r_raw in raw_items:
                            if isinstance(r_raw, dict) and "id" in r_raw:
                                with_fallback = RegionAnalysis(
                                    id=str(r_raw["id"]),
                                    visual_type=str(r_raw.get("visual_type", "photo")),  # type: ignore[arg-type]
                                    description=str(r_raw.get("description", "")),
                                    key_metrics=[str(m) for m in r_raw.get("key_metrics", [])],
                                    confidence=float(r_raw.get("confidence", 0.9)),
                                    )
                                parsed_regions_map[with_fallback.id] = with_fallback

        except Exception as ex:
            logger.warning(
                "Single-pass Gemini visual extraction failed or timed out",
                extra={"page_number": page_number, "error": str(ex)},
            )

        # 4. Assemble results and trigger fallback for any missing regions or empty descriptions
        for reg in filtered_regions:
            analysis_item = parsed_regions_map.get(reg.region_id)
            if analysis_item and analysis_item.description and analysis_item.description.strip():
                results.append(
                    VisualRegionResult(
                        region_id=reg.region_id,
                        visual_type=analysis_item.visual_type,
                        description=clean_vlm_text(analysis_item.description.strip()),
                        key_metrics=analysis_item.key_metrics,
                        confidence=analysis_item.confidence,
                    )
                )
            else:
                # Per-region fallback if single-pass omitted this region or returned empty description
                fallback_res = await self._fallback_extract_region(
                    page_image_bytes=page_image_bytes,
                    region=reg,
                )
                results.append(fallback_res)

        logger.info(
            "Single-pass visual extraction completed",
            extra={
                "page_number": page_number,
                "regions_sent": len(filtered_regions),
                "results_count": len(results),
                "api_calls": 1,
            },
        )
        return results

    # -----------------------------------------------------------------------
    # Fallback for individual omitted regions
    # -----------------------------------------------------------------------

    async def _fallback_extract_region(
        self,
        page_image_bytes: bytes,
        region: VisualRegion,
    ) -> VisualRegionResult:
        """High-fidelity crop extraction with VLM analysis and OCR spatial table support."""
        from io import BytesIO
        from PIL import Image

        try:
            img = Image.open(BytesIO(page_image_bytes))
            w, h = img.size
            x0 = max(0, min(int(region.bbox[0]), w - 1))
            y0 = max(0, min(int(region.bbox[1]), h - 1))
            x1 = max(x0 + 1, min(int(region.bbox[2]), w))
            y1 = max(y0 + 1, min(int(region.bbox[3]), h))

            crop = img.crop((x0, y0, x1, y1))
            buf = BytesIO()
            crop.save(buf, format="PNG")
            crop_bytes = buf.getvalue()

            # Deterministic classification via OCR density
            cls = self._fallback_extractor._classify_via_ocr_density(crop_bytes)
            visual_type = cls.visual_type
            metrics: list[str] = []
            table_md = ""

            if visual_type in {"table", "data_chart", "infographic"}:
                extracted = self._fallback_extractor.extract_table_via_spatial_ocr(
                    image_bytes=crop_bytes, visual_type=visual_type
                )
                if extracted.markdown_table:
                    table_md = extracted.markdown_table
                metrics = extracted.key_metrics

            # Call VLM on the cropped region
            vlm_desc = ""
            try:
                provider = self._get_provider()
                caption_context = f" Caption context: {region.caption_hint}" if region.caption_hint else ""
                crop_prompt = (
                    f"Describe this cropped newspaper image in 2-3 concise, informative sentences detailing visible subjects, "
                    f"actions, and news context.{caption_context} "
                    f"If this is an infographic, chart, or data graphic, describe its key trends. "
                    f"Do not include conversational greetings, inner thoughts, or meta-commentary; begin directly with the visual description."
                )
                vlm_resp = await provider.analyze_image(
                    image_bytes=crop_bytes,
                    prompt=crop_prompt,
                    max_tokens=600,
                )
                if vlm_resp.text and vlm_resp.text.strip():
                    vlm_desc = clean_vlm_text(vlm_resp.text.strip())
            except Exception as v_err:
                logger.debug("VLM crop analysis error", extra={"error": str(v_err)})

            if table_md:
                desc = f"{vlm_desc}\n\n{table_md}".strip() if vlm_desc else table_md
            elif vlm_desc:
                desc = vlm_desc
            elif region.caption_hint:
                desc = f"Editorial news photograph. {region.caption_hint}".strip()
            else:
                desc = "Editorial news photograph."

            return VisualRegionResult(
                region_id=region.region_id,
                visual_type=visual_type,
                description=desc,
                key_metrics=metrics,
                confidence=0.85 if vlm_desc else 0.7,
            )
        except Exception as e:
            logger.warning("Fallback region extraction failed", extra={"error": str(e)})
            return VisualRegionResult(
                region_id=region.region_id,
                visual_type="photo",
                description=region.caption_hint or "Editorial news visual asset.",
                key_metrics=[],
                confidence=0.5,
            )


__all__ = [
    "PageVisualAnalysis",
    "RegionAnalysis",
    "SinglePassVisualExtractor",
    "VisualRegion",
    "VisualRegionResult",
    "clean_vlm_text",
]
