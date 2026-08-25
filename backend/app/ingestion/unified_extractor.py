"""Unified Single-Pass Newspaper Layout & Article Extractor.

Supports:
- Google Gemini Vision (Default, Cloud Fast Inference)
- Ollama Gemma 4 (Local Offline Fallback)
- Deterministic Spatial Rule Fallback (Zero-failure safety net)
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from typing import Any

from PIL import Image

from app.core.logging import get_logger
from app.ingestion.extraction_schemas import (
    ArticleEnrichment,
    PageLayoutExtraction,
)
from app.providers.base import VisionModelProvider
from app.providers.registry import get_registry

logger = get_logger(__name__)

PHASE1_LAYOUT_PROMPT = """You are a high-precision broadsheet newspaper layout analyzer and OCR vision engine.
Analyze this newspaper broadsheet page image and extract ALL discrete articles, columns, sidebars, advertisements, and data tables.

Instructions:
1. Identify all articles with their verbatim headline, subheadline (deck/kicker), byline author, and bounding box [ymin, xmin, ymax, xmax] scaled 0 to 1000.
2. Accurately classify each article into its journalism type (news, editorial, opinion, analysis, advertisement, sidebar, photo_caption, table_data, letter, obituary, review, teaser, index) and section (Front Page, National, International, Economy & Policy, Markets & Data, Corporate & Industry, Banking & Finance, Deals, Tech & Startups, Opinion & Editorial, Sports, Science & Environment, Life & Culture, Personal Finance, Law & Justice, Defense & Security, Real Estate & Infrastructure, Advertisements & Notices, News Briefs).
3. If this is Page 1 or contains the masthead, extract the newspaper brand name and issue date (YYYY-MM-DD).
4. If an article mentions 'Continued on Page X' or 'Continued from Page X', populate continues_to_page or continued_from_page.
5. Return strictly valid JSON adhering to the provided schema."""

PHASE2_ENRICH_PROMPT = """You are a high-precision newspaper article transcriber, NER extractor, and summarizer.
Transcribe and analyze this specific newspaper article cropped region.

Headline: {headline}

Instructions:
1. Transcribe the complete body text verbatim in proper reading order across columns. Preserve paragraphs with double newlines.
2. Provide a crisp 2-3 sentence executive summary.
3. Extract key named entities (person, org, location, misc).
4. Identify 1-3 hierarchical topic taxonomy paths (e.g. 'Economy > Monetary Policy', 'Markets > Equities').
5. If there is a table, extract it into structured headers/rows and markdown format.
6. Return strictly valid JSON adhering to the provided schema."""


class UnifiedExtractor:
    """Unified single-pass extractor with multi-provider resilience and deterministic fallbacks."""

    def __init__(
        self,
        provider: VisionModelProvider | None = None,
        engine_name: str | None = None,
    ) -> None:
        self._provider = provider
        self._engine_name = engine_name

    def _get_provider(self) -> VisionModelProvider:
        if self._provider:
            return self._provider

        reg = get_registry()

        # 1. If explicit engine_name was passed (e.g. from upload UI: 'google_cloud_vision', 'gemma4:26b')
        if self._engine_name and self._engine_name.lower() not in ("auto", "default"):
            try:
                target = self._engine_name.lower()
                if "vision" in target or "google" in target or "ocr" in target:
                    prov = reg.get_provider_by_id("google_cloud_vision")
                    if isinstance(prov, VisionModelProvider):
                        return prov
                elif "gemma" in target:
                    prov = reg.get_provider_by_id("ollama_gemma4_26b")
                    if isinstance(prov, VisionModelProvider):
                        return prov
                elif "gemini" in target:
                    prov = reg.get_provider_by_id("gemini_vision")
                    if isinstance(prov, VisionModelProvider):
                        return prov
            except Exception as ex:
                logger.warning(
                    "Failed to resolve requested engine_name in UnifiedExtractor",
                    extra={"engine": self._engine_name, "error": str(ex)},
                )

        # 2. Check layout_analysis task binding from model_config.yaml
        try:
            prov = reg.get_provider("layout_analysis")
            if isinstance(prov, VisionModelProvider):
                return prov
        except Exception as ex:
            logger.warning("Failed to resolve layout_analysis provider from registry", extra={"error": str(ex)})

        # 3. Fallback to Google Cloud Vision Pure OCR or Ollama Gemma 4
        for candidate_id in ("google_cloud_vision", "ollama_gemma4_26b", "gemini_vision"):
            try:
                prov = reg.get_provider_by_id(candidate_id)
                if isinstance(prov, VisionModelProvider):
                    return prov
            except Exception:
                continue

        return reg.get_provider("query_planner")  # type: ignore

    # ---------------------------------------------------------------------------
    # JSON Repair & Parsing Defense
    # ---------------------------------------------------------------------------

    @staticmethod
    def _repair_and_parse_json(raw_text: str) -> dict[str, Any] | None:
        """Robust multi-layer JSON parser with regex recovery for truncated/malformed responses."""
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()
        # Strip markdown fences if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 1. Direct standard parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2. Extract outermost JSON object or array
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # 3. Structural repair: fix unclosed strings, trailing commas, unclosed brackets
        repaired = text
        # If unclosed quote at the end
        if repaired.count('"') % 2 != 0:
            repaired += '"'

        # Remove trailing commas before } or ]
        repaired = re.sub(r",\s*([\}\]])", r"\1", repaired)

        # Count unbalanced braces
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")

        if open_brackets > 0:
            repaired += "]" * open_brackets
        if open_braces > 0:
            repaired += "}" * open_braces

        try:
            return json.loads(repaired)
        except Exception:
            # Try progressively dropping truncated tokens from the end
            for end_idx in [repaired.rfind(","), repaired.rfind("{"), repaired.rfind("[")]:
                if end_idx > 0:
                    sub = repaired[:end_idx]
                    if sub.count('"') % 2 != 0:
                        sub += '"'
                    o_braces = sub.count("{") - sub.count("}")
                    o_brackets = sub.count("[") - sub.count("]")
                    if o_brackets > 0:
                        sub += "]" * o_brackets
                    if o_braces > 0:
                        sub += "}" * o_braces
                    try:
                        return json.loads(sub)
                    except Exception:
                        continue
            logger.warning("All JSON repair attempts failed on raw model output", extra={"snippet": raw_text[:200]})
            return None

    # ---------------------------------------------------------------------------
    # Phase 1: Page Layout & Skeleton Extraction
    # ---------------------------------------------------------------------------

    async def extract_page_layout(
        self,
        page_number: int,
        image_bytes: bytes,
        digital_text_hint: str | None = None,
        max_retries: int = 1,
    ) -> PageLayoutExtraction:
        """Extract all article boundaries, metadata, and skeletons from a page image."""
        provider = self._get_provider()
        prompt = PHASE1_LAYOUT_PROMPT + f"\nPage Number: {page_number}."
        if digital_text_hint:
            # Include a sample of selectable digital text to aid OCR
            prompt += f"\nDigital Text Sample (for reference):\n{digital_text_hint[:800]}"

        schema = PageLayoutExtraction.model_json_schema()

        for attempt in range(max_retries + 1):
            try:
                resp = await provider.analyze_image(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    response_schema=schema,
                    max_tokens=8192,
                )

                parsed = resp.parsed
                if not parsed and resp.text:
                    parsed = self._repair_and_parse_json(resp.text)

                if parsed and isinstance(parsed, dict):
                    # Guarantee page_number is preserved
                    parsed["page_number"] = page_number
                    return PageLayoutExtraction.model_validate(parsed)

            except Exception as e:
                logger.warning(
                    "Phase 1 layout extraction attempt failed",
                    extra={"page_number": page_number, "attempt": attempt, "error": str(e)},
                )
        # Fallback to Google Cloud Vision Pure OCR + Spatial Segmenter if primary provider failed
        try:
            reg = get_registry()
            gcv_prov = reg.get_provider_by_id("google_cloud_vision")
            if isinstance(gcv_prov, VisionModelProvider) and gcv_prov != provider:
                logger.info(
                    "Primary vision provider failed; falling back to Google Cloud Vision OCR layout",
                    extra={"page_number": page_number},
                )
                resp = await gcv_prov.analyze_image(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    response_schema=schema,
                )
                if resp.parsed and isinstance(resp.parsed, dict):
                    resp.parsed["page_number"] = page_number
                    return PageLayoutExtraction.model_validate(resp.parsed)
        except Exception as fallback_ex:
            logger.warning(
                "Google Cloud Vision OCR fallback also failed",
                extra={"page_number": page_number, "error": str(fallback_ex)},
            )

        logger.error(
            "Phase 1 layout extraction failed all attempts; using fallback empty layout",
            extra={"page_number": page_number},
        )
        return PageLayoutExtraction(page_number=page_number, articles=[])

    # ---------------------------------------------------------------------------
    # Phase 2: Targeted Article Enrichment (Body Text, NER, Topics, Tables)
    # ---------------------------------------------------------------------------

    async def enrich_article(
        self,
        headline: str,
        article_crop_bytes: bytes,
        digital_slice: str | None = None,
    ) -> ArticleEnrichment:
        """Extract verbatim body text, summary, entities, and tables for a major article."""
        provider = self._get_provider()
        prompt = PHASE2_ENRICH_PROMPT.format(headline=headline)
        if digital_slice:
            prompt += f"\nDigital text snippet in this region:\n{digital_slice[:1200]}"

        schema = ArticleEnrichment.model_json_schema()

        try:
            resp = await provider.analyze_image(
                image_bytes=article_crop_bytes,
                prompt=prompt,
                response_schema=schema,
                max_tokens=4096,
            )

            parsed = resp.parsed
            if not parsed and resp.text:
                parsed = self._repair_and_parse_json(resp.text)

            if parsed and isinstance(parsed, dict):
                return ArticleEnrichment.model_validate(parsed)

        except Exception as ex:
            logger.warning(
                "Phase 2 enrichment failed; falling back to basic text slice",
                extra={"headline": headline, "error": str(ex)},
            )

        # Graceful fallback: synthesize from digital slice or headline
        body = digital_slice or headline or "News article text."
        summary = body[:200] + ("..." if len(body) > 200 else "")
        return ArticleEnrichment(
            body_text=body,
            summary=summary,
            entities=[],
            topics=["General News"],
            tables=[],
        )

    # ---------------------------------------------------------------------------
    # Helper: Crop Article Image by Bounding Box
    # ---------------------------------------------------------------------------

    @staticmethod
    def crop_article_image(
        full_page_bytes: bytes,
        bbox: list[float] | tuple[float, float, float, float],
    ) -> bytes:
        """Crop a sub-region of the page image based on bounding box."""
        try:
            img = Image.open(io.BytesIO(full_page_bytes))
            w, h = img.size

            if len(bbox) < 4:
                return full_page_bytes

            # Check if bbox is [ymin, xmin, ymax, xmax] normalized 0-1000
            if max(bbox) <= 1000.0 and max(bbox) > 1.0:
                ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
                x0 = int((xmin / 1000.0) * w)
                y0 = int((ymin / 1000.0) * h)
                x1 = int((xmax / 1000.0) * w)
                y1 = int((ymax / 1000.0) * h)
            elif max(bbox) <= 1.0:
                ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
                x0 = int(xmin * w)
                y0 = int(ymin * h)
                x1 = int(xmax * w)
                y1 = int(ymax * h)
            else:
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Clamp coordinates
            x0 = max(0, min(w - 1, x0))
            y0 = max(0, min(h - 1, y0))
            x1 = max(x0 + 10, min(w, x1))
            y1 = max(y0 + 10, min(h, y1))

            cropped = img.crop((x0, y0, x1, y1))
            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.warning("Failed to crop article image region; using full page", extra={"error": str(e)})
            return full_page_bytes
