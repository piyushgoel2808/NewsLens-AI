"""Google Cloud Vision API provider: high-precision pure OCR engine for broadsheet newspapers.

Uses Google Cloud Vision API (`v1/images:annotate` with `DOCUMENT_TEXT_DETECTION`).
Authenticates via GCP Service Account (OAuth2) or Google API Key.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
from PIL import Image

from app.core.logging import get_logger
from app.providers.base import (
    DocumentLayoutProvider,
    ExtractedDocumentNode,
    MinerUParseResult,
    ModelResponse,
    OCRBlock,
    OCREngine,
    OCRResult,
    ProviderCapability,
    ProviderError,
    VisionModelProvider,
)

logger = get_logger(__name__)

VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


class GoogleCloudVisionOCR(OCREngine, DocumentLayoutProvider, VisionModelProvider):
    """Pure OCR & Document Layout Engine backed by Google Cloud Vision API (DOCUMENT_TEXT_DETECTION)."""

    def __init__(
        self,
        service_account_info: dict[str, Any] | str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._sa_credentials: Any = None

        if service_account_info:
            try:
                from google.oauth2 import service_account

                if isinstance(service_account_info, str):
                    if service_account_info.strip().startswith("{"):
                        info = json.loads(service_account_info)
                        self._sa_credentials = service_account.Credentials.from_service_account_info(
                            info,
                            scopes=["https://www.googleapis.com/auth/cloud-platform"],
                        )
                    else:
                        self._sa_credentials = service_account.Credentials.from_service_account_file(
                            service_account_info,
                            scopes=["https://www.googleapis.com/auth/cloud-platform"],
                        )
                elif isinstance(service_account_info, dict):
                    self._sa_credentials = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
            except Exception as e:
                logger.warning(
                    "Failed to initialize GCP Service Account in GoogleCloudVisionOCR",
                    extra={"error": str(e)},
                )

        # Auto-discover service-account.json or GOOGLE_APPLICATION_CREDENTIALS if not explicitly passed
        if not self._sa_credentials:
            import os
            from pathlib import Path

            candidate_paths = []
            env_gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if env_gac:
                candidate_paths.append(Path(env_gac))

            # Look for service-account.json in workspace directories
            curr = Path.cwd()
            candidate_paths.extend([
                curr / "service-account.json",
                curr / "backend" / "service-account.json",
                curr.parent / "service-account.json",
                Path(__file__).resolve().parents[2] / "service-account.json",
                Path(__file__).resolve().parents[3] / "service-account.json",
            ])

            for p in candidate_paths:
                try:
                    if p.exists() and p.is_file():
                        from google.oauth2 import service_account
                        self._sa_credentials = service_account.Credentials.from_service_account_file(
                            str(p.resolve()),
                            scopes=["https://www.googleapis.com/auth/cloud-platform"],
                        )
                        logger.info("Loaded GCP Service Account credentials from file", extra={"path": str(p.resolve())})
                        break
                except Exception as ex:
                    logger.debug("Candidate service account file load failed", extra={"path": str(p), "error": str(ex)})

        if not self._api_key and not self._sa_credentials:
            raise ProviderError(
                "Google Cloud Vision OCR requires a GCP Service Account key or API key."
            )

        self._capability = ProviderCapability(
            supports_vision=True,
            supports_tool_use=False,
            supports_streaming=False,
            supports_structured_output=False,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "google_cloud_vision"

    def _get_auth_headers_and_params(self) -> tuple[dict[str, str], dict[str, str]]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        params: dict[str, str] = {}

        if self._sa_credentials:
            import google.auth.transport.requests

            request = google.auth.transport.requests.Request()
            self._sa_credentials.refresh(request)
            headers["Authorization"] = f"Bearer {self._sa_credentials.token}"
        elif self._api_key:
            params["key"] = self._api_key

        return headers, params

    async def ocr(
        self,
        image_bytes: bytes,
        lang_hint: str | None = None,
    ) -> OCRResult:
        """Run Document Text Detection on a newspaper page image."""
        t0 = time.monotonic()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Configure DOCUMENT_TEXT_DETECTION feature for dense broadsheet text
        image_context: dict[str, Any] = {}
        if lang_hint:
            image_context["languageHints"] = [lang_hint, "en", "hi"]

        payload = {
            "requests": [
                {
                    "image": {"content": b64_image},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": image_context,
                }
            ]
        }

        headers, params = self._get_auth_headers_and_params()

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                res = await client.post(
                    VISION_API_URL,
                    params=params,
                    json=payload,
                    headers=headers,
                )
            except Exception as e:
                logger.error("Google Cloud Vision API network error", extra={"error": str(e)})
                raise ProviderError(f"Google Cloud Vision network error: {e}") from e

        if res.status_code != 200:
            err_msg = res.text
            try:
                err_json = res.json()
                err_msg = err_json.get("error", {}).get("message", res.text)
            except Exception:
                pass
            raise ProviderError(f"Google Cloud Vision API failed ({res.status_code}): {err_msg}")

        data = res.json()
        responses = data.get("responses", [])
        if not responses:
            return OCRResult(blocks=[], full_text="", mean_confidence=0.0, language=lang_hint)

        resp0 = responses[0]
        full_text_annotation = resp0.get("fullTextAnnotation", {})
        full_text = full_text_annotation.get("text", "")

        blocks: list[OCRBlock] = []
        confidences: list[float] = []

        # Parse hierarchical document structure: pages -> blocks -> paragraphs
        for page in full_text_annotation.get("pages", []):
            for block in page.get("blocks", []):
                # Extract bounding box
                vertices = block.get("boundingBox", {}).get("vertices", [])
                if len(vertices) >= 4:
                    xs = [v.get("x", 0) for v in vertices]
                    ys = [v.get("y", 0) for v in vertices]
                    bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
                else:
                    bbox = (0.0, 0.0, 1000.0, 1000.0)

                conf = float(block.get("confidence", 0.95))
                confidences.append(conf)

                # Reconstruct block text from paragraphs/words/symbols
                para_texts: list[str] = []
                for para in block.get("paragraphs", []):
                    words: list[str] = []
                    for word in para.get("words", []):
                        word_str = "".join(s.get("text", "") for s in word.get("symbols", []))
                        words.append(word_str)
                    para_texts.append(" ".join(words))

                block_text = "\n".join(para_texts).strip()
                if block_text:
                    blocks.append(
                        OCRBlock(
                            text=block_text,
                            bbox=bbox,
                            confidence=conf,
                            language=lang_hint or "en",
                        )
                    )

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.95
        lat_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "Google Cloud Vision OCR extracted",
            extra={
                "blocks_count": len(blocks),
                "text_length": len(full_text),
                "mean_confidence": round(mean_conf, 3),
                "lat_ms": lat_ms,
            },
        )

        return OCRResult(
            blocks=blocks,
            full_text=full_text,
            mean_confidence=mean_conf,
            language=lang_hint or "en",
        )

    async def parse_page_image(
        self,
        image_bytes: bytes,
        page_number: int = 1,
        lang: str = "en",
    ) -> MinerUParseResult:
        """Parse single page image layout and structure into ExtractedDocumentNode items."""
        ocr_res = await self.ocr(image_bytes=image_bytes, lang_hint=lang)
        nodes: list[ExtractedDocumentNode] = []

        for idx, blk in enumerate(ocr_res.blocks):
            # Categorize text nodes vs headings based on length and position
            is_title = len(blk.text) < 150 and "\n" not in blk.text.strip()
            node_type = "title" if is_title else "text"
            level = 1 if is_title else None

            nodes.append(
                ExtractedDocumentNode(
                    node_type=node_type,
                    text=blk.text,
                    bbox=blk.bbox,
                    reading_order=idx,
                    level=level,
                )
            )

        return MinerUParseResult(
            page_number=page_number,
            nodes=nodes,
            markdown_content=ocr_res.full_text,
            is_ocr_fallback=False,
            ocr_confidence=ocr_res.mean_confidence,
        )

    async def parse_pdf_document(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
    ) -> list[MinerUParseResult]:
        """Parse complete PDF document page by page using PyMuPDF rasterization + Google Cloud Vision OCR."""
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        results: list[MinerUParseResult] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            parsed = await self.parse_page_image(
                image_bytes=img_bytes,
                page_number=page_idx + 1,
                lang=lang,
            )
            results.append(parsed)

        return results

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Analyze an image using Google Cloud Vision OCR + deterministic newspaper layout segmentation.

        Implements the VisionModelProvider protocol without calling external generative LLM APIs.
        """
        import io
        import re

        from app.ingestion.detector import check_is_advertisement_text
        from app.ingestion.extraction_schemas import (
            ArticleEnrichment,
            ArticleSkeleton,
            PageLayoutExtraction,
        )
        from app.ingestion.layout_analyzer import LayoutAnalyzer
        from app.ingestion.segmenter import ArticleSegmenter

        # 1. Run Pure Document Text OCR
        ocr_res = await self.ocr(image_bytes=image_bytes)

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            w_px, h_px = pil_img.size
        except Exception:
            w_px, h_px = 2480, 3508

        # Extract page number if present in prompt (e.g. "Page Number: 4.")
        page_num_match = re.search(r"Page Number:\s*(\d+)", prompt, re.IGNORECASE)
        page_number = int(page_num_match.group(1)) if page_num_match else 1

        # Determine target schema: PageLayoutExtraction vs ArticleEnrichment
        is_page_layout = (
            response_schema is not None
            and (
                "articles" in response_schema.get("properties", {})
                or "page_number" in response_schema.get("properties", {})
            )
        ) or "PageLayoutExtraction" in prompt or "broadsheet newspaper layout analyzer" in prompt

        if is_page_layout:
            layout_analyzer = LayoutAnalyzer()
            page_layout = layout_analyzer.analyze_from_text_blocks(
                page_number=page_number,
                width_px=w_px,
                height_px=h_px,
                digital_blocks=[],
                ocr_blocks=ocr_res.blocks,
            )

            # Convert to OrderedReadingBlocks and run ArticleSegmenter
            segmenter = ArticleSegmenter()
            segmented_articles = segmenter.segment_page(
                page_number=page_number,
                ordered_blocks=page_layout.reading_order,
                is_advertisement_page=False,
            )

            articles: list[ArticleSkeleton] = []
            for art in segmented_articles:
                if art.bbox_list:
                    min_x0 = min(b[0] for b in art.bbox_list)
                    min_y0 = min(b[1] for b in art.bbox_list)
                    max_x1 = max(b[2] for b in art.bbox_list)
                    max_y1 = max(b[3] for b in art.bbox_list)
                    envelope = (min_x0, min_y0, max_x1, max_y1)
                else:
                    envelope = (0.0, 0.0, float(w_px), float(h_px))

                bbox_list = [
                    float(envelope[1]),  # ymin
                    float(envelope[0]),  # xmin
                    float(envelope[3]),  # ymax
                    float(envelope[2]),  # xmax
                ] if max(envelope) <= 1000.0 else [
                    float((envelope[1] / h_px) * 1000.0),
                    float((envelope[0] / w_px) * 1000.0),
                    float((envelope[3] / h_px) * 1000.0),
                    float((envelope[2] / w_px) * 1000.0),
                ]

                # Map article genre
                is_ad = check_is_advertisement_text(art.headline or "") or check_is_advertisement_text(art.body_text or "")
                art_type = "advertisement" if is_ad else "news"
                prominence = "major" if art.word_count > 250 else ("minor" if art.word_count < 60 else "standard")

                articles.append(
                    ArticleSkeleton(
                        headline=art.headline or "News Item",
                        subheadline=art.subheadline,
                        byline=art.byline_author,
                        body_text=art.body_text or art.headline,
                        article_type=art_type,
                        section="National",
                        prominence=prominence,
                        bbox=bbox_list,
                        continues_to_page=art.jump_to_page,
                        continued_from_page=art.jump_from_page,
                        has_table=False,
                        has_photo=False,
                    )
                )

            extraction = PageLayoutExtraction(
                page_number=page_number,
                newspaper_brand=None,
                issue_date=None,
                printed_page_number=str(page_number),
                is_advertisement_page=False,
                articles=articles,
            )

            return ModelResponse(
                text=json.dumps(extraction.model_dump()),
                model="google_cloud_vision",
                provider=self.provider_name,
                parsed=extraction.model_dump(),
            )

        else:
            # ArticleEnrichment (Phase 2)
            headline_match = re.search(r"Headline:\s*(.+)", prompt)
            hl = headline_match.group(1).strip() if headline_match else "News Article"
            body_text = ocr_res.full_text.strip() or hl
            summary = body_text[:250] + ("..." if len(body_text) > 250 else "")

            enrichment = ArticleEnrichment(
                body_text=body_text,
                summary=summary,
                entities=[],
                topics=[],
                tables=[],
            )

            return ModelResponse(
                text=json.dumps(enrichment.model_dump()),
                model="google_cloud_vision",
                provider=self.provider_name,
                parsed=enrichment.model_dump(),
            )


