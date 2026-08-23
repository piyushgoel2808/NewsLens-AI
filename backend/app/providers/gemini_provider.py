"""Google Gemini provider: high-performance inference via Google Generative Language API.

Implements:
- ChatModelProvider (complete, complete_stream)
- VisionModelProvider (analyze_image)
- OCREngine (ocr)
- DocumentLayoutProvider (parse_page_image, parse_pdf_document)

Supported models:
- gemini-3.7-flash (default, high-speed multimodal)
- gemini-3.5-flash
- gemini-3.1-pro-preview
- gemini-flash-latest
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pymupdf
from PIL import Image

from app.core.logging import get_logger
from app.providers.base import (
    ChatModelProvider,
    DocumentLayoutProvider,
    ExtractedDocumentNode,
    ExtractedPhotoData,
    ExtractedTableData,
    Message,
    MinerUParseResult,
    ModelResponse,
    OCRBlock,
    OCREngine,
    OCRResult,
    ProviderCapability,
    ProviderError,
    ToolCall,
    ToolDefinition,
    VisionModelProvider,
)

logger = get_logger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

LAYOUT_SYSTEM_INSTRUCTION = (
    "You are an expert broadsheet newspaper layout analyzer and OCR transcription engine.\n"
    "Analyze the provided newspaper broadsheet page image with extreme precision.\n"
    "Identify all discrete layout nodes in logical reading order: articles, headlines, "
    "subheadlines, body text blocks, tables, photos/illustrations, photo captions, "
    "mastheads/headers, and footers.\n"
    "Extract the exact bounding boxes and verbatim transcribed text."
)

LAYOUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_type": {
                        "type": "string",
                        "enum": ["title", "text", "table", "image", "caption", "header", "footer"],
                    },
                    "text": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[ymin, xmin, ymax, xmax] normalized 0-1000",
                    },
                    "level": {"type": "integer"},
                    "caption": {"type": "string"},
                    "table_data": {
                        "type": "object",
                        "properties": {
                            "headers": {"type": "array", "items": {"type": "string"}},
                            "rows": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}},
                            },
                            "markdown": {"type": "string"},
                        },
                    },
                },
                "required": ["node_type", "text", "box_2d"],
            },
        },
        "full_markdown": {"type": "string"},
    },
    "required": ["elements"],
}

OCR_SYSTEM_INSTRUCTION = (
    "You are a high-precision OCR engine for broadsheet newspapers.\n"
    "Extract all readable text blocks and lines from this page image with bounding boxes.\n"
    "Return verbatim text, bounding boxes in [ymin, xmin, ymax, xmax] normalized 0-1000, "
    "and estimated confidence scores."
)

OCR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[ymin, xmin, ymax, xmax] normalized 0-1000",
                    },
                    "confidence": {"type": "number"},
                    "language": {"type": "string"},
                },
                "required": ["text", "box_2d"],
            },
        },
        "full_text": {"type": "string"},
        "mean_confidence": {"type": "number"},
    },
    "required": ["blocks"],
}


def _normalize_box(
    box: list[float] | tuple[float, ...], width_px: int, height_px: int
) -> tuple[float, float, float, float]:
    """Convert Gemini [ymin, xmin, ymax, xmax] coordinates to pixel (x0, y0, x1, y1)."""
    if len(box) < 4:
        return (0.0, 0.0, float(width_px), float(height_px))

    ymin, xmin, ymax, xmax = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    if max(ymin, xmin, ymax, xmax) <= 1000.0 and max(ymax, xmax) > 1.0:
        # Scale 0-1000
        x0 = (xmin / 1000.0) * width_px
        y0 = (ymin / 1000.0) * height_px
        x1 = (xmax / 1000.0) * width_px
        y1 = (ymax / 1000.0) * height_px
    elif max(ymin, xmin, ymax, xmax) <= 1.0:
        # Scale 0.0-1.0
        x0 = xmin * width_px
        y0 = ymin * height_px
        x1 = xmax * width_px
        y1 = ymax * height_px
    else:
        # Already in pixels
        x0, y0, x1, y1 = xmin, ymin, xmax, ymax

    return (
        max(0.0, min(float(width_px), float(x0))),
        max(0.0, min(float(height_px), float(y0))),
        max(0.0, min(float(width_px), float(x1))),
        max(0.0, min(float(height_px), float(y1))),
    )


class GeminiProvider(ChatModelProvider, VisionModelProvider, DocumentLayoutProvider, OCREngine):
    """Hosted LLM, Vision, OCR, and Document Layout provider backed by Google Gemini API."""

    def __init__(self, model: str = "gemini-3.7-flash", api_key: str | None = None) -> None:
        if not api_key:
            raise ProviderError(
                "Google Gemini API key is required. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
            )
        self._model = model.replace("models/", "") if model else "gemini-3.7-flash"
        self._api_key = api_key
        self._capability = ProviderCapability(
            supports_vision=True,
            supports_tool_use=True,
            supports_streaming=True,
            supports_structured_output=True,
            context_window=1000000,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def _to_gemini_contents(
        self, messages: list[Message]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Convert standard Message list to Gemini API systemInstruction and contents."""
        system_instruction: dict[str, Any] | None = None
        contents: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system_text = (
                    m.content
                    if isinstance(m.content, str)
                    else " ".join(
                        p.get("text", "")
                        for p in m.content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                )
                system_instruction = {"parts": [{"text": system_text}]}
            elif m.role in ("user", "human"):
                if isinstance(m.content, str):
                    contents.append({"role": "user", "parts": [{"text": m.content}]})
                else:
                    parts: list[dict[str, Any]] = []
                    for part in m.content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append({"text": part.get("text", "")})
                            elif part.get("type") == "image_url":
                                url = part.get("image_url", {}).get("url", "")
                                if url.startswith("data:image/"):
                                    header, b64data = url.split(",", 1)
                                    mime = header.split(";")[0].replace("data:", "")
                                    parts.append(
                                        {"inlineData": {"mimeType": mime, "data": b64data}}
                                    )
                    contents.append({"role": "user", "parts": parts})
            elif m.role in ("assistant", "model"):
                if isinstance(m.content, str):
                    contents.append({"role": "model", "parts": [{"text": m.content}]})
                else:
                    parts = []
                    for part in m.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                    contents.append({"role": "model", "parts": parts})

        return system_instruction, contents

    def _to_gemini_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert standard ToolDefinition list to Gemini function declarations."""
        declarations = []
        for t in tools:
            declarations.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            )
        return [{"functionDeclarations": declarations}]

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        response_schema: dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Run a chat completion via Google Gemini API."""
        t0 = time.monotonic()
        system_instruction, contents = self._to_gemini_contents(messages)

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if response_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_schema

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = self._to_gemini_tools(tools)

        model_candidates = [self._model]
        for fb in ["gemini-flash-latest", "gemini-3.7-flash"]:
            if fb not in model_candidates:
                model_candidates.append(fb)

        last_error: Exception | None = None
        for m in model_candidates:
            url = f"{GEMINI_API_BASE}/{m}:generateContent?key={self._api_key}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    res = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                except Exception as e:
                    last_error = e
                    continue

            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return ModelResponse(
                        text="",
                        model=m,
                        provider=self.provider_name,
                    )

                candidate = candidates[0]
                content_parts = candidate.get("content", {}).get("parts", [])

                text_pieces: list[str] = []
                tool_calls: list[ToolCall] = []

                for p in content_parts:
                    if "text" in p:
                        text_pieces.append(p["text"])
                    elif "functionCall" in p:
                        fc = p["functionCall"]
                        tool_calls.append(
                            ToolCall(
                                tool_name=fc.get("name", ""),
                                tool_input=fc.get("args", {}),
                                tool_use_id=f"call_{int(time.time()*1000)}",
                            )
                        )

                full_text = "".join(text_pieces)

                parsed_json = None
                if response_schema and full_text:
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed_json = json.loads(full_text)

                usage = data.get("usageMetadata", {})
                in_tok = usage.get("promptTokenCount", 0)
                out_tok = usage.get("candidatesTokenCount", 0)
                lat_ms = int((time.monotonic() - t0) * 1000)

                logger.info(
                    "Gemini completion",
                    extra={
                        "model": m,
                        "in_tok": in_tok,
                        "out_tok": out_tok,
                        "lat_ms": lat_ms,
                    },
                )

                return ModelResponse(
                    text=full_text,
                    tool_calls=tool_calls,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    model=m,
                    provider=self.provider_name,
                    parsed=parsed_json,
                    raw=data,
                )

            err_data = (
                res.json()
                if res.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            err_msg = err_data.get("error", {}).get("message", res.text)
            last_error = ProviderError(f"Gemini API returned {res.status_code}: {err_msg}")
            if res.status_code not in (429, 503, 404):
                break

        raise last_error or ProviderError("Gemini completion failed across all candidate models")

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming chat completion yielding text deltas with automatic model fallback."""
        system_instruction, contents = self._to_gemini_contents(messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        model_candidates = [self._model]
        for fb in ["gemini-flash-latest", "gemini-3.7-flash"]:
            if fb not in model_candidates:
                model_candidates.append(fb)

        last_error: Exception | None = None
        for m in model_candidates:
            url = f"{GEMINI_API_BASE}/{m}:streamGenerateContent?key={self._api_key}&alt=sse"
            try:
                async with (
                    httpx.AsyncClient(timeout=90.0) as client,
                    client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response,
                ):
                    if response.status_code != 200:
                        err_text = await response.aread()
                        err_str = err_text.decode("utf-8")
                        msg = f"Gemini streaming error ({response.status_code}): {err_str}"
                        last_error = ProviderError(msg)
                        if response.status_code in (429, 503, 404):
                            continue
                        raise ProviderError(msg)

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                chunk_data = json.loads(data_str)
                                candidates = chunk_data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for p in parts:
                                        if "text" in p:
                                            yield p["text"]
                            except json.JSONDecodeError:
                                continue
                    return
            except Exception as e:
                last_error = e
                continue

        raise last_error or ProviderError("Gemini streaming failed across all candidate models")

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Analyze an image using multimodal Gemini vision."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        contents: list[dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": b64_image}},
                ],
            }
        ]

        generation_config: dict[str, Any] = {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
        }
        if response_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_schema

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        url = f"{GEMINI_API_BASE}/{self._model}:generateContent?key={self._api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                url, json=payload, headers={"Content-Type": "application/json"}
            )

        if res.status_code != 200:
            raise ProviderError(f"Gemini vision error ({res.status_code}): {res.text}")

        data = res.json()
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if "text" in p)

        parsed_json = None
        if response_schema and text:
            with contextlib.suppress(json.JSONDecodeError):
                parsed_json = json.loads(text)

        usage = data.get("usageMetadata", {})
        return ModelResponse(
            text=text,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            model=self._model,
            provider=self.provider_name,
            parsed=parsed_json,
            raw=data,
        )

    # ---------------------------------------------------------------------------
    # OCREngine Protocol Implementation
    # ---------------------------------------------------------------------------

    async def ocr(
        self,
        image_bytes: bytes,
        lang_hint: str | None = None,
    ) -> OCRResult:
        """Run high-precision OCR on an image using Google Gemini Vision."""
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            w_px, h_px = pil_img.size
        except Exception:
            w_px, h_px = 2480, 3508

        prompt = (
            f"{OCR_SYSTEM_INSTRUCTION}\n"
            f"Language hint: {lang_hint or 'en'}. Image dimensions: {w_px}x{h_px} px."
        )

        try:
            resp = await self.analyze_image(
                image_bytes=image_bytes,
                prompt=prompt,
                response_schema=OCR_JSON_SCHEMA,
                max_tokens=8192,
            )

            blocks: list[OCRBlock] = []
            parsed = resp.parsed or {}
            raw_blocks = parsed.get("blocks", [])

            if not raw_blocks and resp.text:
                try:
                    loaded = json.loads(resp.text)
                    if isinstance(loaded, dict):
                        raw_blocks = loaded.get("blocks", [])
                except Exception:
                    pass

            for b in raw_blocks:
                txt = (b.get("text") or "").strip()
                if not txt:
                    continue
                box_raw = b.get("box_2d", [0, 0, 1000, 1000])
                bbox = _normalize_box(box_raw, w_px, h_px)
                conf = float(b.get("confidence", 0.95))
                lang = b.get("language", lang_hint or "en")
                blocks.append(OCRBlock(text=txt, bbox=bbox, confidence=conf, language=lang))

            full_text = parsed.get("full_text") or "\n".join(b.text for b in blocks)
            mean_conf = (
                sum(b.confidence for b in blocks) / len(blocks)
                if blocks
                else float(parsed.get("mean_confidence", 0.9))
            )

            return OCRResult(
                blocks=blocks,
                full_text=full_text,
                mean_confidence=mean_conf,
                language=lang_hint or "en",
            )
        except Exception as e:
            logger.warning(
                "Gemini OCR extraction failed; returning fallback empty result",
                extra={"error": str(e)},
            )
            return OCRResult(
                blocks=[],
                full_text="",
                mean_confidence=0.0,
                language=lang_hint or "en",
            )

    # ---------------------------------------------------------------------------
    # DocumentLayoutProvider Protocol Implementation
    # ---------------------------------------------------------------------------

    async def parse_page_image(
        self,
        image_bytes: bytes,
        page_number: int = 1,
        lang: str = "en",
    ) -> MinerUParseResult:
        """Parse a single broadsheet page raster image using Gemini multimodal layout analysis."""
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            w_px, h_px = pil_img.size
        except Exception:
            w_px, h_px = 2480, 3508

        prompt = (
            f"{LAYOUT_SYSTEM_INSTRUCTION}\n"
            f"Page Number: {page_number}. Language: {lang}. Dimensions: {w_px}x{h_px} px."
        )

        try:
            resp = await self.analyze_image(
                image_bytes=image_bytes,
                prompt=prompt,
                response_schema=LAYOUT_JSON_SCHEMA,
                max_tokens=8192,
            )

            parsed = resp.parsed or {}
            raw_elements = parsed.get("elements", [])

            if not raw_elements and resp.text:
                try:
                    loaded = json.loads(resp.text)
                    if isinstance(loaded, dict):
                        raw_elements = loaded.get("elements", [])
                except Exception:
                    pass

            nodes: list[ExtractedDocumentNode] = []
            for idx, el in enumerate(raw_elements):
                node_type = el.get("node_type", "text")
                raw_text = (el.get("text") or "").strip()
                box_raw = el.get("box_2d", [0, 0, 1000, 1000])
                bbox = _normalize_box(box_raw, w_px, h_px)

                table_data: ExtractedTableData | None = None
                if node_type == "table":
                    t_info = el.get("table_data", {})
                    headers = t_info.get("headers", [])
                    rows = t_info.get("rows", [])
                    t_md = t_info.get("markdown", raw_text)
                    table_data = ExtractedTableData(
                        bbox=bbox,
                        headers=headers,
                        rows=rows,
                        raw_markdown=t_md,
                    )

                photo_data: ExtractedPhotoData | None = None
                if node_type in ("image", "photo"):
                    photo_data = ExtractedPhotoData(
                        bbox=bbox,
                        caption=el.get("caption"),
                    )

                nodes.append(
                    ExtractedDocumentNode(
                        node_type=node_type,
                        text=raw_text,
                        bbox=bbox,
                        reading_order=idx,
                        level=el.get("level"),
                        table_data=table_data,
                        photo_data=photo_data,
                    )
                )

            full_md = parsed.get("full_markdown") or "\n\n".join(n.text for n in nodes if n.text)

            return MinerUParseResult(
                page_number=page_number,
                nodes=nodes,
                markdown_content=full_md,
                is_ocr_fallback=False,
                ocr_confidence=0.98,
            )
        except Exception as e:
            logger.error(
                "Gemini parse_page_image failed",
                extra={"page_number": page_number, "error": str(e)},
            )
            return MinerUParseResult(
                page_number=page_number,
                nodes=[],
                markdown_content="",
                is_ocr_fallback=True,
                ocr_confidence=0.0,
            )

    async def parse_pdf_document(
        self,
        pdf_bytes: bytes,
        lang: str = "en",
        extract_tables: bool = True,
    ) -> list[MinerUParseResult]:
        """Parse multi-page PDF document concurrently using Gemini 3.7 Flash."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        logger.info(
            "Starting Gemini Vision PDF document parsing",
            extra={"total_pages": total_pages, "model": self._model},
        )

        semaphore = asyncio.Semaphore(4)

        async def _parse_single_page(p_idx: int) -> MinerUParseResult:
            async with semaphore:
                try:
                    page = doc[p_idx]
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    return await self.parse_page_image(
                        image_bytes=img_bytes,
                        page_number=p_idx + 1,
                        lang=lang,
                    )
                except Exception as ex:
                    logger.warning(
                        "Gemini failed on PDF page; extracting native text blocks",
                        extra={"page_number": p_idx + 1, "error": str(ex)},
                    )
                    page = doc[p_idx]
                    blocks = page.get_text("blocks")
                    nodes: list[ExtractedDocumentNode] = []
                    for b_idx, b in enumerate(blocks):
                        x0, y0, x1, y1, b_text, _, _ = b
                        if b_text.strip():
                            nodes.append(
                                ExtractedDocumentNode(
                                    node_type="text",
                                    text=b_text.strip(),
                                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                                    reading_order=b_idx,
                                )
                            )
                    return MinerUParseResult(
                        page_number=p_idx + 1,
                        nodes=nodes,
                        markdown_content=page.get_text(),
                        is_ocr_fallback=True,
                    )

        tasks = [_parse_single_page(idx) for idx in range(total_pages)]
        results = await asyncio.gather(*tasks)
        return list(results)
