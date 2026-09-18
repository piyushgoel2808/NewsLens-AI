"""Ollama provider: local LLM and VLM via the Ollama Python client.

Implements ChatModelProvider and VisionModelProvider for any model served
by a running Ollama instance (http://localhost:11434 by default).

Models with 'vl' in their name are treated as vision-capable.
Cost is always 0.0 (local inference).
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import ollama
from PIL import Image

from app.core.logging import get_logger
from app.providers.base import (
    Message,
    ModelResponse,
    OCRBlock,
    OCRResult,
    ProviderCapability,
    ProviderError,
    ToolCall,
    ToolDefinition,
)

logger = get_logger(__name__)


class OllamaProvider:
    """Local LLM and VLM provider backed by Ollama."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        supports_vision: bool | None = None,
        max_output_tokens: int | None = None,
        is_reasoning_model: bool | None = None,
        reasoning_headroom: int | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._client = ollama.AsyncClient(host=base_url)

        is_vision_model = (
            supports_vision
            if supports_vision is not None
            else (
                "vl" in model.lower()
                or "gemma4:26b" in model.lower()
                or "vision" in model.lower()
                or "llava" in model.lower()
            )
        )
        is_reasoning = (
            is_reasoning_model
            if is_reasoning_model is not None
            else any(k in model.lower() for k in ("r1", "nemotron", "qwq", "thinking"))
        )
        max_out = max_output_tokens if max_output_tokens is not None else (8192 if is_reasoning else 4096)
        headroom = (
            reasoning_headroom
            if reasoning_headroom is not None
            else (2048 if is_reasoning else 0)
        )

        self._capability = ProviderCapability(
            supports_vision=bool(is_vision_model),
            supports_tool_use=True,
            supports_streaming=True,
            supports_structured_output=True,
            context_window=128000,
            max_output_tokens=max_out,
            is_reasoning_model=is_reasoning,
            reasoning_headroom=headroom,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def _to_ollama_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m.content, str):
                result.append({"role": m.role, "content": m.content})
            else:
                # Multimodal: extract text parts and image parts
                text_parts = [p["text"] for p in m.content if p.get("type") == "text"]
                image_parts = [p["data"] for p in m.content if p.get("type") == "image"]
                result.append(
                    {
                        "role": m.role,
                        "content": " ".join(text_parts),
                        "images": image_parts,
                    }
                )
        return result

    def _to_ollama_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        response_schema: dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Run a chat completion via Ollama."""
        t0 = time.monotonic()

        # If structured output requested, append concise instruction to system message
        # (Ollama native format=response_schema enforces the schema via grammar sampling)
        if response_schema:
            system_injection = Message(
                role="system",
                content=(
                    "You MUST respond with a valid JSON object matching the required structure. "
                    "Return only the JSON object, no markdown fences or introductory text."
                ),
            )
            messages = [system_injection, *messages]

        ollama_messages = self._to_ollama_messages(messages)
        # Ensure sufficient context window for multimodal images + JSON schema generation
        ctx_size = max(max_tokens * 2, 16384)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": ollama_messages,
            "options": {
                "num_ctx": ctx_size,
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        has_images = any(
            isinstance(m.content, list)
            and any(isinstance(c, dict) and c.get("type") == "image" for c in m.content)
            for m in messages
        )
        if has_images:
            kwargs["options"]["repeat_penalty"] = 1.15
            if kwargs["options"].get("num_predict", 0) < 1536:
                kwargs["options"]["num_predict"] = 1536

        if response_schema:
            if not has_images and isinstance(response_schema, dict) and len(response_schema) > 1:
                kwargs["format"] = response_schema
            else:
                kwargs["format"] = "json"
        if tools:
            kwargs["tools"] = self._to_ollama_tools(tools)

        try:
            response = await self._client.chat(**kwargs)
        except Exception as e:
            if "format" in kwargs and kwargs["format"] != "json":
                try:
                    kwargs["format"] = "json"
                    response = await self._client.chat(**kwargs)
                except Exception:
                    raise ProviderError(f"Ollama API error: {e}") from e
            else:
                raise ProviderError(f"Ollama API error: {e}") from e

        latency_ms = round((time.monotonic() - t0) * 1000)
        raw_text: str = response.message.content or ""
        thinking_text: str = getattr(response.message, "thinking", "") or ""

        # Fallback to extracting response from thinking tokens if content was starved
        if not raw_text.strip() and thinking_text.strip():
            match = re.search(
                r"(\{[^{}]*\"(?:summary|markdown_table|visual_type|key_metrics|regions)\"[^}]*\}|\{.*\})",
                thinking_text,
                re.DOTALL,
            )
            raw_text = match.group(1) if match else ("" if response_schema else thinking_text)

        # Strip reasoning / thinking tokens (e.g. <thought>...</thought>, <think>...</think>)
        cleaned_text = re.sub(r"<(thought|think)>.*?</\1>", "", raw_text, flags=re.DOTALL).strip()
        text = cleaned_text or raw_text

        # Parse structured output if requested
        parsed: Any | None = None
        if response_schema and text:
            try:
                parsed = json.loads(text.strip())
            except json.JSONDecodeError:
                # 1. Attempt regex recovery for outer [...] or {...}
                match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
                if match:
                    with contextlib.suppress(Exception):
                        parsed = json.loads(match.group(1))

                # 2. Attempt comma-separated objects wrapped as a list
                if not parsed:
                    with contextlib.suppress(Exception):
                        parsed = json.loads(f"[{text.strip().rstrip(',')}]")

                # 3. Stream JSONDecoder.raw_decode recovery across the text
                if not parsed:
                    decoder = json.JSONDecoder()
                    idx = 0
                    objs: list[Any] = []
                    while idx < len(text):
                        start = text.find("{", idx)
                        if start == -1:
                            break
                        try:
                            obj, end = decoder.raw_decode(text, idx=start)
                            objs.append(obj)
                            idx = end
                        except json.JSONDecodeError:
                            idx = start + 1
                    if objs:
                        if len(objs) == 1 and isinstance(objs[0], dict) and "regions" in objs[0]:
                            parsed = objs[0]
                        else:
                            parsed = objs

                # 4. Attempt recovery from thinking tokens if not yet parsed
                if not parsed and thinking_text:
                    match_think = re.search(r"(\[.*\]|\{.*\})", thinking_text, re.DOTALL)
                    if match_think:
                        with contextlib.suppress(Exception):
                            parsed = json.loads(match_think.group(1))
                    if not parsed:
                        with contextlib.suppress(Exception):
                            parsed = json.loads(f"[{thinking_text.strip().rstrip(',')}]")

                if not parsed:
                    logger.warning(
                        "Failed to parse structured JSON from Ollama response",
                        extra={"model": self._model, "text_preview": text[:200]},
                    )

        # Extract tool calls if present
        tool_calls: list[ToolCall] = []
        if hasattr(response.message, "tool_calls") and response.message.tool_calls:
            for tc in response.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        tool_input=tc.function.arguments or {},
                    )
                )

        input_tokens = getattr(response, "prompt_eval_count", 0) or 0
        output_tokens = getattr(response, "eval_count", 0) or 0

        logger.info(
            "Ollama completion",
            extra={
                "model": self._model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
            },
        )

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            provider="ollama",
            parsed=parsed,
            raw=response,
        )

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming text completion via Ollama."""
        ollama_messages = self._to_ollama_messages(messages)
        ctx_size = max(max_tokens * 2, 16384)
        try:
            async for chunk in await self._client.chat(
                model=self._model,
                messages=ollama_messages,
                stream=True,
                options={
                    "num_ctx": ctx_size,
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            ):
                if chunk.message.content:
                    yield chunk.message.content
        except ollama.ResponseError as e:
            raise ProviderError(f"Ollama streaming error: {e}") from e

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Analyze an image using an Ollama vision model."""
        if not self._capability.supports_vision:
            raise ProviderError(
                f"Model {self._model!r} does not support vision. "
                "Use a model with 'vl' in the name (e.g. qwen2.5vl:7b)."
            )
        # Defensive image resizing: clamp max dimension to 1024px to prevent
        # Vision Transformer patch token overflow on local Ollama VLM instances
        try:
            img = Image.open(io.BytesIO(image_bytes))
            max_dim = 1024
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img_resized.save(buf, format="PNG")
                image_bytes = buf.getvalue()
        except Exception:
            pass

        image_b64 = base64.b64encode(image_bytes).decode()
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "image", "data": image_b64},
                    {"type": "text", "text": prompt},
                ],
            )
        ]
        return await self.complete(
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
        )

    async def ocr(
        self,
        image_bytes: bytes,
        lang_hint: str | None = None,
    ) -> OCRResult:
        """Run OCR transcription on an image using Ollama VLM."""
        if not self._capability.supports_vision:
            return OCRResult(blocks=[], full_text="", mean_confidence=0.0)

        prompt = (
            "Transcribe all printed newspaper text accurately from this image. "
            "Preserve paragraphs, columns, and headings verbatim."
        )
        try:
            resp = await self.analyze_image(image_bytes, prompt=prompt)
            full_text = resp.text or ""
            lines = [line.strip() for line in full_text.split("\n") if line.strip()]
            blocks = [
                OCRBlock(
                    text=line,
                    bbox=(0.0, float(i * 20), 1000.0, float((i + 1) * 20)),
                    confidence=0.95,
                    language=lang_hint or "en",
                )
                for i, line in enumerate(lines)
            ]
            return OCRResult(
                blocks=blocks,
                full_text=full_text,
                mean_confidence=0.95 if blocks else 0.0,
                language=lang_hint or "en",
            )
        except Exception as e:
            logger.warning(
                "Ollama VLM OCR failed",
                extra={"model": self._model, "error": str(e)},
            )
            return OCRResult(blocks=[], full_text="", mean_confidence=0.0)
