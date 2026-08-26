"""Ollama provider: local LLM and VLM via the Ollama Python client.

Implements ChatModelProvider and VisionModelProvider for any model served
by a running Ollama instance (http://localhost:11434 by default).

Models with 'vl' in their name are treated as vision-capable.
Cost is always 0.0 (local inference).
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import ollama

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
        self._capability = ProviderCapability(
            supports_vision=bool(is_vision_model),
            supports_tool_use=True,
            supports_streaming=True,
            supports_structured_output=True,
            context_window=128000,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "ollama"

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

        # If structured output requested, append instruction to system message
        if response_schema:
            schema_str = json.dumps(response_schema)
            system_injection = Message(
                role="system",
                content=(
                    f"You MUST respond with valid JSON that matches this schema exactly:\n"
                    f"{schema_str}\n"
                    "Return only the JSON object, no markdown fences."
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
        if response_schema:
            # Pass native response_schema to Ollama for grammar-constrained decoding
            kwargs["format"] = response_schema
        if tools:
            kwargs["tools"] = self._to_ollama_tools(tools)

        try:
            response = await self._client.chat(**kwargs)
        except ollama.ResponseError as e:
            # If Ollama server rejects complex schema dictionary, fallback to format="json"
            if response_schema and "format" in kwargs and kwargs["format"] != "json":
                try:
                    kwargs["format"] = "json"
                    response = await self._client.chat(**kwargs)
                except Exception:
                    raise ProviderError(f"Ollama API error: {e}") from e
            else:
                raise ProviderError(f"Ollama API error: {e}") from e
        except Exception as e:
            if response_schema and "format" in kwargs and kwargs["format"] != "json":
                try:
                    kwargs["format"] = "json"
                    response = await self._client.chat(**kwargs)
                except Exception:
                    raise ProviderError(f"Ollama request failed: {e}") from e
            else:
                raise ProviderError(f"Ollama request failed: {e}") from e

        latency_ms = round((time.monotonic() - t0) * 1000)
        raw_text: str = response.message.content or ""

        # Strip reasoning / thinking tokens (e.g. <thought>...</thought>, <think>...</think>)
        cleaned_text = re.sub(r"<(thought|think)>.*?</\1>", "", raw_text, flags=re.DOTALL).strip()
        text = cleaned_text or raw_text

        # Parse structured output if requested
        parsed: Any | None = None
        if response_schema and text:
            try:
                parsed = json.loads(text.strip())
            except json.JSONDecodeError:
                # Attempt regex recovery if direct json.loads fails
                match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                    except Exception:
                        pass
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
        try:
            async for chunk in await self._client.chat(
                model=self._model,
                messages=ollama_messages,
                stream=True,
                options={"num_predict": max_tokens, "temperature": temperature},
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
