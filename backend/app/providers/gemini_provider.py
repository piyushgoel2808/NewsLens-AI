"""Google Gemini provider: high-performance inference via Google Generative Language API.

Implements ChatModelProvider and VisionModelProvider for models like:
- gemini-3.7-flash
- gemini-3.5-flash
- gemini-3.5-flash-lite
- gemini-3.1-pro-preview
- gemini-2.5-flash
- gemini-flash-latest
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.logging import get_logger
from app.providers.base import (
    Message,
    ModelResponse,
    ProviderCapability,
    ProviderError,
    ToolCall,
    ToolDefinition,
)

logger = get_logger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider:
    """Hosted LLM and Vision provider backed by Google Gemini API."""

    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise ProviderError(
                "Google Gemini API key is required. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
            )
        self._model = model.replace("models/", "") if model else "gemini-flash-latest"
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

        url = f"{GEMINI_API_BASE}/{self._model}:generateContent?key={self._api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                res = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except Exception as e:
                raise ProviderError(f"HTTP request to Gemini API failed: {e}") from e

        if res.status_code != 200:
            err_data = (
                res.json()
                if res.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            err_msg = err_data.get("error", {}).get("message", res.text)
            raise ProviderError(f"Gemini API returned {res.status_code}: {err_msg}")

        data = res.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ModelResponse(
                text="",
                model=self._model,
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

        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", 0)
        out_tok = usage.get("candidatesTokenCount", 0)
        lat_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "Gemini completion",
            extra={
                "model": self._model,
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
            model=self._model,
            provider=self.provider_name,
            raw=data,
        )

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming chat completion yielding text deltas."""
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

        url = f"{GEMINI_API_BASE}/{self._model}:streamGenerateContent?key={self._api_key}&alt=sse"

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
                msg = f"Gemini streaming error ({response.status_code}): {err_text.decode('utf-8')}"
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

        usage = data.get("usageMetadata", {})
        return ModelResponse(
            text=text,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            model=self._model,
            provider=self.provider_name,
            raw=data,
        )
