"""NVIDIA NIM provider: high-performance hosted inference via NVIDIA API Catalog.

Implements ChatModelProvider and VisionModelProvider for models such as:
- nvidia/nemotron-3.5-lightning-30b-a3b (Agentic planning, synthesis with native reasoning)
- meta/llama-3.2-11b-vision-instruct (Multimodal document, chart, and photo reasoning)
- meta/llama-3.2-90b-vision-instruct

Uses the OpenAI-compatible endpoint (https://integrate.api.nvidia.com/v1) via AsyncOpenAI.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI

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

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaProvider:
    """Hosted LLM and Vision provider backed by NVIDIA NIM / NVIDIA API Catalog."""

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        supports_vision: bool | None = None,
        context_window: int = 128000,
        max_output_tokens: int | None = None,
        is_reasoning_model: bool | None = None,
        reasoning_headroom: int | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("NVIDIA API key is required. Set NVIDIA_API_KEY in your .env file.")

        self._model = model
        self._base_url = base_url or DEFAULT_NVIDIA_BASE_URL
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=api_key,
        )

        # Infer vision support if not explicitly set
        is_vision = (
            supports_vision
            if supports_vision is not None
            else any(kw in model.lower() for kw in ("vision", "vl", "vila", "neva", "multimodal"))
        )
        is_reasoning = (
            is_reasoning_model
            if is_reasoning_model is not None
            else any(kw in model.lower() for kw in ("nemotron", "r1", "reasoning"))
        )
        max_out = max_output_tokens if max_output_tokens is not None else (8192 if is_reasoning else 4096)
        headroom = (
            reasoning_headroom
            if reasoning_headroom is not None
            else (2048 if is_reasoning else 0)
        )

        self._capability = ProviderCapability(
            supports_vision=is_vision,
            supports_tool_use=True,
            supports_streaming=True,
            supports_structured_output=True,
            context_window=context_window,
            max_output_tokens=max_out,
            is_reasoning_model=is_reasoning,
            reasoning_headroom=headroom,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "nvidia"

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m.content, str):
                out.append({"role": m.role, "content": m.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in m.content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image":
                        data_str = part.get("data", "")
                        # If already a data URI, pass as-is, otherwise prefix standard base64 png header
                        if not data_str.startswith("data:"):
                            url = f"data:image/png;base64,{data_str}"
                        else:
                            url = data_str
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                out.append({"role": m.role, "content": parts})
        return out

    def _to_openai_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
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
        """Run a chat completion via NVIDIA NIM."""
        t0 = time.monotonic()
        oai_messages = self._to_openai_messages(messages)

        # Structured output prompt guidance when schema is provided
        if response_schema:
            schema_instr = (
                "You MUST respond with a valid JSON object matching the required schema. "
                "Return only the JSON object, with no markdown fences or conversational text."
            )
            if oai_messages and oai_messages[0].get("role") == "system":
                oai_messages[0]["content"] = f"{oai_messages[0]['content']}\n\n{schema_instr}"
            else:
                oai_messages.insert(0, {"role": "system", "content": schema_instr})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ProviderError(f"NVIDIA API completion error for model {self._model}: {e}") from e

        choice = response.choices[0]
        raw_content = choice.message.content or ""
        reasoning_content = getattr(choice.message, "reasoning_content", None) or ""

        # Wrap reasoning trace in <think> tags if present and not already enclosed
        if reasoning_content and "<think>" not in raw_content:
            text = f"<think>\n{reasoning_content.strip()}\n</think>\n\n{raw_content.strip()}".strip()
        else:
            text = raw_content.strip()

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {"raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        tool_input=fn_args,
                        tool_use_id=tc.id or "",
                    )
                )

        parsed: Any = None
        if response_schema and text:
            try:
                # If text contains <think> tags, strip them before JSON parsing
                json_candidate = text
                if "</think>" in json_candidate:
                    json_candidate = json_candidate.split("</think>", 1)[-1].strip()
                parsed = json.loads(json_candidate)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response from NVIDIA model", extra={"raw_text": text[:200]})

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        latency_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "NVIDIA completion",
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
            provider="nvidia",
            parsed=parsed,
            raw=response,
        )

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming completion via NVIDIA NIM.
        
        Transcribes native reasoning deltas into <think>...</think> blocks for progressive UI rendering.
        """
        oai_messages = self._to_openai_messages(messages)
        try:
            stream_resp = await self._client.chat.completions.create(
                model=self._model,
                messages=cast(Any, oai_messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            is_thinking = False
            async for chunk in cast(AsyncIterator[Any], stream_resp):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)

                if reasoning:
                    if not is_thinking:
                        yield "<think>\n"
                        is_thinking = True
                    yield reasoning
                elif content:
                    if is_thinking:
                        yield "\n</think>\n\n"
                        is_thinking = False
                    yield content

            if is_thinking:
                yield "\n</think>\n\n"

        except Exception as e:
            raise ProviderError(f"NVIDIA streaming error for model {self._model}: {e}") from e

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Analyze an image with prompt using NVIDIA multimodal vision endpoints."""
        if not self._capability.supports_vision:
            raise ProviderError(f"Model {self._model} does not support vision/image analysis.")

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": prompt},
                {"type": "image", "data": b64},
            ],
        )
        return await self.complete(
            [msg],
            response_schema=response_schema,
            max_tokens=max_tokens,
        )
