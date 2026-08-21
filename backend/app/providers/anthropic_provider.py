"""Anthropic provider: Claude models via the anthropic Python SDK.

Implements ChatModelProvider and VisionModelProvider.
Reads ANTHROPIC_API_KEY from environment / Settings.
Raises ProviderError at construction time if the key is not set.
"""
from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic

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

# Approximate pricing per million tokens (USD).
# Update when Anthropic adjusts pricing.
_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-3-5": {"input": 0.8, "output": 4.0},
    "claude-opus-4-5": {"input": 15.0, "output": 75.0},
    # Default fallback
    "default": {"input": 3.0, "output": 15.0},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _PRICING.get(model, _PRICING["default"])
    return (
        input_tokens / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )


class AnthropicProvider:
    """Hosted LLM and VLM provider backed by the Anthropic API."""

    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise ProviderError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY in your .env file."
            )
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
        self._capability = ProviderCapability(
            supports_vision=True,
            supports_tool_use=True,
            supports_streaming=True,
            supports_structured_output=True,
            context_window=200000,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _to_anthropic_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split system message out and convert to Anthropic format."""
        system: str | None = None
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system = m.content if isinstance(m.content, str) else str(m.content)
            elif isinstance(m.content, str):
                out.append({"role": m.role, "content": m.content})
            else:
                # Multimodal content list
                parts: list[dict[str, Any]] = []
                for part in m.content:
                    if part["type"] == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image":
                        parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.get("media_type", "image/png"),
                                "data": part["data"],
                            },
                        })
                out.append({"role": m.role, "content": parts})
        return system, out

    def _to_anthropic_tools(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
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
        """Run a chat completion via the Anthropic Messages API."""
        t0 = time.monotonic()

        # Inject structured output instruction
        if response_schema:
            schema_str = json.dumps(response_schema)
            inject = Message(
                role="system",
                content=(
                    f"You MUST respond with valid JSON matching this schema:\n{schema_str}\n"
                    "Return ONLY the JSON object, no markdown fences or explanations."
                ),
            )
            messages = [inject, *messages]

        system, anthro_messages = self._to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthro_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        try:
            response = await self._async_client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise ProviderError(f"Anthropic auth failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise ProviderError(f"Anthropic rate limit: {e}") from e
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {e}") from e

        latency_ms = round((time.monotonic() - t0) * 1000)

        # Extract text and tool calls from response content blocks
        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        tool_name=block.name,
                        tool_input=block.input,
                        tool_use_id=block.id,
                    )
                )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Parse structured output if requested
        parsed: Any | None = None
        if response_schema and text.strip():
            try:
                parsed = json.loads(text.strip())
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse structured JSON from Anthropic response",
                    extra={"model": self._model, "text_preview": text[:200]},
                )

        cost = _estimate_cost(self._model, input_tokens, output_tokens)
        logger.info(
            "Anthropic completion",
            extra={
                "model": self._model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
            },
        )

        result = ModelResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            provider="anthropic",
            parsed=parsed,
            raw=response,
        )
        # Attach cost as an attribute (overrides base property)
        object.__setattr__(result, "_cost_usd", cost)
        return result

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming text completion via Anthropic."""
        system, anthro_messages = self._to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthro_messages,
        }
        if system:
            kwargs["system"] = system
        try:
            async with self._async_client.messages.stream(**kwargs) as stream:
                async for text_chunk in stream.text_stream:
                    yield text_chunk
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic streaming error: {e}") from e

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Analyze an image via Claude vision."""
        image_b64 = base64.b64encode(image_bytes).decode()
        messages = [
            Message(
                role="user",
                content=[
                    {
                        "type": "image",
                        "data": image_b64,
                        "media_type": "image/png",
                    },
                    {"type": "text", "text": prompt},
                ],
            )
        ]
        return await self.complete(
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
        )
