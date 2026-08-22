"""Groq provider: ultra-fast inference via Groq's OpenAI-compatible API.

Implements ChatModelProvider for models like:
- llama-3.3-70b-versatile
- llama-3.1-8b-instant
- deepseek-r1-distill-llama-70b
- mixtral-8x7b-32768
"""

from __future__ import annotations

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

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider:
    """Ultra-fast hosted LLM provider backed by Groq LPU inference."""

    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise ProviderError("Groq API key is required. Set GROQ_API_KEY in your .env file.")
        self._model = model
        self._client = AsyncOpenAI(
            base_url=GROQ_BASE_URL,
            api_key=api_key,
        )
        self._capability = ProviderCapability(
            supports_vision=False,
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
        return "groq"

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m.content, str):
                out.append({"role": m.role, "content": m.content})
            else:
                text_parts = [p["text"] for p in m.content if p.get("type") == "text"]
                out.append({"role": m.role, "content": " ".join(text_parts)})
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
        """Run a chat completion via Groq."""
        t0 = time.monotonic()

        if response_schema:
            schema_str = json.dumps(response_schema)
            inject = {
                "role": "system",
                "content": f"Respond ONLY with valid JSON matching this schema:\n{schema_str}",
            }
            oai_messages = [inject, *self._to_openai_messages(messages)]
        else:
            oai_messages = self._to_openai_messages(messages)

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
            raise ProviderError(f"Groq API error: {e}") from e

        latency_ms = round((time.monotonic() - t0) * 1000)
        choice = response.choices[0]
        text = choice.message.content or ""

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        tool_input=json.loads(tc.function.arguments or "{}"),
                        tool_use_id=tc.id,
                    )
                )

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        parsed: Any | None = None
        if response_schema and text.strip():
            try:
                parsed = json.loads(text.strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from Groq response")

        logger.info(
            "Groq completion",
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
            provider="groq",
            parsed=parsed,
            raw=response,
        )

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming completion via Groq."""
        oai_messages = self._to_openai_messages(messages)
        try:
            stream_resp = await self._client.chat.completions.create(
                model=self._model,
                messages=cast(Any, oai_messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in cast(AsyncIterator[Any], stream_resp):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            raise ProviderError(f"Groq streaming error: {e}") from e
