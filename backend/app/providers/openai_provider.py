"""OpenAI provider: GPT models + text-embedding via the openai Python SDK.

Implements ChatModelProvider and EmbeddingProvider.
Raises ProviderError at construction time if OPENAI_API_KEY is not set.
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

# Approximate pricing per million tokens
_CHAT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "default": {"input": 5.0, "output": 15.0},
}
_EMBED_PRICING: dict[str, float] = {
    "text-embedding-3-large": 0.13,  # per million tokens
    "text-embedding-3-small": 0.02,
    "default": 0.13,
}


class OpenAIProvider:
    """Hosted LLM + Embedding provider backed by the OpenAI API."""

    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise ProviderError("OpenAI API key is required. Set OPENAI_API_KEY in your .env file.")
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)
        self._capability = ProviderCapability(
            supports_vision="gpt-4" in model.lower(),
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
        return "openai"

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m.content, str):
                out.append({"role": m.role, "content": m.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in m.content:
                    if part["type"] == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image":
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{part['data']}"},
                            }
                        )
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
        """Run a chat completion via the OpenAI Chat API."""
        t0 = time.monotonic()

        if response_schema:
            schema_str = json.dumps(response_schema)
            inject = {
                "role": "system",
                "content": (f"Respond ONLY with valid JSON matching this schema:\n{schema_str}"),
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

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {e}") from e

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
        pricing = _CHAT_PRICING.get(self._model, _CHAT_PRICING["default"])
        cost = (
            input_tokens / 1_000_000 * pricing["input"]
            + output_tokens / 1_000_000 * pricing["output"]
        )

        parsed: Any | None = None
        if response_schema and text.strip():
            try:
                parsed = json.loads(text.strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from OpenAI response")

        logger.info(
            "OpenAI completion",
            extra={
                "model": self._model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
            },
        )

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            provider="openai",
            parsed=parsed,
            raw=response,
        )

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming completion via OpenAI."""
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
            raise ProviderError(f"OpenAI streaming error: {e}") from e

    # --- EmbeddingProvider interface ---

    @property
    def embedding_dim(self) -> int:
        if "3-large" in self._model:
            return 3072
        if "3-small" in self._model:
            return 1536
        return 1536  # safe default

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via OpenAI Embeddings API (batches of up to 100)."""
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = await self._client.embeddings.create(model=self._model, input=batch)
            except Exception as e:
                raise ProviderError(f"OpenAI embedding error: {e}") from e
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]
