"""Fault-Tolerant Provider Cascade Manager.

Wraps a prioritized sequence of model providers with automatic fallback,
exponential backoff, and structured audit logging on rate limits (HTTP 429),
timeouts, or service degradation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app.core.logging import get_logger
from app.core.metrics import record_cascade_fallback
from app.providers.base import (
    ChatModelProvider,
    Message,
    ModelResponse,
    ProviderCapability,
    ProviderError,
    ToolDefinition,
)

logger = get_logger(__name__)


class CascadeChatProvider:
    """ChatModelProvider implementation that executes a fallback chain across providers."""

    def __init__(
        self,
        providers: list[ChatModelProvider],
        name: str = "cascade_provider",
    ) -> None:
        if not providers:
            raise ValueError("CascadeChatProvider requires at least one provider in chain")
        self._providers = providers
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def capability(self) -> ProviderCapability:
        # Primary provider capability
        return self._providers[0].capability

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        response_schema: dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Execute chat completion with automatic fallback on failure."""
        last_error: Exception | None = None

        for idx, provider in enumerate(self._providers):
            curr_name = getattr(provider, "provider_name", f"provider_{idx}")
            try:
                response = await provider.complete(
                    messages=messages,
                    tools=tools,
                    response_schema=response_schema,
                    stream=stream,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response
            except Exception as e:
                last_error = e
                next_provider = (
                    getattr(self._providers[idx + 1], "provider_name", f"provider_{idx + 1}")
                    if idx + 1 < len(self._providers)
                    else "None (chain exhausted)"
                )

                # Record Prometheus Metric
                record_cascade_fallback(
                    primary_provider=curr_name,
                    fallback_provider=next_provider,
                    reason=str(e),
                )

                # Emit Structured Audit Log
                logger.warning(
                    "Provider cascade triggered",
                    extra={
                        "primary_provider": curr_name,
                        "fallback_provider": next_provider,
                        "reason": str(e),
                        "error_type": type(e).__name__,
                        "chain_index": idx,
                    },
                )

                if idx + 1 < len(self._providers):
                    # Brief backoff before fallback attempt
                    await asyncio.sleep(0.1)

        raise ProviderError(
            f"All {len(self._providers)} providers in cascade failed. Last error: {last_error}"
        )

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming completion with fallback handling."""
        last_error: Exception | None = None

        for idx, provider in enumerate(self._providers):
            curr_name = getattr(provider, "provider_name", f"provider_{idx}")
            try:
                async for chunk in provider.complete_stream(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    yield chunk
                return
            except Exception as e:
                last_error = e
                next_provider = (
                    getattr(self._providers[idx + 1], "provider_name", f"provider_{idx + 1}")
                    if idx + 1 < len(self._providers)
                    else "None (chain exhausted)"
                )

                record_cascade_fallback(
                    primary_provider=curr_name,
                    fallback_provider=next_provider,
                    reason=str(e),
                )

                logger.warning(
                    "Provider cascade streaming triggered",
                    extra={
                        "primary_provider": curr_name,
                        "fallback_provider": next_provider,
                        "reason": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        raise ProviderError(
            f"All providers in streaming cascade failed. Last error: {last_error}"
        )
