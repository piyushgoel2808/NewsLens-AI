"""Tests for Fault-Tolerant Provider Cascade Manager."""

from __future__ import annotations

from typing import Any

import pytest

from app.providers.base import (
    Message,
    ModelResponse,
    ProviderCapability,
    ProviderError,
    ToolDefinition,
)
from app.providers.cascade import CascadeChatProvider


class FailingProvider:
    """Mock provider that always throws an error."""

    def __init__(self, name: str, error_msg: str) -> None:
        self._name = name
        self._error_msg = error_msg

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability()

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        response_schema: dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ModelResponse:
        raise ProviderError(self._error_msg)

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        raise ProviderError(self._error_msg)
        yield ""


class SuccessfulProvider:
    """Mock provider that succeeds."""

    def __init__(self, name: str, reply: str) -> None:
        self._name = name
        self._reply = reply

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability()

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        response_schema: dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ModelResponse:
        return ModelResponse(
            text=self._reply,
            model="mock_model",
            provider=self._name,
            input_tokens=50,
            output_tokens=20,
        )

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        for word in self._reply.split(" "):
            yield word + " "


@pytest.mark.asyncio
async def test_cascade_fallback_to_secondary_on_primary_failure() -> None:
    """Verify CascadeChatProvider seamlessly falls back to secondary provider."""
    failing = FailingProvider("primary_cloud", "HTTP 429 Rate Limit Exceeded")
    working = SuccessfulProvider("backup_local", "Fallback successful response.")

    cascade = CascadeChatProvider(
        providers=[failing, working],  # type: ignore[list-item]
        name="test_cascade",
    )

    response = await cascade.complete(
        messages=[Message(role="user", content="Hello research query")]
    )

    assert response.text == "Fallback successful response."
    assert response.provider == "backup_local"


@pytest.mark.asyncio
async def test_cascade_raises_when_all_providers_fail() -> None:
    """Verify CascadeChatProvider raises ProviderError when all providers fail."""
    failing1 = FailingProvider("primary", "Error 1")
    failing2 = FailingProvider("secondary", "Error 2")

    cascade = CascadeChatProvider(
        providers=[failing1, failing2],  # type: ignore[list-item]
        name="all_failing_cascade",
    )

    with pytest.raises(ProviderError) as exc_info:
        await cascade.complete(messages=[Message(role="user", content="Hello")])
    assert "All 2 providers in cascade failed" in str(exc_info.value)
