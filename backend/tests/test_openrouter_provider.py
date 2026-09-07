"""Tests for OpenRouter Provider, KeyRotator, and Multimodal Vision."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError

from app.providers.base import (
    ChatModelProvider,
    Message,
    ModelResponse,
    VisionModelProvider,
)
from app.providers.openrouter_provider import (
    KeyRotator,
    OpenRouterProvider,
    RateLimitExhaustedError,
)


class TestProtocolConformance:
    """Verify OpenRouterProvider satisfies ChatModelProvider and VisionModelProvider."""

    def test_satisfies_chat_and_vision_protocols(self) -> None:
        provider = OpenRouterProvider(
            model="google/gemma-4-26b-a4b-it:free",
            api_keys=["key-1", "key-2"],
            supports_vision=True,
        )
        assert isinstance(provider, ChatModelProvider)
        assert isinstance(provider, VisionModelProvider)
        assert provider.provider_name == "openrouter"
        assert provider.capability.supports_vision is True
        assert provider.capability.context_window == 262144

    def test_nemotron_capability(self) -> None:
        provider = OpenRouterProvider(
            model="nvidia/nemotron-3.5-lightning:free",
            api_keys=["key-1", "key-2"],
            supports_vision=False,
        )
        assert provider.capability.supports_vision is False
        assert provider.capability.context_window == 1_000_000


class TestKeyRotator:
    """Verify KeyRotator round-robin, cooldown, and exhaustion behavior."""

    @pytest.mark.asyncio
    async def test_round_robin_rotation(self) -> None:
        rotator = KeyRotator(["key-a", "key-b"])
        k1, idx1 = await rotator.get_next_key()
        k2, idx2 = await rotator.get_next_key()
        k3, idx3 = await rotator.get_next_key()

        assert (k1, idx1) == ("key-a", 0)
        assert (k2, idx2) == ("key-b", 1)
        assert (k3, idx3) == ("key-a", 0)

    @pytest.mark.asyncio
    async def test_cooldown_skips_rate_limited_key(self) -> None:
        rotator = KeyRotator(["key-a", "key-b"])
        # Mark key-a as rate-limited for 10s
        await rotator.mark_rate_limited("key-a", cooldown_seconds=10.0)

        # Both consecutive requests should yield key-b
        k1, idx1 = await rotator.get_next_key()
        k2, idx2 = await rotator.get_next_key()

        assert (k1, idx1) == ("key-b", 1)
        assert (k2, idx2) == ("key-b", 1)

    @pytest.mark.asyncio
    async def test_all_keys_exhausted_raises_error(self) -> None:
        rotator = KeyRotator(["key-a", "key-b"])
        await rotator.mark_rate_limited("key-a", cooldown_seconds=5.0)
        await rotator.mark_rate_limited("key-b", cooldown_seconds=8.0)

        with pytest.raises(RateLimitExhaustedError) as exc_info:
            await rotator.get_next_key()

        assert "rate-limited" in str(exc_info.value)
        assert exc_info.value.retry_after_seconds > 0.0

    @pytest.mark.asyncio
    async def test_are_all_keys_rate_limited(self) -> None:
        rotator = KeyRotator(["key-a", "key-b"])
        assert not rotator.are_all_keys_rate_limited()

        await rotator.mark_rate_limited("key-a", cooldown_seconds=10.0)
        assert not rotator.are_all_keys_rate_limited()

        await rotator.mark_rate_limited("key-b", cooldown_seconds=15.0)
        assert rotator.are_all_keys_rate_limited()
        assert rotator.get_shortest_cooldown_remaining() > 0.0

        await rotator.clear_cooldown("key-a")
        assert not rotator.are_all_keys_rate_limited()


class TestOpenRouterProviderInference:
    """Verify complete() and analyze_image() with failover and formatting."""

    @pytest.mark.asyncio
    async def test_complete_returns_model_response(self) -> None:
        provider = OpenRouterProvider(
            model="google/gemma-4-26b-a4b-it:free",
            api_keys=["key-a", "key-b"],
        )
        mock_choice = MagicMock()
        mock_choice.message.content = "OpenRouter response text"
        mock_choice.message.tool_calls = None
        mock_response = MagicMock(choices=[mock_choice], usage=MagicMock(prompt_tokens=15, completion_tokens=8))

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(provider, "_get_client", return_value=mock_client):
            resp = await provider.complete([Message(role="user", content="Hello OpenRouter")])
            assert isinstance(resp, ModelResponse)
            assert resp.text == "OpenRouter response text"
            assert resp.provider == "openrouter"
            assert resp.input_tokens == 15
            assert resp.output_tokens == 8

    @pytest.mark.asyncio
    async def test_automatic_429_failover_to_second_key(self) -> None:
        provider = OpenRouterProvider(
            model="google/gemma-4-26b-a4b-it:free",
            api_keys=["key-a", "key-b"],
        )

        mock_choice = MagicMock()
        mock_choice.message.content = "Recovered on Key B"
        mock_choice.message.tool_calls = None
        mock_success_response = MagicMock(choices=[mock_choice], usage=MagicMock(prompt_tokens=10, completion_tokens=5))

        client_a = AsyncMock()
        client_a.chat.completions.create.side_effect = Exception("HTTP 429 Too Many Requests: Rate limit exceeded")

        client_b = AsyncMock()
        client_b.chat.completions.create.return_value = mock_success_response

        def client_selector(key: str) -> AsyncMock:
            return client_a if key == "key-a" else client_b

        with patch.object(provider, "_get_client", side_effect=client_selector):
            resp = await provider.complete([Message(role="user", content="Test prompt")])
            assert resp.text == "Recovered on Key B"

    @pytest.mark.asyncio
    async def test_total_exhaustion_raises_rate_limit_exhausted_error(self) -> None:
        provider = OpenRouterProvider(
            model="google/gemma-4-26b-a4b-it:free",
            api_keys=["key-a", "key-b"],
        )

        client_failing = AsyncMock()
        client_failing.chat.completions.create.side_effect = Exception("429 rate limit reached")

        with patch.object(provider, "_get_client", return_value=client_failing):
            with pytest.raises(RateLimitExhaustedError):
                await provider.complete([Message(role="user", content="Will fail")])

    @pytest.mark.asyncio
    async def test_analyze_image_wraps_data_url(self) -> None:
        provider = OpenRouterProvider(
            model="google/gemma-4-26b-a4b-it:free",
            api_keys=["key-a", "key-b"],
            supports_vision=True,
        )

        mock_choice = MagicMock()
        mock_choice.message.content = '{"summary": "Chart of revenue", "markdown_table": "| Q1 | 100 |"}'
        mock_choice.message.tool_calls = None
        mock_resp = MagicMock(choices=[mock_choice], usage=MagicMock(prompt_tokens=50, completion_tokens=20))

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_resp

        # Create 10-byte dummy PNG image
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00"

        with patch.object(provider, "_get_client", return_value=mock_client):
            resp = await provider.analyze_image(
                image_bytes=png_bytes,
                prompt="Extract financial data",
            )
            assert resp.parsed is not None or "Chart of revenue" in resp.text

            # Verify call arguments wrapped in data:image/png;base64,...
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            assert len(messages) == 1
            user_content = messages[0]["content"]
            assert any(
                p.get("type") == "image_url" and "data:image/png;base64," in p["image_url"]["url"]
                for p in user_content
            )

    @pytest.mark.asyncio
    async def test_complete_stream_with_failover(self) -> None:
        provider = OpenRouterProvider(
            model="nvidia/nemotron-3.5-lightning:free",
            api_keys=["key-a", "key-b"],
        )

        async def fake_stream_b():
            for c in ["Here's ", "a response"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=c))]
                yield chunk

        client_a = AsyncMock()
        client_a.chat.completions.create.side_effect = Exception("429 Too Many Requests")

        client_b = AsyncMock()
        client_b.chat.completions.create.return_value = fake_stream_b()

        def client_selector(key: str) -> AsyncMock:
            return client_a if key == "key-a" else client_b

        with patch.object(provider, "_get_client", side_effect=client_selector):
            chunks = []
            async for chunk in provider.complete_stream([Message(role="user", content="Hi")]):
                chunks.append(chunk)

            assert "".join(chunks) == "Here's a response"
            # Verify penalties were applied for Nemotron
            call_kwargs = client_b.chat.completions.create.call_args[1]
            assert call_kwargs["frequency_penalty"] == 0.35
            assert call_kwargs["presence_penalty"] == 0.15
            assert call_kwargs["temperature"] >= 0.3
