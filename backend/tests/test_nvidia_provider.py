"""Tests for NVIDIA NIM Provider, protocol conformance, reasoning extraction, and vision."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.providers.base import (
    ChatModelProvider,
    Message,
    ModelResponse,
    ProviderError,
    VisionModelProvider,
)
from app.providers.nvidia_provider import NvidiaProvider


class TestNvidiaProtocolConformance:
    """Verify NvidiaProvider satisfies ChatModelProvider and VisionModelProvider."""

    def test_satisfies_chat_and_vision_protocols(self) -> None:
        provider = NvidiaProvider(
            model="meta/llama-3.2-11b-vision-instruct",
            api_key="test-key",
            supports_vision=True,
        )
        assert isinstance(provider, ChatModelProvider)
        assert isinstance(provider, VisionModelProvider)
        assert provider.provider_name == "nvidia"
        assert provider.capability.supports_vision is True
        assert provider.capability.supports_streaming is True
        assert provider.capability.supports_tool_use is True

    def test_nemotron_capability(self) -> None:
        provider = NvidiaProvider(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key="test-key",
            supports_vision=False,
        )
        assert provider.capability.supports_vision is False
        assert provider.capability.context_window == 128000

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ProviderError, match="NVIDIA API key is required"):
            NvidiaProvider(model="test-model", api_key=None)


class TestNvidiaCompletions:
    """Verify reasoning parsing, tool calling, and structured JSON parsing."""

    @pytest.mark.asyncio
    async def test_complete_with_reasoning_content(self) -> None:
        provider = NvidiaProvider(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key="test-key",
        )

        mock_choice = MagicMock()
        mock_choice.message.content = "Final synthesized brief."
        mock_choice.message.reasoning_content = "Step 1: Check archives. Step 2: Formulate."
        mock_choice.message.tool_calls = None

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 42
        mock_resp.usage.completion_tokens = 84

        with patch.object(provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_resp
            resp = await provider.complete([Message(role="user", content="Test query")])

            assert isinstance(resp, ModelResponse)
            assert "<think>" in resp.text
            assert "Step 1: Check archives" in resp.text
            assert "Final synthesized brief." in resp.text
            assert resp.input_tokens == 42
            assert resp.output_tokens == 84

    @pytest.mark.asyncio
    async def test_complete_with_structured_schema(self) -> None:
        provider = NvidiaProvider(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key="test-key",
        )

        mock_choice = MagicMock()
        mock_choice.message.content = '{"archetype": "cross_newspaper_comparison", "confidence": 0.98}'
        mock_choice.message.reasoning_content = "Planning steps..."
        mock_choice.message.tool_calls = None

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 30
        mock_resp.usage.completion_tokens = 50

        with patch.object(provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_resp
            resp = await provider.complete(
                [Message(role="user", content="Plan this")],
                response_schema={"type": "object"},
            )

            assert resp.parsed == {"archetype": "cross_newspaper_comparison", "confidence": 0.98}

    @pytest.mark.asyncio
    async def test_streaming_with_reasoning_deltas(self) -> None:
        provider = NvidiaProvider(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key="test-key",
        )

        class MockDelta:
            def __init__(self, content=None, reasoning_content=None):
                self.content = content
                self.reasoning_content = reasoning_content

        class MockChunk:
            def __init__(self, delta):
                self.choices = [MagicMock(delta=delta)]

        async def mock_stream_generator():
            yield MockChunk(MockDelta(reasoning_content="Thinking part 1. "))
            yield MockChunk(MockDelta(reasoning_content="Thinking part 2."))
            yield MockChunk(MockDelta(content="Final answer chunk."))

        with patch.object(provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream_generator()
            tokens = []
            async for token in provider.complete_stream([Message(role="user", content="Hi")]):
                tokens.append(token)

            full_stream = "".join(tokens)
            assert full_stream.startswith("<think>\nThinking part 1. Thinking part 2.\n</think>\n\nFinal answer chunk.")

    @pytest.mark.asyncio
    async def test_analyze_image(self) -> None:
        provider = NvidiaProvider(
            model="meta/llama-3.2-11b-vision-instruct",
            api_key="test-key",
            supports_vision=True,
        )

        mock_choice = MagicMock()
        mock_choice.message.content = "Quarterly revenue chart showing 15% increase."
        mock_choice.message.reasoning_content = None
        mock_choice.message.tool_calls = None

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 120
        mock_resp.usage.completion_tokens = 25

        with patch.object(provider._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_resp
            resp = await provider.analyze_image(b"fake_image_bytes", "Transcribe chart")

            assert "Quarterly revenue chart" in resp.text
            call_kwargs = mock_create.call_args[1]
            user_msg = call_kwargs["messages"][0]["content"]
            assert user_msg[0]["type"] == "text"
            assert user_msg[1]["type"] == "image_url"
