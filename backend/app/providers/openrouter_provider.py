"""OpenRouter Provider: Dual-Account Round-Robin Load Balancer with 429 Failover.

Implements ChatModelProvider and VisionModelProvider for hosted models via OpenRouter
(e.g., google/gemma-4-26b-a4b-it:free, nvidia/nemotron-3.5-lightning:free).

Features:
- Dual-Account KeyRotator distributing requests across multiple API keys in round-robin.
- Transparent HTTP 429 failover with per-key cooldown timers.
- RateLimitExhaustedError raised when all keys hit rate limits simultaneously.
- Strict OpenAI-compatible Multimodal Data URL payload formatting for analyze_image().
- Grammar-constrained / schema-driven JSON structured output parsing.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI, RateLimitError

from app.core.logging import get_logger
from app.providers.base import (
    ChatModelProvider,
    Message,
    ModelResponse,
    ProviderCapability,
    ProviderError,
    ToolCall,
    ToolDefinition,
    VisionModelProvider,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class RateLimitExhaustedError(ProviderError):
    """Raised when all configured API keys have been exhausted due to HTTP 429 rate limits."""

    def __init__(self, message: str, retry_after_seconds: float = 60.0) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


# ---------------------------------------------------------------------------
# Key Rotator with Transparent Failover & Cooldown Tracking
# ---------------------------------------------------------------------------


class KeyRotator:
    """Thread-safe round-robin API key rotator with active cooldown management."""

    def __init__(self, keys: list[str], default_cooldown_seconds: float = 60.0) -> None:
        if not keys:
            raise ProviderError("At least one OpenRouter API key is required.")
        self._keys = [k.strip() for k in keys if k.strip()]
        self._current_idx = 0
        self._cooldowns: dict[str, float] = {}  # key -> timestamp until cooldown expires
        self._lock = asyncio.Lock()
        self._default_cooldown = default_cooldown_seconds

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    async def get_next_key(self) -> tuple[str, int]:
        """Return the next available key not in cooldown and its 0-based index.

        Raises:
            RateLimitExhaustedError: If all keys are currently in cooldown.
        """
        async with self._lock:
            now = time.time()
            n = len(self._keys)

            # Search round-robin for an available key
            for offset in range(n):
                idx = (self._current_idx + offset) % n
                candidate = self._keys[idx]
                cooldown_until = self._cooldowns.get(candidate, 0.0)
                if now >= cooldown_until:
                    # Key is available
                    self._current_idx = (idx + 1) % n
                    return candidate, idx

            # All keys are currently in cooldown
            remaining_times = [
                max(0.0, self._cooldowns.get(k, 0.0) - now)
                for k in self._keys
            ]
            min_remaining = min(remaining_times) if remaining_times else self._default_cooldown
            wait_s = max(1.0, round(min_remaining, 1))

            raise RateLimitExhaustedError(
                f"All {n} OpenRouter API keys are currently rate-limited (HTTP 429). "
                f"Shortest cooldown expires in {wait_s}s.",
                retry_after_seconds=wait_s,
            )

    async def mark_rate_limited(self, key: str, cooldown_seconds: float | None = None) -> None:
        """Mark a specific key as rate-limited for the cooldown duration."""
        async with self._lock:
            cd = cooldown_seconds if cooldown_seconds is not None else self._default_cooldown
            self._cooldowns[key] = time.time() + cd
            logger.warning(
                "OpenRouter key marked as rate-limited",
                extra={"key_preview": f"{key[:8]}...{key[-4:]}", "cooldown_seconds": cd},
            )

    async def clear_cooldown(self, key: str) -> None:
        """Clear cooldown status for a key upon successful response."""
        async with self._lock:
            self._cooldowns.pop(key, None)

    def are_all_keys_rate_limited(self) -> bool:
        """Synchronously check if all keys are currently in cooldown."""
        now = time.time()
        return all(now < self._cooldowns.get(k, 0.0) for k in self._keys)

    def get_shortest_cooldown_remaining(self) -> float:
        """Return remaining seconds until the next key cooldown expires."""
        now = time.time()
        remaining = [
            max(0.0, self._cooldowns.get(k, 0.0) - now)
            for k in self._keys
        ]
        return min(remaining) if remaining else 0.0


# ---------------------------------------------------------------------------
# OpenRouter Provider
# ---------------------------------------------------------------------------


class OpenRouterProvider(ChatModelProvider, VisionModelProvider):
    """Hosted LLM and VLM provider connecting to OpenRouter with multi-key load balancing."""

    def __init__(
        self,
        model: str,
        api_keys: list[str],
        base_url: str = "https://openrouter.ai/api/v1",
        supports_vision: bool | None = None,
        context_window: int | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._rotator = KeyRotator(api_keys)
        self._clients: dict[str, AsyncOpenAI] = {}

        # Pre-initialize clients with OpenRouter headers and max_retries=0
        # (KeyRotator explicitly handles multi-key failover with zero internal SDK sleep-retries)
        for k in api_keys:
            if k.strip():
                self._clients[k.strip()] = AsyncOpenAI(
                    api_key=k.strip(),
                    base_url=self._base_url,
                    max_retries=0,
                    default_headers={
                        "HTTP-Referer": "https://github.com/piyushgoel2808/NewsLens-AI",
                        "X-Title": "NewsLens-AI Newspaper Intelligence",
                    },
                )

        # Infer capabilities
        is_vision = (
            supports_vision
            if supports_vision is not None
            else (
                "gemma-4" in model.lower()
                or "vl" in model.lower()
                or "vision" in model.lower()
                or "4o" in model.lower()
                or "gemini" in model.lower()
            )
        )

        ctx = context_window
        if ctx is None:
            if "nemotron" in model.lower():
                ctx = 1_000_000
            elif "gemma-4" in model.lower():
                ctx = 262_144
            elif "4o" in model.lower():
                ctx = 128_000
            else:
                ctx = 128_000

        self._capability = ProviderCapability(
            supports_vision=bool(is_vision),
            supports_tool_use=True,
            supports_streaming=True,
            supports_structured_output=True,
            context_window=ctx,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def are_all_keys_rate_limited(self) -> bool:
        """Check if all configured OpenRouter keys are in cooldown."""
        return self._rotator.are_all_keys_rate_limited()

    def get_shortest_cooldown_remaining(self) -> float:
        """Return shortest seconds remaining before an OpenRouter key cooldown expires."""
        return self._rotator.get_shortest_cooldown_remaining()

    def _get_client(self, key: str) -> AsyncOpenAI:
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(
                api_key=key,
                base_url=self._base_url,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://github.com/piyushgoel2808/NewsLens-AI",
                    "X-Title": "NewsLens-AI Newspaper Intelligence",
                },
            )
        return self._clients[key]

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
                    elif part.get("type") == "image_url":
                        parts.append(part)
                    elif part.get("type") == "image":
                        # Legacy fallback: convert raw base64 to data URL
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{part['data']}"},
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

    # -----------------------------------------------------------------------
    # ChatModelProvider interface
    # -----------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        response_schema: dict[str, Any] | None = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Run a chat completion via OpenRouter with transparent dual-key 429 failover."""
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

        eff_temp = max(0.3, temperature) if "nemotron" in self._model.lower() else temperature
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": eff_temp,
        }
        if "nemotron" in self._model.lower() or "gemma" in self._model.lower():
            kwargs["frequency_penalty"] = 0.35
            kwargs["presence_penalty"] = 0.15

        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        # Attempt invocation with round-robin key rotation and 429 failover
        attempts = 0
        max_attempts = self._rotator.total_keys

        while attempts < max_attempts:
            attempts += 1
            key, key_idx = await self._rotator.get_next_key()
            client = self._get_client(key)

            try:
                response = await client.chat.completions.create(**kwargs)
                await self._rotator.clear_cooldown(key)
                break
            except RateLimitError as rle:
                logger.warning(
                    "OpenRouter Key %d hit 429 RateLimitError, failing over...",
                    key_idx + 1,
                    extra={"error": str(rle)},
                )
                await self._rotator.mark_rate_limited(key)
                if attempts >= max_attempts:
                    raise RateLimitExhaustedError(
                        f"All OpenRouter keys exhausted after {attempts} attempts.",
                        retry_after_seconds=60.0,
                    ) from rle
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                    logger.warning(
                        "OpenRouter Key %d hit 429 string match, failing over...",
                        key_idx + 1,
                        extra={"error": str(e)},
                    )
                    await self._rotator.mark_rate_limited(key)
                    if attempts >= max_attempts:
                        raise RateLimitExhaustedError(
                            f"All OpenRouter keys exhausted after {attempts} attempts.",
                            retry_after_seconds=60.0,
                        ) from e
                else:
                    raise ProviderError(f"OpenRouter API error: {e}") from e

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

        # Structured JSON parsing with regex fallback
        parsed: Any | None = None
        if response_schema and text.strip():
            # Clean markdown fences or thinking tokens if present
            cleaned = re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.DOTALL).strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                # Try finding outer braces
                match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
                if match:
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed = json.loads(match.group(1))

        logger.info(
            "OpenRouter completion successful",
            extra={
                "model": self._model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "key_index": key_idx + 1,
            },
        )

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            provider="openrouter",
            parsed=parsed,
            raw=response,
        )

    async def complete_stream(
        self,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streaming chat completion via OpenRouter with multi-key 429 failover and loop prevention."""
        oai_messages = self._to_openai_messages(messages)
        eff_temp = max(0.3, temperature) if "nemotron" in self._model.lower() else temperature
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": cast(Any, oai_messages),
            "max_tokens": max_tokens,
            "temperature": eff_temp,
            "stream": True,
        }
        if "nemotron" in self._model.lower() or "gemma" in self._model.lower():
            kwargs["frequency_penalty"] = 0.35
            kwargs["presence_penalty"] = 0.15

        attempts = 0
        max_attempts = self._rotator.total_keys
        stream_resp = None
        active_key = ""

        while attempts < max_attempts:
            attempts += 1
            key, key_idx = await self._rotator.get_next_key()
            active_key = key
            client = self._get_client(key)

            try:
                stream_resp = await client.chat.completions.create(**kwargs)
                await self._rotator.clear_cooldown(key)
                break
            except RateLimitError as rle:
                logger.warning(
                    "OpenRouter Key %d hit 429 RateLimitError during stream setup, failing over...",
                    key_idx + 1,
                    extra={"error": str(rle)},
                )
                await self._rotator.mark_rate_limited(key)
                if attempts >= max_attempts:
                    raise RateLimitExhaustedError(
                        f"All OpenRouter keys exhausted after {attempts} stream attempts.",
                        retry_after_seconds=60.0,
                    ) from rle
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                    logger.warning(
                        "OpenRouter Key %d hit 429 string match during stream setup, failing over...",
                        key_idx + 1,
                        extra={"error": str(e)},
                    )
                    await self._rotator.mark_rate_limited(key)
                    if attempts >= max_attempts:
                        raise RateLimitExhaustedError(
                            f"All OpenRouter keys exhausted after {attempts} stream attempts.",
                            retry_after_seconds=60.0,
                        ) from e
                else:
                    raise ProviderError(f"OpenRouter streaming error: {e}") from e

        if not stream_resp:
            return

        try:
            async for chunk in cast(AsyncIterator[Any], stream_resp):
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str:
                if active_key:
                    await self._rotator.mark_rate_limited(active_key)
                raise RateLimitExhaustedError(
                    f"OpenRouter streaming was aborted by 429 rate limit: {e}",
                    retry_after_seconds=60.0,
                ) from e
            raise ProviderError(f"OpenRouter streaming error mid-stream: {e}") from e

    # -----------------------------------------------------------------------
    # VisionModelProvider interface
    # -----------------------------------------------------------------------

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Analyze an image using OpenRouter multimodal vision model.

        Wraps the image in strict OpenAI-compatible Data URL format:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64_string}"}}
        """
        if not self._capability.supports_vision:
            raise ProviderError(
                f"Model {self._model!r} does not support vision. "
                "Use a multimodal model such as 'google/gemma-4-26b-a4b-it:free'."
            )

        # Detect image format from magic bytes
        if image_bytes.startswith(b"\x89PNG"):
            mime_type = "image/png"
        elif image_bytes.startswith(b"\xff\xd8"):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b"GIF8"):
            mime_type = "image/gif"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_b64}"

        messages = [
            Message(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            )
        ]

        return await self.complete(
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
        )
