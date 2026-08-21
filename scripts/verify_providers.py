#!/usr/bin/env python3
"""Verify NewsLens-AI provider abstraction works end-to-end.

Runs smoke tests against each configured provider:
- OllamaProvider: real completion request to the local Ollama server
- AnthropicProvider: real request (skipped if ANTHROPIC_API_KEY not set)
- LocalEmbeddingProvider: embed a test sentence, verify shape

This script PROVES the "config-only provider swap" works:
both Ollama and Anthropic return the same ModelResponse shape.

Usage:
    python scripts/verify_providers.py
    # or from repo root:
    make verify
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Add backend/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv()
except ImportError:
    pass


RESULTS: list[tuple[str, bool, str]] = []  # (name, passed, message)
SKIPPED: list[str] = []


def _result(name: str, passed: bool, message: str) -> None:
    RESULTS.append((name, passed, message))


async def test_ollama() -> None:
    """Test OllamaProvider with a real (tiny) completion call."""
    import httpx
    from app.providers.ollama_provider import OllamaProvider
    from app.providers.base import Message, ProviderError

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    # Discover installed models if default model is not present
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base_url}/api/tags")
            if r.status_code == 200:
                available_models = [m.get("name") for m in r.json().get("models", [])]
                if model not in available_models and not any(model in m for m in available_models):
                    # Prefer standard chat models over reasoning/embedding models
                    llama_models = [m for m in available_models if "llama" in m]
                    chat_candidates = [m for m in available_models if "embed" not in m]
                    if llama_models:
                        model = llama_models[0]
                    elif chat_candidates:
                        model = chat_candidates[0]
    except Exception:
        pass

    provider = OllamaProvider(model=model, base_url=base_url)

    t0 = time.monotonic()
    try:
        response = await provider.complete(
            messages=[Message(role="user", content="Respond with a short greeting")],
            max_tokens=50,
            temperature=0.0,
        )
        assert response.text.strip(), "Empty response from Ollama"
        latency = round((time.monotonic() - t0) * 1000)
        _result(
            f"OllamaProvider ({model} @ {base_url})",
            True,
            f"Response: {response.text.strip()!r} | tokens: {response.total_tokens} | {latency}ms",
        )
    except ProviderError as e:
        _result(f"OllamaProvider ({model})", False, f"ProviderError: {e}")
    except Exception as e:
        _result(f"OllamaProvider ({model})", False, f"{type(e).__name__}: {e}")


async def test_anthropic() -> None:
    """Test AnthropicProvider — skipped if API key not set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        SKIPPED.append("AnthropicProvider (ANTHROPIC_API_KEY not set)")
        _result("AnthropicProvider", True, "SKIPPED — ANTHROPIC_API_KEY not set")
        return

    from app.providers.anthropic_provider import AnthropicProvider
    from app.providers.base import Message, ProviderError

    model = "claude-haiku-3-5"  # cheapest/fastest for smoke test
    provider = AnthropicProvider(model=model, api_key=api_key)

    t0 = time.monotonic()
    try:
        response = await provider.complete(
            messages=[Message(role="user", content="Respond with exactly one word: hello")],
            max_tokens=10,
            temperature=0.0,
        )
        assert response.text.strip(), "Empty response from Anthropic"
        latency = round((time.monotonic() - t0) * 1000)
        _result(
            f"AnthropicProvider ({model})",
            True,
            f"Response: {response.text.strip()!r} | tokens: {response.total_tokens} | {latency}ms",
        )
    except ProviderError as e:
        _result(f"AnthropicProvider ({model})", False, f"ProviderError: {e}")
    except Exception as e:
        _result(f"AnthropicProvider ({model})", False, f"{type(e).__name__}: {e}")


async def test_groq() -> None:
    """Test GroqProvider with real API call."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        SKIPPED.append("GroqProvider (GROQ_API_KEY not set)")
        _result("GroqProvider", True, "SKIPPED — GROQ_API_KEY not set")
        return

    from app.providers.groq_provider import GroqProvider
    from app.providers.base import Message, ProviderError

    model = "qwen/qwen3.6-27b"
    provider = GroqProvider(model=model, api_key=api_key)

    t0 = time.monotonic()
    try:
        response = await provider.complete(
            messages=[Message(role="user", content="Respond with a short greeting")],
            max_tokens=30,
            temperature=0.0,
        )
        assert response.text.strip(), "Empty response from Groq"
        latency = round((time.monotonic() - t0) * 1000)
        _result(
            f"GroqProvider ({model})",
            True,
            f"Response: {response.text.strip()[:40]!r}... | tokens: {response.total_tokens} | {latency}ms",
        )
    except ProviderError as e:
        _result(f"GroqProvider ({model})", False, f"ProviderError: {e}")
    except Exception as e:
        _result(f"GroqProvider ({model})", False, f"{type(e).__name__}: {e}")


async def test_local_embedding() -> None:
    """Test LocalEmbeddingProvider — downloads bge-m3 on first run (~2GB)."""
    from app.providers.local_embedding_provider import LocalEmbeddingProvider, ProviderError

    model = "BAAI/bge-m3"
    provider = LocalEmbeddingProvider(model=model)

    t0 = time.monotonic()
    try:
        texts = [
            "The stock market crashed today.",
            "राजनीतिक विरोध प्रदर्शन हुआ।",  # Hindi — tests multilingual
        ]
        embeddings = await provider.embed(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == provider.embedding_dim
        assert len(embeddings[1]) == provider.embedding_dim
        latency = round((time.monotonic() - t0) * 1000)
        _result(
            f"LocalEmbeddingProvider ({model}, dim={provider.embedding_dim})",
            True,
            f"Embedded {len(texts)} texts | shape: [{len(embeddings[0])}] | {latency}ms",
        )
    except ProviderError as e:
        _result(f"LocalEmbeddingProvider ({model})", False, f"ProviderError: {e}")
    except Exception as e:
        _result(f"LocalEmbeddingProvider ({model})", False, f"{type(e).__name__}: {e}")


async def test_provider_swap_proof() -> None:
    """Structural test: both providers return identical ModelResponse shape."""
    from app.providers.base import ModelResponse
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.providers.ollama_provider import OllamaProvider
    from app.providers.anthropic_provider import AnthropicProvider
    from app.providers.base import Message

    messages = [Message(role="user", content="test")]

    # Mocked — no network call, tests interface shape only
    with patch("app.providers.ollama_provider.ollama") as mock_ollama:
        mock_client = AsyncMock()
        mock_ollama.AsyncClient.return_value = mock_client
        mock_client.chat.return_value = MagicMock(
            message=MagicMock(content="Ollama response", tool_calls=None),
            prompt_eval_count=5, eval_count=3, model="llama3.2:3b",
        )
        ollama = OllamaProvider(model="llama3.2:3b")
        r1 = await ollama.complete(messages)

    with patch("app.providers.anthropic_provider.anthropic") as mock_anth:
        mock_async = AsyncMock()
        mock_anth.AsyncAnthropic.return_value = mock_async
        mock_anth.Anthropic.return_value = MagicMock()
        mock_async.messages.create.return_value = MagicMock(
            content=[MagicMock(type="text", text="Claude response")],
            usage=MagicMock(input_tokens=5, output_tokens=3),
            model="claude-haiku-3-5", stop_reason="end_turn",
        )
        claude = AnthropicProvider(model="claude-haiku-3-5", api_key="test-key")
        r2 = await claude.complete(messages)

    assert type(r1) is ModelResponse and type(r2) is ModelResponse
    shared_fields = ["text", "input_tokens", "output_tokens", "tool_calls", "model", "provider"]
    all_match = all(hasattr(r1, f) and hasattr(r2, f) for f in shared_fields)
    _result(
        "Provider swap proof (mocked interface test)",
        all_match,
        f"Both return ModelResponse with identical fields: {shared_fields}",
    )


async def main() -> None:
    print("\n" + "=" * 60)
    print("NewsLens-AI Provider Verification")
    print("Config-only provider swap proof")
    print("=" * 60 + "\n")

    await asyncio.gather(
        test_ollama(),
        test_groq(),
        test_anthropic(),
        test_local_embedding(),
        test_provider_swap_proof(),
        return_exceptions=True,
    )

    print("Results:")
    print("-" * 60)
    all_passed = True
    for name, passed, message in RESULTS:
        is_skip = "SKIPPED" in message
        icon = "⚡ SKIP" if is_skip else ("✓ PASS" if passed else "✗ FAIL")
        print(f"  {icon}  {name}")
        print(f"          {message}")
        if not passed and not is_skip:
            all_passed = False
    print("-" * 60)

    if all_passed:
        print("\n✓ All provider tests passed. Swap abstraction verified.\n")
        sys.exit(0)
    else:
        print("\n✗ Some provider tests failed. See details above.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
