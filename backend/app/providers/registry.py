"""Model registry: resolves task→provider bindings and manages provider instances.

The ModelRegistry is a singleton that reads model_config.yaml via Settings
and lazily instantiates concrete provider classes on first access. This is
the single place where provider selection happens — all other code calls
registry.get_provider(task_name) and gets back a properly-typed instance.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.providers.base import (
    ChatModelProvider,
    EmbeddingProvider,
    OCREngine,
    ProviderCapability,
    ProviderError,
    VisionModelProvider,
)

logger = get_logger(__name__)

AnyProvider = (
    ChatModelProvider | EmbeddingProvider | VisionModelProvider | OCREngine
)


class ModelRegistry:
    """Singleton registry for model providers.

    Resolves task names to provider instances via model_config.yaml bindings.
    Providers are instantiated lazily and cached.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_config = settings.load_model_config()
        self._instances: dict[str, AnyProvider] = {}
        self._lock = asyncio.Lock()

    def _instantiate(self, provider_id: str) -> AnyProvider:
        """Instantiate a concrete provider from its ProviderConfig."""
        from app.providers.anthropic_provider import AnthropicProvider
        from app.providers.groq_provider import GroqProvider
        from app.providers.local_embedding_provider import LocalEmbeddingProvider
        from app.providers.ollama_provider import OllamaProvider
        from app.providers.openai_provider import OpenAIProvider
        from app.providers.tesseract_ocr import TesseractOCR

        cfg = self._model_config.providers.get(provider_id)
        if not cfg:
            raise ProviderError(
                f"Provider {provider_id!r} is not defined in model_config.yaml"
            )

        provider_type = cfg.provider
        model = cfg.model or ""

        if provider_type == "ollama":
            return OllamaProvider(
                model=model,
                base_url=cfg.base_url or self._settings.ollama_base_url,
            )
        elif provider_type == "groq":
            return GroqProvider(
                model=model or "llama-3.3-70b-versatile",
                api_key=self._settings.groq_api_key,
            )
        elif provider_type == "anthropic":
            return AnthropicProvider(
                model=model,
                api_key=self._settings.anthropic_api_key,
            )
        elif provider_type == "openai":
            return OpenAIProvider(
                model=model,
                api_key=self._settings.openai_api_key,
            )
        elif provider_type == "local_sentence_transformers":
            return LocalEmbeddingProvider(model=model or "BAAI/bge-m3")
        elif provider_type == "tesseract":
            return TesseractOCR(lang=cfg.lang or "eng")
        else:
            raise ProviderError(
                f"Unknown provider type {provider_type!r} for {provider_id!r}. "
                "Supported: ollama, groq, anthropic, openai, "
                "local_sentence_transformers, tesseract"
            )

    def get_provider(self, task: str) -> AnyProvider:
        """Return the provider bound to the given task.

        Instantiates and caches the provider on first call.

        Args:
            task: Task name (e.g. 'query_planner', 'embedding', 'ocr').

        Raises:
            ProviderError: If no binding exists or provider cannot be created.
        """
        provider_config = self._model_config.get_provider_for_task(task)
        provider_id = self._model_config.task_bindings[task]

        if provider_id in self._instances:
            return self._instances[provider_id]

        # Synchronous instantiation (safe — only happens once per provider_id)
        instance = self._instantiate(provider_id)
        self._instances[provider_id] = instance
        logger.info(
            "Provider instantiated",
            extra={
                "task": task,
                "provider_id": provider_id,
                "provider_type": provider_config.provider,
                "model": provider_config.model,
            },
        )
        return instance

    def validate_task_capability(
        self, task: str, required: str
    ) -> None:
        """Validate that the provider bound to a task has a required capability.

        Args:
            task: Task name.
            required: Capability attribute name (e.g. 'supports_vision').

        Raises:
            ProviderError: If the provider lacks the required capability.
        """
        provider = self.get_provider(task)
        cap = getattr(provider, "capability", None)
        if (
            cap is not None
            and isinstance(cap, ProviderCapability)
            and not getattr(cap, required, False)
        ):
            raise ProviderError(
                f"Provider bound to task {task!r} does not support {required}. "
                f"Check the task_bindings in model_config.yaml and ensure "
                f"the selected provider has '{required}: true'."
            )

    async def get_available_providers(self) -> list[dict[str, Any]]:
        """Introspect all configured providers and check reachability.

        Returns a list of dicts suitable for the /api/models/available endpoint.
        Does not fail if a provider is unreachable — marks is_reachable=False.
        """
        results: list[dict[str, Any]] = []
        for provider_id, cfg in self._model_config.providers.items():
            is_reachable = await self._check_reachable(provider_id, cfg.provider)
            results.append({
                "id": provider_id,
                "provider": cfg.provider,
                "model": cfg.model,
                "is_reachable": is_reachable,
                "capabilities": {
                    "supports_vision": cfg.supports_vision,
                    "supports_tool_use": cfg.supports_tool_use,
                    "context_window": cfg.context_window,
                    "embedding_dim": cfg.embedding_dim,
                },
            })
        return results

    async def _check_reachable(
        self, provider_id: str, provider_type: str
    ) -> bool | None:
        """Quick reachability check for a provider. Returns None on timeout."""
        try:
            if provider_type == "ollama":
                cfg = self._model_config.providers[provider_id]
                base_url = cfg.base_url or self._settings.ollama_base_url
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r = await client.get(f"{base_url}/api/version")
                    return r.status_code == 200
            elif provider_type == "groq":
                return bool(self._settings.groq_api_key)
            elif provider_type == "anthropic":
                return bool(self._settings.anthropic_api_key)
            elif provider_type == "openai":
                return bool(self._settings.openai_api_key)
            elif provider_type in ("local_sentence_transformers", "tesseract"):
                return True  # Always "reachable" (local, no network needed)
            else:
                return None
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the singleton ModelRegistry. Initialised on first call."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(get_settings())
    return _registry


def reset_registry() -> None:
    """Reset the singleton (used in tests)."""
    global _registry
    _registry = None
