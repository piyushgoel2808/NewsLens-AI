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
    DocumentLayoutProvider,
    EmbeddingProvider,
    OCREngine,
    ProviderCapability,
    ProviderError,
    VisionModelProvider,
)

logger = get_logger(__name__)

AnyProvider = (
    ChatModelProvider | EmbeddingProvider | VisionModelProvider | OCREngine | DocumentLayoutProvider
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
        from app.providers.gemini_provider import GeminiProvider
        from app.providers.groq_provider import GroqProvider
        from app.providers.local_embedding_provider import LocalEmbeddingProvider
        from app.providers.ollama_provider import OllamaProvider
        from app.providers.openai_provider import OpenAIProvider

        cfg = self._model_config.providers.get(provider_id)
        if not cfg:
            raise ProviderError(f"Provider {provider_id!r} is not defined in model_config.yaml")

        provider_type = cfg.provider
        model = cfg.model or ""

        if provider_type == "ollama":
            return OllamaProvider(
                model=model,
                base_url=cfg.base_url or self._settings.ollama_base_url,
                supports_vision=cfg.supports_vision,
            )
        elif provider_type == "groq":
            return GroqProvider(
                model=model or "llama-3.3-70b-versatile",
                api_key=self._settings.groq_api_key,
            )
        elif provider_type in (
            "gemini",
            "gemini_ocr",
            "gemini_vlm",
            "gemini_vision",
            "gemini_layout",
        ):
            return GeminiProvider(
                model=model or "gemini-3.7-flash",
                api_key=self._settings.gemini_api_key or self._settings.google_api_key,
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
        elif provider_type in ("docling", "docling_parser"):
            from app.providers.docling_provider import DoclingProvider

            return DoclingProvider(lang=cfg.lang or "en")
        elif provider_type in ("mineru", "magic_pdf", "tesseract"):
            from app.providers.mineru_provider import MinerUProvider

            return MinerUProvider(lang=cfg.lang or "en")
        else:
            raise ProviderError(
                f"Unknown provider type {provider_type!r} for {provider_id!r}. "
                "Supported: ollama, groq, gemini, anthropic, openai, "
                "local_sentence_transformers, docling, mineru"
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

    def get_provider_by_id(self, provider_id: str) -> AnyProvider:
        """Return a provider directly by its configured provider ID."""
        if provider_id in self._instances:
            return self._instances[provider_id]
        instance = self._instantiate(provider_id)
        self._instances[provider_id] = instance
        return instance

    def get_chat_provider(self, model_name_or_id: str | None = None) -> ChatModelProvider:
        """Return a ChatModelProvider instance by name, ID, alias, or bound default."""
        if not model_name_or_id:
            provider = self.get_provider("answerer")
            if isinstance(provider, ChatModelProvider):
                return provider
            raise ProviderError("Bound 'answerer' provider does not implement ChatModelProvider")

        target_id = model_name_or_id.strip()

        # 1. Exact match in configured providers
        if target_id in self._model_config.providers:
            provider = self.get_provider_by_id(target_id)
            if isinstance(provider, ChatModelProvider):
                return provider

        # 2. Alias resolution
        alias_map = {
            "gemini": "gemini_flash",
            "gemini_flash": "gemini_flash",
            "gemini_pro": "gemini_pro",
            "groq": "groq_compound",
            "groq_compound": "groq_compound",
            "groq_qwen": "groq_qwen",
            "groq_llama": "groq_compound",
            "groq_gpt_oss": "groq_gpt_oss",
            "ollama": "ollama_llama3",
            "ollama_chat": "ollama_llama3",
            "ollama_llama": "ollama_llama3",
            "ollama_llama3": "ollama_llama3",
            "ollama_nemotron": "ollama_nemotron",
            "nemotron": "ollama_nemotron",
            "nvidia": "ollama_nemotron",
            "ollama_deepseek": "ollama_deepseek",
            "deepseek": "ollama_deepseek",
            "openai": "openai_gpt4o",
            "openai_gpt4o": "openai_gpt4o",
            "openai_gpt4o_mini": "openai_gpt4o_mini",
            "gpt4o": "openai_gpt4o",
            "gpt4o_mini": "openai_gpt4o_mini",
            "anthropic": "anthropic_sonnet",
            "gemma": "ollama_gemma4_12b",
            "gemma4": "ollama_gemma4_12b",
            "gemma4:12b": "ollama_gemma4_12b",
            "gemma4:26b": "ollama_gemma4_26b",
            "ollama_gemma4_12b": "ollama_gemma4_12b",
            "ollama_gemma4_26b": "ollama_gemma4_26b",
        }
        if target_id.lower() in alias_map:
            resolved_id = alias_map[target_id.lower()]
            if resolved_id in self._model_config.providers:
                provider = self.get_provider_by_id(resolved_id)
                if isinstance(provider, ChatModelProvider):
                    return provider

        # 3. Dynamic provider instantiation based on prefix/content
        if "gemma" in target_id.lower():
            from app.providers.ollama_provider import OllamaProvider

            m_name = "gemma4:26b" if "26b" in target_id.lower() else "gemma4:12b"
            return OllamaProvider(
                model=m_name,
                base_url=self._settings.ollama_base_url,
                supports_vision="26b" in target_id.lower(),
            )
        elif "gemini" in target_id.lower():
            from app.providers.gemini_provider import GeminiProvider

            return GeminiProvider(
                model=target_id if target_id.startswith("gemini-") else "gemini-3.7-flash",
                api_key=self._settings.gemini_api_key or self._settings.google_api_key,
            )
        elif (
            "groq" in target_id.lower()
            or "qwen" in target_id.lower()
            or "llama" in target_id.lower()
        ):
            from app.providers.groq_provider import GroqProvider

            is_full_model_name = "/" in target_id or "-" in target_id
            m_name = target_id if is_full_model_name else "groq/compound"
            return GroqProvider(
                model=m_name,
                api_key=self._settings.groq_api_key,
            )
        elif "nemotron" in target_id.lower():
            from app.providers.ollama_provider import OllamaProvider

            return OllamaProvider(
                model="nemotron-3.5-lightning:latest",
                base_url=self._settings.ollama_base_url,
            )
        elif "deepseek" in target_id.lower():
            from app.providers.ollama_provider import OllamaProvider

            return OllamaProvider(
                model="deepseek-r1:14b",
                base_url=self._settings.ollama_base_url,
            )
        elif "openai" in target_id.lower() or "gpt" in target_id.lower():
            from app.providers.openai_provider import OpenAIProvider

            m_name = "gpt-4o-mini" if "mini" in target_id.lower() else "gpt-4o"
            return OpenAIProvider(
                model=m_name,
                api_key=self._settings.openai_api_key,
            )
        elif "ollama" in target_id.lower():
            from app.providers.ollama_provider import OllamaProvider

            return OllamaProvider(
                model=target_id,
                base_url=self._settings.ollama_base_url,
            )

        # Fallback to bound answerer
        fallback = self.get_provider("answerer")
        if isinstance(fallback, ChatModelProvider):
            return fallback
        raise ProviderError(f"Could not resolve chat provider for {model_name_or_id!r}")

    def invalidate_task(self, task: str) -> None:
        """Reload configuration for the specified task."""
        self._instances.clear()
        object.__setattr__(self._settings, "_model_config_data", None)
        self._model_config = self._settings.load_model_config()

    def invalidate_all(self) -> None:
        """Clear cached provider instances and reload configuration from disk."""
        self._instances.clear()
        object.__setattr__(self._settings, "_model_config_data", None)
        self._model_config = self._settings.load_model_config()

    def validate_task_capability(self, task: str, required: str) -> None:
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

        Returns rich provider metadata with specific names, descriptions, and reachability.
        """
        display_names = {
            "gemini_flash": "Google Gemini 3.7 Flash (Grounding)",
            "gemini_pro": "Google Gemini Pro",
            "groq_compound": "Groq Compound AI (Ultra-Fast)",
            "groq_qwen": "Groq Qwen 3.6 27B (Reasoning)",
            "groq_gpt_oss": "Groq OpenAI GPT-OSS 120B",
            "ollama_nemotron": "NVIDIA Nemotron 3.5 Lightning (Local 25GB)",
            "ollama_deepseek": "DeepSeek R1 14B (Local Reasoning)",
            "ollama_llama3": "Meta Llama 3.1 8B (Local)",
            "ollama_chat": "Meta Llama 3.1 8B (Local Chat)",
            "ollama_vlm": "Qwen 2.5 VL 7B (Local Vision)",
            "openai_gpt4o": "OpenAI GPT-4o (Omni)",
            "openai_gpt4o_mini": "OpenAI GPT-4o Mini",
            "docling_parser": "Docling Document Layout Engine",
            "mineru_parser": "MinerU Magic-PDF Layout Engine",
            "local_embed_bge": "BAAI BGE-M3 Multilingual Embedding",
        }

        local_providers = (
            "ollama",
            "local_sentence_transformers",
            "docling",
            "mineru",
            "tesseract",
        )

        results: list[dict[str, Any]] = []
        for provider_id, cfg in self._model_config.providers.items():
            is_reachable = await self._check_reachable(provider_id, cfg.provider)
            fallback_name = f"{cfg.provider.title()} ({cfg.model or provider_id})"
            name_label = display_names.get(provider_id, fallback_name)
            results.append(
                {
                    "id": provider_id,
                    "name": name_label,
                    "provider": cfg.provider,
                    "model": cfg.model,
                    "is_local": cfg.provider in local_providers,
                    "is_reachable": is_reachable,
                    "capabilities": {
                        "supports_vision": cfg.supports_vision,
                        "supports_tool_use": cfg.supports_tool_use,
                        "context_window": cfg.context_window,
                        "embedding_dim": cfg.embedding_dim,
                    },
                }
            )
        return results

    async def _check_reachable(self, provider_id: str, provider_type: str) -> bool | None:
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
