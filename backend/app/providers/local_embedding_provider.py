"""Local embedding provider using sentence-transformers.

Implements EmbeddingProvider. Default model: BAAI/bge-m3 (multilingual, ~2GB).
Model is loaded lazily on first embed() call to avoid slow startup.
"""

from __future__ import annotations

import asyncio
import time
from typing import cast

from app.core.logging import get_logger
from app.providers.base import ProviderCapability, ProviderError

logger = get_logger(__name__)

# Known embedding dimensions for common models
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class LocalEmbeddingProvider:
    """Local embedding provider backed by sentence-transformers."""

    def __init__(self, model: str = "BAAI/bge-m3") -> None:
        self._model_name = model
        self._model: object | None = None
        self._lock = asyncio.Lock()
        self._dim = _KNOWN_DIMS.get(model, 1024)

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability(embedding_dim=self._dim)

    @property
    def provider_name(self) -> str:
        return "local_sentence_transformers"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    async def _ensure_loaded(self) -> object:
        """Lazily load the model on first use (thread-safe)."""
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            logger.info(
                "Loading local embedding model (first use, may take a moment)",
                extra={"model": self._model_name},
            )
            t0 = time.monotonic()
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(None, self._load_model)
            # Update actual dimension from loaded model
            get_dim_fn = getattr(
                model,
                "get_embedding_dimension",
                getattr(model, "get_sentence_embedding_dimension", None),
            )
            if callable(get_dim_fn):
                actual_dim = get_dim_fn()
                if actual_dim:
                    self._dim = int(actual_dim)
            self._model = model
            logger.info(
                "Local embedding model loaded",
                extra={
                    "model": self._model_name,
                    "dim": self._dim,
                    "load_time_ms": round((time.monotonic() - t0) * 1000),
                },
            )
            return self._model

    def _detect_device(self) -> str:
        """Detect best available hardware accelerator (CUDA -> MPS -> CPU)."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_model(self) -> object:
        try:
            from sentence_transformers import SentenceTransformer

            device = self._detect_device()
            return SentenceTransformer(self._model_name, device=device)
        except ImportError as e:
            raise ProviderError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise ProviderError(f"Failed to load embedding model {self._model_name!r}: {e}") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        if not texts:
            return []
        model = await self._ensure_loaded()
        loop = asyncio.get_event_loop()

        def _encode() -> list[list[float]]:
            vecs = model.encode(  # type: ignore[attr-defined]
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if hasattr(vecs, "tolist"):
                return cast(list[list[float]], vecs.tolist())
            return [list(v) for v in vecs]

        return await loop.run_in_executor(None, _encode)

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]
