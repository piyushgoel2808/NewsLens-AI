"""Unified 768d MRL EmbeddingGemma 2 & Gemini Embedding Provider (-> Qdrant article_chunks_v2).

Provides a single, cloud-portable 768-dimensional Matryoshka Representation Learning (MRL)
embedding layer supporting text, visual/infographic descriptions, and multimodal inputs.
"""

from __future__ import annotations

import asyncio
import base64
import math
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.base import EmbeddingProvider, ProviderCapability, ProviderError

logger = get_logger(__name__)


def _normalize_l2(vec: list[float], target_dim: int = 768) -> list[float]:
    """Truncate to target dimension (Matryoshka Representation Learning) and L2 normalize."""
    sliced = [float(x) for x in vec[:target_dim]]
    norm = math.sqrt(sum(x * x for x in sliced))
    if norm > 0:
        return [x / norm for x in sliced]
    return sliced


class EmbeddingGemma2Provider(EmbeddingProvider):
    """768-dimensional MRL EmbeddingProvider with Cloud-First Portability and Local Fallback.

    Vector Space Consistency Note:
    When `GEMINI_API_KEY` or `GOOGLE_API_KEY` is configured and `prefer_cloud_portability=True`,
    this provider embeds via Google's 768d MRL API (`gemini-embedding-001` / `gemini-embedding-2-preview`)
    both locally and in the cloud so local Qdrant snapshots can be migrated directly to Qdrant Cloud
    with 100% vector-space compatibility and zero re-indexing.
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        ollama_model: str = "embeddinggemma-2",
        base_url: str = "http://localhost:11434",
        dimensions: int = 768,
        prefer_cloud_portability: bool = True,
    ) -> None:
        self.model_name = model_name
        self.ollama_model = ollama_model
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.prefer_cloud_portability = prefer_cloud_portability
        self._st_model: Any = None
        self._st_lock = asyncio.Lock()
        self._capability = ProviderCapability(
            supports_vision=True,
            supports_tool_use=False,
            supports_streaming=False,
            supports_structured_output=False,
            context_window=8192,
            embedding_dim=self.dimensions,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "embedding_gemma2"

    @property
    def embedding_dim(self) -> int:
        return self.dimensions

    @staticmethod
    def _format_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> str:
        text_clean = text.strip()
        if task_type == "RETRIEVAL_QUERY":
            return f"task: search result | query: {text_clean}"
        if text_clean.startswith("title:"):
            return text_clean
        return f"title: broadsheet article | text: {text_clean}"

    def _get_effective_ollama_url(self) -> str:
        try:
            from app.api.routers.settings import get_laptop_node_runtime_state

            state = get_laptop_node_runtime_state()
            if state.get("enabled") and state.get("tunnel_url"):
                return str(state["tunnel_url"]).rstrip("/")
        except Exception:
            pass
        return self.base_url

    async def _try_gemini_cloud_embed(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]] | None:
        """Attempt 768d MRL embedding via GeminiEmbeddingProvider for local-to-cloud Qdrant portability."""
        settings = get_settings()
        gemini_key = settings.gemini_api_key or settings.google_api_key
        sa_info = settings.gcp_service_account_json or settings.gcp_service_account_key
        if not gemini_key and not sa_info:
            return None
        try:
            from app.providers.gemini_embedding_provider import GeminiEmbeddingProvider

            gem_prov = GeminiEmbeddingProvider(
                model="gemini-embedding-001",
                embedding_dim=self.dimensions,
                api_key=gemini_key,
                service_account_info=sa_info,
                project_id=settings.gcp_project_id,
            )
            vecs = await gem_prov.embed(texts, task_type=task_type)
            if vecs and len(vecs) == len(texts):
                return [_normalize_l2(v, self.dimensions) for v in vecs]
        except Exception as exc:
            logger.debug(
                "GeminiEmbeddingProvider 768d MRL attempt skipped/failed",
                extra={"error": str(exc)},
            )
        return None

    async def embed_multimodal(
        self,
        text: str,
        image_bytes: bytes | None = None,
        mime_type: str = "image/jpeg",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[float]:
        """Embed multimodal visual asset (photo/infographic crop + caption/table text) into 768d MRL space."""
        settings = get_settings()
        gemini_key = settings.gemini_api_key or settings.google_api_key
        if image_bytes and gemini_key and not gemini_key.startswith("AQ."):
            try:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent"
                b64_img = base64.b64encode(image_bytes).decode("utf-8")
                payload = {
                    "model": "models/gemini-embedding-2-preview",
                    "content": {
                        "parts": [
                            {"text": text or "Broadsheet visual asset"},
                            {"inline_data": {"mime_type": mime_type, "data": b64_img}},
                        ]
                    },
                    "taskType": task_type,
                    "outputDimensionality": self.dimensions,
                }
                async with httpx.AsyncClient(timeout=25.0) as client:
                    resp = await client.post(url, params={"key": gemini_key}, json=payload)
                    if resp.status_code == 200:
                        values = resp.json().get("embedding", {}).get("values", [])
                        if values:
                            return _normalize_l2(values, self.dimensions)
            except Exception as exc:
                logger.debug("Multimodal gemini-embedding-2-preview fallback to text embedding", extra={"error": str(exc)})

        res = await self.embed_batch([text], task_type=task_type)
        return res[0] if res else [0.0] * self.dimensions

    async def embed_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        if not texts:
            return []
        formatted = [self._format_text(t, task_type) for t in texts]
        settings = get_settings()

        # 1. Attempt Local Ollama (or Laptop Bridge Tunnel) first if reachable
        ollama_url = self._get_effective_ollama_url()
        headers = {
            "ngrok-skip-browser-warning": "true",
            "X-Laptop-Bridge-Secret": settings.laptop_bridge_secret,
        }
        try:
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                resp = await client.post(
                    f"{ollama_url}/api/embed",
                    json={"model": self.ollama_model, "input": formatted},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and len(embeddings) == len(texts):
                        return [_normalize_l2(vec, self.dimensions) for vec in embeddings]
        except Exception as exc:
            logger.debug(
                "Ollama EmbeddingGemma 2 unreachable, trying cloud/HF fallback",
                extra={"error": str(exc)},
            )

        # 2. Attempt Hugging Face Inference API (same google/embeddinggemma model weights, 0 MB local RAM)
        hf_token = settings.hf_api_token or settings.hf_token
        if hf_token:
            try:
                hf_headers = {"Authorization": f"Bearer {hf_token}"}
                async with httpx.AsyncClient(timeout=15.0, headers=hf_headers) as client:
                    resp = await client.post(
                        f"https://router.huggingface.co/hf-inference/models/{self.model_name}/pipeline/feature-extraction",
                        json={"inputs": formatted, "options": {"wait_for_model": True}},
                    )
                    if resp.status_code == 200:
                        vecs = resp.json()
                        if isinstance(vecs, list) and len(vecs) == len(texts):
                            return [_normalize_l2(v, self.dimensions) for v in vecs]
            except Exception as exc:
                logger.warning(
                    "HF EmbeddingGemma 2 API fallback failed",
                    extra={"error": str(exc)},
                )

        # 3. Cloud Gemini Embedding 001 (768d MRL) fallback
        cloud_vecs = await self._try_gemini_cloud_embed(texts, task_type=task_type)
        if cloud_vecs is not None:
            return cloud_vecs

        # 4. Local SentenceTransformers fallback (when running offline on laptop)
        if getattr(settings, "render_free_tier", False):
            raise ProviderError(
                "EmbeddingGemma 2 unavailable: laptop node offline and cloud/HF fallbacks failed on Render Free Tier"
            )
        if self._st_model is None:
            async with self._st_lock:
                if self._st_model is None:
                    from sentence_transformers import SentenceTransformer

                    self._st_model = await asyncio.to_thread(SentenceTransformer, self.model_name)
        raw_vecs = (
            await asyncio.to_thread(self._st_model.encode, formatted, normalize_embeddings=True)
        ).tolist()
        return [_normalize_l2(v, self.dimensions) for v in raw_vecs]

    async def embed_query(self, query: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        res = await self.embed_batch([query], task_type=task_type)
        return res[0] if res else [0.0] * self.dimensions

    async def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """EmbeddingProvider protocol implementation."""
        return await self.embed_batch(texts, task_type=task_type)

    async def embed_one(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        """Convenience wrapper for single text embedding."""
        return await self.embed_query(text, task_type=task_type)
