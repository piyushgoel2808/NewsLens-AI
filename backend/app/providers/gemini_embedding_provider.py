"""Google Gemini Embedding Provider for NewsLens-AI.

Implements EmbeddingProvider using Google's gemini-embedding-001 model with
Matryoshka Representation Learning (MRL) for configurable dimensionality (default: 768d).

Features:
- Dual endpoint support:
  1. Vertex AI Express Mode (via GEMINI_API_KEY with 'AQ.' prefix)
  2. Vertex AI Regional Endpoint (asia-south1 / us-central1 via GCP Service Account / ADC)
  3. Google AI Studio fallback (via standard API key)
- Asymmetric task types:
  - Ingestion: task_type="RETRIEVAL_DOCUMENT"
  - Retrieval query: task_type="RETRIEVAL_QUERY"
- Zero container RAM footprint (drops PyTorch/HuggingFace dependencies in cloud mode).
- Automatic batching and retry backoff.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import httpx

from app.core.logging import get_logger
from app.providers.base import ProviderCapability, ProviderError

logger = get_logger(__name__)

VERTEX_AI_EXPRESS_BASE = "https://aiplatform.googleapis.com/v1/publishers/google/models"
AI_STUDIO_EMBED_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiEmbeddingProvider:
    """EmbeddingProvider backed by Google's gemini-embedding-001."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        embedding_dim: int = 768,
        api_key: str | None = None,
        service_account_info: dict[str, Any] | str | None = None,
        project_id: str | None = None,
        location: str = "asia-south1",
        base_url: str | None = None,
    ) -> None:
        self._model = model.replace("models/", "") if model else "gemini-embedding-001"
        self._dim = int(embedding_dim) if embedding_dim else 768
        self._api_key = api_key
        self._project_id = project_id or "newslens-ai-prod"
        self._location = location
        self._base_url = base_url
        self._sa_credentials: Any = None

        # Set up Service Account credentials if provided
        if service_account_info:
            try:
                from google.oauth2 import service_account

                scopes = ["https://www.googleapis.com/auth/cloud-platform"]
                if isinstance(service_account_info, str):
                    if service_account_info.strip().startswith("{"):
                        info = json.loads(service_account_info)
                        self._sa_credentials = service_account.Credentials.from_service_account_info(
                            info, scopes=scopes
                        )
                    else:
                        self._sa_credentials = service_account.Credentials.from_service_account_file(
                            service_account_info, scopes=scopes
                        )
                elif isinstance(service_account_info, dict):
                    self._sa_credentials = service_account.Credentials.from_service_account_info(
                        service_account_info, scopes=scopes
                    )
            except Exception as e:
                logger.warning(
                    "Failed to initialize GCP Service Account credentials in GeminiEmbeddingProvider",
                    extra={"error": str(e)},
                )

        # If no explicit SA provided and no express API key, try Application Default Credentials
        if not self.is_express_mode and not self._sa_credentials:
            try:
                import google.auth

                creds, proj = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self._sa_credentials = creds
                if proj and not project_id:
                    self._project_id = proj
            except Exception:
                pass

        if not self._api_key and not self._sa_credentials:
            raise ProviderError(
                "Google Gemini Embedding requires an API key or GCP credentials. "
                "Set GEMINI_API_KEY, GOOGLE_API_KEY, or GCP_SERVICE_ACCOUNT_KEY in .env."
            )

        self._capability = ProviderCapability(
            supports_vision=False,
            supports_tool_use=False,
            supports_streaming=False,
            supports_structured_output=False,
            embedding_dim=self._dim,
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def provider_name(self) -> str:
        return "gemini_embedding"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def is_express_mode(self) -> bool:
        """Return True if using Vertex AI Express Mode API key (starts with 'AQ.')."""
        return bool(self._api_key and self._api_key.startswith("AQ."))

    def _get_target_endpoint(self) -> tuple[str, dict[str, str], dict[str, str]]:
        """Resolve request URL, headers, and params based on available auth credentials."""
        headers = {"Content-Type": "application/json"}
        params: dict[str, str] = {}

        if self._base_url:
            url = f"{self._base_url.rstrip('/')}/{self._model}:predict"
            if self._api_key:
                headers["x-goog-api-key"] = self._api_key
            return url, headers, params

        if self.is_express_mode:
            url = f"{VERTEX_AI_EXPRESS_BASE}/{self._model}:predict"
            headers["x-goog-api-key"] = str(self._api_key)
            return url, headers, params

        if self._sa_credentials:
            try:
                import google.auth.transport.requests

                request = google.auth.transport.requests.Request()
                self._sa_credentials.refresh(request)
                headers["Authorization"] = f"Bearer {self._sa_credentials.token}"
                url = (
                    f"https://{self._location}-aiplatform.googleapis.com/v1/"
                    f"projects/{self._project_id}/locations/{self._location}/"
                    f"publishers/google/models/{self._model}:predict"
                )
                return url, headers, params
            except Exception as e:
                logger.warning(
                    "Service account token refresh failed, attempting fallback",
                    extra={"error": str(e)},
                )

        if self._api_key:
            url = f"{AI_STUDIO_EMBED_BASE}/{self._model}:batchEmbedContents"
            params["key"] = self._api_key
            return url, headers, params

        raise ProviderError("No valid authentication method found for GeminiEmbeddingProvider")

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        params: dict[str, str],
        payload: dict[str, Any],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Send HTTP POST with exponential backoff for transient errors."""
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = (2**attempt) + random.uniform(0.1, 0.5)
                    logger.warning(
                        "Transient error from Gemini Embedding API, retrying",
                        extra={"status": resp.status_code, "attempt": attempt + 1, "wait_s": wait},
                    )
                    await asyncio.sleep(wait)
                    # Refresh SA credentials if expired
                    if self._sa_credentials and attempt > 0:
                        try:
                            import google.auth.transport.requests

                            req = google.auth.transport.requests.Request()
                            self._sa_credentials.refresh(req)
                            headers["Authorization"] = f"Bearer {self._sa_credentials.token}"
                        except Exception:
                            pass
                    continue

                error_text = resp.text[:400]
                raise ProviderError(
                    f"Gemini embedding API returned HTTP {resp.status_code}: {error_text}"
                )
            except httpx.RequestError as exc:
                last_err = exc
                wait = (2**attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    "Network error calling Gemini Embedding API, retrying",
                    extra={"error": str(exc), "attempt": attempt + 1},
                )
                await asyncio.sleep(wait)

        raise ProviderError(f"Gemini embedding failed after {max_retries} retries: {last_err}")

    async def embed(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Embed a list of texts using gemini-embedding-001 with outputDimensionality.

        Args:
            texts: List of strings to embed.
            task_type: "RETRIEVAL_DOCUMENT" for ingestion, "RETRIEVAL_QUERY" for search.

        Returns:
            List of embedding vectors, each having length `self._dim`.
        """
        if not texts:
            return []

        # Batch in chunks of up to 100 to prevent payload size overflow
        batch_size = 100
        all_embeddings: list[list[float]] = []

        url, headers, params = self._get_target_endpoint()
        is_ai_studio = "generativelanguage.googleapis.com" in url

        async with httpx.AsyncClient(timeout=45.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                if is_ai_studio:
                    # Google AI Studio batchEmbedContents format
                    payload = {
                        "requests": [
                            {
                                "model": f"models/{self._model}",
                                "content": {"parts": [{"text": t}]},
                                "taskType": task_type,
                                "outputDimensionality": self._dim,
                            }
                            for t in batch
                        ]
                    }
                    data = await self._post_with_retry(client, url, headers, params, payload)
                    embeddings_data = data.get("embeddings", [])
                    for item in embeddings_data:
                        vec = item.get("values", [])
                        all_embeddings.append(vec)
                else:
                    # Vertex AI :predict format
                    payload = {
                        "instances": [
                            {"content": t, "task_type": task_type}
                            for t in batch
                        ],
                        "parameters": {
                            "outputDimensionality": self._dim,
                        },
                    }
                    data = await self._post_with_retry(client, url, headers, params, payload)
                    predictions = data.get("predictions", [])
                    for item in predictions:
                        vec = item.get("embeddings", {}).get("values", [])
                        all_embeddings.append(vec)

        if len(all_embeddings) != len(texts):
            logger.warning(
                "Embedding count mismatch from Gemini API",
                extra={"expected": len(texts), "received": len(all_embeddings)},
            )

        return all_embeddings

    async def embed_one(
        self,
        text: str,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> list[float]:
        """Embed a single text string (default: RETRIEVAL_QUERY for search)."""
        result = await self.embed([text], task_type=task_type)
        if not result:
            raise ProviderError("Gemini embedding returned empty vector for input text")
        return result[0]
