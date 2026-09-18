"""Unit tests for GeminiEmbeddingProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import httpx

from app.providers.base import ProviderError
from app.providers.gemini_embedding_provider import GeminiEmbeddingProvider


class TestGeminiEmbeddingProvider:
    """Test suite for Google Gemini Embedding Provider."""

    def test_provider_initialization_defaults(self) -> None:
        provider = GeminiEmbeddingProvider(
            api_key="AQ.mock_express_key",
            embedding_dim=768,
        )
        assert provider.provider_name == "gemini_embedding"
        assert provider.model_name == "gemini-embedding-001"
        assert provider.embedding_dim == 768
        assert provider.is_express_mode is True
        assert provider.capability.embedding_dim == 768

    def test_provider_missing_credentials_raises(self) -> None:
        with patch("google.auth.default", side_effect=Exception("No ADC")):
            with pytest.raises(ProviderError) as exc_info:
                GeminiEmbeddingProvider(api_key=None, service_account_info=None)
            assert "requires an API key or GCP credentials" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_vertex_express_mode(self) -> None:
        provider = GeminiEmbeddingProvider(
            api_key="AQ.mock_express_key",
            embedding_dim=768,
        )

        mock_resp = {
            "predictions": [
                {"embeddings": {"values": [0.01] * 768}},
                {"embeddings": {"values": [0.02] * 768}},
            ]
        }

        mock_http_resp = httpx.Response(
            status_code=200,
            json=mock_resp,
            request=httpx.Request("POST", "https://aiplatform.googleapis.com"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_http_resp

            texts = ["Newspaper paragraph 1", "Newspaper paragraph 2"]
            vectors = await provider.embed(texts, task_type="RETRIEVAL_DOCUMENT")

            assert len(vectors) == 2
            assert len(vectors[0]) == 768
            assert len(vectors[1]) == 768
            assert mock_post.called

            # Check request payload structure
            call_kwargs = mock_post.call_args[1]
            assert "x-goog-api-key" in call_kwargs["headers"]
            assert call_kwargs["headers"]["x-goog-api-key"] == "AQ.mock_express_key"
            payload = call_kwargs["json"]
            assert payload["parameters"]["outputDimensionality"] == 768
            assert payload["instances"][0]["task_type"] == "RETRIEVAL_DOCUMENT"

    @pytest.mark.asyncio
    async def test_embed_one_query(self) -> None:
        provider = GeminiEmbeddingProvider(
            api_key="AQ.mock_express_key",
            embedding_dim=768,
        )

        mock_resp = {
            "predictions": [
                {"embeddings": {"values": [0.05] * 768}},
            ]
        }
        mock_http_resp = httpx.Response(
            status_code=200,
            json=mock_resp,
            request=httpx.Request("POST", "https://aiplatform.googleapis.com"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_http_resp

            query_vec = await provider.embed_one("election results query")
            assert len(query_vec) == 768

            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            assert payload["instances"][0]["task_type"] == "RETRIEVAL_QUERY"

    @pytest.mark.asyncio
    async def test_embed_batching_large_list(self) -> None:
        provider = GeminiEmbeddingProvider(
            api_key="AQ.mock_express_key",
            embedding_dim=768,
        )

        # 150 texts should be split into 2 batches (100 + 50)
        texts = [f"Text chunk {i}" for i in range(150)]

        def make_resp(request: httpx.Request) -> httpx.Response:
            import json
            data = json.loads(request.content)
            count = len(data["instances"])
            return httpx.Response(
                status_code=200,
                json={"predictions": [{"embeddings": {"values": [0.1] * 768}} for _ in range(count)]},
                request=request,
            )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = lambda url, **kwargs: make_resp(
                httpx.Request("POST", url, content=httpx._types.RequestContent(kwargs.get("json", {})))
            )

            # Use simpler mock
            mock_post.side_effect = [
                httpx.Response(200, json={"predictions": [{"embeddings": {"values": [0.1] * 768}} for _ in range(100)]}, request=httpx.Request("POST", "https://test")),
                httpx.Response(200, json={"predictions": [{"embeddings": {"values": [0.1] * 768}} for _ in range(50)]}, request=httpx.Request("POST", "https://test")),
            ]

            vectors = await provider.embed(texts)
            assert len(vectors) == 150
            assert mock_post.call_count == 2
