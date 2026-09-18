"""Unit tests for Docling dual mode (Local vs Hosted IBM Cloud SaaS API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.core.config import get_settings
from app.ingestion.parsers.docling import DoclingLayoutParser, DoclingParsedItem
from app.providers.registry import ModelRegistry, reset_registry


class TestDoclingDualMode:
    """Test suite for dual local/cloud Docling parsing architecture."""

    def test_local_initialization(self) -> None:
        """Local mode should have is_cloud=False and provider_name='docling'."""
        parser = DoclingLayoutParser(is_cloud=False)
        assert parser.is_cloud is False
        assert parser.provider_name == "docling"
        assert parser._client is None

    def test_cloud_initialization_with_credentials(self) -> None:
        """Cloud mode with credentials should initialize DoclingServiceClient."""
        with patch("docling.service_client.DoclingServiceClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            parser = DoclingLayoutParser(
                is_cloud=True,
                api_key="test_api_key_12345",
                service_url="https://api.test.docling.ibm.com/v1",
            )
            assert parser.is_cloud is True
            assert parser.provider_name == "docling_cloud"
            assert parser._client is not None
            mock_client_cls.assert_called_once_with(
                url="https://api.test.docling.ibm.com/v1",
                api_key="test_api_key_12345",
            )

    def test_cloud_mode_fallback_when_no_api_key(self) -> None:
        """If cloud mode is requested without an API key, fall back gracefully to local."""
        with patch.object(get_settings(), "docling_api_key", None):
            parser = DoclingLayoutParser(is_cloud=True, api_key=None)
            assert parser.is_cloud is False
            assert parser.provider_name == "docling"
            assert parser._client is None

    def test_registry_instantiates_docling_cloud(self) -> None:
        """ModelRegistry should correctly instantiate DoclingLayoutParser in cloud mode."""
        reset_registry()
        settings = get_settings()
        registry = ModelRegistry(settings)

        with patch("docling.service_client.DoclingServiceClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            provider = registry.get_provider_by_id("docling_cloud_parser")
            assert isinstance(provider, DoclingLayoutParser)
            assert provider.is_cloud is True

    def test_registry_instantiates_docling_local(self) -> None:
        """ModelRegistry should instantiate DoclingLayoutParser in local mode for docling_parser."""
        reset_registry()
        settings = get_settings()
        registry = ModelRegistry(settings)

        provider = registry.get_provider_by_id("docling_parser")
        assert isinstance(provider, DoclingLayoutParser)
        assert provider.is_cloud is False

    def test_cloud_docling_fallback_to_local_on_conversion_error(self) -> None:
        """If cloud conversion fails, it should seamlessly fallback to local converter."""
        with patch("docling.service_client.DoclingServiceClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.convert.side_effect = RuntimeError("503 Service Unavailable")
            mock_client_cls.return_value = mock_client

            parser = DoclingLayoutParser(
                is_cloud=True,
                api_key="test_key",
                service_url="https://mock.service.url",
            )
            assert parser.is_cloud is True

            # Mock the local converter so we don't do real heavy OCR in this unit test
            mock_local_converter = MagicMock()
            mock_doc = MagicMock()
            mock_doc.iterate_items.return_value = []
            mock_local_converter.convert.return_value.document = mock_doc
            parser._converter = mock_local_converter

            # Call parse_docling_document — cloud will fail, local will succeed
            result = parser.parse_docling_document(
                pdf_bytes=b"%PDF-1.4 mock",
                page_number=1,
                width_px=800,
                height_px=1200,
            )
            assert isinstance(result, list)
            mock_client.convert.assert_called_once()
            mock_local_converter.convert.assert_called_once()
