"""Unit tests verifying runtime model switching and zero unwanted Docling invocations."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.core.config import Settings
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.ocr_service import OCRService
from app.providers.ollama_provider import OllamaProvider
from app.providers.registry import ModelRegistry, get_registry


@pytest.mark.asyncio
async def test_runtime_model_swapping_live_update() -> None:
    """Verify that updating bindings via PUT actually updates registry and live instances."""
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Reset to default Gemma 4 configuration
        reset_resp = await client.post("/api/settings/model-bindings/reset")
        assert reset_resp.status_code == 200
        data = reset_resp.json()
        assert data["task_bindings"]["layout_analysis"] == "ollama_gemma4_26b"
        assert data["task_bindings"]["query_planner"] == "ollama_gemma4_12b"

        # Check registry resolution
        reg = get_registry()
        prov = reg.get_provider("layout_analysis")
        assert isinstance(prov, OllamaProvider)
        assert prov._model == "gemma4:26b"

        # 2. Swap layout_analysis to ollama_vlm
        put_resp = await client.put(
            "/api/settings/model-bindings",
            json={"task_bindings": {"layout_analysis": "ollama_vlm"}},
        )
        assert put_resp.status_code == 200
        put_data = put_resp.json()
        assert put_data["task_bindings"]["layout_analysis"] == "ollama_vlm"

        # Check that registry immediately yields the swapped provider
        swapped_prov = reg.get_provider("layout_analysis")
        assert isinstance(swapped_prov, OllamaProvider)
        assert swapped_prov._model == "qwen2.5vl:7b"

        # 3. Reset back to Gemma 4
        reset_back = await client.post("/api/settings/model-bindings/reset")
        assert reset_back.status_code == 200
        restored_prov = reg.get_provider("layout_analysis")
        assert isinstance(restored_prov, OllamaProvider)
        assert restored_prov._model == "gemma4:26b"


@pytest.mark.asyncio
async def test_layout_analyzer_and_ocr_use_gemma_without_docling() -> None:
    """Verify LayoutAnalyzer and OCR resolve Gemma 4 directly without instantiating Docling."""
    settings = Settings()
    reg = ModelRegistry(settings=settings)

    # Ensure default bindings point to Gemma 4
    layout_prov = reg.get_provider("layout_analysis")
    assert isinstance(layout_prov, OllamaProvider)
    assert layout_prov._model == "gemma4:26b"

    ocr_prov = reg.get_provider("ocr")
    assert isinstance(ocr_prov, OllamaProvider)

    # Verify LayoutAnalyzer loads OllamaProvider
    analyzer = LayoutAnalyzer()
    active_layout_prov = await analyzer._get_layout_provider()
    assert isinstance(active_layout_prov, OllamaProvider)

    # Verify OCRService loads OllamaProvider or Tesseract without touching Docling
    mock_db = AsyncMock()
    with patch("app.providers.docling_provider.DoclingProvider") as mock_docling:
        OCRService(db=mock_db)
        assert mock_docling.call_count == 0
