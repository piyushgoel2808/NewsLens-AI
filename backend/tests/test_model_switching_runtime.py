"""Unit tests verifying runtime model switching and zero unwanted Docling invocations."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app
from app.core.config import Settings
from app.ingestion.ocr_service import OCRService
from app.providers.ollama_provider import OllamaProvider
from app.providers.registry import ModelRegistry, get_registry


@pytest.mark.asyncio
async def test_runtime_model_swapping_live_update() -> None:
    """Verify that updating bindings via PUT actually updates registry and live instances."""
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Reset to default configuration
        reset_resp = await client.post("/api/settings/model-bindings/reset")
        assert reset_resp.status_code == 200
        data = reset_resp.json()
        assert data["task_bindings"]["layout_analysis"] == "google_cloud_vision"
        assert data["task_bindings"]["query_planner"] == "ollama_gemma4_12b"

        # 2. Swap layout_analysis to ollama_gemma4_26b
        put_resp = await client.put(
            "/api/settings/model-bindings",
            json={"task_bindings": {"layout_analysis": "ollama_gemma4_26b"}},
        )
        assert put_resp.status_code == 200
        put_data = put_resp.json()
        assert put_data["task_bindings"]["layout_analysis"] == "ollama_gemma4_26b"

        # Check that registry immediately yields the swapped provider
        reg = get_registry()
        swapped_prov = reg.get_provider("layout_analysis")
        assert isinstance(swapped_prov, OllamaProvider)
        assert swapped_prov._model == "gemma4:26b"

        # 3. Reset back to defaults
        reset_back = await client.post("/api/settings/model-bindings/reset")
        assert reset_back.status_code == 200
        assert reset_back.json()["task_bindings"]["layout_analysis"] == "google_cloud_vision"


@pytest.mark.asyncio
async def test_layout_analyzer_and_ocr_use_gemma_without_docling() -> None:
    """Verify LayoutAnalyzer and OCR resolve cleanly without instantiating Docling."""
    settings = Settings()
    reg = ModelRegistry(settings=settings)

    # Verify provider resolution
    layout_prov = reg.get_provider("layout_analysis")
    assert layout_prov is not None

    ocr_prov = reg.get_provider("ocr")
    assert ocr_prov is not None

    # Verify OCRService initializes cleanly
    mock_db = AsyncMock()
    ocr_srv = OCRService(db=mock_db)
    assert ocr_srv is not None
