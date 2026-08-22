"""Unit tests for Settings endpoints: GET & PUT /api/settings/model-bindings."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app


@pytest.mark.asyncio
async def test_get_and_update_model_bindings() -> None:
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. GET current bindings
        get_resp = await client.get("/api/settings/model-bindings")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert "task_bindings" in get_data
        assert "configured_providers" in get_data

        # 2. PUT valid update
        put_resp = await client.put(
            "/api/settings/model-bindings",
            json={
                "task_bindings": {
                    "query_planner": "ollama_chat",
                    "answerer": "ollama_chat",
                }
            },
        )
        assert put_resp.status_code == 200
        put_data = put_resp.json()
        assert put_data["status"] == "updated"
        assert put_data["task_bindings"]["query_planner"] == "ollama_chat"

        # 3. PUT invalid provider ID should return 400
        invalid_resp = await client.put(
            "/api/settings/model-bindings",
            json={
                "task_bindings": {
                    "query_planner": "non_existent_provider_xyz",
                }
            },
        )
        assert invalid_resp.status_code == 400
        assert "not defined" in invalid_resp.json()["detail"]
