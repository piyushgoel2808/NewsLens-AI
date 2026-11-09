"""Unit tests for Phase 2 Hardening: API Auth, PIN Lockout, Unlimited Qdrant Date Ranges, ContextVar Bindings, and Distributed Rate Limiting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client.http import models as qmodels

from app.api.main import create_app
from app.core.config import QdrantSettings, get_settings
from app.core.security import pin_attempt_tracker
from app.providers.registry import ModelRegistry
from app.storage.qdrant_store import QdrantStore


@pytest.mark.asyncio
async def test_api_auth_enforced_when_configured() -> None:
    """Verify mutating and query endpoints enforce X-API-Key / Bearer token when API_AUTH_KEY is set."""
    app = create_app()
    settings = get_settings()
    transport = ASGITransport(app=app)

    with patch.object(settings, "api_auth_key", "super-secret-prod-key"):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Unauthenticated request is rejected with 401
            unauth_resp = await client.post("/api/model-bindings/reset" if False else "/api/settings/model-bindings/reset")
            assert unauth_resp.status_code == 401

            # 2. Invalid key is rejected with 401
            bad_resp = await client.post(
                "/api/settings/model-bindings/reset",
                headers={"X-API-Key": "wrong-key"},
            )
            assert bad_resp.status_code == 401

            # 3. Valid X-API-Key succeeds
            ok_resp = await client.post(
                "/api/settings/model-bindings/reset",
                headers={"X-API-Key": "super-secret-prod-key"},
            )
            assert ok_resp.status_code == 200

            # 4. Valid Bearer token succeeds
            bearer_resp = await client.post(
                "/api/settings/model-bindings/reset",
                headers={"Authorization": "Bearer super-secret-prod-key"},
            )
            assert bearer_resp.status_code == 200


@pytest.mark.asyncio
async def test_laptop_node_pin_brute_force_lockout() -> None:
    """Verify 5 failed PIN attempts lock out the client IP with HTTP 429."""
    pin_attempt_tracker.reset("127.0.0.1")
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(5):
            r = await client.post(
                "/api/settings/laptop-node/toggle",
                json={"enabled": True},
                headers={"X-Admin-Pin": "0000"},
            )
            assert r.status_code == 403

        # 6th attempt is locked out with 429 Too Many Requests
        locked_resp = await client.post(
            "/api/settings/laptop-node/toggle",
            json={"enabled": True},
            headers={"X-Admin-Pin": "1234"},
        )
        assert locked_resp.status_code == 429

    pin_attempt_tracker.reset("127.0.0.1")


def test_qdrant_unlimited_multi_year_date_range_filter() -> None:
    """Verify Qdrant _build_filter uses MatchAny for <=366d and Range on issue_date_int for >366d or open-ended ranges."""
    store = QdrantStore(QdrantSettings())

    # 1. Short range (<= 366 days) uses MatchAny on issue_date for backward compatibility
    short_filter = store._build_filter({"issue_date": {"gte": "2026-08-01", "lte": "2026-08-03"}})
    assert short_filter is not None
    cond_short = short_filter.must[0]
    assert isinstance(cond_short, qmodels.FieldCondition)
    assert cond_short.key == "issue_date"
    assert isinstance(cond_short.match, qmodels.MatchAny)
    assert len(cond_short.match.any) == 3

    # 2. Multi-year range (> 366 days) uses Range on issue_date_int (YYYYMMDD)
    multi_year_filter = store._build_filter({"issue_date": {"gte": "2022-01-01", "lte": "2026-08-22"}})
    assert multi_year_filter is not None
    cond_long = multi_year_filter.must[0]
    assert isinstance(cond_long, qmodels.FieldCondition)
    assert cond_long.key == "issue_date_int"
    assert isinstance(cond_long.range, qmodels.Range)
    assert cond_long.range.gte == 20220101.0
    assert cond_long.range.lte == 20260822.0

    # 3. Open-ended range (only gte) uses Range on issue_date_int
    open_filter = store._build_filter({"issue_date": {"gte": "2025-06-15"}})
    assert open_filter is not None
    cond_open = open_filter.must[0]
    assert isinstance(cond_open, qmodels.FieldCondition)
    assert cond_open.key == "issue_date_int"
    assert cond_open.range.gte == 20250615.0
    assert cond_open.range.lte is None


def test_contextvar_scoped_task_bindings_isolation() -> None:
    """Verify ModelRegistry.scoped_task_bindings isolates overrides via ContextVar without mutating global config."""
    settings = get_settings()
    reg = ModelRegistry(settings)
    original_doc_parser = reg._model_config.task_bindings["document_parser"]

    with patch.object(reg, "_instantiate", return_value=MagicMock()) as mock_instantiate:
        with reg.scoped_task_bindings({"document_parser": "docling_cloud_parser"}):
            # Global config dictionary remains untouched
            assert reg._model_config.task_bindings["document_parser"] == original_doc_parser
            # Inside context, get_provider resolves the scoped binding
            _ = reg.get_provider("document_parser")
            mock_instantiate.assert_called_with("docling_cloud_parser")

        # Outside context, restores original binding
        mock_instantiate.reset_mock()
        _ = reg.get_provider("document_parser")
        mock_instantiate.assert_called_with(original_doc_parser)


@pytest.mark.asyncio
async def test_rate_limit_middleware_blocks_excess_requests() -> None:
    """Verify RateLimitMiddleware enforces RATE_LIMIT_RPM per client IP."""
    app = create_app()
    settings = get_settings()
    transport = ASGITransport(app=app)

    with patch.object(settings, "rate_limit_rpm", 3):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(3):
                r = await client.get("/api/models")
                assert r.status_code != 429
            # 4th request within the same minute returns 429
            blocked = await client.get("/api/models")
            assert blocked.status_code == 429
