"""Model provider introspection endpoints.

GET  /api/models/available       — list all configured providers + reachability
GET  /api/settings/model-bindings — read current task→provider bindings
PUT  /api/settings/model-bindings — update bindings (admin, Phase 6+)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.registry import get_registry

router = APIRouter(tags=["models"])
logger = get_logger(__name__)


@router.get(
    "/models/available",
    summary="List available model providers",
    tags=["models"],
)
async def get_available_models() -> dict[str, Any]:
    """Introspect all configured providers and check reachability.

    Returns:
        JSON with 'providers' list (each with id, provider, model,
        is_reachable, capabilities) and 'task_bindings' dict.
    """
    registry = get_registry()
    providers = await registry.get_available_providers()
    model_cfg = get_settings().load_model_config()
    return {
        "providers": providers,
        "models": providers,
        "task_bindings": model_cfg.task_bindings,
    }
