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
        "task_bindings": model_cfg.task_bindings,
    }


@router.get(
    "/settings/model-bindings",
    summary="Get current model task bindings",
    tags=["models"],
)
async def get_model_bindings() -> dict[str, Any]:
    """Return the current task→provider bindings from model_config.yaml."""
    model_cfg = get_settings().load_model_config()
    return {
        "task_bindings": model_cfg.task_bindings,
        "providers": {
            k: v.model_dump() for k, v in model_cfg.providers.items()
        },
    }


@router.put(
    "/settings/model-bindings",
    summary="Update model task bindings (admin)",
    tags=["models"],
)
async def update_model_bindings(bindings: dict[str, str]) -> dict[str, Any]:
    """Update task→provider bindings in model_config.yaml.

    Note: This endpoint is stubbed in Phase 0 and will be fully implemented
    in Phase 6 (admin UI). Currently validates the bindings and returns them.

    Args:
        bindings: Dict of {task_name: provider_id}.
    """
    # Phase 0 stub: validate that provided provider_ids exist
    model_cfg = get_settings().load_model_config()
    unknown_providers = [
        pid for pid in bindings.values() if pid not in model_cfg.providers
    ]
    if unknown_providers:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider IDs: {unknown_providers}. "
                   f"Available: {list(model_cfg.providers.keys())}",
        )
    logger.info(
        "Model bindings update requested (stub — not persisted in Phase 0)",
        extra={"bindings": bindings},
    )
    return {"message": "Binding update validated (persistence implemented in Phase 6)", "bindings": bindings}
