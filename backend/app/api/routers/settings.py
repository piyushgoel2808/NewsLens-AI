"""FastAPI router for Model Settings and Runtime Provider Binding Swapping."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.registry import get_registry

logger = get_logger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


class UpdateBindingsRequest(BaseModel):
    """Request payload to update task provider bindings."""

    task_bindings: dict[str, str] = Field(
        ...,
        description="Mapping of task names to configured provider IDs",
        examples=[
            {
                "query_planner": "ollama_llama",
                "answerer": "ollama_llama",
                "layout_analysis": "ollama_vlm",
                "embedding": "local_embed_bge",
                "ocr": "tesseract_ocr",
            }
        ],
    )


@router.get("/model-bindings", summary="Get active task-provider bindings and provider status")
async def get_model_bindings() -> dict[str, Any]:
    """Retrieve current task bindings and reachability status for all configured providers."""
    settings = get_settings()
    model_cfg = settings.load_model_config()
    registry = get_registry()

    providers_status = await registry.get_available_providers()

    return {
        "task_bindings": model_cfg.task_bindings,
        "configured_providers": [
            {
                "id": p_id,
                "provider": p_cfg.provider,
                "model": p_cfg.model,
                "base_url": p_cfg.base_url,
                "context_window": p_cfg.context_window,
                "supports_vision": p_cfg.supports_vision,
                "supports_tool_use": p_cfg.supports_tool_use,
                "embedding_dim": p_cfg.embedding_dim,
            }
            for p_id, p_cfg in model_cfg.providers.items()
        ],
        "provider_reachability": providers_status,
    }


@router.put("/model-bindings", summary="Update task-provider bindings at runtime")
async def update_model_bindings(
    request: UpdateBindingsRequest,
) -> dict[str, Any]:
    """Update task bindings dynamically and persist to model_config.yaml."""
    settings = get_settings()
    model_cfg = settings.load_model_config()

    # Validate that all requested providers exist in configuration
    for _task, provider_id in request.task_bindings.items():
        if provider_id not in model_cfg.providers:
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{provider_id}' is not defined in model_config.yaml",
            )

    # Update in-memory model config
    model_cfg.task_bindings.update(request.task_bindings)

    # Persist to model_config.yaml on disk
    saved = settings.save_model_config(model_cfg)

    # Clear registry cache and reload fresh configuration
    registry = get_registry()
    registry.invalidate_all()

    logger.info(
        "Updated task model bindings at runtime and persisted to disk",
        extra={"new_bindings": request.task_bindings, "saved_to_disk": saved},
    )

    return {
        "status": "updated",
        "saved_to_disk": saved,
        "task_bindings": model_cfg.task_bindings,
    }
