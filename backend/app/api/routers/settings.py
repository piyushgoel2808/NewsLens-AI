"""FastAPI router for Model Settings and Runtime Provider Binding Swapping."""

from __future__ import annotations

import hmac
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException
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

    result: dict[str, Any] = {
        "status": "updated",
        "saved_to_disk": saved,
        "task_bindings": model_cfg.task_bindings,
    }
    if not saved:
        result["warning"] = (
            "Task bindings updated in memory for the active session, "
            "but could not be persisted to model_config.yaml on disk. "
            "Verify host volume mount permissions."
        )
    return result


@router.post("/model-bindings/reset", summary="Reset task-provider bindings to system defaults")
async def reset_model_bindings() -> dict[str, Any]:
    """Reset task model bindings back to default configuration."""
    from app.core.config import DEFAULT_TASK_BINDINGS
    settings = get_settings()
    model_cfg = settings.load_model_config()

    default_bindings = dict(DEFAULT_TASK_BINDINGS)

    # Ensure all default providers exist or fall back safely
    for task, provider_id in default_bindings.items():
        if provider_id in model_cfg.providers:
            model_cfg.task_bindings[task] = provider_id

    saved = settings.save_model_config(model_cfg)
    registry = get_registry()
    registry.invalidate_all()

    logger.info(
        "Reset model task bindings to default configuration",
        extra={"saved_to_disk": saved},
    )

    result: dict[str, Any] = {
        "status": "reset_to_default",
        "saved_to_disk": saved,
        "task_bindings": model_cfg.task_bindings,
    }
    if not saved:
        result["warning"] = (
            "Task bindings reset in memory for the active session, "
            "but could not be persisted to model_config.yaml on disk. "
            "Verify host volume mount permissions."
        )
    return result


class LaptopNodeToggleRequest(BaseModel):
    enabled: bool
    tunnel_url: str | None = None
    default_model: str | None = None


_LAPTOP_NODE_RUNTIME: dict[str, Any] = {
    "enabled": False,
    "tunnel_url": "",
    "default_model": "llama3.1:8b",
    "available_models": [],
    "online": False,
    "_initialized": False,
}


def get_laptop_node_runtime_state() -> dict[str, Any]:
    """Retrieve global in-memory runtime state for Sovereign Laptop Node."""
    settings = get_settings()
    if not _LAPTOP_NODE_RUNTIME["_initialized"]:
        _LAPTOP_NODE_RUNTIME["_initialized"] = True
        _LAPTOP_NODE_RUNTIME["tunnel_url"] = settings.laptop_tunnel_url
        _LAPTOP_NODE_RUNTIME["enabled"] = settings.laptop_node_enabled
        _LAPTOP_NODE_RUNTIME["default_model"] = settings.laptop_default_model
    return _LAPTOP_NODE_RUNTIME


@router.get("/laptop-node/status", summary="Get Sovereign Laptop Node status and discovered local models")
async def get_laptop_node_status(
    x_admin_pin: str | None = Header(None, alias="X-Admin-Pin"),
) -> dict[str, Any]:
    """Public status endpoint to check if host laptop is online and list installed Ollama models."""
    settings = get_settings()
    state = get_laptop_node_runtime_state()

    is_admin = bool(
        x_admin_pin
        and settings.laptop_admin_pin.strip()
        and hmac.compare_digest(x_admin_pin.strip().encode(), settings.laptop_admin_pin.strip().encode())
    )

    if not state["enabled"]:
        return {
            "enabled": False,
            "online": False,
            "available_models": [],
            "default_model": state.get("default_model", "llama3.1:8b"),
            "tunnel_url": state.get("tunnel_url", "") if is_admin else None,
        }

    tunnel_url = str(state.get("tunnel_url") or settings.laptop_tunnel_url).rstrip("/")
    headers = {
        "ngrok-skip-browser-warning": "true",
        "X-Laptop-Bridge-Secret": settings.laptop_bridge_secret,
    }

    try:
        async with httpx.AsyncClient(timeout=2.5, headers=headers) as client:
            resp = await client.get(f"{tunnel_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", []) if "name" in m]
                state["available_models"] = models
                state["online"] = True
                return {
                    "enabled": True,
                    "online": True,
                    "available_models": models,
                    "default_model": state.get("default_model", "llama3.1:8b"),
                    "tunnel_url": tunnel_url if is_admin else None,
                }
    except Exception as exc:
        logger.debug("Laptop node health check failed", extra={"error": str(exc), "tunnel_url": tunnel_url})

    state["online"] = False
    return {
        "enabled": True,
        "online": False,
        "available_models": [],
        "default_model": state.get("default_model", "llama3.1:8b"),
        "tunnel_url": tunnel_url if is_admin else None,
    }


@router.post("/laptop-node/toggle", summary="Toggle Sovereign Laptop Node ON or OFF (Admin PIN Protected)")
async def toggle_laptop_node(
    payload: LaptopNodeToggleRequest,
    x_admin_pin: str | None = Header(None, alias="X-Admin-Pin"),
) -> dict[str, Any]:
    """Toggle laptop node access and update tunnel URL. Requires admin PIN."""
    settings = get_settings()
    expected_pin = settings.laptop_admin_pin.strip()

    if not expected_pin or not x_admin_pin or not hmac.compare_digest(
        x_admin_pin.strip().encode(), expected_pin.encode()
    ):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing X-Admin-Pin")

    tunnel: str | None = None
    if payload.tunnel_url:
        tunnel = payload.tunnel_url.strip().rstrip("/")
        if not (tunnel.startswith("http://") or tunnel.startswith("https://")):
            raise HTTPException(status_code=400, detail="Invalid tunnel URL: must start with http:// or https://")

    state = get_laptop_node_runtime_state()
    state["enabled"] = payload.enabled
    if tunnel:
        state["tunnel_url"] = tunnel
    if payload.default_model:
        state["default_model"] = payload.default_model.strip()

    # Clear provider cache so newly activated tunnel settings take immediate effect
    registry = get_registry()
    registry.invalidate_all()

    # If enabled, refresh status and discover models
    if state["enabled"]:
        status_res = await get_laptop_node_status(x_admin_pin=x_admin_pin)
        return {
            "status": "updated",
            "enabled": state["enabled"],
            "online": status_res.get("online", False),
            "available_models": status_res.get("available_models", []),
            "default_model": state["default_model"],
            "tunnel_url": state["tunnel_url"],
        }

    state["online"] = False
    state["available_models"] = []
    return {
        "status": "updated",
        "enabled": False,
        "online": False,
        "available_models": [],
        "default_model": state["default_model"],
        "tunnel_url": state["tunnel_url"],
    }
