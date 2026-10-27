"""Shared LLM Provider Failover Resolution, JSON Extraction, and Text Cleaning Utilities.

Eliminates duplicate provider resolution, `<think>` tag stripping, and JSON extraction
across QueryPlanner, AnswerSynthesizer, EvidenceEvaluator, AnswerVerifier, and ToolMaker.
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

from app.core.logging import get_logger
from app.providers.base import ChatModelProvider
from app.providers.registry import get_registry

logger = get_logger(__name__)


def resolve_provider_candidates(
    task_name: str | None = None,
    model_override: str | None = None,
    injected_provider: ChatModelProvider | None = None,
    fallback_task_keys: tuple[str, ...] = (),
    prefer_local: bool = False,
    registry_getter: Any = None,
) -> list[ChatModelProvider]:
    """Return a deduplicated, ordered list of viable ChatModelProviders for resilient failover.

    Prioritizes cloud candidates after the primary task model when `prefer_local=False` so
    dead local Ollama ports never stall failover across multiple local models.
    """
    if injected_provider is not None and not model_override:
        return [injected_provider]

    candidates: list[ChatModelProvider] = []
    seen_keys: set[str] = set()

    def _add(p: Any) -> None:
        if p is not None and hasattr(p, "complete"):
            ident = f"{getattr(p, 'provider_name', '')}:{getattr(p, '_model', getattr(p, 'model_name', ''))}:{id(p) if not getattr(p, '_model', None) else ''}"
            if ident not in seen_keys:
                seen_keys.add(ident)
                candidates.append(p)

    try:
        reg_fn = registry_getter or get_registry
        reg = reg_fn()
        if model_override:
            with contextlib.suppress(Exception):
                _add(reg.get_chat_provider(model_override))
        elif task_name:
            with contextlib.suppress(Exception):
                _add(reg.get_provider(task_name))
        else:
            with contextlib.suppress(Exception):
                _add(reg.get_chat_provider(None))

        for t_key in fallback_task_keys:
            with contextlib.suppress(Exception):
                _add(reg.get_provider(t_key))

        for f_key in reg.get_chat_failover_candidates(prefer_local=prefer_local):
            with contextlib.suppress(Exception):
                _add(reg.get_chat_provider(f_key))
    except Exception as e:
        logger.warning(
            "Could not resolve LLM provider candidates from registry",
            extra={"task": task_name, "override": model_override, "error": str(e)},
        )

    return candidates


def strip_thinking_tags(text: str) -> str:
    """Strip `<think>...</think>` or `<thought>...</thought>` reasoning blocks from model output."""
    if not text:
        return ""
    cleaned = re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^<(thought|think)>.*?</\1>", "", cleaned, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract and parse a JSON dictionary from raw LLM output, stripping thinking tags and markdown fences."""
    if not text:
        return None
    cleaned = strip_thinking_tags(text)

    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    with contextlib.suppress(Exception):
        res = json.loads(cleaned)
        if isinstance(res, dict):
            return res

    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        with contextlib.suppress(Exception):
            res = json.loads(match.group(1))
            if isinstance(res, dict):
                return res
    return None


__all__ = [
    "extract_json_object",
    "resolve_provider_candidates",
    "strip_thinking_tags",
]
