"""Canonical Tool Factory and Parameter Reconciler for NewsLens-AI Query Planning.

Provides single-source construction for PlannedToolCall instances and centralized
ground-truth parameter reconciliation and hallucination pruning.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.models import PlannedToolCall

_GENERIC_FILLER_QUERIES: frozenset[str] = frozenset({
    "newspaper coverage comparison",
    "coverage comparison",
    "coverage analysis",
    "newspaper comparison",
    "compare newspapers",
    "all available newspapers",
    "all newspaper",
    "cross newspaper comparison",
    "cross-edition frontpage and lead story comparison",
})


def sanitize_generic_filler_query(
    query: str,
    raw_query_arg: str | None = None,
    category_filter: str | None = None,
) -> str:
    """Detect and sanitize prompt few-shot filler phrases into substantive topic queries."""
    candidate = (raw_query_arg or "").strip().lower()
    if candidate in _GENERIC_FILLER_QUERIES:
        if category_filter:
            return f"{category_filter.lower()} related news"
        return query.strip()
    return raw_query_arg.strip() if raw_query_arg and raw_query_arg.strip() else query.strip()


def reconcile_and_sanitize_arguments(
    tool_name: str,
    args: dict[str, Any],
    extracted: dict[str, Any],
    query: str,
    active_issue_date: str | None = None,
    active_newspapers: list[str] | None = None,
) -> dict[str, Any]:
    """Reconcile tool arguments against query ground truth and active context, pruning hallucinations."""
    sanitized = dict(args)
    q_lower = query.lower()

    # 1. Newspaper Brand Ground Truth & Active Context Retention
    valid_brands: list[str] = (
        extracted.get("target_newspapers")
        or ([extracted["newspaper_name"]] if extracted.get("newspaper_name") else [])
    )
    if sanitized.get("newspaper_name"):
        sn_lower = str(sanitized["newspaper_name"]).strip().lower()
        if valid_brands:
            matched_valid = next(
                (b for b in valid_brands if b.lower() == sn_lower or sn_lower in b.lower() or b.lower() in sn_lower),
                None,
            )
            if matched_valid:
                sanitized["newspaper_name"] = matched_valid
            else:
                sanitized["newspaper_name"] = valid_brands[0]
        else:
            is_active_brand = (
                bool(active_newspapers)
                and any(sn_lower == str(an).strip().lower() for an in (active_newspapers or []))
            )
            if not is_active_brand:
                brand_tokens = [w.lower() for w in str(sanitized["newspaper_name"]).split() if w.lower() not in {"the", "of", "and"}]
                if not any(tok in q_lower for tok in brand_tokens):
                    sanitized.pop("newspaper_name", None)

    # Reconcile comparison_newspaper if present
    if sanitized.get("comparison_newspaper"):
        cn_lower = str(sanitized["comparison_newspaper"]).strip().lower()
        if valid_brands:
            matched_comp = next(
                (b for b in valid_brands if b.lower() == cn_lower or cn_lower in b.lower() or b.lower() in cn_lower),
                None,
            )
            if matched_comp:
                sanitized["comparison_newspaper"] = matched_comp
            elif extracted.get("comparison_newspaper"):
                sanitized["comparison_newspaper"] = extracted["comparison_newspaper"]

    # Reconcile source_newspaper if present
    if sanitized.get("source_newspaper"):
        src_lower = str(sanitized["source_newspaper"]).strip().lower()
        if valid_brands:
            matched_src = next(
                (b for b in valid_brands if b.lower() == src_lower or src_lower in b.lower() or b.lower() in src_lower),
                None,
            )
            if matched_src:
                sanitized["source_newspaper"] = matched_src
            elif extracted.get("source_newspaper"):
                sanitized["source_newspaper"] = extracted["source_newspaper"]

    # 2. Issue Date & Date Range Ground Truth & Active Context Retention
    valid_dates: list[str] = (
        extracted.get("target_dates")
        or ([extracted["issue_date"]] if extracted.get("issue_date") else [])
    )
    if sanitized.get("issue_date"):
        if valid_dates:
            if sanitized["issue_date"] not in valid_dates:
                sanitized["issue_date"] = valid_dates[0]
        else:
            is_active_date = bool(active_issue_date) and str(sanitized["issue_date"]).strip() == str(active_issue_date).strip()
            if not is_active_date:
                d_parts = [p for p in str(sanitized["issue_date"]).split("-") if len(p) >= 2]
                if not any(part in query for part in d_parts):
                    sanitized.pop("issue_date", None)

    for d_key in ("date_from", "date_to", "target_date"):
        if sanitized.get(d_key):
            d_val = str(sanitized[d_key]).strip()
            if valid_dates:
                if d_key == "target_date" and d_val not in valid_dates:
                    sanitized[d_key] = valid_dates[0]
            else:
                is_active_date = bool(active_issue_date) and d_val == str(active_issue_date).strip()
                if not is_active_date:
                    d_parts = [p for p in d_val.split("-") if len(p) >= 2]
                    if not any(part in query for part in d_parts):
                        sanitized.pop(d_key, None)

    # 3. Category Filter Ground Truth
    if sanitized.get("category_filter"):
        cat_val = str(sanitized["category_filter"]).strip().lower()
        valid_cat = extracted.get("category_filter")
        if valid_cat:
            sanitized["category_filter"] = valid_cat
        else:
            cat_tokens = [w for w in re.split(r"[^a-zA-Z0-9]+", cat_val) if len(w) >= 4]
            if not any(tok in q_lower for tok in cat_tokens):
                sanitized.pop("category_filter", None)

    # 4. Page Filter Ground Truth
    if sanitized.get("page_filter") is not None:
        p_val = str(sanitized["page_filter"]).strip()
        if not re.search(rf"\bpage\s*{re.escape(p_val)}\b|\bp\.?\s*{re.escape(p_val)}\b", query, re.I):
            sanitized.pop("page_filter", None)

    # 5. Issue ID Ground Truth
    if extracted.get("issue_id") is not None and "issue_id" not in sanitized:
        sanitized["issue_id"] = extracted["issue_id"]

    # 6. Generic Filler Query Sanitization
    if "query" in sanitized and sanitized["query"]:
        sanitized["query"] = sanitize_generic_filler_query(
            query=query,
            raw_query_arg=str(sanitized["query"]),
            category_filter=sanitized.get("category_filter") or extracted.get("category_filter"),
        )

    return sanitized


# ---------------------------------------------------------------------------
# Tool Construction Helpers
# ---------------------------------------------------------------------------

def build_sql_summary_tool(
    analysis_type: str = "issue_summary",
    newspaper_name: str | None = None,
    issue_date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    issue_id: int | None = None,
    page_filter: str | None = None,
    category_filter: str | None = None,
    query: str | None = None,
    target_date: str | None = None,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned sql_analytics tool invocation with non-null arguments."""
    raw_args: dict[str, Any] = {
        "analysis_type": analysis_type,
        "newspaper_name": newspaper_name,
        "issue_date": issue_date,
        "date_from": date_from,
        "date_to": date_to,
        "issue_id": issue_id,
        "page_filter": page_filter,
        "category_filter": category_filter,
        "query": query,
        "target_date": target_date,
    }
    args = {k: v for k, v in raw_args.items() if v is not None}
    default_purpose = f"SQL analytics manifest ({analysis_type})"
    return PlannedToolCall("sql_analytics", args, purpose or default_purpose)


def build_sql_difference_tool(
    source_newspaper: str,
    comparison_newspaper: str,
    query: str,
    issue_date: str | None = None,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned sql_analytics coverage_difference tool invocation."""
    args: dict[str, Any] = {
        "analysis_type": "coverage_difference",
        "newspaper_name": source_newspaper,
        "comparison_newspaper": comparison_newspaper,
        "query": query,
    }
    if issue_date:
        args["issue_date"] = issue_date
    default_purpose = f"Compute stories in {source_newspaper} absent from {comparison_newspaper}"
    return PlannedToolCall("sql_analytics", args, purpose or default_purpose)


def build_sql_coverage_comparison_tool(
    target_date: str,
    query: str,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned sql_analytics coverage_comparison tool invocation."""
    args: dict[str, Any] = {
        "analysis_type": "coverage_comparison",
        "target_date": target_date,
        "query": query,
    }
    default_purpose = f"SQL coverage comparison on {target_date}"
    return PlannedToolCall("sql_analytics", args, purpose or default_purpose)


def build_hybrid_search_tool(
    query: str,
    newspaper_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page_filter: str | None = None,
    category_filter: str | None = None,
    top_k: int = 6,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned hybrid_search tool invocation with non-null bounds."""
    raw_args: dict[str, Any] = {
        "query": query,
        "newspaper_name": newspaper_name,
        "date_from": date_from,
        "date_to": date_to,
        "page_filter": page_filter,
        "category_filter": category_filter,
        "top_k": top_k,
    }
    args = {k: v for k, v in raw_args.items() if v is not None}
    default_purpose = "Search for factual broadsheet evidence"
    return PlannedToolCall("hybrid_search", args, purpose or default_purpose)


def build_coverage_analysis_tool(
    query: str,
    target_date: str | None = None,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned coverage_analysis tool invocation."""
    args: dict[str, Any] = {"query": query}
    if target_date:
        args["target_date"] = target_date
    default_purpose = "Archive-wide negative coverage audit"
    return PlannedToolCall("coverage_analysis", args, purpose or default_purpose)


def build_timeline_tool(
    query: str,
    limit: int = 25,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned timeline_builder tool invocation."""
    default_purpose = "Aggregate chronological milestones across archive"
    return PlannedToolCall("timeline_builder", {"query": query, "limit": limit}, purpose or default_purpose)


def build_entity_search_tool(
    entity_name: str,
    top_k: int = 10,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned entity_search tool invocation."""
    default_purpose = f"Profile entity '{entity_name}'"
    return PlannedToolCall("entity_search", {"entity_name": entity_name, "top_k": top_k}, purpose or default_purpose)


def build_web_search_tool(
    query: str,
    num_results: int = 5,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned web_search tool invocation."""
    default_purpose = "Live internet search for external context"
    return PlannedToolCall("web_search", {"query": query, "num_results": num_results}, purpose or default_purpose)


__all__ = [
    "_GENERIC_FILLER_QUERIES",
    "build_coverage_analysis_tool",
    "build_entity_search_tool",
    "build_hybrid_search_tool",
    "build_sql_coverage_comparison_tool",
    "build_sql_difference_tool",
    "build_sql_summary_tool",
    "build_timeline_tool",
    "build_web_search_tool",
    "reconcile_and_sanitize_arguments",
    "sanitize_generic_filler_query",
]
