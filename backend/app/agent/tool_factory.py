"""Canonical Tool Factory and Parameter Reconciler for NewsLens-AI Query Planning.

Provides single-source construction for PlannedToolCall instances and centralized
ground-truth parameter reconciliation and hallucination pruning.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    is_archive_wide_newspaper_query,
)
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
    purpose: str | None = None,
) -> dict[str, Any]:
    """Reconcile and normalize tool arguments against ground truth without destructive deletion."""
    sanitized = {k: v for k, v in args.items() if v is not None and v != ""}
    q_lower = query.lower()

    if tool_name == "dynamic_analysis":
        if "query" not in sanitized:
            sanitized["query"] = query
        return sanitized

    if tool_name == "inspect_visual_asset":
        target_date = extracted.get("issue_date") or sanitized.get("issue_date") or active_issue_date
        attached_date = extracted.get("attached_issue_date")
        date_conflict = bool(target_date and attached_date and target_date != attached_date)

        if not date_conflict:
            if extracted.get("attached_photo_id") and "photo_id" not in sanitized:
                sanitized["photo_id"] = extracted["attached_photo_id"]
            if extracted.get("attached_article_id") and "article_id" not in sanitized:
                sanitized["article_id"] = extracted["attached_article_id"]
        if "query" not in sanitized:
            sanitized["query"] = query
        if extracted.get("newspaper_name") and "newspaper_name" not in sanitized:
            sanitized["newspaper_name"] = extracted["newspaper_name"]
        if extracted.get("issue_date") and "issue_date" not in sanitized:
            sanitized["issue_date"] = extracted["issue_date"]
        elif active_issue_date and "issue_date" not in sanitized:
            sanitized["issue_date"] = active_issue_date
        if (extracted.get("page_filter") or extracted.get("page_number")) and "page_filter" not in sanitized:
            sanitized["page_filter"] = str(extracted.get("page_filter") or extracted["page_number"])
        return sanitized

    # Archive-wide newspaper inquiry: clear publication filter if query doesn't name a brand
    is_archive_np = is_archive_wide_newspaper_query(query)
    has_date_in_query = bool(extracted.get("issue_date") or extracted.get("date_from"))
    named_brand_in_query = any(pat.search(query) for pat, _ in _KNOWN_BRANDS_PATTERNS)
    if is_archive_np and not named_brand_in_query:
        sanitized.pop("newspaper_name", None)
        sanitized.pop("comparison_newspaper", None)
        sanitized.pop("source_newspaper", None)
        if not has_date_in_query:
            sanitized.pop("issue_date", None)
            sanitized.pop("date_from", None)
            sanitized.pop("date_to", None)
        if tool_name == "sql_analytics" and sanitized.get("analysis_type") in ("issue_summary", None):
            sanitized["analysis_type"] = "count_issues"

    # Newspaper Brand Normalization
    valid_brands: list[str] = (
        extracted.get("target_newspapers")
        or ([extracted["newspaper_name"]] if extracted.get("newspaper_name") else [])
    )
    if sanitized.get("newspaper_name"):
        sn_lower = str(sanitized["newspaper_name"]).strip().lower()
        if valid_brands:
            matched = next(
                (b for b in valid_brands if b.lower() == sn_lower or sn_lower in b.lower() or b.lower() in sn_lower),
                None,
            )
            sanitized["newspaper_name"] = matched or valid_brands[0]
        else:
            is_active = bool(active_newspapers) and any(sn_lower == str(an).strip().lower() for an in (active_newspapers or []))
            if not is_active:
                brand_tokens = [w.lower() for w in str(sanitized["newspaper_name"]).split() if w.lower() not in {"the", "of", "and"}]
                if not any(tok in q_lower for tok in brand_tokens):
                    sanitized.pop("newspaper_name", None)
    else:
        # Check purpose for specific newspaper brand match if available
        matched_from_purpose = None
        if purpose and valid_brands:
            p_lower = purpose.lower()
            for b in valid_brands:
                if b.lower() in p_lower:
                    matched_from_purpose = b
                    break
            if not matched_from_purpose:
                for pat, canonical in _KNOWN_BRANDS_PATTERNS:
                    if pat.search(purpose):
                        for b in valid_brands:
                            if canonical.lower() in b.lower() or b.lower() in canonical.lower():
                                matched_from_purpose = b
                                break
                        if not matched_from_purpose:
                            matched_from_purpose = canonical
                        break
        if matched_from_purpose:
            sanitized["newspaper_name"] = matched_from_purpose
        elif tool_name == "hybrid_search" and len(valid_brands) >= 2:
            # For cross-newspaper comparative queries, leave hybrid_search unconstrained to search across editions
            pass
        elif extracted.get("newspaper_name") and named_brand_in_query and len(valid_brands) <= 1:
            sanitized["newspaper_name"] = extracted["newspaper_name"]
        elif tool_name == "sql_analytics" and extracted.get("newspaper_name") and len(valid_brands) == 1:
            sanitized["newspaper_name"] = extracted["newspaper_name"]

    for np_field in ("comparison_newspaper", "source_newspaper"):
        if sanitized.get(np_field):
            field_lower = str(sanitized[np_field]).strip().lower()
            if valid_brands:
                matched = next(
                    (b for b in valid_brands if b.lower() == field_lower or field_lower in b.lower() or b.lower() in field_lower),
                    None,
                )
                if matched:
                    sanitized[np_field] = matched
                elif extracted.get(np_field):
                    sanitized[np_field] = extracted[np_field]

    # Shared / differential coverage analysis type and newspaper pairing reconciliation
    if tool_name == "sql_analytics":
        if extracted.get("is_differential"):
            sanitized["analysis_type"] = "coverage_difference"
        elif extracted.get("is_shared") and sanitized.get("analysis_type") in ("coverage_difference", None, "issue_summary"):
            sanitized["analysis_type"] = "shared_coverage"
        elif not sanitized.get("analysis_type"):
            if sanitized.get("issue_date") or extracted.get("issue_date") or sanitized.get("page_filter") or extracted.get("page_filter"):
                sanitized["analysis_type"] = "issue_summary"

    if sanitized.get("analysis_type") in ("shared_coverage", "coverage_difference"):
        if not sanitized.get("comparison_newspaper") and extracted.get("comparison_newspaper"):
            sanitized["comparison_newspaper"] = extracted["comparison_newspaper"]
        elif not sanitized.get("comparison_newspaper") and len(valid_brands) >= 2:
            sanitized["comparison_newspaper"] = valid_brands[1]
        if not sanitized.get("newspaper_name") and (extracted.get("source_newspaper") or extracted.get("newspaper_name")):
            sanitized["newspaper_name"] = extracted.get("source_newspaper") or extracted.get("newspaper_name")
        elif not sanitized.get("newspaper_name") and valid_brands:
            sanitized["newspaper_name"] = valid_brands[0]

    # Date normalization & ground truth retention
    if is_archive_np and not has_date_in_query and not named_brand_in_query:
        sanitized.pop("issue_date", None)
        sanitized.pop("date_from", None)
        sanitized.pop("date_to", None)
    elif extracted.get("issue_date"):
        sanitized["issue_date"] = extracted["issue_date"]
        if not sanitized.get("date_from"):
            sanitized["date_from"] = extracted["issue_date"]
        if not sanitized.get("date_to"):
            sanitized["date_to"] = extracted["issue_date"]
    elif sanitized.get("issue_date"):
        if not sanitized.get("date_from"):
            sanitized["date_from"] = sanitized["issue_date"]
        if not sanitized.get("date_to"):
            sanitized["date_to"] = sanitized["issue_date"]
    elif active_issue_date and not extracted.get("date_from") and not is_archive_np:
        if "issue_date" not in sanitized:
            sanitized["issue_date"] = active_issue_date
            if not sanitized.get("date_from"):
                sanitized["date_from"] = active_issue_date
                sanitized["date_to"] = active_issue_date

    if extracted.get("date_from"):
        sanitized["date_from"] = extracted["date_from"]
        sanitized["date_to"] = extracted.get("date_to", extracted["date_from"])

    if not extracted.get("issue_date") and not extracted.get("date_from") and not (active_issue_date and not is_archive_np):
        sanitized.pop("issue_date", None)
        sanitized.pop("date_from", None)
        sanitized.pop("date_to", None)

    if sanitized.get("date_from") and sanitized.get("date_to") and not extracted.get("issue_date"):
        sanitized.pop("issue_date", None)
        if tool_name == "sql_analytics" and sanitized.get("analysis_type") == "issue_summary" and not sanitized.get("newspaper_name"):
            sanitized["analysis_type"] = "count_issues"

    if tool_name == "coverage_analysis":
        if not sanitized.get("target_date") and (extracted.get("issue_date") or extracted.get("date_from") or active_issue_date):
            sanitized["target_date"] = extracted.get("issue_date") or extracted.get("date_from") or active_issue_date
        if not sanitized.get("newspaper_name") and extracted.get("newspaper_name"):
            sanitized["newspaper_name"] = extracted.get("newspaper_name")

    if tool_name == "entity_search":
        if not sanitized.get("newspaper_name") and extracted.get("newspaper_name"):
            sanitized["newspaper_name"] = extracted.get("newspaper_name")
        if not sanitized.get("entity_name") and extracted.get("entity_name"):
            sanitized["entity_name"] = extracted.get("entity_name")

    # Category and Page filter normalization
    if extracted.get("category_filter") and "category_filter" not in sanitized:
        sanitized["category_filter"] = extracted["category_filter"]

    if extracted.get("page_filter") and "page_filter" not in sanitized:
        sanitized["page_filter"] = str(extracted["page_filter"]).strip()

    if sanitized.get("page_filter") is not None:
        p_val = str(sanitized["page_filter"]).strip()
        is_front_page = p_val == "1" and bool(re.search(r"\b(?:front[\s-]*page|cover[\s-]*page|page\s*(?:1|one))\b", query, re.I))
        if not is_front_page and not re.search(rf"\b(?:page|pg|p\.?)\s*{re.escape(p_val)}\b", query, re.I):
            sanitized.pop("page_filter", None)
        else:
            sanitized["page_filter"] = p_val

    if extracted.get("issue_id") is not None and "issue_id" not in sanitized:
        sanitized["issue_id"] = extracted["issue_id"]

    # Query sanitization
    if "query" in sanitized and sanitized["query"]:
        sanitized["query"] = sanitize_generic_filler_query(
            query=query,
            raw_query_arg=str(sanitized["query"]),
            category_filter=sanitized.get("category_filter") or extracted.get("category_filter"),
        )
    elif "query" not in sanitized or not sanitized["query"]:
        sanitized["query"] = query

    return sanitized


# ---------------------------------------------------------------------------
# Tool Construction Helpers
# ---------------------------------------------------------------------------

def build_sql_tool(analysis_type: str, **kwargs: Any) -> PlannedToolCall:
    """Unified constructor for planned sql_analytics tool invocations."""
    purpose = kwargs.pop("purpose", None) or f"SQL analytics ({analysis_type})"
    raw_args = {"analysis_type": analysis_type, **kwargs}
    args = {k: v for k, v in raw_args.items() if v is not None and v != ""}
    return PlannedToolCall("sql_analytics", args, purpose)


def build_sql_summary_tool(**kwargs: Any) -> PlannedToolCall:
    """Legacy alias delegating to build_sql_tool."""
    analysis_type = kwargs.pop("analysis_type", "issue_summary")
    return build_sql_tool(analysis_type=analysis_type, **kwargs)


def build_sql_difference_tool(source_newspaper: str, comparison_newspaper: str, query: str, issue_date: str | None = None, purpose: str = "") -> PlannedToolCall:
    """Legacy alias delegating to build_sql_tool."""
    return build_sql_tool(
        analysis_type="coverage_difference",
        newspaper_name=source_newspaper,
        comparison_newspaper=comparison_newspaper,
        query=query,
        issue_date=issue_date,
        purpose=purpose or f"Compute stories in {source_newspaper} absent from {comparison_newspaper}",
    )


def build_sql_shared_coverage_tool(newspaper_a: str, newspaper_b: str, issue_date: str | None = None, query: str | None = None, purpose: str = "") -> PlannedToolCall:
    """Legacy alias delegating to build_sql_tool."""
    return build_sql_tool(
        analysis_type="shared_coverage",
        newspaper_name=newspaper_a,
        comparison_newspaper=newspaper_b,
        issue_date=issue_date,
        query=query,
        purpose=purpose or f"Compute verified shared syndicated wire coverage between {newspaper_a} and {newspaper_b}",
    )


def build_sql_coverage_comparison_tool(target_date: str, query: str, purpose: str = "") -> PlannedToolCall:
    """Legacy alias delegating to build_sql_tool."""
    return build_sql_tool(
        analysis_type="coverage_comparison",
        target_date=target_date,
        query=query,
        purpose=purpose or f"SQL coverage comparison on {target_date}",
    )



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
    newspaper_name: str | None = None,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned coverage_analysis tool invocation."""
    args: dict[str, Any] = {"query": query}
    if target_date:
        args["target_date"] = target_date
    if newspaper_name:
        args["newspaper_name"] = newspaper_name
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
    newspaper_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned entity_search tool invocation."""
    args: dict[str, Any] = {"entity_name": entity_name, "top_k": top_k}
    if newspaper_name:
        args["newspaper_name"] = newspaper_name
    if date_from:
        args["date_from"] = date_from
    if date_to:
        args["date_to"] = date_to
    default_purpose = f"Profile entity '{entity_name}'"
    return PlannedToolCall("entity_search", args, purpose or default_purpose)


def build_web_search_tool(
    query: str,
    num_results: int = 5,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned web_search tool invocation."""
    default_purpose = "Live internet search for external context"
    return PlannedToolCall("web_search", {"query": query, "num_results": num_results}, purpose or default_purpose)


def build_inspect_visual_asset_tool(
    photo_id: int | None = None,
    article_id: int | None = None,
    query: str = "",
    newspaper_name: str = "",
    issue_date: str = "",
    page_filter: str | int | None = None,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned inspect_visual_asset tool invocation."""
    args: dict[str, Any] = {}
    if photo_id is not None:
        args["photo_id"] = photo_id
    if article_id is not None:
        args["article_id"] = article_id
    if query:
        args["query"] = query
    if newspaper_name:
        args["newspaper_name"] = newspaper_name
    if issue_date:
        args["issue_date"] = issue_date
    if page_filter is not None and str(page_filter).strip():
        args["page_filter"] = str(page_filter).strip()
    default_purpose = "Deep multimodal visual inspection and numerical data table transcription"
    return PlannedToolCall("inspect_visual_asset", args, purpose or default_purpose)


def build_dynamic_analysis_tool(
    query: str,
    analysis_description: str = "",
    newspaper_name: str | None = None,
    issue_date: str | None = None,
    purpose: str = "",
) -> PlannedToolCall:
    """Build a planned dynamic_analysis tool invocation."""
    args: dict[str, Any] = {
        "query": query,
        "analysis_description": analysis_description or f"Statistical computation: {query}",
    }
    if newspaper_name:
        args["newspaper_name"] = newspaper_name
    if issue_date:
        args["issue_date"] = issue_date
    default_purpose = "Synthesize and execute dynamic Python analytical tool"
    return PlannedToolCall("dynamic_analysis", args, purpose or default_purpose)


__all__ = [
    "_GENERIC_FILLER_QUERIES",
    "build_coverage_analysis_tool",
    "build_dynamic_analysis_tool",
    "build_entity_search_tool",
    "build_hybrid_search_tool",
    "build_inspect_visual_asset_tool",
    "build_sql_coverage_comparison_tool",
    "build_sql_difference_tool",
    "build_sql_shared_coverage_tool",
    "build_sql_summary_tool",
    "build_sql_tool",
    "build_timeline_tool",
    "build_web_search_tool",
    "reconcile_and_sanitize_arguments",
    "sanitize_generic_filler_query",
]
