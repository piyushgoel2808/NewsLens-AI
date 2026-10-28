"""Post-Planning Completeness Enforcement and Guardrail Helpers for QueryPlanner."""

from __future__ import annotations

import re
from typing import Any

from app.agent.decision_layer import (
    is_dynamic_analysis_permitted,
    is_visual_query,
    is_volume_query,
)
from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    build_targeted_web_query,
)
from app.agent.models import PlannedToolCall
from app.agent.tool_factory import (
    build_coverage_analysis_tool,
    build_entity_search_tool,
    build_hybrid_search_tool,
    build_inspect_visual_asset_tool,
    build_sql_coverage_comparison_tool,
    build_sql_difference_tool,
    build_sql_shared_coverage_tool,
    build_sql_summary_tool,
    build_web_search_tool,
)


def _deduplicate_tool_calls(tool_calls: list[PlannedToolCall]) -> list[PlannedToolCall]:
    """Deduplicate PlannedToolCall instances by (tool_name, sorted arguments)."""
    deduped: list[PlannedToolCall] = []
    seen_keys: set[str] = set()
    for t in tool_calls:
        c_key = f"{t.tool_name}:{sorted((k, str(v)) for k, v in t.arguments.items())}"
        if c_key not in seen_keys:
            seen_keys.add(c_key)
            deduped.append(t)
    return deduped


def _extract_clean_entity(query: str) -> str:
    """Strip conversational prefixes and newspaper brand names to isolate an entity name."""
    clean_ent = re.sub(
        r"(?i)^(?:find\s+(?:key\s+)?(?:news|coverage|reports?)\s+(?:and\s+(?:news|coverage)\s+)?(?:regarding|about|on)|"
        r"everything\s+about|all\s+mentions\s+of|mentions\s+of|profile\s+the\s+coverage\s+of|profile\s+of|"
        r"coverage\s+regarding|news\s+regarding|involvement\s+of|role\s+of)\s*",
        "",
        query,
    ).strip("?:!.,\"' ") or query
    for pat, _ in _KNOWN_BRANDS_PATTERNS:
        clean_ent = pat.sub("", clean_ent).strip("?:!.,\"' ")
    return re.sub(r"(?i)\s+in\s*$", "", clean_ent).strip("?:!.,\"' ")


def _build_comparative_tool_calls(
    query: str,
    src_np: str | None,
    comp_np: str | None,
    target_dt: str | None,
    target_pages: list[str],
    category_filter: str | None,
    is_diff: bool,
    is_shared: bool,
) -> list[PlannedToolCall]:
    """Construct comparative SQL manifest + hybrid search tool sequence across newspapers or pages."""
    calls: list[PlannedToolCall] = []
    if src_np and comp_np and is_diff:
        calls.append(build_sql_difference_tool(src_np, comp_np, query, target_dt))
        calls.append(build_hybrid_search_tool(query, newspaper_name=src_np, date_from=target_dt, date_to=target_dt, top_k=10, purpose=f"Retrieve articles from {src_np}"))
    elif src_np and comp_np and is_shared:
        calls.append(build_sql_shared_coverage_tool(src_np, comp_np, issue_date=target_dt, query=query))
        calls.append(build_hybrid_search_tool(query, date_from=target_dt, date_to=target_dt, top_k=10, purpose=f"Shared coverage between {src_np} and {comp_np}"))
    elif src_np and comp_np:
        for np_name in (src_np, comp_np):
            calls.append(build_sql_summary_tool(newspaper_name=np_name, issue_date=target_dt, category_filter=category_filter, query=query, purpose=f"Retrieve manifest for {np_name}"))
        calls.append(build_hybrid_search_tool(query=query, date_from=target_dt, date_to=target_dt, category_filter=category_filter, top_k=12, purpose=f"Comparative articles across {src_np} and {comp_np}"))
    elif len(target_pages) >= 2:
        for p in target_pages:
            calls.append(
                build_sql_summary_tool(
                    newspaper_name=src_np,
                    issue_date=target_dt,
                    page_filter=p,
                    category_filter=category_filter,
                    query=query,
                    purpose=f"Retrieve manifest for Page {p}{f' of {src_np}' if src_np else ''}",
                )
            )
        calls.append(
            build_hybrid_search_tool(
                query=query,
                newspaper_name=src_np,
                date_from=target_dt,
                date_to=target_dt,
                category_filter=category_filter,
                top_k=12,
                purpose=f"Comparative articles across pages {', '.join(target_pages)}",
            )
        )
    elif not target_dt:
        calls.append(build_hybrid_search_tool(query=query, category_filter=category_filter, top_k=12, purpose="Comparative articles across broadsheet editions"))
        if any(w in query.lower() for w in ("omit", "miss", "exclusive", "gap", "audit", "coverage analysis", "unreported")):
            calls.append(build_coverage_analysis_tool(query=query, purpose="Archive-wide coverage comparison"))
    else:
        calls.append(build_sql_summary_tool(newspaper_name=src_np, issue_date=target_dt, category_filter=category_filter, query=query, purpose=f"Manifest across all newspapers on {target_dt}" if not src_np else f"SQL article manifest for {src_np}"))
        calls.append(
            build_hybrid_search_tool(
                query=query,
                newspaper_name=src_np,
                date_from=target_dt,
                date_to=target_dt,
                page_filter="1" if not src_np and not category_filter else None,
                category_filter=category_filter,
                top_k=12,
                purpose="Comparative article excerpts",
            )
        )
        if not category_filter or any(w in query.lower() for w in ("omit", "miss", "exclusive", "gap", "audit")):
            calls.append(build_coverage_analysis_tool(query=query, target_date=target_dt))
        if not category_filter and not src_np:
            calls.append(build_sql_coverage_comparison_tool(target_date=target_dt, query=query))
    return calls



def _enforce_direct_tool_completeness(
    tool_calls: list[PlannedToolCall],
    query: str,
    archetype: str,
    extracted: dict[str, Any],
    is_comparative: bool,
    active_issue_date: str | None = None,
    active_newspapers: list[str] | None = None,
) -> None:
    """Ensure multi-newspaper, multi-page, and volume count tool calls are complete and sanitized."""
    q_lower = query.lower()
    target_nps = extracted.get("target_newspapers", [])

    if archetype == "cross_newspaper_comparison" and len(target_nps) >= 2:
        target_dt = extracted.get("issue_date") or active_issue_date
        p_filt = extracted.get("page_filter")
        for c in tool_calls:
            if c.tool_name == "sql_analytics" and not c.arguments.get("analysis_type"):
                c.arguments["analysis_type"] = "issue_summary"

        sql_summary_calls = [
            c for c in tool_calls
            if c.tool_name == "sql_analytics" and c.arguments.get("analysis_type") == "issue_summary"
        ]
        assigned_nps: set[str] = set()
        calls_to_reassign: list[PlannedToolCall] = []
        for c in sql_summary_calls:
            np_val = (c.arguments.get("newspaper_name") or "").strip().lower()
            if np_val and np_val not in assigned_nps:
                assigned_nps.add(np_val)
            else:
                calls_to_reassign.append(c)

        for req_np in target_nps:
            if not any(req_np.lower() in an or an in req_np.lower() for an in assigned_nps) and calls_to_reassign:
                call_to_fix = calls_to_reassign.pop(0)
                call_to_fix.arguments["newspaper_name"] = req_np
                if target_dt and "issue_date" not in call_to_fix.arguments:
                    call_to_fix.arguments["issue_date"] = target_dt
                if p_filt and "page_filter" not in call_to_fix.arguments:
                    call_to_fix.arguments["page_filter"] = str(p_filt)
                call_to_fix.purpose = f"Retrieve manifest for {req_np}"
                assigned_nps.add(req_np.lower())

        for req_np in target_nps:
            if not any(req_np.lower() in an or an in req_np.lower() for an in assigned_nps):
                tool_calls.append(
                    build_sql_summary_tool(
                        newspaper_name=req_np,
                        issue_date=target_dt,
                        page_filter=p_filt,
                        query=query,
                        purpose=f"Retrieve manifest for {req_np}",
                    )
                )
                assigned_nps.add(req_np.lower())

    if archetype == "cross_newspaper_comparison" and len(target_nps) < 2 and (extracted.get("issue_date") or active_issue_date):
        target_dt = extracted.get("issue_date") or active_issue_date
        has_summary = any(
            c.tool_name == "sql_analytics" and c.arguments.get("analysis_type") == "issue_summary"
            for c in tool_calls
        )
        if not has_summary:
            sql_calls = [c for c in tool_calls if c.tool_name == "sql_analytics"]
            if sql_calls:
                sql_calls[0].arguments["analysis_type"] = "issue_summary"
                sql_calls[0].arguments["issue_date"] = target_dt
                sql_calls[0].arguments.pop("newspaper_name", None)
                sql_calls[0].purpose = f"Retrieve manifests for all newspapers on {target_dt}"
            else:
                tool_calls.insert(
                    0,
                    build_sql_summary_tool(
                        newspaper_name=None,
                        issue_date=target_dt,
                        page_filter=extracted.get("page_filter"),
                        query=query,
                        purpose=f"Retrieve manifests for all newspapers on {target_dt}",
                    ),
                )

    target_pages: list[str] = [str(p).strip() for p in extracted.get("target_pages", []) if str(p).strip()]
    if len(target_pages) >= 2:
        target_dt = extracted.get("issue_date") or active_issue_date
        req_np = extracted.get("newspaper_name") or (active_newspapers[0] if active_newspapers else None)
        for c in tool_calls:
            if c.tool_name == "sql_analytics" and not c.arguments.get("analysis_type"):
                c.arguments["analysis_type"] = "issue_summary"

        sql_summary_calls = [
            c for c in tool_calls
            if c.tool_name == "sql_analytics" and c.arguments.get("analysis_type") == "issue_summary"
        ]
        assigned_pages: set[str] = set()
        calls_to_reassign = []
        for c in sql_summary_calls:
            p_val = str(c.arguments.get("page_filter") or "").strip()
            if p_val and p_val in target_pages and p_val not in assigned_pages:
                assigned_pages.add(p_val)
            else:
                calls_to_reassign.append(c)

        for req_p in target_pages:
            if req_p not in assigned_pages and calls_to_reassign:
                call_to_fix = calls_to_reassign.pop(0)
                call_to_fix.arguments["page_filter"] = req_p
                if target_dt and "issue_date" not in call_to_fix.arguments:
                    call_to_fix.arguments["issue_date"] = target_dt
                if req_np and "newspaper_name" not in call_to_fix.arguments:
                    call_to_fix.arguments["newspaper_name"] = req_np
                call_to_fix.purpose = f"Retrieve manifest for Page {req_p}"
                assigned_pages.add(req_p)

        for req_p in target_pages:
            if req_p not in assigned_pages:
                tool_calls.append(
                    build_sql_summary_tool(
                        newspaper_name=req_np,
                        issue_date=target_dt,
                        page_filter=req_p,
                        query=query,
                        purpose=f"Retrieve manifest for Page {req_p}",
                    )
                )
                assigned_pages.add(req_p)

        for c in tool_calls:
            if c.tool_name == "hybrid_search" and "page_filter" in c.arguments:
                c.arguments.pop("page_filter", None)

    vol_q = (not is_comparative) and is_volume_query(query)
    if (archetype == "quantitative_trend" or vol_q) and archetype != "cross_newspaper_comparison":
        sql_found = False
        for c in tool_calls:
            if c.tool_name == "sql_analytics":
                sql_found = True
                if vol_q and c.arguments.get("analysis_type") in ("issue_summary", None, ""):
                    c.arguments["analysis_type"] = "count_issues"
                    if not extracted.get("issue_date"):
                        c.arguments.pop("issue_date", None)
                        c.arguments.pop("date_from", None)
                        c.arguments.pop("date_to", None)
        if not sql_found:
            named_brand = extracted.get("newspaper_name")
            analysis_t = "count_articles" if "article" in q_lower and not vol_q else "count_issues"
            iss_dt = extracted.get("issue_date") if not vol_q else None
            tool_calls.insert(
                0,
                build_sql_summary_tool(
                    analysis_type=analysis_t,
                    newspaper_name=named_brand,
                    issue_date=iss_dt,
                    date_from=extracted.get("date_from") if not vol_q else None,
                    date_to=extracted.get("date_to") if not vol_q else None,
                    query=query,
                    purpose=f"Count verified {analysis_t.split('_')[-1]} in archive",
                ),
            )

    for c in tool_calls:
        if c.tool_name == "hybrid_search" and c.arguments.get("query"):
            q_val = str(c.arguments["query"]).strip()
            cleaned_q = re.sub(
                r"^(?:please\s+)?(?:compare|contrast|difference\s+between|list|show\s+me)\s+(?:all\s+)?(?:the\s+)?(?:available\s+)?(?:newspapers?|broadsheets?|dailies|papers?)\s+(?:available\s+)?(?:on|dated|for)?\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})?\s*(?:regarding|about|on)?\s*",
                "",
                q_val,
                flags=re.I,
            ).strip()
            c.arguments["query"] = cleaned_q if cleaned_q and len(cleaned_q) >= 3 else "top lead stories front page headlines news"

def _apply_post_plan_guardrails(
    tool_calls: list[PlannedToolCall],
    query: str,
    extracted: dict[str, Any],
    enable_web_search: bool = False,
    active_issue_date: str | None = None,
    active_newspapers: list[str] | None = None,
    attached_article_id: int | None = None,
    attached_photo_id: int | None = None,
) -> list[PlannedToolCall]:
    """Enforce post-planning tool guardrails, entity/coverage fallbacks, and deduplication."""
    q_lower = query.lower()
    if not is_visual_query(query, attached_photo_id):
        tool_calls = [t for t in tool_calls if t.tool_name != "inspect_visual_asset"]
    if not is_dynamic_analysis_permitted(query):
        tool_calls = [t for t in tool_calls if t.tool_name != "dynamic_analysis"]

    is_entity_query = bool(
        re.search(
            r"\b(?:mentions?\s+of|coverage\s+regarding|news\s+regarding|involvement\s+of|role\s+of|profile\s+of|profile\s+the\s+coverage\s+of|everything\s+about|all\s+mentions\s+of|tracked\s+entity)\b",
            q_lower,
        )
    )
    if is_entity_query and not any(t.tool_name == "entity_search" for t in tool_calls):
        clean_ent = _extract_clean_entity(query)
        ent_np = (active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name")
        if clean_ent:
            tool_calls.insert(
                0,
                build_entity_search_tool(
                    entity_name=clean_ent,
                    top_k=10,
                    newspaper_name=ent_np,
                    purpose=f"Profile entity '{clean_ent}'",
                ),
            )

    is_coverage_audit = bool(
        re.search(
            r"\b(?:did\s+.+\s+(?:report\s+on|cover|publish|carry)|was\s+.+\s+(?:reported|covered|published|carried)\s+(?:in|by)|(?:coverage|reporting)\s+(?:of|on)\s+.+\s+(?:in|by)|check\s+(?:if|whether)\s+.+\s+(?:reported|covered))\b",
            q_lower,
        )
    )
    if is_coverage_audit and not any(t.tool_name == "coverage_analysis" for t in tool_calls):
        cov_np = (active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name")
        cov_dt = active_issue_date or extracted.get("issue_date") or extracted.get("date_from")
        topic_clean = re.sub(r"(?i)\b(?:did|was|were|is|are|has|have)\s+(?:the\s+)?[\w\s]+\s+(?:report\s+on|cover|publish|carry|mention)\s*", "", query)
        topic_clean = re.sub(r"(?i)\b(?:on|dated?)\s+\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", topic_clean).strip("?:!.,\"' ") or query
        tool_calls.insert(
            0,
            build_coverage_analysis_tool(
                query=topic_clean,
                target_date=cov_dt,
                newspaper_name=cov_np,
                purpose=f"Audit coverage presence for {cov_np or 'archive'}",
            ),
        )

    if not any(t.tool_name in ("hybrid_search", "sql_analytics", "dynamic_analysis", "timeline", "entity_search", "coverage_analysis") for t in tool_calls):
        tool_calls.append(
            build_hybrid_search_tool(
                query=query,
                newspaper_name=(active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name"),
                date_from=active_issue_date or extracted.get("issue_date"),
                date_to=active_issue_date or extracted.get("issue_date"),
                top_k=8,
                purpose=f"Search broadsheet reporting on {query[:50]}",
            )
        )

    if attached_photo_id and not any(t.tool_name == "inspect_visual_asset" for t in tool_calls):
        np_target = (active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name") or ""
        dt_target = active_issue_date or extracted.get("issue_date") or ""
        tool_calls.insert(
            0,
            build_inspect_visual_asset_tool(
                photo_id=attached_photo_id,
                article_id=attached_article_id,
                query=query,
                newspaper_name=np_target,
                issue_date=dt_target,
                purpose="Inspect attached visual asset and transcribe metadata",
            ),
        )

    if enable_web_search and not any(t.tool_name == "web_search" for t in tool_calls):
        tool_calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))

    return _deduplicate_tool_calls(tool_calls)


__all__ = [
    "_apply_post_plan_guardrails",
    "_build_comparative_tool_calls",
    "_deduplicate_tool_calls",
    "_enforce_direct_tool_completeness",
    "_extract_clean_entity",
]
