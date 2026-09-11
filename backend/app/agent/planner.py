"""Query Planner: True Agentic Direct Tool Planning and High-Cohesion Orchestration.

Coordinates LLM-driven structured tool sequence planning with single-pass deterministic
heuristic fallback, parameter reconciliation, and seamless multi-provider failover.
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    _SECTION_PATTERNS,
    _build_targeted_web_query,
    build_targeted_web_query,
    extract_parameters_from_query,
)
from app.agent.models import (
    AgentPlan,
    ExtractedToolArguments,
    PlannedToolCall,
    PlanResult,
    QueryArchetype,
    QueryPlan,
    ToolCallSpec,
    ToolName,
)
from app.agent.tool_factory import (
    build_coverage_analysis_tool,
    build_entity_search_tool,
    build_hybrid_search_tool,
    build_inspect_visual_asset_tool,
    build_sql_coverage_comparison_tool,
    build_sql_difference_tool,
    build_sql_summary_tool,
    build_timeline_tool,
    build_web_search_tool,
    reconcile_and_sanitize_arguments,
)
from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Planner System Prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are the expert Query Planner for NewsLens-AI, an agentic intelligence system over broadsheet newspapers.
Analyze the user's query, understand their underlying intent, produce step-by-step reasoning, and directly schedule the optimal ordered sequence of 1 to 3 tool calls.

### 🛠️ AVAILABLE RETRIEVAL TOOLS
1. `sql_analytics`: Relational system of record.
   - Arguments: {"analysis_type": "issue_summary" | "count_articles" | "coverage_difference", "newspaper_name": str, "issue_date": "YYYY-MM-DD", "category_filter": str, "page_filter": str, "query": str}
   - Use for: Catalogs, section manifests, whole issue overviews, article counts, or cross-newspaper article differences.
2. `hybrid_search`: Dense vector + BM25 keyword search for factual answers, quotes, and specific events.
   - Arguments: {"query": str, "newspaper_name": str, "date_from": str, "date_to": str, "page_filter": str, "category_filter": str, "top_k": int}
   - Use for: Point-in-time facts, quotes, event details, or targeted content.
3. `timeline_builder`: Chronological evolution and milestone articles.
   - Arguments: {"query": str, "limit": int}
   - Use for: Evolution over time, trajectories, and multi-date developments.
4. `entity_search`: Multi-hop entity network search and profiles.
   - Arguments: {"entity_name": str, "top_k": int}
   - Use for: Deep profiles of specific people or corporations.
5. `coverage_analysis`: Unreported news and negative coverage audit.
   - Arguments: {"query": str, "target_date": str}
   - Use for: Identifying what a newspaper omitted or missed across the archive.
6. `web_search`: Live internet search.
   - Arguments: {"query": str, "num_results": int}
   - Use for: Real-time current events outside the archive.
7. `inspect_visual_asset`: Deep multimodal visual inspection, numerical table extraction, and chart axis reading from broadsheet visual crops.
   - Arguments: {"photo_id": int, "article_id": int, "query": str, "newspaper_name": str, "issue_date": str, "page_filter": str}
   - Use for: Extracting specific numbers, data tables, infographic graphics, charts, and captions from an attached, cited, or inquired broadsheet visual asset, or checking if an article/page has infographics or graphs.

### 📚 FEW-SHOT EXAMPLES
Query: "What happened to Tata Power on page 3?"
Output: {"thought_process": "Factual question about Tata Power on page 3. Direct hybrid search bounded to page 3.", "archetype": "factual_lookup", "tool_calls": [{"tool_name": "hybrid_search", "arguments": {"query": "Tata Power", "page_filter": "3", "top_k": 6}, "purpose": "Search page 3 for Tata Power reporting"}]}

Query: "What are the exact figures and routes shown in this infographic?" (Attached asset photo_id: 5382)
Output: {"thought_process": "User is asking about specific figures and content in an attached infographic. Schedule inspect_visual_asset.", "archetype": "factual_lookup", "tool_calls": [{"tool_name": "inspect_visual_asset", "arguments": {"photo_id": 5382, "query": "transport routes and figures"}, "purpose": "Transcribe and analyze infographic visual crop"}]}

Query: "Does the Brics article on page 4 have any infographics or graphs with it?"
Output: {"thought_process": "Inquiring about visual assets, charts, or infographics attached to an article on page 4.", "archetype": "factual_lookup", "tool_calls": [{"tool_name": "inspect_visual_asset", "arguments": {"query": "Brics multipolarity Global South agenda", "page_filter": "4"}, "purpose": "Inspect visual charts and infographics on page 4"}, {"tool_name": "hybrid_search", "arguments": {"query": "Brics multipolarity Global South agenda", "page_filter": "4", "top_k": 4}, "purpose": "Retrieve article textual context"}]}

Query: "List all health news in The Goan on 2026-08-01"
Output: {"thought_process": "Relational catalog query for health articles in The Goan.", "archetype": "article_catalog", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"newspaper_name": "The Goan", "issue_date": "2026-08-01", "category_filter": "Health", "analysis_type": "issue_summary"}, "purpose": "Retrieve complete manifest of Health articles"}]}

Query: "Compare all available newspapers dated 1/8/2026 on health related news"
Output: {"thought_process": "Cross-newspaper domain comparison on health. First fetch SQL manifest for all newspapers on that date, then retrieve comparative excerpts.", "archetype": "cross_newspaper_comparison", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"issue_date": "2026-08-01", "category_filter": "Health", "analysis_type": "issue_summary", "query": "health related news"}, "purpose": "Fetch complete manifest of health articles across all newspapers"}, {"tool_name": "hybrid_search", "arguments": {"query": "health related news", "category_filter": "Health", "date_from": "2026-08-01", "date_to": "2026-08-01", "top_k": 12}, "purpose": "Retrieve comparative excerpts across broadsheet editions"}]}

### ⚡ REASONING & OUTPUT INSTRUCTIONS
- Keep internal chain-of-thought concise (<80 words).
- CRITICAL DATE RESTRAINT: NEVER invent or hallucinate date ranges (e.g. "2020-01-01" to "2022-12-31") or historical years when the user query does NOT specify any dates! If the query contains no dates, leave `date_from`, `date_to`, `issue_date`, and `target_date` empty or omitted so the retrieval tools search across the entire broadsheet archive.
- ARCHETYPE SELECTION:
  * For queries citing specific statements, article quotes, headlines, or factual claims without explicit multi-newspaper comparative keywords, choose `factual_lookup` and schedule targeted `hybrid_search`.
  * Only select `cross_newspaper_comparison` when the user explicitly asks to compare across publications (e.g. "compare newspapers", "across editions", "coverage differences").
- You MUST respond with a valid JSON object matching the required schema. Return only the JSON object, with no markdown fences or conversational text.
"""


# ---------------------------------------------------------------------------
# Backward-Compatible Helper
# ---------------------------------------------------------------------------

def resolve_tool_sequence(archetype: str, query: str, **kwargs: Any) -> list[PlannedToolCall]:
    """Backward-compatible helper delegating to QueryPlanner deterministic heuristic planner."""
    planner = QueryPlanner()
    res = planner._plan_query_heuristic(query)
    return res.tool_calls


# ---------------------------------------------------------------------------
# QueryPlanner Class
# ---------------------------------------------------------------------------

class QueryPlanner:
    """Agentic Query Planner with True Direct Tool Calling and Lean Deterministic Fallback."""

    def __init__(self, provider: ChatModelProvider | None = None) -> None:
        self._provider = provider

    def _get_provider_candidates(self, model_override: str | None = None) -> list[ChatModelProvider]:
        """Resolve LLM provider candidates for planning failover."""
        if self._provider is not None and not model_override:
            return [self._provider]

        candidates: list[ChatModelProvider] = []
        seen_keys: set[str] = set()

        try:
            reg = get_registry()
            primary = reg.get_chat_provider(model_override) if model_override else reg.get_provider("query_planner")
            if isinstance(primary, ChatModelProvider):
                candidates.append(primary)
                seen_keys.add(f"{getattr(primary, 'provider_name', '')}:{getattr(primary, '_model', '')}")
        except Exception as e:
            logger.warning("Could not resolve primary provider for QueryPlanner", extra={"error": str(e)})

        failover_keys = [
            "nvidia_nemotron",
            "openrouter_nemotron",
            "openrouter_gemma4_26b",
            "gemini_flash",
            "groq_compound",
            "openai_gpt4o_mini",
            "ollama_llama3",
            "ollama_deepseek",
        ]
        try:
            reg = get_registry()
            for k in failover_keys:
                with contextlib.suppress(Exception):
                    p = reg.get_chat_provider(k)
                    if p:
                        ident = f"{getattr(p, 'provider_name', '')}:{getattr(p, '_model', '')}"
                        if ident not in seen_keys:
                            seen_keys.add(ident)
                            candidates.append(p)
        except Exception:
            pass

        return candidates

    async def plan_query_async(
        self,
        query: str,
        enable_web_search: bool = False,
        model_override: str | None = None,
        archive_context: str | None = None,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
    ) -> PlanResult:
        """Plan query using true direct agentic tool calling with graceful heuristic fallback."""
        providers = self._get_provider_candidates(model_override)
        for provider in providers:
            try:
                user_content = f"Analyze and plan the following broadsheet research query:\nQuery: \"{query}\"\n"
                if archive_context:
                    user_content += f"\nACTIVE ARCHIVE STATE:\n{archive_context}\n"
                if active_issue_date or active_newspapers:
                    user_content += f"\nCONVERSATION WORKING CONTEXT:\nActive Date: {active_issue_date or 'None'}\nActive Newspapers: {', '.join(active_newspapers or []) or 'None'}\n"
                if attached_photo_id or attached_article_id:
                    user_content += f"\nATTACHED VISUAL ASSET CONTEXT:\nAttached Photo ID: {attached_photo_id or 'None'}\nAttached Article ID: {attached_article_id or 'None'}\n"
                user_content += "\nSchedule the exact tool calls needed to gather evidence for this query."

                messages = [
                    Message(role="system", content=PLANNER_SYSTEM_PROMPT),
                    Message(role="user", content=user_content),
                ]
                resp = await provider.complete(
                    messages=messages,
                    response_schema=AgentPlan.model_json_schema(),
                    temperature=0.0,
                    max_tokens=2048,
                )

                parsed_dict = resp.parsed if isinstance(resp.parsed, dict) else self._parse_json_plan(resp.text)
                if parsed_dict:
                    plan_obj = AgentPlan.model_validate(parsed_dict)
                    return self._build_plan_from_structured_model(
                        query=query,
                        plan_obj=plan_obj,
                        enable_web_search=enable_web_search,
                        active_issue_date=active_issue_date,
                        active_newspapers=active_newspapers,
                        attached_article_id=attached_article_id,
                        attached_photo_id=attached_photo_id,
                    )
                logger.warning(
                    "Provider returned unparseable plan output, attempting failover candidate",
                    extra={"provider": getattr(provider, "provider_name", ""), "raw": (resp.text or "")[:150]},
                )
            except Exception as ex:
                logger.warning(
                    "LLM Agentic Planning attempt failed on provider, trying failover candidate",
                    extra={"query": query[:50], "provider": getattr(provider, "provider_name", ""), "error": str(ex)},
                )

        # Fallback to lean deterministic router
        return self._plan_query_heuristic(
            query,
            enable_web_search=enable_web_search,
            active_issue_date=active_issue_date,
            active_newspapers=active_newspapers,
            attached_article_id=attached_article_id,
            attached_photo_id=attached_photo_id,
        )

    def plan_query(
        self,
        query: str,
        enable_web_search: bool = False,
        archive_context: str | None = None,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
    ) -> PlanResult:
        """Synchronous planning interface."""
        return self._plan_query_heuristic(
            query,
            enable_web_search=enable_web_search,
            active_issue_date=active_issue_date,
            active_newspapers=active_newspapers,
            attached_article_id=attached_article_id,
            attached_photo_id=attached_photo_id,
        )

    def classify_archetype(self, query: str) -> tuple[str, str]:
        """Classify query archetype synchronously."""
        res = self._plan_query_heuristic(query)
        return res.archetype, res.reasoning

    @staticmethod
    def _parse_json_plan(text: str) -> dict[str, Any] | None:
        """Extract and parse JSON from text, stripping reasoning tags and markdown fences."""
        if not text:
            return None
        cleaned = re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^<(thought|think)>.*?</\1>", "", cleaned, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = text.strip()

        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

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

    def _build_plan_from_structured_model(
        self,
        query: str,
        plan_obj: AgentPlan,
        enable_web_search: bool = False,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
    ) -> PlanResult:
        """Translate AgentPlan into PlanResult, executing direct tool calls or legacy adapter."""
        extracted = extract_parameters_from_query(query)
        if attached_article_id is not None:
            extracted["attached_article_id"] = attached_article_id
        if attached_photo_id is not None:
            extracted["attached_photo_id"] = attached_photo_id
        tool_calls: list[PlannedToolCall] = []

        archetype = plan_obj.archetype
        if archetype == "macro_summary":
            archetype = "quantitative_trend"
        elif archetype == "negative_coverage_audit":
            archetype = "cross_newspaper_comparison"

        # 1. Direct tool calling (Option 2)
        if plan_obj.tool_calls:
            for spec in plan_obj.tool_calls:
                args = reconcile_and_sanitize_arguments(
                    spec.tool_name,
                    spec.arguments,
                    extracted,
                    query,
                    active_issue_date=active_issue_date,
                    active_newspapers=active_newspapers,
                )
                tool_calls.append(PlannedToolCall(tool_name=spec.tool_name, arguments=args, purpose=spec.purpose))

        # 2. Legacy adapter: if mock/LLM provided primary_tool or arguments without tool_calls
        elif plan_obj.primary_tool or plan_obj.arguments:
            raw_args: dict[str, Any] = {}
            if isinstance(plan_obj.arguments, dict):
                raw_args = dict(plan_obj.arguments)
            elif hasattr(plan_obj.arguments, "model_dump"):
                raw_args = plan_obj.arguments.model_dump(exclude_none=True)

            primary_tool_name = plan_obj.primary_tool or "hybrid_search"
            args = reconcile_and_sanitize_arguments(
                primary_tool_name,
                raw_args,
                extracted,
                query,
                active_issue_date=active_issue_date,
                active_newspapers=active_newspapers,
            )

            if archetype == "cross_newspaper_comparison":
                target_dt = args.get("issue_date") or extracted.get("issue_date")
                comp_np = args.get("comparison_newspaper") or extracted.get("comparison_newspaper")
                src_np = args.get("newspaper_name") or extracted.get("newspaper_name")
                is_diff = (
                    extracted.get("is_differential", False)
                    or plan_obj.archetype == "negative_coverage_audit"
                    or args.get("analysis_type") == "coverage_difference"
                )

                if src_np and comp_np and is_diff:
                    tool_calls.append(build_sql_difference_tool(src_np, comp_np, args.get("query", query), target_dt))
                    tool_calls.append(build_hybrid_search_tool(args.get("query", query), newspaper_name=src_np, top_k=10, purpose=f"Articles from {src_np}"))
                elif plan_obj.primary_tool == "coverage_analysis" and not src_np and not args.get("category_filter") and not target_dt:
                    tool_calls.append(build_coverage_analysis_tool(args.get("query", query)))
                else:
                    tool_calls.append(build_sql_summary_tool(
                        analysis_type="issue_summary",
                        newspaper_name=src_np,
                        issue_date=target_dt,
                        category_filter=args.get("category_filter"),
                        query=args.get("query", query),
                        purpose="SQL article manifest",
                    ))
                    tool_calls.append(build_hybrid_search_tool(
                        query=args.get("query", query),
                        newspaper_name=src_np,
                        date_from=target_dt,
                        date_to=target_dt,
                        category_filter=args.get("category_filter"),
                        top_k=12,
                        purpose="Comparative article excerpts",
                    ))
                    if not src_np and target_dt and not args.get("category_filter"):
                        tool_calls.append(build_coverage_analysis_tool(args.get("query", query), target_date=target_dt))
            else:
                primary = plan_obj.primary_tool or ("sql_analytics" if archetype in ("quantitative_trend", "article_catalog") else "hybrid_search")
                tool_calls.append(PlannedToolCall(tool_name=primary, arguments=args, purpose=f"Execute {primary}"))
                if getattr(plan_obj, "include_secondary_hybrid_search", False) and primary != "hybrid_search":
                    tool_calls.append(build_hybrid_search_tool(getattr(plan_obj, "secondary_search_query", None) or query, top_k=6, purpose="Corroborating search"))

        # 3. Fallback if no tool calls produced
        if not tool_calls:
            heur = self._plan_query_heuristic(
                query,
                enable_web_search=enable_web_search,
                active_issue_date=active_issue_date,
                active_newspapers=active_newspapers,
                attached_article_id=attached_article_id,
                attached_photo_id=attached_photo_id,
            )
            tool_calls = heur.tool_calls

        if attached_photo_id and not any(t.tool_name == "inspect_visual_asset" for t in tool_calls):
            np_target = (active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name") or ""
            dt_target = active_issue_date or extracted.get("issue_date") or ""
            tool_calls.insert(0, build_inspect_visual_asset_tool(
                photo_id=attached_photo_id,
                article_id=attached_article_id,
                query=query,
                newspaper_name=np_target,
                issue_date=dt_target,
                purpose="Inspect attached visual asset and transcribe metadata",
            ))

        if enable_web_search and not any(t.tool_name == "web_search" for t in tool_calls):
            tool_calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))

        return PlanResult(archetype=archetype, reasoning=plan_obj.thought_process or f"Structured agentic plan ({archetype})", tool_calls=tool_calls)

    def _plan_query_heuristic(
        self,
        query: str,
        enable_web_search: bool = False,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
    ) -> PlanResult:
        """Clean, deterministic single-pass intent routing for offline fallback."""
        q_lower = query.lower().strip()
        params = extract_parameters_from_query(query)

        newspaper = params.get("newspaper_name")
        issue_id = params.get("issue_id")
        issue_date = params.get("issue_date") or active_issue_date
        date_from = params.get("date_from") or issue_date
        date_to = params.get("date_to") or issue_date
        target_dates = params.get("target_dates") or ([issue_date] if issue_date else [])
        category = params.get("category_filter")
        is_diff = params.get("is_differential", False)
        comp_newspaper = params.get("comparison_newspaper")

        # Page extraction
        page_filter = None
        p_match = re.search(r"\b(?:page|pg|p\.?)\s*(\d{1,3})\b", q_lower)
        if p_match:
            page_filter = p_match.group(1)

        tool_calls: list[PlannedToolCall] = []

        # 0. Visual Asset / Infographic Inspection
        is_visual_query = any(w in q_lower for w in ["infographic", "data chart", "chart", "diagram", "table", "graph", "visual", "figure", "photograph", "photo"])
        has_visual_trigger = any(w in q_lower for w in ["this", "the infographic", "attached", "chart", "diagram", "table", "above", "shown", "it have", "have any", "has any", "with it", "there any"])
        if attached_photo_id or (attached_article_id and is_visual_query) or (is_visual_query and has_visual_trigger):
            archetype = "factual_lookup"
            tool_calls.append(build_inspect_visual_asset_tool(
                photo_id=attached_photo_id,
                article_id=attached_article_id,
                query=query,
                newspaper_name=newspaper or "",
                issue_date=issue_date or "",
                page_filter=page_filter or "",
                purpose="Inspect visual crop and transcribe numerical data table",
            ))
            tool_calls.append(build_hybrid_search_tool(
                query=query,
                newspaper_name=newspaper,
                date_from=date_from,
                date_to=date_to,
                page_filter=page_filter,
                top_k=4,
                purpose="Contextual article evidence",
            ))
            if enable_web_search:
                tool_calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))
            return PlanResult(archetype=archetype, reasoning=f"Deterministic visual inspection plan ({archetype})", tool_calls=tool_calls)

        # 1. Timeline / Chronological Trajectory
        if any(w in q_lower for w in ["timeline", "chronology", "chronological", "evolution", "over time", "history of", "progression"]):
            archetype = "thematic_timeline"
            tool_calls.append(build_timeline_tool(query=query, limit=25))
            tool_calls.append(build_hybrid_search_tool(query=query, top_k=8, purpose="Retrieve anchor articles"))

        # 2. Entity Deep Dive
        elif any(w in q_lower for w in ["everything about", "all mentions of", "profile the coverage", "profile of"]):
            archetype = "entity_deep_dive"
            clean_ent = re.sub(r"(?i)^(?:everything about|all mentions of|profile the coverage of|profile of)\s*", "", query).strip("?:!.,\"' ") or query
            tool_calls.append(build_entity_search_tool(entity_name=clean_ent, top_k=10))
            tool_calls.append(build_hybrid_search_tool(query=query, top_k=8, purpose="Semantic context"))

        # 3. Single Newspaper Multi-Issue Comparison (same newspaper brand across multiple dates)
        elif newspaper and not comp_newspaper and len(target_dates) >= 2:
            archetype = "quantitative_trend"
            for dt_val in target_dates:
                tool_calls.append(build_sql_summary_tool(newspaper_name=newspaper, issue_date=dt_val, query=query, purpose=f"Manifest for {newspaper} on {dt_val}"))
            tool_calls.append(build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, top_k=10, purpose="Multi-issue articles"))

        # 4. Cross-Newspaper Comparison
        elif (
            comp_newspaper is not None
            or is_diff
            or any(w in q_lower for w in ["across newspapers", "across different papers", "all available", "all the available", "all newspapers", "both newspapers", "different papers", "different newspapers"])
            or bool(re.search(r"\b(?:compa[a-z]*|contrast[a-z]*|diff[a-z]*|versus|vs\.?)\b", q_lower))
        ):
            archetype = "cross_newspaper_comparison"
            target_dt = issue_date or date_from
            if newspaper and comp_newspaper and is_diff:
                tool_calls.append(build_sql_difference_tool(newspaper, comp_newspaper, query, target_dt))
                tool_calls.append(build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=target_dt, date_to=target_dt, top_k=10, purpose=f"Retrieve articles from {newspaper}"))
            elif newspaper and comp_newspaper:
                for np_name in [newspaper, comp_newspaper]:
                    tool_calls.append(build_sql_summary_tool(newspaper_name=np_name, issue_date=target_dt, category_filter=category, query=query, purpose=f"Retrieve manifest for {np_name}"))
                tool_calls.append(build_hybrid_search_tool(query=query, date_from=target_dt, date_to=target_dt, category_filter=category, top_k=12, purpose=f"Comparative articles across {newspaper} and {comp_newspaper}"))
            elif not target_dt:
                tool_calls.append(build_hybrid_search_tool(query=query, category_filter=category, top_k=12, purpose="Comparative articles across broadsheet editions"))
                if any(w in q_lower for w in ["omit", "miss", "exclusive", "gap", "audit", "coverage analysis", "unreported"]):
                    tool_calls.append(build_coverage_analysis_tool(query=query, purpose="Archive-wide coverage comparison"))
            else:
                tool_calls.append(build_sql_summary_tool(issue_date=target_dt, category_filter=category, query=query, purpose=f"Manifest across all newspapers on {target_dt}"))
                hs_page = "1" if not newspaper and target_dt and not category else None
                tool_calls.append(build_hybrid_search_tool(query=query, date_from=target_dt, date_to=target_dt, page_filter=hs_page, category_filter=category, top_k=12, purpose="Diverse articles across editions"))

                if not category or any(w in q_lower for w in ["omit", "miss", "exclusive", "gap", "audit"]):
                    tool_calls.append(build_coverage_analysis_tool(query=query, target_date=target_dt))

                if not category and not newspaper:
                    tool_calls.append(build_sql_coverage_comparison_tool(target_date=target_dt, query=query))

        # 5. Quantitative Trend / Article Catalog / Issue Manifest / Page Listings / Counts
        elif (
            (page_filter and any(w in q_lower for w in ["article", "story", "stories", "no of", "how many", "list"]))
            or any(w in q_lower for w in [
                "how many", "count", "number of", "no of", "frequency", "trend", "distribution",
                "volume", "statistics", "summarize", "overview", "whole", "entire", "all articles",
                "list", "manifest", "today's paper", "edition", "what articles", "articles on",
            ])
            or (category and any(w in q_lower for w in ["news", "articles", "stories", "headlines"]))
        ):
            is_count = any(w in q_lower for w in ["how many", "total articles", "number of articles", "count of articles"]) and not page_filter
            is_whole_issue_or_count = (
                bool(page_filter)
                or is_count
                or any(w in q_lower for w in [
                    "today's paper", "edition", "whole", "entire", "overview", "summarize",
                    "how many", "count of", "number of", "no of", "distribution", "frequency", "trend", "statistics", "volume",
                ])
            )
            is_catalog_request = any(w in q_lower for w in ["list", "catalog", "manifest", "all articles", "all news", "all stories"]) or bool(category)

            if is_catalog_request and not is_whole_issue_or_count:
                archetype = "article_catalog"
            else:
                archetype = "quantitative_trend"

            analysis_type = "count_articles" if is_count else "issue_summary"

            tool_calls.append(build_sql_summary_tool(
                analysis_type=analysis_type,
                newspaper_name=newspaper,
                issue_date=issue_date,
                date_from=date_from,
                date_to=date_to,
                issue_id=issue_id,
                page_filter=page_filter,
                category_filter=category,
                query=query,
            ))
            if page_filter and any(w in q_lower for w in ["list", "articles on", "stories on", "all articles"]):
                tool_calls.append(build_hybrid_search_tool(
                    query=query,
                    newspaper_name=newspaper,
                    date_from=date_from,
                    date_to=date_to,
                    page_filter=page_filter,
                    top_k=6,
                    purpose=f"Articles on Page {page_filter}",
                ))

        # 6. Factual Lookup (Default)
        else:
            archetype = "factual_lookup"
            tool_calls.append(build_hybrid_search_tool(
                query=query,
                newspaper_name=newspaper,
                date_from=date_from,
                date_to=date_to,
                page_filter=page_filter,
                top_k=6,
                purpose="Search for factual evidence",
            ))

        if enable_web_search:
            tool_calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))

        return PlanResult(archetype=archetype, reasoning=f"Deterministic heuristic plan ({archetype})", tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Public Re-Exports for 100% Backward Compatibility
# ---------------------------------------------------------------------------

__all__ = [
    "ToolName",
    "QueryArchetype",
    "PlannedToolCall",
    "PlanResult",
    "ToolCallSpec",
    "AgentPlan",
    "QueryPlan",
    "ExtractedToolArguments",
    "QueryPlanner",
    "extract_parameters_from_query",
    "_KNOWN_BRANDS_PATTERNS",
    "_SECTION_PATTERNS",
    "build_targeted_web_query",
    "_build_targeted_web_query",
    "resolve_tool_sequence",
]
