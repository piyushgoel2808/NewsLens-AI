"""Query Planner: True Agentic Direct Tool Planning and High-Cohesion Orchestration.

Coordinates LLM-driven structured tool sequence planning with pre-planning ToolDecisionPolicy,
single-pass deterministic heuristic fallback, parameter reconciliation, and multi-provider failover.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.archive_context import (
    get_known_publications,
)
from app.agent.blueprints import (
    DEFAULT_BLUEPRINTS,
    build_heuristic_answer_blueprint,
)
from app.agent.decision_layer import (
    MATH_AGGREGATE_PATTERNS,
    ToolDecisionPolicy,
    is_availability_query,
    is_comparative_query,
    is_dynamic_analysis_permitted,
    is_macro_edition_query,
    is_ordinal_lead_story_query,
    is_visual_query,
    is_volume_query,
)
from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    _SECTION_PATTERNS,
    _build_targeted_web_query,
    build_targeted_web_query,
    extract_parameters_from_query,
    is_archive_wide_newspaper_query,
)
from app.agent.llm_helpers import (
    extract_json_object,
    resolve_provider_candidates,
)
from app.agent.models import (
    AgentPlan,
    AnswerBlueprint,
    ExtractedToolArguments,
    PlannedToolCall,
    PlanResult,
    QueryArchetype,
    QueryPlan,
    SectionSpec,
    ToolCallSpec,
    ToolDecision,
    ToolName,
)
from app.agent.plan_enforcer import (
    _apply_post_plan_guardrails,
    _build_comparative_tool_calls,
    _deduplicate_tool_calls,
    _enforce_direct_tool_completeness,
    _extract_clean_entity,
)
from app.agent.planner_prompts import (
    _PROMPT_BODY,
    _PROMPT_PREAMBLE,
    PLANNER_SYSTEM_PROMPT,
    build_replanner_prompt,
)
from app.agent.state import ToolExecutionRecord
from app.agent.tool_factory import (
    build_coverage_analysis_tool,
    build_dynamic_analysis_tool,
    build_entity_search_tool,
    build_hybrid_search_tool,
    build_inspect_visual_asset_tool,
    build_sql_summary_tool,
    build_timeline_tool,
    build_web_search_tool,
    reconcile_and_sanitize_arguments,
)
from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry

logger = get_logger(__name__)


def resolve_tool_sequence(archetype: str, query: str, **kwargs: Any) -> list[PlannedToolCall]:
    """Compatibility wrapper: resolve tool sequence using QueryPlanner heuristic."""
    planner = QueryPlanner()
    res = planner._plan_query_heuristic(
        query,
        enable_web_search=kwargs.get("enable_web_search", False),
        active_issue_date=kwargs.get("active_issue_date"),
        active_newspapers=kwargs.get("active_newspapers"),
    )
    return res.tool_calls


class QueryPlanner:
    """Agentic Query Planner with True Direct Tool Calling and Lean Deterministic Fallback."""

    _build_heuristic_answer_blueprint = staticmethod(build_heuristic_answer_blueprint)

    def __init__(self, provider: ChatModelProvider | None = None) -> None:
        self._provider = provider

    def _get_provider_candidates(self, model_override: str | None = None) -> list[ChatModelProvider]:
        """Resolve LLM provider candidates for planning failover via shared llm_helpers."""
        return resolve_provider_candidates(
            task_name="query_planner",
            model_override=model_override,
            injected_provider=self._provider,
            prefer_local=False,
            registry_getter=get_registry,
        )

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
        chat_history: list[dict[str, Any]] | None = None,
        extracted_params: dict[str, Any] | None = None,
    ) -> PlanResult:
        """Plan query using pre-LLM ToolDecisionPolicy + direct agentic tool calling with heuristic fallback."""
        extracted = extracted_params if extracted_params is not None else extract_parameters_from_query(query)
        pre_decision = ToolDecisionPolicy.decide(
            query=query,
            extracted=extracted,
            active_issue_date=active_issue_date,
            active_newspapers=active_newspapers,
            attached_article_id=attached_article_id,
            attached_photo_id=attached_photo_id,
            enable_web_search=enable_web_search,
        )
        if pre_decision.intent == "conversational_greeting":
            return PlanResult(
                archetype="conversational_meta_query",
                reasoning="Conversational greeting. No retrieval tools required.",
                tool_calls=[],
                answer_blueprint=AnswerBlueprint(
                    user_intent="greeting",
                    overall_tone="conversational",
                    sections=[],
                ),
                decision=pre_decision,
            )
        for provider in self._get_provider_candidates(model_override):
            try:
                vis_query = is_visual_query(query, attached_photo_id)
                user_content = (
                    f"Analyze and plan the following broadsheet research query:\n"
                    f"Query: \"{query}\"\n"
                    f"Decision Policy Hint: Intent={pre_decision.intent}, "
                    f"Archetype={pre_decision.archetype}, "
                    f"CandidateTools={pre_decision.selected_tools}"
                    + (f", AnalysisType={pre_decision.recommended_analysis_type}\n" if pre_decision.recommended_analysis_type else "\n")
                )
                if archive_context:
                    user_content += f"\nACTIVE ARCHIVE STATE:\n{archive_context}\n"
                if chat_history:
                    hist_lines = []
                    for turn in chat_history[-4:]:
                        role = str(turn.get("role", "user")).capitalize()
                        content = str(turn.get("content", "")).strip()
                        if len(content) > 300:
                            content = content[:300] + "..."
                        if content:
                            hist_lines.append(f"{role}: {content}")
                    if hist_lines:
                        user_content += "\nRECENT CONVERSATION HISTORY:\n" + "\n".join(hist_lines) + "\n"
                if active_issue_date or active_newspapers or attached_article_id:
                    ctx_lines = []
                    if active_issue_date:
                        ctx_lines.append(f"Active Date: {active_issue_date}")
                    if active_newspapers:
                        ctx_lines.append(f"Active Newspapers: {', '.join(active_newspapers)}")
                    if attached_article_id:
                        ctx_lines.append(f"Referenced Article ID: {attached_article_id}")
                    user_content += "\nCONVERSATION WORKING CONTEXT:\n" + "\n".join(ctx_lines) + "\n"
                if attached_photo_id or (attached_article_id and vis_query):
                    user_content += f"\nATTACHED VISUAL ASSET CONTEXT:\nAttached Photo ID: {attached_photo_id or 'None'}\nAttached Article ID: {attached_article_id or 'None'}\n"
                user_content += "\nSchedule the exact tool calls needed to gather evidence for this query."

                resp = await provider.complete(
                    messages=[
                        Message(role="system", content=PLANNER_SYSTEM_PROMPT),
                        Message(role="user", content=user_content),
                    ],
                    response_schema=AgentPlan.model_json_schema(),
                    temperature=0.0,
                    max_tokens=2048,
                    thinking_budget=0,
                )

                parsed_dict = resp.parsed if isinstance(resp.parsed, dict) else self._parse_json_plan(resp.text)
                if parsed_dict:
                    plan_obj = AgentPlan.model_validate(parsed_dict)
                    plan_res = self._build_plan_from_structured_model(
                        query=query,
                        plan_obj=plan_obj,
                        enable_web_search=enable_web_search,
                        active_issue_date=active_issue_date,
                        active_newspapers=active_newspapers,
                        attached_article_id=attached_article_id,
                        attached_photo_id=attached_photo_id,
                        extracted_params=extracted,
                    )
                    plan_res.decision = pre_decision
                    return plan_res
                logger.warning(
                    "Provider returned unparseable plan output, attempting failover candidate",
                    extra={"provider": getattr(provider, "provider_name", ""), "raw": (resp.text or "")[:150]},
                )
            except Exception as ex:
                logger.warning(
                    "LLM Agentic Planning attempt failed on provider, trying failover candidate",
                    extra={"query": query[:50], "provider": getattr(provider, "provider_name", ""), "error": str(ex)},
                )

        fallback_res = self._plan_query_heuristic(
            query,
            enable_web_search=enable_web_search,
            active_issue_date=active_issue_date,
            active_newspapers=active_newspapers,
            attached_article_id=attached_article_id,
            attached_photo_id=attached_photo_id,
            extracted_params=extracted,
        )
        fallback_res.decision = pre_decision
        return fallback_res

    def plan_query(
        self,
        query: str,
        enable_web_search: bool = False,
        archive_context: str | None = None,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
        chat_history: list[dict[str, Any]] | None = None,
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
        return extract_json_object(text)

    def _build_plan_from_structured_model(
        self,
        query: str,
        plan_obj: AgentPlan,
        enable_web_search: bool = False,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
        extracted_params: dict[str, Any] | None = None,
    ) -> PlanResult:
        """Translate AgentPlan into PlanResult, executing direct tool calls or legacy adapter."""
        extracted = dict(extracted_params) if extracted_params is not None else extract_parameters_from_query(query)
        if attached_article_id is not None:
            extracted["attached_article_id"] = attached_article_id
        if attached_photo_id is not None:
            extracted["attached_photo_id"] = attached_photo_id
        tool_calls: list[PlannedToolCall] = []

        archetype = plan_obj.archetype
        if archetype == "macro_summary" and not is_macro_edition_query(query):
            archetype = "quantitative_trend"
        elif archetype == "negative_coverage_audit":
            archetype = "cross_newspaper_comparison"

        q_lower = query.lower()
        target_nps = extracted.get("target_newspapers", [])
        is_comparative = (
            archetype == "cross_newspaper_comparison"
            or plan_obj.archetype in ("cross_newspaper_comparison", "negative_coverage_audit")
            or is_comparative_query(query, target_nps)
            or any(w in q_lower for w in ("between", "audit negative", "negative coverage"))
        )
        if is_comparative:
            archetype = "cross_newspaper_comparison"
        elif (
            bool(re.search(r"\b(how\s+many|count\s+of|number\s+of|total\s+number\s+of|no\s+of)\s+.*(?:newspapers?|publications?|dailies|issues?|articles?|pages?|ads?|advertisements?)\b", q_lower))
            or is_archive_wide_newspaper_query(query)
        ) and archetype not in ("quantitative_trend", "analytical_computation") and not any(
            w in q_lower for w in ("list", "catalog", "headlines", "headings", "what articles")
        ):
            archetype = "quantitative_trend"

        # 1. Direct tool calling
        if plan_obj.tool_calls:
            for spec in plan_obj.tool_calls:
                args = reconcile_and_sanitize_arguments(
                    spec.tool_name,
                    spec.arguments,
                    extracted,
                    query,
                    active_issue_date=active_issue_date,
                    active_newspapers=active_newspapers,
                    purpose=spec.purpose,
                )
                tool_calls.append(PlannedToolCall(tool_name=spec.tool_name, arguments=args, purpose=spec.purpose))

            self._enforce_direct_tool_completeness(
                tool_calls=tool_calls,
                query=query,
                archetype=archetype,
                extracted=extracted,
                is_comparative=is_comparative,
                active_issue_date=active_issue_date,
                active_newspapers=active_newspapers,
            )

        # 2. Legacy adapter: if mock/legacy caller provided primary_tool or arguments without tool_calls
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
                if src_np and not comp_np and primary_tool_name == "coverage_analysis":
                    cov_target_dt = target_dt or args.get("target_date")
                    tool_calls.append(build_coverage_analysis_tool(args.get("query", query), target_date=cov_target_dt, newspaper_name=src_np))
                else:
                    is_diff = bool(
                        extracted.get("is_differential", False)
                        or plan_obj.archetype == "negative_coverage_audit"
                        or args.get("analysis_type") == "coverage_difference"
                    )
                    is_shared = bool(
                        extracted.get("is_shared", False)
                        or args.get("analysis_type") in ("shared_coverage", "similar_articles", "common_stories")
                        or any(w in q_lower for w in ("similar", "shared", "common", "same article", "same stories", "both"))
                    )
                    target_pages = [str(p).strip() for p in extracted.get("target_pages", []) if str(p).strip()]
                    tool_calls.extend(
                        _build_comparative_tool_calls(
                            query=args.get("query", query),
                            src_np=src_np,
                            comp_np=comp_np,
                            target_dt=target_dt,
                            target_pages=target_pages,
                            category_filter=args.get("category_filter"),
                            is_diff=is_diff,
                            is_shared=is_shared,
                        )
                    )
            else:
                tool_calls.append(PlannedToolCall(tool_name=primary_tool_name, arguments=args, purpose=f"Execute {primary_tool_name}"))
                if getattr(plan_obj, "include_secondary_hybrid_search", False) and primary_tool_name != "hybrid_search":
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
                extracted_params=extracted,
            )
            tool_calls = heur.tool_calls

        tool_calls = self._apply_post_plan_guardrails(
            tool_calls=tool_calls,
            query=query,
            extracted=extracted,
            enable_web_search=enable_web_search,
            active_issue_date=active_issue_date,
            active_newspapers=active_newspapers,
            attached_article_id=attached_article_id,
            attached_photo_id=attached_photo_id,
        )

        blueprint = getattr(plan_obj, "answer_blueprint", None) or self._build_heuristic_answer_blueprint(query, archetype, extracted)
        return PlanResult(
            archetype=archetype,
            reasoning=plan_obj.thought_process or f"Structured agentic plan ({archetype})",
            tool_calls=tool_calls,
            answer_blueprint=blueprint,
        )

    _enforce_direct_tool_completeness = staticmethod(_enforce_direct_tool_completeness)
    _apply_post_plan_guardrails = staticmethod(_apply_post_plan_guardrails)

    async def replan_with_feedback_async(
        self,
        query: str,
        previous_plan: list[dict[str, Any]] | list[PlannedToolCall],
        tool_executions: list[dict[str, Any]] | list[ToolExecutionRecord],
        gap_diagnosis: str | None = None,
        evidence_summary: str | None = None,
        model_override: str | None = None,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
    ) -> PlanResult:
        """Adaptive closed-loop re-planner: inspects gap diagnosis and previous executions to generate targeted corrective tool calls."""
        q_lower = query.lower()
        extracted = extract_parameters_from_query(query)

        tried_calls: list[tuple[str, dict[str, Any]]] = []
        for t in tool_executions:
            t_name = t.get("tool_name") if isinstance(t, dict) else getattr(t, "tool_name", "")
            t_inp = t.get("tool_input", {}) if isinstance(t, dict) else getattr(t, "tool_input", {})
            if t_name:
                tried_calls.append((t_name, dict(t_inp)))

        prompt = build_replanner_prompt(query=query, tried_calls=tried_calls, gap_diagnosis=gap_diagnosis)
        is_dyn_allowed = is_dynamic_analysis_permitted(query)

        planned_calls: list[PlannedToolCall] = []
        archetype = "factual_lookup"
        reasoning = "Adaptive re-planning"

        for prov in self._get_provider_candidates(model_override):
            try:
                resp = await prov.complete(
                    messages=[
                        Message(role="system", content="You are a precise adaptive query replanner. Respond ONLY in valid JSON."),
                        Message(role="user", content=prompt),
                    ],
                    max_tokens=400,
                    temperature=0.0,
                    thinking_budget=0,
                )
                data = extract_json_object(resp.text or "")
                if data:
                    archetype = data.get("archetype", "factual_lookup")
                    reasoning = data.get("thought_process", "Targeted recovery re-plan")
                    for rc in data.get("tool_calls", []):
                        t_name = rc.get("tool_name", "")
                        if not t_name or (t_name == "dynamic_analysis" and not is_dyn_allowed):
                            continue
                        raw_args = rc.get("arguments", {})
                        purpose = rc.get("purpose") or f"Recovery {t_name}"
                        san_args = reconcile_and_sanitize_arguments(
                            tool_name=t_name,
                            args=raw_args,
                            extracted=extracted,
                            query=query,
                            active_issue_date=active_issue_date,
                            active_newspapers=active_newspapers,
                            purpose=purpose,
                        )
                        is_repeated = any(
                            prev_name == t_name
                            and (
                                all(str(san_args.get(k, "")) == str(prev_args.get(k, "")) for k in ("query", "newspaper_name", "date_from", "date_to", "analysis_type"))
                                or all(str(raw_args.get(k, "")) == str(prev_args.get(k, "")) for k in ("query", "newspaper_name", "date_from", "date_to", "analysis_type"))
                            )
                            for prev_name, prev_args in tried_calls
                        )
                        if is_repeated:
                            if t_name == "hybrid_search":
                                san_args.pop("date_from", None)
                                san_args.pop("date_to", None)
                                san_args["top_k"] = max(8, int(san_args.get("top_k", 6)) + 4)
                            elif t_name == "sql_analytics":
                                san_args.pop("issue_date", None)
                                san_args.pop("date_from", None)
                                san_args.pop("date_to", None)
                        elif t_name == "hybrid_search" and "top_k" not in san_args:
                            san_args["top_k"] = 8

                        planned_calls.append(PlannedToolCall(tool_name=t_name, arguments=san_args, purpose=purpose))

                    if planned_calls:
                        break
            except Exception as e:
                logger.debug("Adaptive replanner candidate failed", extra={"error": str(e)})
                continue

        if not planned_calls:
            diag = (gap_diagnosis or "").lower()
            newspaper = extracted.get("newspaper_name")

            if "missing_newspaper_coverage" in diag or "missing articles from" in diag:
                cov_dt = extracted.get("issue_date") or extracted.get("date_from") or active_issue_date
                cov_pfilt = extracted.get("page_filter")
                for np_name in get_known_publications():
                    if np_name.lower() in diag:
                        if cov_dt:
                            planned_calls.append(
                                build_sql_summary_tool(
                                    newspaper_name=np_name,
                                    issue_date=cov_dt,
                                    page_filter=cov_pfilt,
                                    query=query,
                                    purpose=f"Recover relational manifest for {np_name} on {cov_dt}",
                                )
                            )
                        planned_calls.append(
                            build_hybrid_search_tool(
                                query=query,
                                newspaper_name=np_name,
                                date_from=cov_dt,
                                date_to=cov_dt,
                                page_filter=cov_pfilt,
                                top_k=8,
                                purpose=f"Recover missing coverage from {np_name}",
                            )
                        )
            elif "missing_quantitative_payload" in diag or any(w in q_lower for w in ("advertisement", "ad count", "ads count")):
                is_ad = any(w in q_lower for w in ("advertisement", "ad", "commercial"))
                analysis_type = "count_advertisements" if is_ad else "count_articles"
                planned_calls.append(
                    build_sql_summary_tool(
                        analysis_type=analysis_type,
                        newspaper_name=newspaper,
                        query=query,
                        purpose=f"Recover quantitative aggregate ({analysis_type})",
                    )
                )
            else:
                planned_calls.append(
                    build_hybrid_search_tool(
                        query=query,
                        top_k=8,
                        purpose="Broadened archive retrieval fallback",
                    )
                )

        return PlanResult(
            archetype=archetype,
            reasoning=reasoning,
            tool_calls=_deduplicate_tool_calls(planned_calls),
            answer_blueprint=self._build_heuristic_answer_blueprint(query, archetype, extracted),
        )

    def _plan_query_heuristic(
        self,
        query: str,
        enable_web_search: bool = False,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
        extracted_params: dict[str, Any] | None = None,
    ) -> PlanResult:
        """Clean, deterministic single-pass intent routing for offline fallback."""
        q_lower = query.lower().strip()
        params = dict(extracted_params) if extracted_params is not None else extract_parameters_from_query(query)

        is_archive_np = is_archive_wide_newspaper_query(query) or bool(
            re.search(r"\b(?:no|number|count|how many|all|total|which|list)\s+(?:of\s+)?newspapers?\b", q_lower)
            or any(w in q_lower for w in ("all available", "all newspaper", "both newspaper", "across newspaper"))
        )
        vol_query = is_volume_query(query)
        raw_np = params.get("newspaper_name")
        brand_in_q = False
        if raw_np:
            b_low = raw_np.lower()
            b_clean = re.sub(r"^the\s+", "", b_low).strip()
            brand_in_q = b_low in q_lower or (bool(b_clean) and b_clean in q_lower)
        newspaper = raw_np if raw_np and (not is_archive_np or brand_in_q) else None
        issue_id = params.get("issue_id") if newspaper else None

        has_explicit_range = bool(params.get("date_from") and params.get("date_to"))
        has_explicit_date = bool(params.get("issue_date"))
        if is_archive_np or vol_query:
            issue_date = params.get("issue_date") if has_explicit_date else None
        else:
            issue_date = params.get("issue_date") or (None if has_explicit_range else active_issue_date)
        date_from = params.get("date_from") or issue_date
        date_to = params.get("date_to") or issue_date
        target_dates = params.get("target_dates") or ([issue_date] if issue_date else [])
        category = params.get("category_filter")
        is_diff = params.get("is_differential", False)
        comp_newspaper = params.get("comparison_newspaper")

        target_pages: list[str] = [str(p).strip() for p in params.get("target_pages", []) if str(p).strip()]
        page_filter = params.get("page_filter")
        if not page_filter:
            if re.search(r"\b(?:front[\s-]*page|cover[\s-]*page|page\s*(?:1|one))\b", q_lower):
                page_filter = "1"
            else:
                p_match = re.search(r"\b(?:page|pg|p\.?)\s*(\d{1,3})\b", q_lower)
                page_filter = p_match.group(1) if p_match else None
        if page_filter is not None:
            page_filter = str(page_filter).strip()
        if not target_pages and page_filter:
            target_pages = [page_filter]

        is_target_art = bool(
            re.search(r"[\"“][^\"”]{8,150}[\"”]", query)
            or re.search(r"\b(?:tell me about|explain|summ[ae]ri[sz]e|find|read|what does|describe|details? of|takeaways? from)\s+(?:about|on|for)?\s*(?:this|the|that|an?|ir)?\s*(?:[\w-]+\s+){0,4}(?:article|story|piece|it|report)\b", q_lower)
            or re.search(r"\b(?:in\s+(?:under\s+|less\s+than\s+|at\s+most\s+)?\d+\s+words?|under\s+\d+\s+words?|brief\s+summary|key\s+takeaways?)\b", q_lower)
        ) and not any(w in q_lower for w in ("how many", "count", "list all", "catalog", "compare all")) and not is_macro_edition_query(query)

        # 0. Visual Asset / Infographic Inspection
        is_vis = is_visual_query(query, attached_photo_id)
        has_vis_trigger = any(w in q_lower for w in ("this", "the infographic", "attached", "chart", "diagram", "table", "above", "shown", "it have", "have any", "has any", "with it", "there any", "what does the photo", "show the photo"))
        if attached_photo_id or (is_vis and has_vis_trigger):
            calls = [
                build_inspect_visual_asset_tool(photo_id=attached_photo_id, article_id=attached_article_id, query=query, newspaper_name=newspaper or "", issue_date=issue_date or "", page_filter=page_filter or "", purpose="Inspect visual crop and transcribe numerical data table"),
                build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, page_filter=page_filter, top_k=4, purpose="Contextual article evidence"),
            ]
            if enable_web_search:
                calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))
            return PlanResult(
                archetype="factual_lookup",
                reasoning="Deterministic visual inspection plan (factual_lookup)",
                tool_calls=calls,
                answer_blueprint=self._build_heuristic_answer_blueprint(query, "factual_lookup", params),
            )

        # 1. Timeline / Chronological Trajectory
        if any(w in q_lower for w in ("timeline", "chronology", "chronological", "evolution", "over time", "history of", "progression", "trace back", "trace it back", "where it began", "from the beginning", "how it unfolded", "how it started")):
            archetype = "thematic_timeline"
            calls = [build_timeline_tool(query=query, limit=25), build_hybrid_search_tool(query=query, top_k=8, purpose="Retrieve anchor articles")]

        # 2. Entity Deep Dive
        elif bool(re.search(r"\b(?:mentions?\s+of|coverage\s+regarding|news\s+regarding|involvement\s+of|role\s+of|profile\s+of|profile\s+the\s+coverage\s+of|everything\s+about|all\s+mentions\s+of|tracked\s+entity)\b", q_lower)):
            archetype = "entity_deep_dive"
            clean_ent = _extract_clean_entity(query)
            calls = [
                build_entity_search_tool(entity_name=clean_ent, top_k=10, newspaper_name=newspaper, date_from=date_from, date_to=date_to),
                build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, top_k=8, purpose="Semantic context"),
            ]

        # 3. Analytical Computation / Statistical Analysis
        elif any(re.search(pat, q_lower) for pat in MATH_AGGREGATE_PATTERNS) and is_dynamic_analysis_permitted(query):
            archetype = "analytical_computation"
            calls = [
                build_dynamic_analysis_tool(
                    query=query,
                    analysis_description=f"Statistical computation: {query}",
                    newspaper_name=newspaper,
                    issue_date=issue_date or date_from,
                    purpose="Execute custom analytical computation",
                ),
            ]
            if (issue_date or date_from) and (newspaper or page_filter):
                calls.append(
                    build_sql_summary_tool(
                        analysis_type="issue_summary",
                        newspaper_name=newspaper,
                        issue_date=issue_date if not has_explicit_range else None,
                        date_from=date_from if has_explicit_range else None,
                        date_to=date_to if has_explicit_range else None,
                        page_filter=page_filter,
                        query=query,
                        purpose="Ground-truth broadsheet issue summary and page metrics",
                    )
                )
            calls.append(build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, page_filter=page_filter, top_k=4, purpose="Contextual evidence for analytical results"))

        # 4. Single Newspaper Multi-Issue Comparison
        elif newspaper and not comp_newspaper and len(target_dates) >= 2:
            archetype = "quantitative_trend"
            calls = [build_sql_summary_tool(newspaper_name=newspaper, issue_date=dt, query=query, purpose=f"Manifest for {newspaper} on {dt}") for dt in target_dates]
            calls.append(build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, top_k=10, purpose="Multi-issue articles"))

        # 5. Cross-Newspaper Comparison or Cross-Page Comparison
        elif comp_newspaper is not None or len(target_pages) >= 2 or is_diff or is_comparative_query(query):
            archetype = "cross_newspaper_comparison"
            target_dt = issue_date or date_from
            is_shared = params.get("is_shared", False) or any(w in q_lower for w in ("similar", "shared", "common", "same article", "same stories", "both"))
            calls = _build_comparative_tool_calls(
                query=query,
                src_np=newspaper,
                comp_np=comp_newspaper,
                target_dt=target_dt,
                target_pages=target_pages,
                category_filter=category,
                is_diff=is_diff,
                is_shared=is_shared,
            )

        # 5b. Single Newspaper Coverage Audit / Presence Check
        elif newspaper and not comp_newspaper and bool(re.search(r"\b(?:did\s+.+\s+(?:report\s+on|cover|publish|carry)|was\s+.+\s+(?:reported|covered|published|carried)\s+(?:in|by)|(?:coverage|reporting)\s+(?:of|on)\s+.+\s+(?:in|by)|check\s+(?:if|whether)\s+.+\s+(?:reported|covered))\b", q_lower)):
            archetype = "factual_lookup"
            target_dt = issue_date or date_from
            topic_clean = re.sub(r"(?i)\b(?:did|was|were|is|are|has|have)\s+(?:the\s+)?[\w\s]+\s+(?:report\s+on|cover|publish|carry|mention)\s*", "", query)
            topic_clean = re.sub(r"(?i)\b(?:on|dated?)\s+\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", topic_clean).strip("?:!.,\"' ") or query
            calls = [
                build_coverage_analysis_tool(query=topic_clean, target_date=target_dt, newspaper_name=newspaper, purpose=f"Audit coverage presence for {newspaper}"),
                build_hybrid_search_tool(query=topic_clean, newspaper_name=newspaper, date_from=date_from, date_to=date_to, top_k=6, purpose=f"Search candidate articles in {newspaper}"),
            ]

        # 6. Quantitative Trend / Article Catalog / Issue Manifest / Counts / Availability / Ordinal Page Queries
        elif (not is_target_art) and (
            is_archive_np
            or (page_filter and any(w in q_lower for w in ("article", "story", "stories", "no of", "how many", "list", "news", "headline", "headlines", "lead", "top", "main", "1st", "first", "2nd", "second", "3rd", "third")))
            or any(w in q_lower for w in (
                "how many", "count", "number of", "no of", "frequency", "trend", "distribution", "volume", "statistics", "summarize", "overview", "whole", "entire", "all articles", "list", "manifest", "today's paper", "edition", "what articles", "articles on", "is any newspaper available", "are there any newspapers", "available for dated", "newspaper available", "newspapers available", "paper available", "issues available",
            ))
            or (category and any(w in q_lower for w in ("news", "articles", "stories", "headlines")))
        ):
            q_no_np = re.sub(r"\bnewspapers?\b", "", q_lower)
            has_article_words = bool(re.search(r"\b(articles?|story|stories|news)\b", q_no_np))
            is_ordinal_page = bool(page_filter and is_ordinal_lead_story_query(query))
            is_availability = is_availability_query(query)
            is_count = is_availability or any(w in q_lower for w in ("how many", "total articles", "number of articles", "count of articles", "no of articles", "count of", "number of issues", "total issues", "no of newspaper", "how many issues", "count of pages", "number of pages"))
            is_whole_or_count = is_availability or is_count or any(w in q_lower for w in ("today's paper", "edition", "whole", "entire", "overview", "summarize", "how many", "count of", "number of", "no of", "distribution", "frequency", "trend", "statistics", "volume"))
            is_catalog = (not is_availability) and (is_ordinal_page or any(w in q_lower for w in ("list", "catalog", "manifest", "all articles", "all news", "all stories", "headlines", "headings", "what articles", "articles on")) or bool(category) or (bool(page_filter) and not is_count))

            if is_ordinal_page:
                archetype = "article_catalog"
            elif is_archive_np and not has_article_words:
                archetype = "quantitative_trend"
            else:
                archetype = "article_catalog" if is_catalog and not is_whole_or_count else "quantitative_trend"

            is_ad = any(w in q_lower for w in ("advertisement", "advertisements", "ad count", "ads count", "number of ads", "how many ads", "commercials", "notices and ads", "commercial notices"))
            is_photo = is_count and any(w in q_lower for w in ("photo", "photos", "picture", "pictures", "image", "images"))

            if is_archive_np and not any(pat.search(query) for pat, _ in _KNOWN_BRANDS_PATTERNS):
                newspaper = None

            if is_ad:
                atype = "count_advertisements"
            elif is_photo:
                atype = "count_photos"
            elif is_availability or (is_archive_np and not has_article_words) or (is_count and ("issue" in q_lower or "newspaper" in q_lower) and not has_article_words):
                atype = "count_issues"
            elif is_count and not page_filter:
                atype = "count_articles"
            else:
                atype = "issue_summary"

            if len(target_pages) >= 2:
                calls = [
                    build_sql_summary_tool(
                        analysis_type=atype,
                        newspaper_name=newspaper,
                        issue_date=issue_date,
                        date_from=date_from,
                        date_to=date_to,
                        issue_id=issue_id,
                        page_filter=p,
                        category_filter=category,
                        query=query,
                        purpose=f"Articles on Page {p}",
                    )
                    for p in target_pages
                ]
                calls.append(
                    build_hybrid_search_tool(
                        query=query,
                        newspaper_name=newspaper,
                        date_from=date_from,
                        date_to=date_to,
                        top_k=10,
                        purpose=f"Articles across pages {', '.join(target_pages)}",
                    )
                )
            else:
                calls = [build_sql_summary_tool(analysis_type=atype, newspaper_name=newspaper, issue_date=issue_date, date_from=date_from, date_to=date_to, issue_id=issue_id, page_filter=page_filter, category_filter=category, query=query)]
                if page_filter and (is_ordinal_page or any(w in q_lower for w in ("list", "articles on", "stories on", "all articles", "news", "lead", "top", "first", "1st"))):
                    calls.append(build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, page_filter=page_filter, top_k=6, purpose=f"Articles on Page {page_filter}"))

        # 7. Factual Lookup (Default)
        else:
            archetype = "factual_lookup"
            search_q = params.get("headline") or query
            calls = [build_hybrid_search_tool(query=search_q, newspaper_name=newspaper, date_from=date_from, date_to=date_to, page_filter=page_filter, top_k=4 if is_target_art else 6, purpose="Search for factual evidence")]

        if enable_web_search:
            calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))

        return PlanResult(
            archetype=archetype,
            reasoning=f"Deterministic heuristic plan ({archetype})",
            tool_calls=calls,
            answer_blueprint=self._build_heuristic_answer_blueprint(query, archetype, params),
        )


__all__ = [
    "AgentPlan",
    "AnswerBlueprint",
    "DEFAULT_BLUEPRINTS",
    "ExtractedToolArguments",
    "MATH_AGGREGATE_PATTERNS",
    "PLANNER_SYSTEM_PROMPT",
    "PlanResult",
    "PlannedToolCall",
    "QueryArchetype",
    "QueryPlan",
    "QueryPlanner",
    "SectionSpec",
    "ToolCallSpec",
    "ToolDecision",
    "ToolName",
    "_KNOWN_BRANDS_PATTERNS",
    "_PROMPT_BODY",
    "_PROMPT_PREAMBLE",
    "_SECTION_PATTERNS",
    "_build_targeted_web_query",
    "build_heuristic_answer_blueprint",
    "build_targeted_web_query",
    "extract_parameters_from_query",
    "is_dynamic_analysis_permitted",
    "resolve_tool_sequence",
]
