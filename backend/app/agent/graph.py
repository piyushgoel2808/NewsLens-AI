"""LangGraph State Machine for Newspaper Intelligence Agentic RAG."""

from __future__ import annotations

import contextlib
import re
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.archive_context import get_archive_and_schema_context
from app.agent.condenser import (
    CLEAN_SESSION_CLARIFICATION_MESSAGE,
    condense_conversational_query,
    is_ambiguous_standalone_query,
    is_in_context_meta_query,
    needs_condensation,
)
from app.retrieval import resolve_conversation_working_context
from app.agent.extractor import extract_parameters_from_query
from app.agent.evaluator import EvidenceEvaluator
from app.agent.executor import ToolExecutor
from app.agent.sql_dispatcher import SQLAnalyticsDispatcher
from app.agent.planner import QueryPlanner
from app.agent.state import AgentState, ToolExecutionRecord
from app.agent.answer_verifier import AnswerVerifier
from app.agent.synthesizer import AnswerSynthesizer
from app.agent.sandbox import ASTSafetyScanner, SandboxedExecutor
from app.agent.tool_maker import ToolMaker
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import record_agent_query
from app.models.query import QueryLog
from app.retrieval.coverage_analyzer import CoverageAnalyzer
from app.retrieval.entity_filter import EntitySearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.sql_analytics import SQLAnalyticsEngine
from app.retrieval.timeline_builder import TimelineBuilder
from app.retrieval.visual_inspector import VisualInspectionEngine
from app.retrieval.web_search import WebSearchEngine
from app.storage.cache_store import CacheStore, compute_query_cache_key

logger = get_logger(__name__)


class AgentWorkflow:
    """Compiled LangGraph workflow executing the agentic RAG lifecycle."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._planner = QueryPlanner()
        self._synthesizer = AnswerSynthesizer()
        self._hybrid_search = HybridSearchEngine(session_factory=session_factory)
        self._entity_search = EntitySearchEngine(session_factory=session_factory)
        self._timeline_builder = TimelineBuilder(session_factory=session_factory)
        self._sql_analytics = SQLAnalyticsEngine(session_factory=session_factory)
        self._coverage_analyzer = CoverageAnalyzer(
            session_factory=session_factory,
            hybrid_search_engine=self._hybrid_search,
        )
        self._web_search = WebSearchEngine()
        self._visual_inspector = VisualInspectionEngine(session_factory=session_factory)
        self._cache = CacheStore()

        # Dynamic Tool Maker (LLM-as-Tool-Maker)
        settings = get_settings()
        tool_maker = None
        if settings.enable_dynamic_tools:
            scanner = ASTSafetyScanner()
            sandbox = SandboxedExecutor(
                db_url_readonly=settings.mysql_readonly_url,
                timeout_seconds=settings.dynamic_tool_timeout_seconds,
                max_memory_mb=settings.dynamic_tool_max_memory_mb,
            )
            tool_maker = ToolMaker(scanner=scanner, sandbox=sandbox)
        self._tool_maker = tool_maker

        # Modular Tool Execution and Evidence Evaluation Engines
        self._sql_dispatcher = SQLAnalyticsDispatcher(
            sql_analytics=self._sql_analytics,
            coverage_analyzer=self._coverage_analyzer,
        )
        self._executor = ToolExecutor(
            session_factory=session_factory,
            hybrid_search=self._hybrid_search,
            entity_search=self._entity_search,
            timeline_builder=self._timeline_builder,
            sql_analytics=self._sql_analytics,
            coverage_analyzer=self._coverage_analyzer,
            web_search=self._web_search,
            tool_maker=tool_maker,
            visual_inspector=self._visual_inspector,
            sql_dispatcher=self._sql_dispatcher,
        )
        self._evaluator = EvidenceEvaluator(
            entity_search=self._entity_search,
            web_search=self._web_search,
            tool_maker=tool_maker,
            sql_analytics=self._sql_analytics,
        )
        self._verifier = AnswerVerifier()

        # Build Graph
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        workflow.add_node("classify_and_plan", self._classify_and_plan_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("evaluate_and_fallback", self._evaluate_and_fallback_node)
        workflow.add_node("execute_adaptive_replan", self._execute_adaptive_replan_node)
        workflow.add_node("execute_dynamic_code", self._execute_dynamic_code_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        workflow.add_node("verify_answer", self._verify_answer_node)
        workflow.add_node("log_query", self._log_query_node)

        workflow.add_edge(START, "classify_and_plan")

        # Native conditional routing after planning
        workflow.add_conditional_edges(
            "classify_and_plan",
            self._route_after_planning,
            {
                "execute_tools": "execute_tools",
                "synthesize_answer": "synthesize_answer",
                "log_query": "log_query",
            },
        )
        workflow.add_edge("execute_tools", "evaluate_and_fallback")

        # Closed-loop reflexive routing after evidence evaluation
        workflow.add_conditional_edges(
            "evaluate_and_fallback",
            self._route_after_evaluation,
            {
                "synthesize_answer": "synthesize_answer",
                "execute_adaptive_replan": "execute_adaptive_replan",
                "execute_dynamic_code": "execute_dynamic_code",
            },
        )
        workflow.add_edge("execute_adaptive_replan", "synthesize_answer")
        workflow.add_edge("execute_dynamic_code", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "verify_answer")

        # Closed-loop routing after LLM answer verification
        workflow.add_conditional_edges(
            "verify_answer",
            self._route_after_verification,
            {
                "log_query": "log_query",
                "execute_dynamic_code": "execute_dynamic_code",
            },
        )
        workflow.add_edge("log_query", END)

        return workflow.compile()

    def _route_after_evaluation(self, state: AgentState) -> str:
        """Route conditionally based on reflexive evaluation verdict with 1-cycle ceiling."""
        if state.get("recovery_attempts", 0) >= 1:
            return "synthesize_answer"

        verdict = state.get("evaluation_verdict")
        if not verdict or verdict.get("is_sufficient", True):
            return "synthesize_answer"

        action = verdict.get("recommended_action")
        if action == "synthesize_dynamic_tool":
            if self._tool_maker:
                return "execute_dynamic_code"
            return "execute_adaptive_replan"
        elif action == "replan_static_tools":
            return "execute_adaptive_replan"

        return "synthesize_answer"

    def _route_after_verification(self, state: AgentState) -> str:
        """Route conditionally based on LLM answer verification with 1-cycle ceiling."""
        if state.get("recovery_attempts", 0) >= 1 or state.get("verification_attempts", 0) >= 1:
            return "log_query"

        verdict = state.get("answer_verification")
        if not verdict:
            return "log_query"

        action = verdict.get("recommended_action")
        if action == "fallback_to_dynamic_tool" and self._tool_maker:
            return "execute_dynamic_code"

        return "log_query"

    @staticmethod
    def _route_after_planning(state: AgentState) -> str:
        """Route conditionally based on planning outcome to prevent unnecessary node execution."""
        archetype = state.get("archetype")
        if archetype == "clarification_needed":
            return "log_query"
        if archetype == "conversational_meta_query" or not state.get("plan"):
            return "synthesize_answer"
        return "execute_tools"

    async def _classify_and_plan_node(self, state: AgentState) -> dict[str, Any]:
        """Classify archetype and produce multi-step tool execution plan."""
        query = state["query"]
        chat_history = state.get("chat_history", [])
        original_query = state.get("original_query") or query

        has_attached = bool(state.get("attached_photo_id") or state.get("attached_article_id"))
        if is_ambiguous_standalone_query(query, chat_history, has_attached_asset=has_attached):
            return {
                "query": query,
                "original_query": original_query,
                "archetype": "clarification_needed",
                "plan": [],
                "synthesized_answer": CLEAN_SESSION_CLARIFICATION_MESSAGE,
                "evidence_items": [],
                "citations": [],
            }

        # In-Context Meta-Query Detection (e.g. asking for dates, newspapers, citations)
        if is_in_context_meta_query(query, chat_history):
            return {
                "query": query,
                "original_query": original_query,
                "archetype": "conversational_meta_query",
                "plan": [],
                "evidence_items": [],
            }

        # Use working context already initialized in state with current_query grounding
        active_issue_id = state.get("active_issue_id")
        active_newspaper_name = state.get("active_newspaper_name")
        active_issue_date = state.get("active_issue_date")
        active_newspapers = [active_newspaper_name] if active_newspaper_name else []

        attached_asset = state.get("attached_asset")
        has_attached_asset = bool(
            has_attached
            or (attached_asset and (attached_asset.get("photo_id") or attached_asset.get("article_id") or attached_asset.get("newspaper_name")))
        )
        is_followup = bool(
            (chat_history or has_attached_asset)
            and needs_condensation(query, chat_history, attached_asset=attached_asset)
        )

        if state.get("is_condensed"):
            condensed_query = query
        else:
            condensed_query = await condense_conversational_query(
                query=query,
                chat_history=chat_history,
                model_override=state.get("model_override"),
                active_issue_id=active_issue_id,
                active_newspaper_name=active_newspaper_name,
                active_issue_date=active_issue_date,
                attached_asset=attached_asset,
            )

        archive_context_str = await get_archive_and_schema_context(self._session_factory)

        # Only pass active_issue_date and active_newspapers to planner if an asset is attached,
        # or if this is an active follow-up query. Do NOT constrain standalone text queries.
        planner_active_date = active_issue_date if (has_attached_asset or is_followup) else None
        planner_active_newspapers = active_newspapers if (has_attached_asset or is_followup) else []

        plan_res = await self._planner.plan_query_async(
            condensed_query,
            enable_web_search=state.get("enable_web_search", False),
            model_override=state.get("model_override"),
            archive_context=archive_context_str,
            active_issue_date=planner_active_date,
            active_newspapers=planner_active_newspapers,
            attached_article_id=state.get("attached_article_id"),
            attached_photo_id=state.get("attached_photo_id"),
        )

        planned_calls = [
            {
                "tool_name": c.tool_name,
                "arguments": c.arguments,
                "purpose": c.purpose,
            }
            for c in plan_res.tool_calls
        ]

        blueprint_dict = (
            plan_res.answer_blueprint.model_dump()
            if hasattr(plan_res.answer_blueprint, "model_dump")
            else (plan_res.answer_blueprint if isinstance(plan_res.answer_blueprint, dict) else None)
        )

        extracted_params = extract_parameters_from_query(condensed_query)

        return {
            "query": condensed_query,
            "original_query": original_query,
            "archetype": plan_res.archetype,
            "plan": planned_calls,
            "answer_blueprint": blueprint_dict,
            "extracted_params": extracted_params,
            "is_condensed": True,
        }

    async def _execute_single_tool(
        self,
        call: dict[str, Any],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], ToolExecutionRecord, dict[str, Any]]:
        """Backward-compatible proxy delegating to ToolExecutor."""
        return await self._executor.execute_single_tool(call, state)

    async def _execute_tools_node(self, state: AgentState) -> dict[str, Any]:
        """Execute scheduled tools concurrently via ToolExecutor."""
        if state.get("archetype") == "clarification_needed" or not state.get("plan"):
            return {
                "evidence_items": [],
                "tool_executions": [],
            }

        plan = state.get("plan", [])
        evidence_items, tool_records, ctx_updates = await self._executor.execute_tools(plan, state)

        return {
            "evidence_items": evidence_items,
            "tool_executions": tool_records,
            "active_issue_id": ctx_updates.get("active_issue_id", state.get("active_issue_id")),
            "active_newspaper_name": ctx_updates.get("active_newspaper_name", state.get("active_newspaper_name")),
            "active_issue_date": ctx_updates.get("active_issue_date", state.get("active_issue_date")),
        }

    async def _evaluate_and_fallback_node(self, state: AgentState) -> dict[str, Any]:
        """Corrective RAG (CRAG) Node: Grade retrieval quality using reflexive evaluator."""
        evidence = state.get("evidence_items", [])
        filtered_evidence = self._evaluator.filter_evidence(
            evidence,
            state.get("query", ""),
            state.get("archetype", "factual_lookup"),
        )
        verdict = await self._evaluator.evaluate_evidence_async(
            filtered_evidence,
            state,
            model_override=state.get("model_override"),
        )
        return {
            "evidence_items": filtered_evidence,
            "evaluation_verdict": verdict.to_dict(),
            "gap_diagnosis": verdict.gap_reason,
            "recovery_attempts": state.get("recovery_attempts", 0),
        }

    async def _execute_adaptive_replan_node(self, state: AgentState) -> dict[str, Any]:
        """Adaptive closed-loop recovery: re-plan targeted tools and execute them."""
        query = state.get("query", "")
        plan = state.get("plan", [])
        tool_records = list(state.get("tool_executions", []))
        existing_evidence = list(state.get("evidence_items", []))
        gap_diag = state.get("gap_diagnosis")

        logger.info(
            "Adaptive replan recovery initiated",
            extra={"query": query, "gap_diagnosis": gap_diag},
        )

        replan_res = await self._planner.replan_with_feedback_async(
            query=query,
            previous_plan=plan,
            tool_executions=tool_records,
            gap_diagnosis=gap_diag,
            model_override=state.get("model_override"),
            active_issue_date=state.get("active_issue_date"),
            active_newspapers=[state["active_newspaper_name"]] if state.get("active_newspaper_name") else [],
        )

        new_tool_calls = [
            {
                "tool_name": c.tool_name,
                "arguments": c.arguments,
                "purpose": c.purpose,
            }
            for c in replan_res.tool_calls
        ]

        combined_evidence = existing_evidence
        combined_records = tool_records
        ctx_updates: dict[str, Any] = {}

        if new_tool_calls:
            new_evidence, new_records, ctx_updates = await self._executor.execute_tools(new_tool_calls, state)
            for rec in new_records:
                rec["is_recovery"] = True
            combined_evidence = new_evidence + existing_evidence
            combined_records = tool_records + new_records

        # If still empty and web search is enabled, execute web fallback
        if not combined_evidence and state.get("enable_web_search", False):
            t_web = time.monotonic()
            web_res = await self._web_search.search(query=query, num_results=4)
            if web_res:
                web_items = [
                    {
                        "article_id": 0,
                        "headline": wr.title,
                        "newspaper_name": wr.source,
                        "issue_date": wr.published_date or "Live Web",
                        "pages": [1],
                        "snippet": f"[CRAG Web Fallback]: {wr.snippet}",
                        "url": wr.url,
                        "is_web": True,
                        "prominence_score": 0.8,
                        "source_tool": "crag_web_fallback",
                    }
                    for wr in web_res
                ]
                combined_evidence = web_items
                dur_ms = round((time.monotonic() - t_web) * 1000)
                combined_records.append(
                    ToolExecutionRecord(
                        tool_name="crag_web_fallback",
                        tool_input={"query": query},
                        results_count=len(web_res),
                        execution_time_ms=dur_ms,
                    )
                )

        replan_blueprint = (
            replan_res.answer_blueprint.model_dump()
            if hasattr(replan_res.answer_blueprint, "model_dump")
            else state.get("answer_blueprint")
        )

        return {
            "evidence_items": combined_evidence,
            "tool_executions": combined_records,
            "recovery_attempts": state.get("recovery_attempts", 0) + 1,
            "active_issue_id": ctx_updates.get("active_issue_id", state.get("active_issue_id")),
            "active_newspaper_name": ctx_updates.get("active_newspaper_name", state.get("active_newspaper_name")),
            "active_issue_date": ctx_updates.get("active_issue_date", state.get("active_issue_date")),
            "answer_blueprint": replan_blueprint,
        }

    async def _execute_dynamic_code_node(self, state: AgentState) -> dict[str, Any]:
        """Dynamic Python/SQL tool synthesis node via ToolMaker."""
        existing_evidence = list(state.get("evidence_items", []))
        tool_records = list(state.get("tool_executions", []))

        if not self._tool_maker:
            return {
                "recovery_attempts": state.get("recovery_attempts", 0) + 1,
            }

        t_dyn = time.monotonic()
        context: dict[str, Any] = {
            "available_newspapers": [],
            "available_dates": [],
            "categories": [],
        }
        with contextlib.suppress(Exception):
            meta = await self._sql_analytics.get_archive_metadata()
            context["available_newspapers"] = meta.get("newspapers", [])
            context["available_dates"] = list(meta.get("available_dates", {}).keys())
            context["categories"] = meta.get("categories", [])

        # Inject pre-normalized parameters from extracted state
        extracted = state.get("extracted_params") or {}
        if extracted.get("issue_date"):
            context["target_date"] = extracted["issue_date"]
        if extracted.get("newspaper_name"):
            context["newspaper_name"] = extracted["newspaper_name"]
        if extracted.get("date_from"):
            context["date_from"] = extracted["date_from"]
        if extracted.get("date_to"):
            context["date_to"] = extracted["date_to"]
        if extracted.get("category_filter"):
            context["category"] = extracted["category_filter"]

        dyn_res = await self._tool_maker.generate_and_execute(
            query=state.get("query", ""),
            context=context,
            model_override=state.get("model_override"),
            attempted_tools=tool_records,
            gap_diagnosis=state.get("gap_diagnosis"),
        )

        if dyn_res.success and dyn_res.evidence_items:
            combined_evidence = dyn_res.evidence_items + existing_evidence
            dur_ms = round((time.monotonic() - t_dyn) * 1000)
            tool_records.append(
                ToolExecutionRecord(
                    tool_name="crag_dynamic_tool_fallback",
                    tool_input={
                        "query": state.get("query", ""),
                        "gap_diagnosis": state.get("gap_diagnosis"),
                    },
                    results_count=len(dyn_res.evidence_items),
                    execution_time_ms=dur_ms,
                )
            )
            return {
                "evidence_items": combined_evidence,
                "tool_executions": tool_records,
                "recovery_attempts": state.get("recovery_attempts", 0) + 1,
            }

        return {
            "recovery_attempts": state.get("recovery_attempts", 0) + 1,
        }

    async def _synthesize_answer_node(self, state: AgentState) -> dict[str, Any]:
        """Formulate grounded answer with source citations."""
        if state.get("archetype") == "clarification_needed" and state.get("synthesized_answer"):
            return {
                "synthesized_answer": state["synthesized_answer"],
                "citations": [],
                "cost_usd": 0.0,
            }

        query = state["query"]
        archetype = state.get("archetype", "factual_lookup")
        evidence = state.get("evidence_items", [])
        model_override = state.get("model_override")

        answer, citations, cost_usd = await self._synthesizer.synthesize(
            query=query,
            archetype=archetype,
            evidence_items=evidence,
            model_override=model_override,
            chat_history=state.get("chat_history", []),
            answer_blueprint=state.get("answer_blueprint"),
        )

        return {
            "synthesized_answer": answer,
            "citations": citations,
            "cost_usd": cost_usd,
        }

    async def _verify_answer_node(self, state: AgentState) -> dict[str, Any]:
        """LLM Critic Fact-Checking: Verify synthesized answer against retrieved evidence."""
        query = state["query"]
        draft_answer = state.get("synthesized_answer", "")
        evidence = state.get("evidence_items", [])
        archetype = state.get("archetype", "factual_lookup")
        model_override = state.get("model_override")
        blueprint = state.get("answer_blueprint")

        if archetype in ("clarification_needed", "conversational_meta_query") or not draft_answer:
            return {
                "verification_attempts": state.get("verification_attempts", 0) + 1,
            }

        res = await self._verifier.verify_answer_async(
            query=query,
            draft_answer=draft_answer,
            evidence_items=evidence,
            archetype=archetype,
            model_override=model_override,
            answer_blueprint=blueprint,
        )

        updates: dict[str, Any] = {
            "answer_verification": res.to_dict(),
            "verification_attempts": state.get("verification_attempts", 0) + 1,
        }

        if not res.is_valid:
            logger.warning(
                "AnswerVerifier flagged ungrounded answer",
                extra={
                    "critique": res.critique,
                    "factual_errors": res.factual_errors,
                    "recommended_action": res.recommended_action,
                },
            )
            if res.refined_answer:
                updates["synthesized_answer"] = res.refined_answer
            if res.evidence_gap_detected and res.dynamic_tool_hint:
                updates["gap_diagnosis"] = res.dynamic_tool_hint

        return updates

    async def _log_query_node(self, state: AgentState) -> dict[str, Any]:
        """Persist execution audit and query history in MySQL."""
        async with self._session_factory() as db:
            log_record = QueryLog(
                user_id=state.get("user_id"),
                query_text=state["query"],
                query_type=state.get("archetype"),
                plan_json={
                    "plan": state.get("plan", []),
                    "answer_blueprint": state.get("answer_blueprint"),
                },
                tool_calls_json={"tools": state.get("tool_executions", [])},
                answer_text=state.get("synthesized_answer"),
                citations_json={"citations": state.get("citations", [])},
                latency_ms=state.get("latency_ms", 0),
                cost_usd=state.get("cost_usd", 0.0),
            )
            db.add(log_record)
            await db.commit()

        return {}

    async def run(
        self,
        query: str,
        chat_history: list[dict[str, Any]] | None = None,
        user_id: str | None = None,
        model_override: str | None = None,
        enable_web_search: bool = False,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
        attached_issue_date: str | None = None,
        attached_newspaper_name: str | None = None,
    ) -> AgentState:
        """Execute the complete agentic query cycle with caching and metrics."""
        t0 = time.monotonic()
        history = chat_history or []

        # 1. Deterministic Redis Cache Check
        cache_key = compute_query_cache_key(
            query=f"{query}__web_{enable_web_search}__art_{attached_article_id}__ph_{attached_photo_id}",
            model_id=model_override or "",
        )
        cached_result = await self._cache.get_query(cache_key)
        if cached_result:
            dur = time.monotonic() - t0
            record_agent_query(
                archetype=cached_result.get("archetype", "cached"),
                status="cached",
                model=model_override or "default",
                duration_seconds=dur,
            )
            return cached_result  # type: ignore[return-value]

        working_ctx = await resolve_conversation_working_context(
            query=query,
            chat_history=history,
            attached_photo_id=attached_photo_id,
            attached_article_id=attached_article_id,
            attached_issue_date=attached_issue_date,
            attached_newspaper_name=attached_newspaper_name,
            session_factory=self._session_factory,
        )

        initial_state: AgentState = {
            "query": query,
            "original_query": query,
            "chat_history": history,
            "archetype": "factual_lookup",
            "plan": [],
            "tool_executions": [],
            "evidence_items": [],
            "synthesized_answer": "",
            "citations": [],
            "cost_usd": 0.0,
            "latency_ms": 0,
            "user_id": user_id,
            "model_override": model_override,
            "enable_web_search": enable_web_search,
            "web_search_results": [],
            "active_issue_id": working_ctx.active_ctx.get("issue_id"),
            "active_newspaper_name": working_ctx.active_ctx.get("newspaper_name"),
            "active_issue_date": working_ctx.active_ctx.get("issue_date"),
            "attached_article_id": working_ctx.eff_attached_article_id,
            "attached_photo_id": working_ctx.eff_attached_photo_id,
            "attached_asset": working_ctx.attached_ctx,
            "error": None,
            "evaluation_verdict": None,
            "recovery_attempts": 0,
            "gap_diagnosis": None,
            "answer_blueprint": None,
            "answer_verification": None,
            "verification_attempts": 0,
        }

        try:
            final_state: AgentState = await self._graph.ainvoke(initial_state)
            dur = time.monotonic() - t0
            final_state["latency_ms"] = round(dur * 1000)

            # Record Prometheus Metrics
            record_agent_query(
                archetype=final_state.get("archetype", "factual_lookup"),
                status="success",
                model=model_override or "default",
                duration_seconds=dur,
            )

            # 2. Store in Redis Cache
            await self._cache.set_query(cache_key, dict(final_state), ttl_seconds=3600)

            return final_state
        except Exception as e:
            dur = time.monotonic() - t0
            record_agent_query(
                archetype="unknown",
                status="error",
                model=model_override or "default",
                duration_seconds=dur,
            )
            raise e


__all__ = [
    "AgentWorkflow",
]
