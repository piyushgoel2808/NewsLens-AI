"""LangGraph State Machine for Newspaper Intelligence Agentic RAG."""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.condenser import (
    CLEAN_SESSION_CLARIFICATION_MESSAGE,
    condense_conversational_query,
    extract_active_issue_from_history,
    is_ambiguous_standalone_query,
    is_in_context_meta_query,
    needs_condensation,
    parse_inline_citation,
    resolve_attached_asset_context,
)
from app.agent.extractor import extract_parameters_from_query
from app.agent.evaluator import EvidenceEvaluator
from app.agent.executor import ToolExecutor
from app.agent.planner import QueryPlanner
from app.agent.state import AgentState, ToolExecutionRecord
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

        # Modular Tool Execution and Evidence Evaluation Engines
        self._executor = ToolExecutor(
            session_factory=session_factory,
            hybrid_search=self._hybrid_search,
            entity_search=self._entity_search,
            timeline_builder=self._timeline_builder,
            sql_analytics=self._sql_analytics,
            coverage_analyzer=self._coverage_analyzer,
            web_search=self._web_search,
            tool_maker=tool_maker,
        )
        self._evaluator = EvidenceEvaluator(
            entity_search=self._entity_search,
            web_search=self._web_search,
            tool_maker=tool_maker,
            sql_analytics=self._sql_analytics,
        )

        # Build Graph
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        workflow.add_node("classify_and_plan", self._classify_and_plan_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("evaluate_and_fallback", self._evaluate_and_fallback_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
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
        workflow.add_edge("evaluate_and_fallback", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "log_query")
        workflow.add_edge("log_query", END)

        return workflow.compile()

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

        condensed_query = await condense_conversational_query(
            query=query,
            chat_history=chat_history,
            model_override=state.get("model_override"),
            active_issue_id=active_issue_id,
            active_newspaper_name=active_newspaper_name,
            active_issue_date=active_issue_date,
            attached_asset=attached_asset,
        )

        archive_context_str: str | None = None
        try:
            meta = await self._sql_analytics.get_archive_metadata()
            available_dates = meta.get("available_dates", {})
            if available_dates:
                dates_info = [f"- {dt}: {', '.join(nps)}" for dt, nps in list(available_dates.items())[:10]]
                cats_info = ", ".join(meta.get("categories", []))
                archive_context_str = "Available Issues:\n" + "\n".join(dates_info) + f"\nCanonical Categories: {cats_info}"
        except Exception as e:
            logger.warning("Failed to fetch archive metadata for planner", extra={"error": str(e)})

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

        return {
            "query": condensed_query,
            "original_query": original_query,
            "archetype": plan_res.archetype,
            "plan": planned_calls,
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
        """Corrective RAG (CRAG) Node: Grade retrieval quality and execute corrective fallback if needed."""
        evidence = state.get("evidence_items", [])
        tool_records = list(state.get("tool_executions", []))

        filtered_evidence, fallback_records = await self._evaluator.evaluate_and_fallback(evidence, state)
        return {
            "evidence_items": filtered_evidence,
            "tool_executions": tool_records + fallback_records,
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
        )

        return {
            "synthesized_answer": answer,
            "citations": citations,
            "cost_usd": cost_usd,
        }

    async def _log_query_node(self, state: AgentState) -> dict[str, Any]:
        """Persist execution audit and query history in MySQL."""
        async with self._session_factory() as db:
            log_record = QueryLog(
                user_id=state.get("user_id"),
                query_text=state["query"],
                query_type=state.get("archetype"),
                plan_json={"plan": state.get("plan", [])},
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

        attached_ctx = await resolve_attached_asset_context(
            session_factory=self._session_factory,
            attached_photo_id=attached_photo_id,
            attached_article_id=attached_article_id,
            attached_issue_date=attached_issue_date,
            attached_newspaper_name=attached_newspaper_name,
        )
        res_date = attached_ctx.get("issue_date")
        res_np = attached_ctx.get("newspaper_name")
        res_art_id = attached_ctx.get("article_id")
        res_iss_id = attached_ctx.get("issue_id")
        q_params = extract_parameters_from_query(query) if query else {}
        q_cite = parse_inline_citation(query or "")
        explicit_q_date = q_params.get("issue_date") or q_cite.get("issue_date")
        explicit_q_np = q_params.get("newspaper_name") or q_cite.get("newspaper_name")

        has_date_conflict = bool(explicit_q_date and res_date and explicit_q_date != res_date)
        has_np_conflict = bool(
            explicit_q_np
            and res_np
            and explicit_q_np.lower() not in res_np.lower()
            and res_np.lower() not in explicit_q_np.lower()
        )

        eff_attached_article_id = (res_art_id or attached_article_id) if not has_date_conflict and not has_np_conflict else None
        eff_attached_photo_id = attached_photo_id if not has_date_conflict and not has_np_conflict else None

        active_ctx = extract_active_issue_from_history(
            history,
            current_query=query,
            attached_photo_id=eff_attached_photo_id,
            attached_article_id=eff_attached_article_id,
            attached_issue_date=res_date if not has_date_conflict else None,
            attached_newspaper_name=res_np if not has_np_conflict else None,
            attached_headline=attached_ctx.get("headline") if not has_date_conflict and not has_np_conflict else None,
        )

        if not has_date_conflict and not has_np_conflict:
            if res_date:
                active_ctx["issue_date"] = res_date
            if res_np:
                active_ctx["newspaper_name"] = res_np
            if res_iss_id:
                active_ctx["issue_id"] = res_iss_id
            if eff_attached_article_id:
                active_ctx["article_id"] = eff_attached_article_id
            if eff_attached_photo_id:
                active_ctx["photo_id"] = eff_attached_photo_id
            if attached_ctx.get("headline"):
                active_ctx["headline"] = attached_ctx["headline"]
        else:
            if explicit_q_date:
                active_ctx["issue_date"] = explicit_q_date
            if explicit_q_np:
                active_ctx["newspaper_name"] = explicit_q_np

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
            "active_issue_id": active_ctx.get("issue_id"),
            "active_newspaper_name": active_ctx.get("newspaper_name"),
            "active_issue_date": active_ctx.get("issue_date"),
            "attached_article_id": eff_attached_article_id,
            "attached_photo_id": eff_attached_photo_id,
            "attached_asset": attached_ctx,
            "error": None,
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
