"""LangGraph State Machine for Newspaper Intelligence Agentic RAG."""
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.planner import QueryPlanner
from app.agent.state import AgentState, ToolExecutionRecord
from app.agent.synthesizer import AnswerSynthesizer
from app.core.logging import get_logger
from app.models.query import QueryLog
from app.retrieval.entity_filter import EntitySearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine, SearchFilter
from app.retrieval.sql_analytics import SQLAnalyticsEngine
from app.retrieval.timeline_builder import TimelineBuilder

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

        # Build Graph
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        workflow.add_node("classify_and_plan", self._classify_and_plan_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        workflow.add_node("log_query", self._log_query_node)

        workflow.add_edge(START, "classify_and_plan")
        workflow.add_edge("classify_and_plan", "execute_tools")
        workflow.add_edge("execute_tools", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "log_query")
        workflow.add_edge("log_query", END)

        return workflow.compile()

    async def _classify_and_plan_node(self, state: AgentState) -> dict[str, Any]:
        """Classify archetype and produce multi-step tool execution plan."""
        query = state["query"]
        plan_res = self._planner.plan_query(query)

        planned_calls = [
            {
                "tool_name": c.tool_name,
                "arguments": c.arguments,
                "purpose": c.purpose,
            }
            for c in plan_res.tool_calls
        ]

        return {
            "archetype": plan_res.archetype,
            "plan": planned_calls,
        }

    async def _execute_tools_node(self, state: AgentState) -> dict[str, Any]:
        """Execute scheduled tools and collect evidence items."""
        plan = state.get("plan", [])
        evidence_items: list[dict[str, Any]] = []
        tool_records: list[ToolExecutionRecord] = []

        for call in plan:
            t_start = time.monotonic()
            name = call.get("tool_name")
            args = call.get("arguments", {})
            hits_count = 0

            try:
                if name == "hybrid_search":
                    filters = None
                    if "newspaper_id" in args or "date_from" in args or "date_to" in args:
                        filters = SearchFilter(
                            newspaper_id=args.get("newspaper_id"),
                            date_from=args.get("date_from"),
                            date_to=args.get("date_to"),
                        )
                    hybrid_results = await self._hybrid_search.search(
                        query=args.get("query", ""),
                        top_k=args.get("top_k", 6),
                        filters=filters,
                    )
                    hits_count = len(hybrid_results)
                    for hr in hybrid_results:
                        evidence_items.append(
                            {
                                "article_id": hr.article_id,
                                "headline": hr.headline,
                                "newspaper_name": hr.newspaper_name,
                                "issue_date": hr.issue_date,
                                "pages": hr.pages,
                                "snippet": hr.snippet,
                                "prominence_score": hr.prominence_score,
                                "source_tool": "hybrid_search",
                            }
                        )

                elif name == "entity_search":
                    entity_results = await self._entity_search.search_by_entity(
                        entity_name=args.get("entity_name"),
                        entity_type=args.get("entity_type"),
                        top_k=args.get("top_k", 10),
                    )
                    hits_count = len(entity_results)
                    for er in entity_results:
                        snip_str = (
                            f"Entity [{er.entity_name} ({er.entity_type}) - "
                            f"Salience {er.salience_score}]: {er.summary}"
                        )
                        evidence_items.append(
                            {
                                "article_id": er.article_id,
                                "headline": er.headline,
                                "newspaper_name": er.newspaper_name,
                                "issue_date": er.issue_date,
                                "pages": er.pages,
                                "snippet": snip_str,
                                "prominence_score": er.prominence_score,
                                "source_tool": "entity_search",
                            }
                        )

                elif name == "timeline_builder":
                    tl_result = await self._timeline_builder.build_timeline(
                        query=args.get("query"),
                        limit=args.get("limit", 20),
                    )
                    hits_count = tl_result.total_articles
                    for g in tl_result.date_groups:
                        for m in g.milestones:
                            evidence_items.append(
                                {
                                    "article_id": m.article_id,
                                    "headline": m.headline,
                                    "newspaper_name": g.newspaper_name,
                                    "issue_date": g.date,
                                    "pages": m.pages,
                                    "snippet": f"Timeline Event ({g.date}): {m.summary}",
                                    "prominence_score": m.prominence_score,
                                    "source_tool": "timeline_builder",
                                }
                            )

                elif name == "sql_analytics":
                    analysis_type = args.get("analysis_type")
                    if analysis_type == "entity_trends":
                        trends = await self._sql_analytics.get_entity_mention_trends(
                            entity_name=args.get("term", ""),
                        )
                        hits_count = len(trends)
                        trend_items = [
                            f"{t['date']}: {t['article_count']} articles "
                            f"({t['total_mentions']} mentions)"
                            for t in trends[:5]
                        ]
                        summary_str = (
                            f"Mention Trends for '{args.get('term')}': "
                            + ", ".join(trend_items)
                        )
                        evidence_items.append(
                            {
                                "article_id": 0,
                                "headline": f"Statistical Trends: {args.get('term')}",
                                "newspaper_name": "Aggregated Archive Analytics",
                                "issue_date": trends[0]["date"] if trends else "Overview",
                                "pages": [1],
                                "snippet": summary_str,
                                "source_tool": "sql_analytics",
                            }
                        )
                    elif analysis_type == "topic_distribution":
                        dist = await self._sql_analytics.get_topic_distribution()
                        hits_count = len(dist)
                        dist_str = "Topic Distribution: " + ", ".join(
                            f"{d['section']} ({d['count']} articles)" for d in dist[:5]
                        )
                        evidence_items.append(
                            {
                                "article_id": 0,
                                "headline": "Topic Volume Breakdown",
                                "newspaper_name": "Aggregated Archive Analytics",
                                "issue_date": "Overview",
                                "pages": [1],
                                "snippet": dist_str,
                                "source_tool": "sql_analytics",
                            }
                        )

            except Exception as e:
                logger.error(f"Error executing tool '{name}'", extra={"error": str(e)})

            dur_ms = round((time.monotonic() - t_start) * 1000)
            tool_records.append(
                ToolExecutionRecord(
                    tool_name=name or "unknown",
                    tool_input=args,
                    results_count=hits_count,
                    execution_time_ms=dur_ms,
                )
            )

        return {
            "evidence_items": evidence_items,
            "tool_executions": tool_records,
        }

    async def _synthesize_answer_node(self, state: AgentState) -> dict[str, Any]:
        """Formulate grounded answer with source citations."""
        query = state["query"]
        archetype = state.get("archetype", "factual_lookup")
        evidence = state.get("evidence_items", [])

        answer, citations, cost_usd = await self._synthesizer.synthesize(
            query=query,
            archetype=archetype,
            evidence_items=evidence,
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

    async def run(self, query: str, user_id: str | None = None) -> AgentState:
        """Execute the complete agentic query cycle."""
        t0 = time.monotonic()
        initial_state: AgentState = {
            "query": query,
            "archetype": "factual_lookup",
            "plan": [],
            "tool_executions": [],
            "evidence_items": [],
            "synthesized_answer": "",
            "citations": [],
            "cost_usd": 0.0,
            "latency_ms": 0,
            "user_id": user_id,
            "error": None,
        }

        final_state: AgentState = await self._graph.ainvoke(initial_state)
        final_state["latency_ms"] = round((time.monotonic() - t0) * 1000)

        return final_state
