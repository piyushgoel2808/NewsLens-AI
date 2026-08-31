"""LangGraph State Machine for Newspaper Intelligence Agentic RAG."""

from __future__ import annotations

import re
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
)
from app.agent.planner import QueryPlanner
from app.agent.state import AgentState, ToolExecutionRecord
from app.agent.synthesizer import AnswerSynthesizer
from app.core.logging import get_logger
from app.core.metrics import record_agent_query
from app.models.newspaper import Issue, Newspaper
from app.models.query import QueryLog
from app.retrieval.coverage_analyzer import CoverageAnalyzer
from app.retrieval.entity_filter import EntitySearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine, SearchFilter
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
        workflow.add_edge("classify_and_plan", "execute_tools")
        workflow.add_edge("execute_tools", "evaluate_and_fallback")
        workflow.add_edge("evaluate_and_fallback", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "log_query")
        workflow.add_edge("log_query", END)

        return workflow.compile()

    async def _classify_and_plan_node(self, state: AgentState) -> dict[str, Any]:
        """Classify archetype and produce multi-step tool execution plan."""
        query = state["query"]
        chat_history = state.get("chat_history", [])
        original_query = state.get("original_query") or query

        # Ambiguity Guardrail for Clean Sessions (e.g. 'summarize it' on turn 1)
        if is_ambiguous_standalone_query(query, chat_history):
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

        # Coreference Resolution & Query Condensation for follow-up turns
        active_ctx = extract_active_issue_from_history(chat_history)
        condensed_query = await condense_conversational_query(
            query=query,
            chat_history=chat_history,
            model_override=state.get("model_override"),
            active_issue_id=state.get("active_issue_id") or active_ctx.get("issue_id"),
            active_newspaper_name=state.get("active_newspaper_name") or active_ctx.get("newspaper_name"),
            active_issue_date=state.get("active_issue_date") or active_ctx.get("issue_date"),
        )

        plan_res = await self._planner.plan_query_async(
            condensed_query,
            enable_web_search=state.get("enable_web_search", False),
            model_override=state.get("model_override"),
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

    async def _execute_tools_node(self, state: AgentState) -> dict[str, Any]:
        """Execute scheduled tools and collect evidence items."""
        if state.get("archetype") == "clarification_needed" or not state.get("plan"):
            return {
                "evidence_items": [],
                "tool_executions": [],
            }

        plan = state.get("plan", [])
        evidence_items: list[dict[str, Any]] = []
        tool_records: list[ToolExecutionRecord] = []
        active_issue_id: int | None = state.get("active_issue_id")
        active_newspaper_name: str | None = state.get("active_newspaper_name")
        active_issue_date: str | None = state.get("active_issue_date")

        for call in plan:
            t_start = time.monotonic()
            name = call.get("tool_name")
            args = call.get("arguments", {})
            hits_count = 0

            try:
                if name == "hybrid_search":
                    filters = None
                    np_id = args.get("newspaper_id")
                    np_name = args.get("newspaper_name")
                    if not np_id and np_name:
                        try:
                            from sqlalchemy import select
                            async with self._session_factory() as db:
                                stmt = select(Newspaper.id).where(Newspaper.name.ilike(f"%{np_name}%")).limit(1)
                                res = await db.execute(stmt)
                                np_id = res.scalar_one_or_none()
                        except Exception as e:
                            logger.warning("Could not resolve newspaper_id by name in hybrid_search", extra={"name": np_name, "error": str(e)})

                    filter_keys = (
                        "newspaper_id",
                        "newspaper_name",
                        "date_from",
                        "date_to",
                        "page_filter",
                        "page_number",
                        "printed_page",
                    )
                    has_filter = any(k in args for k in filter_keys) or (np_id is not None)
                    if has_filter:
                        p_filt = args.get("page_filter") or args.get("printed_page")
                        p_num = args.get("page_number")
                        if p_filt and not p_num and str(p_filt).isdigit():
                            p_num = int(p_filt)
                        filters = SearchFilter(
                            newspaper_id=np_id,
                            date_from=args.get("date_from"),
                            date_to=args.get("date_to"),
                            page_number=p_num,
                            printed_page=str(p_filt) if p_filt else None,
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
                                "issue_id": hr.issue_id,
                                "headline": hr.headline,
                                "newspaper_name": hr.newspaper_name,
                                "issue_date": hr.issue_date,
                                "pages": hr.pages,
                                "bboxes": hr.bboxes,
                                "printed_pages": hr.printed_pages,
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
                                "issue_id": er.issue_id,
                                "headline": er.headline,
                                "newspaper_name": er.newspaper_name,
                                "issue_date": er.issue_date,
                                "pages": er.pages,
                                "bboxes": er.bboxes,
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
                                    "issue_id": m.issue_id,
                                    "headline": m.headline,
                                    "newspaper_name": g.newspaper_name,
                                    "issue_date": g.date,
                                    "pages": m.pages,
                                    "bboxes": m.bboxes,
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
                        summary_str = f"Mention Trends for '{args.get('term')}': " + ", ".join(
                            trend_items
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
                    elif analysis_type == "issue_summary":
                        page_filter = args.get("page_filter")
                        
                        # Context inheritance guardrails:
                        # 1. Do NOT inherit newspaper_name from chat history if query is cross-newspaper comparison
                        #    or if this tool call explicitly targets all newspapers on an issue_date.
                        # 2. Do NOT inherit newspaper_name or issue_id if the tool call specifies an issue_date
                        #    that differs from the active_issue_date in history.
                        is_comparative = (
                            state.get("archetype") == "cross_newspaper_comparison"
                            or any(w in str(state.get("query", "")).lower() for w in ["all available", "all newspaper", "across newspaper", "both newspaper", "different newspaper"])
                        )
                        date_mismatch = bool(args.get("issue_date") and active_issue_date and args.get("issue_date") != active_issue_date)
                        target_all_on_date = bool(args.get("issue_date") and not args.get("newspaper_name"))
                        inherit_history = not is_comparative and not date_mismatch and not target_all_on_date

                        np_arg = args.get("newspaper_name") or (active_newspaper_name if inherit_history else None)
                        iss_d_arg = args.get("issue_date") or active_issue_date
                        iss_id_arg = args.get("issue_id") or (active_issue_id if inherit_history else None)

                        # Multi-issue mode: when issue_date is specified without newspaper_name,
                        # retrieve ALL issues on that date for cross-newspaper comparison
                        is_multi_issue_date = (
                            iss_d_arg
                            and not np_arg
                            and not iss_id_arg
                        )

                        if is_multi_issue_date:
                            from sqlalchemy import select as sa_select
                            from sqlalchemy.orm import selectinload as sa_selectinload
                            async with self._session_factory() as db:
                                stmt = (
                                    sa_select(Issue)
                                    .where(Issue.issue_date == iss_d_arg)
                                    .options(
                                        sa_selectinload(Issue.newspaper),
                                        sa_selectinload(Issue.pages),
                                    )
                                    .order_by(Issue.id)
                                )
                                res = await db.execute(stmt)
                                date_issues = res.scalars().all()

                            if date_issues:
                                for di in date_issues:
                                    summary = await self._sql_analytics.get_issue_summary(
                                        issue_id=di.id,
                                        query=args.get("query", state["query"]),
                                        page_filter=page_filter,
                                        exclude_page_filter=args.get("exclude_page_filter"),
                                        category_filter=args.get("category_filter"),
                                    )
                                    if "error" in summary:
                                        continue

                                    total_arts = summary.get("total_articles", 0)
                                    total_pgs = summary.get("total_pages", 0)
                                    sec_breakdown = ", ".join(
                                        f"{k}: {v}" for k, v in summary.get("section_breakdown", {}).items()
                                    )
                                    articles_list = summary.get("articles", [])
                                    hits_count += total_arts

                                    manifest_lines = []
                                    for idx, a in enumerate(articles_list[:30], 1):
                                        pr_page = str(a.get("printed_page") or "").strip()
                                        pg_num = a.get("page_number", 1)
                                        if (
                                            pr_page
                                            and not pr_page.startswith("Unnumbered")
                                            and not pr_page.startswith("PDF p.")
                                            and pr_page != str(pg_num)
                                        ):
                                            folio_info = f"Page {pr_page} (PDF p.{pg_num})"
                                        else:
                                            folio_info = f"Page {pg_num}"
                                        author_info = (
                                            f" by {a['byline_author']}" if a.get("byline_author") else ""
                                        )
                                        manifest_lines.append(
                                            f'{idx}. [{a["section"]}] "{a["headline"]}" '
                                            f"({folio_info}{author_info}, {a['word_count']} words)"
                                        )
                                    manifest_text = "\n".join(manifest_lines)

                                    np_title = summary.get("newspaper", "Archive")
                                    iss_d = summary.get("issue_date", "")
                                    summary_str = (
                                        f"=== RELATIONAL ARCHIVE MANIFEST FOR {np_title} "
                                        f"({iss_d}) ===\n"
                                        f"• Total Articles Ingested: {total_arts}\n"
                                        f"• Total Issue Pages: {total_pgs}\n"
                                        f"• Sections Breakdown: {sec_breakdown}\n\n"
                                        f"Article Manifest:\n{manifest_text}"
                                    )
                                    evidence_items.append(
                                        {
                                            "article_id": 0,
                                            "headline": (
                                                f"Issue Manifest: {np_title} ({iss_d})"
                                            ),
                                            "newspaper_name": np_title,
                                            "issue_date": iss_d,
                                            "pages": [1],
                                            "snippet": summary_str,
                                            "prominence_score": 1.0,
                                            "source_tool": "sql_analytics",
                                        }
                                    )
                            else:
                                evidence_items.append(
                                    {
                                        "article_id": 0,
                                        "headline": f"No issues found for date {iss_d_arg}",
                                        "newspaper_name": "Archive",
                                        "issue_date": iss_d_arg,
                                        "pages": [1],
                                        "snippet": f"⚠️ No newspaper issues were found in the archive for date {iss_d_arg}.",
                                        "prominence_score": 1.0,
                                        "source_tool": "sql_analytics",
                                    }
                                )
                        else:
                            # Standard single-issue summary
                            summary = await self._sql_analytics.get_issue_summary(
                                newspaper_name=np_arg,
                                issue_date=iss_d_arg,
                                issue_id=iss_id_arg,
                                page_filter=page_filter,
                                exclude_page_filter=args.get("exclude_page_filter"),
                                category_filter=args.get("category_filter"),
                                query=args.get("query", state["query"]),
                            )
                            if "error" in summary:
                                summary_str = f"⚠️ {summary['error']}"
                                hits_count = 0
                            else:
                                active_issue_id = summary.get("issue_id")
                                active_newspaper_name = summary.get("newspaper")
                                active_issue_date = summary.get("issue_date")
                                total_arts = summary.get("total_articles", 0)
                                total_pgs = summary.get("total_pages", 0)
                                sec_breakdown = ", ".join(
                                    f"{k}: {v}" for k, v in summary.get("section_breakdown", {}).items()
                                )
                                articles_list = summary.get("articles", [])
                                hits_count = total_arts

                                manifest_lines = []
                                for idx, a in enumerate(articles_list[:50], 1):
                                    pr_page = str(a.get("printed_page") or "").strip()
                                    pg_num = a.get("page_number", 1)
                                    if (
                                        pr_page
                                        and not pr_page.startswith("Unnumbered")
                                        and not pr_page.startswith("PDF p.")
                                        and pr_page != str(pg_num)
                                    ):
                                        folio_info = f"Page {pr_page} (PDF p.{pg_num})"
                                    else:
                                        folio_info = f"Page {pg_num}"
                                    author_info = (
                                        f" by {a['byline_author']}" if a.get("byline_author") else ""
                                    )
                                    manifest_lines.append(
                                        f'{idx}. [{a["section"]}] "{a["headline"]}" '
                                        f"({folio_info}{author_info}, {a['word_count']} words)"
                                    )
                                manifest_text = "\n".join(manifest_lines)

                                np_title = summary.get("newspaper", "Archive")
                                iss_d = summary.get("issue_date", "")
                                if page_filter:
                                    no_arts_msg = (
                                        "No editorial articles found on this page "
                                        "(Page may be a full-page advertisement, "
                                        "photo gallery, or unindexed wrap)."
                                    )
                                    body_content = manifest_text if manifest_lines else no_arts_msg
                                    summary_str = (
                                        f"=== RELATIONAL ARCHIVE MANIFEST FOR {np_title} "
                                        f"({iss_d}) - PAGE {page_filter} ===\n"
                                        f"• Total Articles on Page {page_filter}: {total_arts}\n"
                                        f"• Total Issue Pages: {total_pgs}\n\n"
                                        f"Articles on Page {page_filter}:\n{body_content}"
                                    )
                                else:
                                    summary_str = (
                                        f"=== RELATIONAL ARCHIVE MANIFEST FOR {np_title} "
                                        f"({iss_d}) ===\n"
                                        f"• Total Articles Ingested: {total_arts}\n"
                                        f"• Total Issue Pages: {total_pgs}\n"
                                        f"• Sections Breakdown: {sec_breakdown}\n\n"
                                        f"Article Manifest:\n{manifest_text}"
                                    )

                            evidence_items.append(
                                {
                                    "article_id": 0,
                                    "headline": (
                                        f"Issue Manifest: {summary.get('newspaper', 'Archive')} "
                                        f"({summary.get('issue_date', '')})"
                                    ),
                                    "newspaper_name": summary.get("newspaper", "Archive"),
                                    "issue_date": summary.get("issue_date", "Overview"),
                                    "pages": [1],
                                    "snippet": summary_str,
                                    "prominence_score": 1.0,
                                    "source_tool": "sql_analytics",
                                }
                            )

                    elif analysis_type == "count_articles":
                        count_res = await self._sql_analytics.count_articles(
                            newspaper_name=args.get("newspaper_name"),
                            issue_date=args.get("issue_date"),
                            section=args.get("section") or args.get("category_filter"),
                            article_type=args.get("article_type"),
                        )
                        c_val = count_res.get("count", 0)
                        hits_count = c_val
                        filt_info = ", ".join(f"{k}: {v}" for k, v in count_res.get("filters", {}).items() if v)
                        summary_str = (
                            f"=== RELATIONAL ARTICLE COUNT AUDIT ===\n"
                            f"• Total Matching Articles: {c_val}\n"
                            f"• Active Filters: {filt_info or 'None (Total Ingested)'}\n"
                        )
                        evidence_items.append(
                            {
                                "article_id": 0,
                                "headline": f"Article Count Analysis: {c_val} articles found",
                                "newspaper_name": args.get("newspaper_name") or "Archive",
                                "issue_date": args.get("issue_date") or "Overview",
                                "pages": [1],
                                "snippet": summary_str,
                                "prominence_score": 1.0,
                                "source_tool": "sql_analytics",
                            }
                        )
                    elif analysis_type == "topic_distribution":
                        topics_dist = await self._sql_analytics.get_topic_distribution()
                        hits_count = len(topics_dist)
                        top_lines = [
                            f"• [{t['section']} / {t['article_type']}]: {t['count']} articles (avg prominence: {t['avg_prominence']})"
                            for t in topics_dist[:10]
                        ]
                        summary_str = "=== TOPIC & SECTION DISTRIBUTION ===\n" + "\n".join(top_lines)
                        evidence_items.append(
                            {
                                "article_id": 0,
                                "headline": "Archive Topic & Section Breakdown",
                                "newspaper_name": "Aggregated Archive Analytics",
                                "issue_date": "Overview",
                                "pages": [1],
                                "snippet": summary_str,
                                "prominence_score": 1.0,
                                "source_tool": "sql_analytics",
                            }
                        )
                    elif analysis_type == "frontpage_ratio":
                        ratio_res = await self._sql_analytics.get_frontpage_prominence_ratio()
                        hits_count = ratio_res.get("total_articles", 0)
                        summary_str = (
                            f"=== FRONTPAGE PROMINENCE RATIO ===\n"
                            f"• Total Articles: {ratio_res.get('total_articles')}\n"
                            f"• Frontpage Articles (Page 1): {ratio_res.get('frontpage_articles')}\n"
                            f"• Frontpage Ratio: {round(ratio_res.get('frontpage_ratio', 0) * 100, 2)}%\n"
                        )
                        evidence_items.append(
                            {
                                "article_id": 0,
                                "headline": "Frontpage Prominence Analysis",
                                "newspaper_name": "Aggregated Archive Analytics",
                                "issue_date": "Overview",
                                "pages": [1],
                                "snippet": summary_str,
                                "prominence_score": 1.0,
                                "source_tool": "sql_analytics",
                            }
                        )
                    elif analysis_type == "coverage_comparison":
                        cov_matrix = await self._coverage_analyzer.generate_coverage_matrix(
                            query_or_event=args.get("query", state["query"]),
                            target_date=args.get("target_date") or args.get("issue_date"),
                        )
                        hits_count = cov_matrix.covered_count
                        lines = [
                            f"=== 3-TIER COVERAGE RECONCILIATION MATRIX: '{cov_matrix.target_query_or_event}' ===",
                            f"• Total Publications Audited: {cov_matrix.total_publications}",
                            f"• Confirmed Coverage: {cov_matrix.covered_count}",
                            f"• Confirmed Omissions (Not Found): {cov_matrix.not_found_count}",
                            f"• Uncertain / Borderline: {cov_matrix.uncertain_count}",
                            f"• Processing Errors / Incomplete: {cov_matrix.processing_error_count}\n",
                        ]
                        for pub_name, rep in cov_matrix.reports.items():
                            hls = f" (Headlines: {', '.join(rep.matched_headlines[:2])})" if rep.matched_headlines else ""
                            lines.append(f"• {pub_name}: [{rep.status}] Confidence {round(rep.confidence * 100, 1)}%{hls} - {rep.audit_notes}")
                        evidence_items.append(
                            {
                                "article_id": 0,
                                "headline": f"Coverage Audit: {cov_matrix.target_query_or_event}",
                                "newspaper_name": "Multi-Newspaper Audit",
                                "issue_date": "Comparative Matrix",
                                "pages": [1],
                                "snippet": "\n".join(lines),
                                "prominence_score": 1.0,
                                "source_tool": "sql_analytics",
                            }
                        )
                    elif analysis_type == "coverage_difference":
                        src_np = args.get("newspaper_name") or args.get("source_newspaper")
                        cmp_np = args.get("comparison_newspaper")
                        iss_dt = args.get("issue_date") or args.get("target_date") or active_issue_date

                        diff_res = await self._sql_analytics.get_newspaper_coverage_difference(
                            source_newspaper=src_np,
                            comparison_newspaper=cmp_np,
                            issue_date=iss_dt,
                        )
                        if "error" in diff_res:
                            evidence_items.append(
                                {
                                    "article_id": 0,
                                    "headline": f"Coverage Difference Error: {diff_res['error']}",
                                    "newspaper_name": src_np or "Archive",
                                    "issue_date": iss_dt or "Overview",
                                    "pages": [1],
                                    "snippet": f"⚠️ {diff_res['error']}",
                                    "prominence_score": 1.0,
                                    "source_tool": "sql_analytics",
                                }
                            )
                        else:
                            exclusives = diff_res.get("exclusive_articles", [])
                            hits_count = len(exclusives)

                            ex_lines = []
                            for idx, ex in enumerate(exclusives[:40], 1):
                                p_str = f"Page {ex.get('page_number')} (PDF Page {ex.get('page_number')})"
                                ex_lines.append(
                                    f"{idx}. [{p_str}] ({ex.get('section')}) \"{ex.get('headline')}\""
                                )

                            diff_manifest_text = "\n".join(ex_lines)
                            summary_str = (
                                f"=== VERIFIED EXCLUSIVE COVERAGE: {diff_res['source_newspaper']} "
                                f"({diff_res['issue_date']}) NOT PRESENT IN {diff_res['comparison_newspaper']} ===\n"
                                f"• Total Source Articles: {diff_res['total_source_articles']}\n"
                                f"• Total Comparison Articles: {diff_res['total_comparison_articles']}\n"
                                f"• Verified Exclusive Articles to {diff_res['source_newspaper']}: {diff_res['exclusive_count']}\n"
                                f"• Shared Cross-Newspaper Stories: {diff_res['shared_count']}\n\n"
                                f"Exclusive Articles Manifest:\n{diff_manifest_text}"
                            )
                            evidence_items.append(
                                {
                                    "article_id": 0,
                                    "headline": (
                                        f"Verified Exclusive Articles: {diff_res['source_newspaper']} vs {diff_res['comparison_newspaper']}"
                                    ),
                                    "newspaper_name": diff_res["source_newspaper"],
                                    "issue_date": diff_res["issue_date"],
                                    "pages": [1],
                                    "snippet": summary_str,
                                    "prominence_score": 1.0,
                                    "source_tool": "sql_analytics",
                                }
                            )

                elif name == "coverage_analysis":
                    cov_matrix = await self._coverage_analyzer.generate_coverage_matrix(
                        query_or_event=args.get("query", state["query"]),
                        target_date=args.get("target_date"),
                    )
                    hits_count = cov_matrix.covered_count
                    lines = [
                        f"=== 3-TIER COVERAGE RECONCILIATION MATRIX: '{cov_matrix.target_query_or_event}' ===",
                        f"• Total Publications Audited: {cov_matrix.total_publications}",
                        f"• Confirmed Coverage: {cov_matrix.covered_count}",
                        f"• Confirmed Omissions (Not Found): {cov_matrix.not_found_count}",
                        f"• Uncertain / Borderline: {cov_matrix.uncertain_count}",
                        f"• Processing Errors / Incomplete: {cov_matrix.processing_error_count}\n",
                    ]
                    for pub_name, rep in cov_matrix.reports.items():
                        hls = f" (Headlines: {', '.join(rep.matched_headlines[:2])})" if rep.matched_headlines else ""
                        lines.append(f"• {pub_name}: [{rep.status}] Confidence {round(rep.confidence * 100, 1)}%{hls} - {rep.audit_notes}")
                    evidence_items.append(
                        {
                            "article_id": 0,
                            "headline": f"Coverage Matrix: {cov_matrix.target_query_or_event}",
                            "newspaper_name": "Multi-Newspaper Audit",
                            "issue_date": "Comparative Matrix",
                            "pages": [1],
                            "snippet": "\n".join(lines),
                            "prominence_score": 1.0,
                            "source_tool": "coverage_analysis",
                        }
                    )

                elif name == "web_search":
                    web_results = await self._web_search.search(
                        query=args.get("query", state["query"]),
                        num_results=args.get("num_results", 5),
                    )
                    hits_count = len(web_results)
                    for wr in web_results:
                        evidence_items.append(
                            {
                                "article_id": 0,
                                "headline": wr.title,
                                "newspaper_name": wr.source,
                                "issue_date": wr.published_date or "Live Web",
                                "pages": [1],
                                "snippet": wr.snippet,
                                "url": wr.url,
                                "is_web": True,
                                "prominence_score": 0.8,
                                "source_tool": "web_search",
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
            "active_issue_id": active_issue_id,
            "active_newspaper_name": active_newspaper_name,
            "active_issue_date": active_issue_date,
        }

    async def _evaluate_and_fallback_node(self, state: AgentState) -> dict[str, Any]:
        """Corrective RAG (CRAG) Node: Grade retrieval quality and execute corrective fallback if needed."""
        evidence = state.get("evidence_items", [])
        archetype = state.get("archetype", "factual_lookup")
        tool_records = list(state.get("tool_executions", []))

        # Skip fallback evaluation for meta queries or user clarification states
        if archetype in ("clarification_needed", "conversational_meta_query"):
            return {"evidence_items": evidence, "tool_executions": tool_records}

        # Stop-words to exclude from core query terms
        stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "once", "here", "there", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "can",
            "will", "just", "should", "now", "tell", "what", "which", "who",
            "whom", "this", "that", "these", "those", "have", "has", "had",
            "give", "details", "information", "report", "news", "articles", "anything",
            "something", "know", "find", "show", "read", "say", "said",
        }
        raw_query_words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", state["query"].lower())
        query_tokens = [w for w in raw_query_words if w not in stop_words]

        def _stem(word: str) -> str:
            w = word.lower()
            if len(w) > 4 and w.endswith("es"):
                return w[:-2]
            if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
                return w[:-1]
            if len(w) > 5 and w.endswith("ing"):
                return w[:-3]
            if len(w) > 4 and w.endswith("ed"):
                return w[:-2]
            return w

        def _score_relevance(item: dict[str, Any]) -> float:
            if not query_tokens:
                return 1.0
            hl = (item.get("headline") or "").lower()
            snip = (item.get("snippet") or item.get("full_text") or item.get("summary") or "").lower()
            combined = f"{hl} {snip}"
            corpus_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", combined))
            corpus_stems = {_stem(w) for w in corpus_words}

            matches = 0.0
            for qt in query_tokens:
                stem = _stem(qt)
                if qt in hl or stem in hl:
                    matches += 2.0
                elif qt in combined or stem in corpus_stems:
                    matches += 1.0

            return matches / max(1.0, float(len(query_tokens)))

        # For macro-structural tools or cross-newspaper comparative archetypes,
        # structured manifests and audit matrices must never be thrown away by lexical query matching.
        def _is_structural_or_macro_evidence(item: dict[str, Any]) -> bool:
            return (
                archetype in ("cross_newspaper_comparison", "quantitative_trend")
                or item.get("source_tool") in ("sql_analytics", "coverage_analysis")
                or item.get("article_id") == 0
            )

        # Retain evidence items that have positive relevance to query tokens OR are structural manifests
        filtered_evidence = [
            item for item in evidence
            if _is_structural_or_macro_evidence(item) or _score_relevance(item) > 0.0
        ]
        filtered_evidence.sort(key=lambda it: (1.0 if _is_structural_or_macro_evidence(it) else _score_relevance(it)), reverse=True)

        # Check if retrieval yielded grounded items matching query
        has_grounded_content = bool(
            filtered_evidence and any(len((item.get("snippet") or "").strip()) >= 20 for item in filtered_evidence)
        )

        if not has_grounded_content:
            logger.info("CRAG triggered: 0 high-confidence articles retrieved, attempting corrective fallback")
            t_start = time.monotonic()
            fallback_items: list[dict[str, Any]] = list(filtered_evidence)

            # 1. Fallback: Entity Graph / Taxonomy Search if a prominent entity is detected in the query
            query_terms = [w.strip() for w in state["query"].split() if len(w) > 3 and w[0].isupper()]
            if query_terms:
                ent_target = query_terms[0]
                ent_res = await self._entity_search.search_by_entity(
                    entity_name=ent_target,
                    top_k=5,
                )
                if ent_res:
                    for er in ent_res:
                        fallback_items.append(
                            {
                                "article_id": er.article_id,
                                "issue_id": er.issue_id,
                                "headline": er.headline,
                                "newspaper_name": er.newspaper_name,
                                "issue_date": er.issue_date,
                                "pages": er.pages,
                                "bboxes": er.bboxes,
                                "snippet": f"[CRAG Entity Fallback - {er.entity_name}]: {er.summary}",
                                "prominence_score": er.prominence_score,
                                "source_tool": "crag_entity_fallback",
                            }
                        )
                    dur_ms = round((time.monotonic() - t_start) * 1000)
                    tool_records.append(
                        ToolExecutionRecord(
                            tool_name="crag_entity_fallback",
                            tool_input={"entity_name": ent_target},
                            results_count=len(ent_res),
                            execution_time_ms=dur_ms,
                        )
                    )

            # 2. Fallback: Live Web Search if web search is enabled and archive is empty
            if not fallback_items and state.get("enable_web_search", False):
                t_web = time.monotonic()
                web_res = await self._web_search.search(
                    query=state["query"],
                    num_results=4,
                )
                if web_res:
                    for wr in web_res:
                        fallback_items.append(
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
                        )
                    dur_ms = round((time.monotonic() - t_web) * 1000)
                    tool_records.append(
                        ToolExecutionRecord(
                            tool_name="crag_web_fallback",
                            tool_input={"query": state["query"]},
                            results_count=len(web_res),
                            execution_time_ms=dur_ms,
                        )
                    )

            return {
                "evidence_items": fallback_items,
                "tool_executions": tool_records,
            }

        return {"evidence_items": filtered_evidence, "tool_executions": tool_records}

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
    ) -> AgentState:
        """Execute the complete agentic query cycle with caching and metrics."""
        t0 = time.monotonic()
        history = chat_history or []

        # 1. Deterministic Redis Cache Check
        cache_key = compute_query_cache_key(
            query=f"{query}__web_{enable_web_search}",
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
            # Reconstruct typed AgentState from cached dict
            return cached_result  # type: ignore[return-value]

        active_ctx = extract_active_issue_from_history(history, current_query=query)
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
