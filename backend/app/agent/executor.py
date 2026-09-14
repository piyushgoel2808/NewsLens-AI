"""Tool Execution Engine for NewsLens-AI Agentic Workflow.

Provides unified dispatch, concurrent execution, parameter sanitization, and standardized
evidence formatting across all broadsheet retrieval tools with zero dynamic imports.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    is_archive_wide_newspaper_query,
)
from app.agent.state import AgentState, ToolExecutionRecord
from app.agent.tool_maker import ToolMaker
from app.core.logging import get_logger
from app.models.article import Photo
from app.retrieval.coverage_analyzer import CoverageAnalyzer, CoverageMatrix
from app.retrieval.entity_filter import EntitySearchEngine
from app.retrieval.formatters import (
    format_coverage_difference_snippet,
    format_coverage_matrix_snippet,
    format_issue_manifest,
    format_shared_coverage_snippet,
)
from app.retrieval.hybrid_search import HybridSearchEngine, SearchFilter
from app.retrieval.sanitizer import repair_text_ligatures
from app.retrieval.sql_analytics import SQLAnalyticsEngine, sanitize_headline
from app.agent.sql_dispatcher import SQLAnalyticsDispatcher
from app.retrieval.timeline_builder import TimelineBuilder
from app.retrieval.visual_inspector import VisualInspectionEngine
from app.retrieval.web_search import WebSearchEngine

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ToolExecutor Engine
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Concurrently executes planned tool calls and normalizes broadsheet evidence items."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        hybrid_search: HybridSearchEngine,
        entity_search: EntitySearchEngine,
        timeline_builder: TimelineBuilder,
        sql_analytics: SQLAnalyticsEngine,
        coverage_analyzer: CoverageAnalyzer,
        web_search: WebSearchEngine,
        tool_maker: ToolMaker | None = None,
        visual_inspector: VisualInspectionEngine | None = None,
        sql_dispatcher: SQLAnalyticsDispatcher | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._hybrid_search = hybrid_search
        self._entity_search = entity_search
        self._timeline_builder = timeline_builder
        self._sql_analytics = sql_analytics
        self._coverage_analyzer = coverage_analyzer
        self._web_search = web_search
        self._tool_maker = tool_maker
        self._visual_inspector = visual_inspector or VisualInspectionEngine(session_factory=session_factory)
        self._sql_dispatcher = sql_dispatcher or SQLAnalyticsDispatcher(
            sql_analytics=sql_analytics,
            coverage_analyzer=coverage_analyzer,
        )

    async def execute_tools(
        self,
        plan: list[dict[str, Any]],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], list[ToolExecutionRecord], dict[str, Any]]:
        """Concurrently execute all planned tool calls with error isolation."""
        if not plan:
            return [], [], {}

        evidence_items: list[dict[str, Any]] = []
        tool_records: list[ToolExecutionRecord] = []
        context_updates: dict[str, Any] = {}

        tasks = [self.execute_single_tool(call, state) for call in plan]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for call, res in zip(plan, results, strict=True):
            if isinstance(res, BaseException):
                logger.error(
                    f"Error in parallel execution of tool '{call.get('tool_name')}'",
                    extra={"error": str(res)},
                )
                tool_records.append(
                    ToolExecutionRecord(
                        tool_name=call.get("tool_name") or "unknown",
                        tool_input=call.get("arguments", {}),
                        results_count=0,
                        execution_time_ms=0,
                    )
                )
            else:
                items, record, ctx_updates = res
                evidence_items.extend(items)
                tool_records.append(record)
                context_updates.update({k: v for k, v in ctx_updates.items() if v is not None})

        return evidence_items, tool_records, context_updates

    async def execute_single_tool(
        self,
        call: dict[str, Any],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], ToolExecutionRecord, dict[str, Any]]:
        """Execute an individual planned tool call with resilience and adaptive fallbacks."""
        t_start = time.monotonic()
        name = call.get("tool_name") if isinstance(call, dict) else getattr(call, "tool_name", None)
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        hits_count = 0
        evidence_items: list[dict[str, Any]] = []
        context_updates: dict[str, Any] = {}

        active_issue_id: int | None = state.get("active_issue_id")
        active_newspaper_name: str | None = state.get("active_newspaper_name")
        active_issue_date: str | None = state.get("active_issue_date")

        try:
            if name == "hybrid_search":
                evidence_items, hits_count = await self._execute_hybrid_search(args)

            elif name == "entity_search":
                evidence_items, hits_count = await self._execute_entity_search(args)

            elif name == "timeline_builder":
                evidence_items, hits_count = await self._execute_timeline_builder(args)

            elif name == "sql_analytics":
                evidence_items, hits_count, context_updates = await self._execute_sql_analytics(
                    args=args,
                    state=state,
                    active_issue_id=active_issue_id,
                    active_newspaper_name=active_newspaper_name,
                    active_issue_date=active_issue_date,
                )

            elif name == "coverage_analysis":
                evidence_items, hits_count = await self._execute_coverage_analysis(args, state)

            elif name == "web_search":
                evidence_items, hits_count = await self._execute_web_search(args, state)

            elif name == "inspect_visual_asset":
                evidence_items, hits_count = await self._execute_inspect_visual_asset(args, state)

            elif name == "dynamic_analysis":
                evidence_items, hits_count = await self._execute_dynamic_analysis(args, state)

        except Exception as e:
            logger.error(f"Error executing tool '{name}'", extra={"error": str(e)})

        dur_ms = round((time.monotonic() - t_start) * 1000)
        tool_record = ToolExecutionRecord(
            tool_name=name or "unknown",
            tool_input=args,
            results_count=hits_count,
            execution_time_ms=dur_ms,
        )
        return evidence_items, tool_record, context_updates

    # -----------------------------------------------------------------------
    # Tool-Specific Implementations
    # -----------------------------------------------------------------------

    async def _execute_hybrid_search(
        self,
        args: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        filters: SearchFilter | None = None
        np_id = args.get("newspaper_id")
        np_name = args.get("newspaper_name")
        if not np_id and np_name:
            try:
                np_id = await self._sql_analytics.get_newspaper_id_by_name(np_name)
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
            "category_filter",
            "category_name",
        )
        has_filter = any(k in args for k in filter_keys) or (np_id is not None)
        if has_filter:
            p_filt = args.get("page_filter") or args.get("printed_page")
            p_num = args.get("page_number")
            if p_filt and not p_num and str(p_filt).isdigit():
                p_num = int(p_filt)
            filters = SearchFilter(
                newspaper_id=np_id,
                newspaper_name=np_name,
                date_from=args.get("date_from"),
                date_to=args.get("date_to"),
                page_number=p_num,
                printed_page=str(p_filt) if p_filt else None,
                category_name=args.get("category_filter") or args.get("category_name"),
            )

        hybrid_results = await self._hybrid_search.search(
            query=args.get("query", ""),
            top_k=args.get("top_k", 6),
            filters=filters,
        )
        hits_count = len(hybrid_results)

        # Adaptive Fallback: If filtered search returned 0 hits and category filter was active,
        # retry unconstrained search without the category filter
        if hits_count == 0 and filters and filters.category_name:
            logger.info("Hybrid search with category '%s' returned 0 hits, retrying without category filter", filters.category_name)
            filters_no_cat = SearchFilter(
                newspaper_id=filters.newspaper_id,
                newspaper_name=filters.newspaper_name,
                date_from=filters.date_from,
                date_to=filters.date_to,
                page_number=filters.page_number,
                printed_page=filters.printed_page,
                category_name=None,
            )
            fallback_results = await self._hybrid_search.search(
                query=args.get("query", ""),
                top_k=args.get("top_k", 6),
                filters=filters_no_cat,
            )
            if fallback_results:
                hybrid_results = fallback_results
                hits_count = len(fallback_results)

        items: list[dict[str, Any]] = []
        for hr in hybrid_results:
            clean_hl, eff_byline = sanitize_headline(
                hr.headline,
                subheadline=hr.subheadline,
                byline_author=hr.byline_author,
                snippet=hr.snippet,
            )
            clean_hl = repair_text_ligatures(clean_hl)
            clean_snip = repair_text_ligatures(hr.snippet)
            items.append(
                {
                    "article_id": hr.article_id,
                    "issue_id": hr.issue_id,
                    "headline": clean_hl,
                    "byline_author": eff_byline or hr.byline_author,
                    "newspaper_name": hr.newspaper_name,
                    "issue_date": hr.issue_date,
                    "pages": hr.pages,
                    "bboxes": hr.bboxes,
                    "printed_pages": hr.printed_pages,
                    "snippet": clean_snip,
                    "prominence_score": hr.prominence_score,
                    "source_tool": "hybrid_search",
                    "section": hr.section,
                    "word_count": getattr(hr, "word_count", 0),
                    "photos": hr.photos,
                    "has_visual_data": hr.has_visual_data,
                    "visual_type": hr.visual_type,
                    "parent_article_text": hr.parent_article_text,
                }
            )
        return items, hits_count

    async def _execute_entity_search(
        self,
        args: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        entity_results = await self._entity_search.search_by_entity(
            entity_name=args.get("entity_name"),
            entity_type=args.get("entity_type"),
            top_k=args.get("top_k", 10),
        )
        items: list[dict[str, Any]] = []
        for er in entity_results:
            snip_str = (
                f"Entity [{er.entity_name} ({er.entity_type}) - "
                f"Salience {er.salience_score}]: {er.summary}"
            )
            items.append(
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
        return items, len(entity_results)

    async def _execute_timeline_builder(
        self,
        args: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        tl_result = await self._timeline_builder.build_timeline(
            query=args.get("query"),
            limit=args.get("limit", 20),
        )
        items: list[dict[str, Any]] = []
        for g in tl_result.date_groups:
            for m in g.milestones:
                items.append(
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
        return items, tl_result.total_articles

    async def _execute_sql_analytics(
        self,
        args: dict[str, Any],
        state: AgentState,
        active_issue_id: int | None,
        active_newspaper_name: str | None,
        active_issue_date: str | None,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        result = await self._sql_dispatcher.dispatch(
            args=args,
            state=state,
            active_issue_id=active_issue_id,
            active_newspaper_name=active_newspaper_name,
            active_issue_date=active_issue_date,
        )
        if result is not None:
            return result

        # Unsupported or custom analysis type requested by LLM (e.g. count_pages, word_count_by_author)
        # Delegate directly to ToolMaker dynamic code synthesis.
        if self._tool_maker:
            analysis_type = args.get("analysis_type")
            logger.info(
                f"Unsupported sql_analytics analysis_type '{analysis_type}'; delegating to dynamic tool synthesis."
            )
            dyn_items, dyn_hits = await self._execute_dynamic_analysis(args, state)
            return dyn_items, dyn_hits, {}

        return [], 0, {}

    async def _execute_coverage_analysis(
        self,
        args: dict[str, Any],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], int]:
        cov_matrix = await self._coverage_analyzer.generate_coverage_matrix(
            query_or_event=args.get("query", state["query"]),
            target_date=args.get("target_date"),
        )
        items = [
            {
                "article_id": 0,
                "headline": f"Coverage Matrix: {cov_matrix.target_query_or_event}",
                "newspaper_name": "Multi-Newspaper Audit",
                "issue_date": "Comparative Matrix",
                "pages": [1],
                "snippet": format_coverage_matrix_snippet(cov_matrix),
                "prominence_score": 1.0,
                "source_tool": "coverage_analysis",
            }
        ]
        return items, cov_matrix.covered_count

    async def _execute_web_search(
        self,
        args: dict[str, Any],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], int]:
        web_results = await self._web_search.search(
            query=args.get("query", state["query"]),
            num_results=args.get("num_results", 5),
        )
        items = [
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
            for wr in web_results
        ]
        return items, len(web_results)

    async def _execute_inspect_visual_asset(
        self,
        args: dict[str, Any],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], int]:
        """Deep multimodal visual inspection, on-demand VLM extraction, and table transcription."""
        return await self._visual_inspector.inspect_visual_asset(args, state_context=dict(state))

    def _resolve_visual_target_params(
        self,
        args: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Backward-compatible proxy delegating to VisualInspectionEngine."""
        return self._visual_inspector.resolve_visual_target_params(args, dict(state))

    async def _resolve_photos_for_inspection(
        self,
        session: AsyncSession,
        params: dict[str, Any],
        args: dict[str, Any],
        state: AgentState,
    ) -> tuple[list[Photo], list[dict[str, Any]] | None]:
        """Backward-compatible proxy delegating to VisualInspectionEngine."""
        return await self._visual_inspector.resolve_photos_for_inspection(session, params, args, dict(state))


    async def _execute_dynamic_analysis(
        self,
        args: dict[str, Any],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], int]:
        """Execute dynamically synthesized Python analysis tool via ToolMaker."""
        if not self._tool_maker:
            logger.warning("dynamic_analysis tool requested but ToolMaker is not configured")
            return [], 0

        query = args.get("query") or state.get("query", "")
        extracted = state.get("extracted_params") or {}
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

        # Inject pre-normalized parameters so ToolMaker directly receives validated context
        target_date = args.get("issue_date") or extracted.get("issue_date")
        target_np = args.get("newspaper_name") or extracted.get("newspaper_name")
        date_from = args.get("date_from") or extracted.get("date_from")
        date_to = args.get("date_to") or extracted.get("date_to")
        category = args.get("category_filter") or extracted.get("category_filter")

        if target_date:
            context["target_date"] = target_date
            context["issue_date"] = target_date
            context["date"] = target_date
        if target_np:
            context["newspaper_name"] = target_np
            context["newspaper"] = target_np
            context["publication"] = target_np
        if date_from:
            context["date_from"] = date_from
        if date_to:
            context["date_to"] = date_to
        if category:
            context["category"] = category
            context["category_filter"] = category

        model_override = state.get("model_override")
        res = await self._tool_maker.generate_and_execute(
            query=query,
            context=context,
            model_override=model_override,
            attempted_tools=state.get("tool_executions", []),
            gap_diagnosis=state.get("gap_diagnosis"),
        )

        if res.success:
            return res.evidence_items, len(res.evidence_items)

        logger.warning("Dynamic tool generation/execution failed", extra={"error": res.error})
        return [], 0


__all__ = [
    "ToolExecutor",
    "format_coverage_difference_snippet",
    "format_coverage_matrix_snippet",
    "format_issue_manifest",
    "format_shared_coverage_snippet",
]
