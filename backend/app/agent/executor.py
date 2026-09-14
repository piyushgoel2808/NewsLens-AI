"""Tool Execution Engine for NewsLens-AI Agentic Workflow.

Provides unified dispatch, concurrent execution, parameter sanitization, and standardized
evidence formatting across all broadsheet retrieval tools with zero dynamic imports.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.state import AgentState, ToolExecutionRecord
from app.agent.tool_maker import ToolMaker
from app.core.logging import get_logger
from app.retrieval.coverage_analyzer import CoverageAnalyzer, CoverageMatrix
from app.retrieval.entity_filter import EntitySearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine, SearchFilter
from app.retrieval.sanitizer import repair_text_ligatures
from app.retrieval.sql_analytics import SQLAnalyticsEngine, sanitize_headline
from app.retrieval.timeline_builder import TimelineBuilder
from app.retrieval.web_search import WebSearchEngine

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Presentation & Manifest Formatting Helpers
# ---------------------------------------------------------------------------

def format_issue_manifest(
    summary: dict[str, Any],
    page_filter: str | None = None,
    category_filter: str | None = None,
    max_articles: int = 50,
) -> str:
    """Render a unified, human-readable broadsheet manifest string from a structured issue summary."""
    articles_list = summary.get("articles", [])
    total_arts = summary.get("total_articles", 0)
    total_pgs = summary.get("total_pages", 0)
    np_title = summary.get("newspaper", "Archive")
    iss_d = summary.get("issue_date", "")
    sec_breakdown = ", ".join(
        f"{k}: {v}" for k, v in summary.get("section_breakdown", {}).items()
    )

    manifest_lines: list[str] = []
    for idx, a in enumerate(articles_list[:max_articles], 1):
        pg_num = a.get("page_number", 1)
        author_info = f" by {a['byline_author']}" if a.get("byline_author") else ""
        clean_a_hl = repair_text_ligatures(a.get("headline") or "")
        manifest_lines.append(
            f'{idx}. [{a.get("section", "General")}] "{clean_a_hl}" '
            f"(Page {pg_num}{author_info}, {a.get('word_count', 0)} words)"
        )

    manifest_text = "\n".join(manifest_lines)

    if page_filter:
        no_arts_msg = (
            "No editorial articles found on this page "
            "(Page may be a full-page advertisement, "
            "photo gallery, or unindexed wrap)."
        )
        body_content = manifest_text if manifest_lines else no_arts_msg
        return (
            f"=== RELATIONAL ARCHIVE MANIFEST FOR {np_title} "
            f"({iss_d}) - PAGE {page_filter} ===\n"
            f"• Total Articles on Page {page_filter}: {total_arts}\n"
            f"• Total Issue Pages: {total_pgs}\n\n"
            f"Articles on Page {page_filter}:\n{body_content}"
        )

    cat_hdr = f" - CATEGORY: {category_filter}" if category_filter else ""
    return (
        f"=== RELATIONAL ARCHIVE MANIFEST FOR {np_title} "
        f"({iss_d}){cat_hdr} ===\n"
        f"• Total Articles Ingested: {total_arts}\n"
        f"• Total Issue Pages: {total_pgs}\n"
        f"• Sections Breakdown: {sec_breakdown}\n\n"
        f"Article Manifest:\n{manifest_text}"
    )


def format_coverage_matrix_snippet(cov_matrix: CoverageMatrix) -> str:
    """Render a unified 3-tier coverage reconciliation matrix string."""
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
    return "\n".join(lines)


def format_coverage_difference_snippet(diff_res: dict[str, Any], max_articles: int = 40) -> str:
    """Render a verified exclusive coverage difference manifest string."""
    exclusives = diff_res.get("exclusive_articles", [])
    ex_lines: list[str] = []
    for idx, ex in enumerate(exclusives[:max_articles], 1):
        p_str = f"Page {ex.get('page_number', 1)}"
        ex_lines.append(
            f"{idx}. [{p_str}] ({ex.get('section')}) \"{ex.get('headline')}\""
        )
    diff_manifest_text = "\n".join(ex_lines)
    return (
        f"=== VERIFIED EXCLUSIVE COVERAGE: {diff_res.get('source_newspaper')} "
        f"({diff_res.get('issue_date')}) NOT PRESENT IN {diff_res.get('comparison_newspaper')} ===\n"
        f"• Total Source Articles: {diff_res.get('total_source_articles')}\n"
        f"• Total Comparison Articles: {diff_res.get('total_comparison_articles')}\n"
        f"• Verified Exclusive Articles to {diff_res.get('source_newspaper')}: {diff_res.get('exclusive_count')}\n"
        f"• Shared Cross-Newspaper Stories: {diff_res.get('shared_count')}\n\n"
        f"Exclusive Articles Manifest:\n{diff_manifest_text}"
    )


def format_shared_coverage_snippet(shared_res: dict[str, Any], max_articles: int = 40) -> str:
    """Render a verified shared syndicated wire coverage manifest string."""
    shared_stories = shared_res.get("shared_stories", [])
    sh_lines: list[str] = []
    for idx, story in enumerate(shared_stories[:max_articles], 1):
        tier_tag = f"[{story.get('match_tier', 'Match')}]"
        sh_lines.append(
            f"{idx}. {tier_tag} \"{story.get('headline_a')}\" ({story.get('newspaper_a')}, P.{story.get('page_a')}, {story.get('section_a')}) "
            f"↔ \"{story.get('headline_b')}\" ({story.get('newspaper_b')}, P.{story.get('page_b')}, {story.get('section_b')}) "
            f"[Overlap: {story.get('shared_keywords', '')}]"
        )
    shared_manifest_text = "\n".join(sh_lines)
    return (
        f"=== VERIFIED SHARED SYNDICATED WIRE COVERAGE: {shared_res.get('newspaper_a')} "
        f"AND {shared_res.get('newspaper_b')} ({shared_res.get('issue_date')}) ===\n"
        f"• Total {shared_res.get('newspaper_a')} Articles: {shared_res.get('total_newspaper_a_articles')}\n"
        f"• Total {shared_res.get('newspaper_b')} Articles: {shared_res.get('total_newspaper_b_articles')}\n"
        f"• Total Verified Shared Wire Stories: {shared_res.get('shared_count')}\n\n"
        f"Shared Stories Manifest:\n{shared_manifest_text}"
    )


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
    ) -> None:
        self._session_factory = session_factory
        self._hybrid_search = hybrid_search
        self._entity_search = entity_search
        self._timeline_builder = timeline_builder
        self._sql_analytics = sql_analytics
        self._coverage_analyzer = coverage_analyzer
        self._web_search = web_search
        self._tool_maker = tool_maker

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
        analysis_type = args.get("analysis_type")
        items: list[dict[str, Any]] = []
        hits_count = 0
        context_updates: dict[str, Any] = {}

        if analysis_type == "entity_trends":
            trends = await self._sql_analytics.get_entity_mention_trends(entity_name=args.get("term", ""))
            hits_count = len(trends)
            trend_items = [
                f"{t['date']}: {t['article_count']} articles ({t['total_mentions']} mentions)"
                for t in trends[:5]
            ]
            summary_str = f"Mention Trends for '{args.get('term')}': " + ", ".join(trend_items)
            items.append(
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

            is_multi_issue_date = iss_d_arg and not np_arg and not iss_id_arg

            if is_multi_issue_date and iss_d_arg:
                date_issues = await self._sql_analytics.get_issues_by_date(iss_d_arg)
                if date_issues:
                    summaries_collected = []
                    total_found = 0
                    for di in date_issues:
                        summary = await self._sql_analytics.get_issue_summary(
                            issue_id=di.id,
                            query=args.get("query", state["query"]),
                            page_filter=page_filter,
                            exclude_page_filter=args.get("exclude_page_filter"),
                            category_filter=args.get("category_filter"),
                        )
                        if "error" not in summary:
                            summaries_collected.append(summary)
                            total_found += summary.get("total_articles", 0)

                    # Adaptive fallback: if 0 hits with category filter, retry unconstrained
                    if total_found == 0 and args.get("category_filter"):
                        logger.info("SQL analytics category '%s' returned 0 articles across %s; retrying unconstrained", args.get("category_filter"), iss_d_arg)
                        summaries_collected = []
                        for di in date_issues:
                            summary = await self._sql_analytics.get_issue_summary(
                                issue_id=di.id,
                                query=args.get("query", state["query"]),
                                page_filter=page_filter,
                                exclude_page_filter=args.get("exclude_page_filter"),
                                category_filter=None,
                            )
                            if "error" not in summary:
                                summaries_collected.append(summary)
                                total_found += summary.get("total_articles", 0)

                    for summary in summaries_collected:
                        hits_count += summary.get("total_articles", 0)
                        summary_str = format_issue_manifest(
                            summary=summary,
                            page_filter=page_filter,
                            category_filter=args.get("category_filter"),
                            max_articles=30,
                        )
                        items.append(
                            {
                                "article_id": 0,
                                "headline": f"Issue Manifest: {summary.get('newspaper', 'Archive')} ({summary.get('issue_date', '')})",
                                "newspaper_name": summary.get("newspaper", "Archive"),
                                "issue_date": summary.get("issue_date", iss_d_arg),
                                "pages": [1],
                                "snippet": summary_str,
                                "prominence_score": 1.0,
                                "source_tool": "sql_analytics",
                            }
                        )
                else:
                    items.append(
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
                summary = await self._sql_analytics.get_issue_summary(
                    newspaper_name=np_arg,
                    issue_date=iss_d_arg,
                    issue_id=iss_id_arg,
                    page_filter=page_filter,
                    exclude_page_filter=args.get("exclude_page_filter"),
                    category_filter=args.get("category_filter"),
                    query=args.get("query", state["query"]),
                )
                if "error" not in summary and summary.get("total_articles", 0) == 0 and args.get("category_filter"):
                    logger.info("Single issue summary with category '%s' returned 0 articles; retrying unconstrained", args.get("category_filter"))
                    summary = await self._sql_analytics.get_issue_summary(
                        newspaper_name=np_arg,
                        issue_date=iss_d_arg,
                        issue_id=iss_id_arg,
                        page_filter=page_filter,
                        exclude_page_filter=args.get("exclude_page_filter"),
                        category_filter=None,
                        query=args.get("query", state["query"]),
                    )

                if "error" in summary:
                    items.append(
                        {
                            "article_id": 0,
                            "headline": f"Issue Summary Error: {summary['error']}",
                            "newspaper_name": np_arg or "Archive",
                            "issue_date": iss_d_arg or "Overview",
                            "pages": [1],
                            "snippet": f"⚠️ {summary['error']}",
                            "prominence_score": 1.0,
                            "source_tool": "sql_analytics",
                        }
                    )
                else:
                    context_updates["active_issue_id"] = summary.get("issue_id")
                    context_updates["active_newspaper_name"] = summary.get("newspaper")
                    context_updates["active_issue_date"] = summary.get("issue_date")
                    hits_count = summary.get("total_articles", 0)
                    summary_str = format_issue_manifest(
                        summary=summary,
                        page_filter=page_filter,
                        category_filter=args.get("category_filter"),
                        max_articles=50,
                    )
                    items.append(
                        {
                            "article_id": 0,
                            "headline": f"Issue Manifest: {summary.get('newspaper', 'Archive')} ({summary.get('issue_date', '')})",
                            "newspaper_name": summary.get("newspaper", "Archive"),
                            "issue_date": summary.get("issue_date", "Overview"),
                            "pages": [1],
                            "snippet": summary_str,
                            "prominence_score": 1.0,
                            "source_tool": "sql_analytics",
                        }
                    )
                    for a in summary.get("articles", [])[:30]:
                        art_id = a.get("id") or a.get("article_id")
                        if art_id:
                            items.append(
                                {
                                    "article_id": art_id,
                                    "headline": a.get("headline", ""),
                                    "newspaper_name": summary.get("newspaper", "Archive"),
                                    "issue_date": summary.get("issue_date", "Overview"),
                                    "pages": [a.get("page_number", 1)],
                                    "snippet": f"\"{a.get('headline')}\" published in {summary.get('newspaper', 'Archive')} on Page {a.get('page_number', 1)} ({summary.get('issue_date', '')}). Section: {a.get('section', 'General')}, Word count: {a.get('word_count', 0)}.",
                                    "prominence_score": 0.85,
                                    "source_tool": "sql_analytics_manifest",
                                }
                            )

        elif analysis_type in ("count_advertisements", "count_ads", "advertisements", "ad_counts") or (
            analysis_type == "count_articles" and (args.get("article_type") == "advertisement" or "advertis" in str(args.get("section") or "").lower())
        ):
            np_name = args.get("newspaper_name") or active_newspaper_name
            iss_date = args.get("issue_date") or active_issue_date
            ad_res = await self._sql_analytics.count_advertisements(
                newspaper_name=np_name,
                issue_date=iss_date,
                date_from=args.get("date_from"),
                date_to=args.get("date_to"),
                issue_id=args.get("issue_id") or active_issue_id,
            )
            c_val = ad_res.get("count", 0)
            hits_count = c_val
            ads_list = ad_res.get("advertisements", [])
            filt_info = ", ".join(f"{k}: {v}" for k, v in ad_res.get("filters", {}).items() if v)

            lines = [
                "=== RELATIONAL ADVERTISEMENT COUNT AUDIT ===",
                f"• Publication: {np_name or 'Archive'}",
                f"• Issue Date: {iss_date or 'Archive'}",
                f"• Total Matching Advertisements: {c_val}",
                f"• Active Filters: {filt_info or 'None'}",
            ]
            if ads_list:
                lines.append("\nVerified Advertisements in Archive:")
                for ad in ads_list[:20]:
                    lines.append(
                        f"• Page {ad.get('page_number', 1)}: [{ad.get('article_id')}] \"{ad.get('headline')}\" "
                        f"({ad.get('section')}, {ad.get('word_count')} words)"
                    )
            summary_str = "\n".join(lines)

            # 1. Macro overview snippet (article_id: 0)
            items.append(
                {
                    "article_id": 0,
                    "headline": f"Advertisement Count Analysis: {c_val} advertisements found in {np_name or 'Archive'}",
                    "newspaper_name": np_name or "Archive",
                    "issue_date": iss_date or "Overview",
                    "pages": [ad.get("page_number", 1) for ad in ads_list] or [1],
                    "snippet": summary_str,
                    "prominence_score": 1.0,
                    "source_tool": "sql_analytics",
                }
            )

            # 2. Individual article evidence items with real article_id > 0 for citations
            for ad in ads_list:
                items.append(
                    {
                        "article_id": ad["article_id"],
                        "headline": ad["headline"],
                        "newspaper_name": ad["newspaper_name"],
                        "issue_date": ad["issue_date"],
                        "pages": [ad["page_number"]],
                        "snippet": (
                            f"[Advertisement] \"{ad['headline']}\" published on Page {ad['page_number']} "
                            f"in {ad['newspaper_name']} ({ad['issue_date']}). "
                            f"Section: {ad['section']}, Word count: {ad['word_count']}."
                        ),
                        "prominence_score": 0.85,
                        "source_tool": "sql_analytics_advertisement",
                    }
                )

        elif analysis_type in ("count_issues", "issue_counts", "total_issues", "newspaper_availability", "check_availability"):
            np_name = args.get("newspaper_name") or active_newspaper_name
            iss_date = args.get("issue_date") or args.get("date") or active_issue_date
            d_from = args.get("date_from")
            d_to = args.get("date_to")
            iss_res = await self._sql_analytics.count_issues(
                newspaper_name=np_name,
                issue_date=iss_date,
                date_from=d_from,
                date_to=d_to,
            )
            c_val = iss_res.get("count", 0)
            hits_count = c_val
            filt_info = ", ".join(f"{k}: {v}" for k, v in iss_res.get("filters", {}).items() if v)
            target_date_val = iss_res.get("filters", {}).get("issue_date") or iss_date or "Overview"

            if c_val > 0:
                nps_str = ", ".join(iss_res.get("newspapers", [])) or (np_name or "All Newspapers")
                issues_sample = ", ".join(
                    f"{iss.get('newspaper')} ({iss.get('issue_date')})"
                    for iss in iss_res.get("issues", [])[:5]
                )
                summary_str = (
                    f"=== RELATIONAL ISSUE COUNT AUDIT ===\n"
                    f"• Total Matching Issues: {c_val}\n"
                    f"• Target Date: {target_date_val}\n"
                    f"• Newspaper(s): {nps_str}\n"
                    f"• Active Filters: {filt_info or 'None'}\n"
                    f"• Issues Found: {issues_sample}\n"
                )
                hl_text = f"Issue Count Analysis: {c_val} issues found for {target_date_val}"
            else:
                rng = iss_res.get("archive_range")
                rng_str = f"{rng['start']} to {rng['end']}" if rng else "Archive Range Available"
                all_nps = ", ".join(iss_res.get("archive_newspapers", [])[:10])
                summary_str = (
                    f"=== RELATIONAL ISSUE COUNT AUDIT ===\n"
                    f"• Target Date: {target_date_val}\n"
                    f"• Total Matching Issues: 0\n"
                    f"• Newspaper Scope: {np_name or 'All Newspapers'}\n"
                    f"• Verification Status: No newspaper issues are available in the archive for {target_date_val}.\n"
                    f"• Archive Coverage Range: {rng_str}\n"
                    f"• Available Publications in Archive: {all_nps or 'None'}\n"
                )
                hl_text = f"Archive Availability Audit: 0 issues found for {target_date_val}"

            items.append(
                {
                    "article_id": 0,
                    "headline": hl_text,
                    "newspaper_name": np_name or "Archive",
                    "issue_date": target_date_val,
                    "pages": [1],
                    "snippet": summary_str,
                    "prominence_score": 1.0,
                    "source_tool": "sql_analytics",
                    "metadata": {
                        "count": c_val,
                        "target_date": target_date_val,
                        "newspapers": iss_res.get("newspapers", []),
                        "archive_range": iss_res.get("archive_range"),
                        "archive_newspapers": iss_res.get("archive_newspapers", []),
                        "filters": iss_res.get("filters", {}),
                    },
                }
            )

        elif analysis_type == "count_articles":
            count_res = await self._sql_analytics.count_articles(
                newspaper_name=args.get("newspaper_name"),
                issue_date=args.get("issue_date"),
                date_from=args.get("date_from"),
                date_to=args.get("date_to"),
                section=args.get("section") or args.get("category_filter"),
                article_type=args.get("article_type"),
            )
            c_val = count_res.get("count", 0)
            hits_count = c_val
            art_list = count_res.get("articles", [])
            filt_info = ", ".join(f"{k}: {v}" for k, v in count_res.get("filters", {}).items() if v)

            lines = [
                "=== RELATIONAL ARTICLE COUNT AUDIT ===",
                f"• Total Matching Articles: {c_val}",
                f"• Active Filters: {filt_info or 'None (Total Ingested)'}",
            ]
            if art_list:
                lines.append("\nVerified Matching Articles in Archive:")
                for a in art_list[:20]:
                    lines.append(
                        f"• Page {a.get('page_number', 1)}: [{a.get('article_id')}] \"{a.get('headline')}\" "
                        f"({a.get('section')}, {a.get('word_count')} words)"
                    )
            summary_str = "\n".join(lines)

            # 1. Macro overview snippet (article_id: 0)
            items.append(
                {
                    "article_id": 0,
                    "headline": f"Article Count Analysis: {c_val} articles found",
                    "newspaper_name": args.get("newspaper_name") or "Archive",
                    "issue_date": args.get("issue_date") or "Overview",
                    "pages": [a.get("page_number", 1) for a in art_list] or [1],
                    "snippet": summary_str,
                    "prominence_score": 1.0,
                    "source_tool": "sql_analytics",
                }
            )

            # 2. Individual article evidence items with real article_id > 0 for citations
            for a in art_list:
                items.append(
                    {
                        "article_id": a["article_id"],
                        "headline": a["headline"],
                        "newspaper_name": a["newspaper_name"],
                        "issue_date": a["issue_date"],
                        "pages": [a["page_number"]],
                        "snippet": (
                            f"\"{a['headline']}\" published in {a['newspaper_name']} on Page {a['page_number']} "
                            f"({a['issue_date']}). Section: {a['section']}, Type: {a['article_type']}, Word count: {a['word_count']}."
                        ),
                        "prominence_score": 0.85,
                        "source_tool": "sql_analytics_article",
                    }
                )

        elif analysis_type in ("photo_count_per_section", "count_photos", "photo_counts", "photos_by_section"):
            np_name = args.get("newspaper_name") or active_newspaper_name
            iss_date = args.get("issue_date") or active_issue_date
            sec_filter = args.get("section") or args.get("category_filter")

            photo_res = await self._sql_analytics.get_photo_counts_by_section(
                newspaper_name=np_name,
                issue_date=iss_date,
                date_from=args.get("date_from"),
                date_to=args.get("date_to"),
                issue_id=args.get("issue_id") or active_issue_id,
                section=sec_filter,
            )
            total_photos = photo_res.get("total_photos", 0)
            hits_count = total_photos
            sec_counts = photo_res.get("section_counts", [])

            table_rows = [f"| {sc['section']} | {sc['count']} |" for sc in sec_counts]
            table_md = "| Section | Photo Count |\n| :--- | :--- |\n" + "\n".join(table_rows) if table_rows else "No photos found."

            filt_info = ", ".join(f"{k}: {v}" for k, v in photo_res.get("filters", {}).items() if v)
            summary_str = (
                f"=== RELATIONAL PHOTO COUNT AUDIT ===\n"
                f"• Total Matching Photos: {total_photos}\n"
                f"• Active Filters: {filt_info or 'None'}\n\n"
                f"{table_md}\n"
            )
            items.append(
                {
                    "article_id": 0,
                    "headline": f"Photo Count Analysis: {total_photos} photos found",
                    "newspaper_name": np_name or "Archive",
                    "issue_date": iss_date or "Overview",
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
            items.append(
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
            items.append(
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
            items.append(
                {
                    "article_id": 0,
                    "headline": f"Coverage Audit: {cov_matrix.target_query_or_event}",
                    "newspaper_name": "Multi-Newspaper Audit",
                    "issue_date": "Comparative Matrix",
                    "pages": [1],
                    "snippet": format_coverage_matrix_snippet(cov_matrix),
                    "prominence_score": 1.0,
                    "source_tool": "sql_analytics",
                }
            )

        elif analysis_type == "coverage_difference":
            src_np = str(args.get("newspaper_name") or args.get("source_newspaper") or "").strip()
            cmp_np = str(args.get("comparison_newspaper") or "").strip()
            iss_dt = args.get("issue_date") or args.get("target_date") or active_issue_date

            if not src_np or not cmp_np:
                items.append(
                    {
                        "article_id": 0,
                        "headline": "Coverage Difference Error: Missing newspaper arguments",
                        "newspaper_name": src_np or "Archive",
                        "issue_date": iss_dt or "Overview",
                        "pages": [1],
                        "snippet": "⚠️ Coverage difference requires both source_newspaper and comparison_newspaper.",
                        "prominence_score": 1.0,
                        "source_tool": "sql_analytics",
                    }
                )
            else:
                diff_res = await self._sql_analytics.get_newspaper_coverage_difference(
                    source_newspaper=src_np,
                    comparison_newspaper=cmp_np,
                    issue_date=iss_dt,
                )
                if "error" in diff_res:
                    items.append(
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
                    hits_count = len(diff_res.get("exclusive_articles", []))
                    items.append(
                        {
                            "article_id": 0,
                            "headline": f"Verified Exclusive Articles: {diff_res['source_newspaper']} vs {diff_res['comparison_newspaper']}",
                            "newspaper_name": diff_res["source_newspaper"],
                            "issue_date": diff_res["issue_date"],
                            "pages": [1],
                            "snippet": format_coverage_difference_snippet(diff_res),
                            "prominence_score": 1.0,
                            "source_tool": "sql_analytics",
                        }
                    )
                    for a in diff_res.get("exclusive_articles", [])[:30]:
                        art_id = a.get("id") or a.get("article_id")
                        if art_id:
                            items.append(
                                {
                                    "article_id": art_id,
                                    "headline": a.get("headline", ""),
                                    "newspaper_name": diff_res["source_newspaper"],
                                    "issue_date": diff_res["issue_date"],
                                    "pages": [a.get("page_number", 1)],
                                    "snippet": f"[Exclusive] \"{a.get('headline')}\" published in {diff_res['source_newspaper']} on Page {a.get('page_number', 1)} ({diff_res['issue_date']}). Section: {a.get('section', 'General')}.",
                                    "prominence_score": 0.85,
                                    "source_tool": "sql_analytics_exclusive",
                                }
                            )

        elif analysis_type in ("shared_coverage", "similar_articles", "common_stories", "shared_stories"):
            src_np = str(args.get("newspaper_name") or args.get("source_newspaper") or "").strip()
            cmp_np = str(args.get("comparison_newspaper") or "").strip()
            iss_dt = args.get("issue_date") or args.get("target_date") or active_issue_date

            if not src_np or not cmp_np:
                items.append(
                    {
                        "article_id": 0,
                        "headline": "Shared Coverage Error: Missing newspaper arguments",
                        "newspaper_name": src_np or "Archive",
                        "issue_date": iss_dt or "Overview",
                        "pages": [1],
                        "snippet": "⚠️ Shared coverage requires two distinct newspapers (e.g. newspaper_name and comparison_newspaper).",
                        "prominence_score": 1.0,
                        "source_tool": "sql_analytics",
                    }
                )
            else:
                shared_res = await self._sql_analytics.get_newspaper_shared_coverage(
                    newspaper_a=src_np,
                    newspaper_b=cmp_np,
                    issue_date=iss_dt,
                )
                if "error" in shared_res:
                    items.append(
                        {
                            "article_id": 0,
                            "headline": f"Shared Coverage Error: {shared_res['error']}",
                            "newspaper_name": f"{src_np} & {cmp_np}",
                            "issue_date": iss_dt or "Overview",
                            "pages": [1],
                            "snippet": f"⚠️ {shared_res['error']}",
                            "prominence_score": 1.0,
                            "source_tool": "sql_analytics",
                        }
                    )
                else:
                    shared_stories = shared_res.get("shared_stories", [])
                    hits_count = len(shared_stories)

                    # 1. Macro overview snippet (article_id: 0)
                    items.append(
                        {
                            "article_id": 0,
                            "headline": f"Verified Shared Wire Coverage: {shared_res['newspaper_a']} & {shared_res['newspaper_b']}",
                            "newspaper_name": f"{shared_res['newspaper_a']} & {shared_res['newspaper_b']}",
                            "issue_date": shared_res["issue_date"],
                            "pages": [1],
                            "snippet": format_shared_coverage_snippet(shared_res),
                            "prominence_score": 1.0,
                            "source_tool": "sql_analytics",
                        }
                    )

                    # 2. Individual article evidence items with real article_id > 0 for citation pills
                    for story in shared_stories:
                        art_id_a = story.get("article_id_a")
                        art_id_b = story.get("article_id_b")
                        np_a = story.get("newspaper_a")
                        np_b = story.get("newspaper_b")
                        hl_a = story.get("headline_a") or ""
                        hl_b = story.get("headline_b") or ""
                        pg_a = story.get("page_a", 1)
                        pg_b = story.get("page_b", 1)
                        sec_a = story.get("section_a", "General")
                        sec_b = story.get("section_b", "General")
                        tier = story.get("match_tier", "Match")
                        kw = story.get("shared_keywords", "")

                        if art_id_a:
                            items.append(
                                {
                                    "article_id": art_id_a,
                                    "headline": hl_a,
                                    "newspaper_name": np_a,
                                    "issue_date": shared_res["issue_date"],
                                    "pages": [pg_a],
                                    "snippet": (
                                        f"[{tier}] \"{hl_a}\" in {np_a} (Page {pg_a}, {sec_a}). "
                                        f"Shared syndicated wire reporting with {np_b} (Page {pg_b}, \"{hl_b}\"). "
                                        f"Identified overlap: {kw}."
                                    ),
                                    "prominence_score": 0.95,
                                    "source_tool": "sql_analytics_shared",
                                }
                            )
                        if art_id_b:
                            items.append(
                                {
                                    "article_id": art_id_b,
                                    "headline": hl_b,
                                    "newspaper_name": np_b,
                                    "issue_date": shared_res["issue_date"],
                                    "pages": [pg_b],
                                    "snippet": (
                                        f"[{tier}] \"{hl_b}\" in {np_b} (Page {pg_b}, {sec_b}). "
                                        f"Shared syndicated wire reporting with {np_a} (Page {pg_a}, \"{hl_a}\"). "
                                        f"Identified overlap: {kw}."
                                    ),
                                    "prominence_score": 0.95,
                                    "source_tool": "sql_analytics_shared",
                                }
                            )

        else:
            # Unsupported or custom analysis type requested by LLM (e.g. count_pages, word_count_by_author)
            # Delegate directly to ToolMaker dynamic code synthesis.
            if self._tool_maker:
                logger.info(
                    f"Unsupported sql_analytics analysis_type '{analysis_type}'; delegating to dynamic tool synthesis."
                )
                dyn_items, dyn_hits = await self._execute_dynamic_analysis(args, state)
                return dyn_items, dyn_hits, context_updates

        return items, hits_count, context_updates

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
        import contextlib
        import re
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        from app.models.article import Article, ArticlePage, Photo
        from app.models.newspaper import Issue, Newspaper
        from app.agent.condenser import parse_inline_citation
        from app.agent.extractor import extract_parameters_from_query

        photo_id = args.get("photo_id") or state.get("attached_photo_id")
        article_id = args.get("article_id") or state.get("attached_article_id")
        query_text = (args.get("query") or state.get("query") or "").strip()
        newspaper_name = (args.get("newspaper_name") or state.get("active_newspaper_name") or "").strip()
        issue_date = (args.get("issue_date") or state.get("active_issue_date") or "").strip()
        page_filter = args.get("page_filter") or ""

        target_headline: str | None = None

        # 0. Check for authoritative inline citation or extracted parameters in query_text or current state
        q_citation = (
            parse_inline_citation(query_text)
            or parse_inline_citation(state.get("query") or "")
            or parse_inline_citation(state.get("original_query") or "")
        )
        if q_citation.get("newspaper_name") and not newspaper_name:
            newspaper_name = q_citation["newspaper_name"]
        if q_citation.get("issue_date") and not issue_date:
            issue_date = q_citation["issue_date"]
        if q_citation.get("page_number") and not page_filter:
            page_filter = str(q_citation["page_number"])
        if q_citation.get("headline"):
            target_headline = q_citation["headline"]

        q_ext = (
            extract_parameters_from_query(query_text)
            or extract_parameters_from_query(state.get("query") or "")
            or extract_parameters_from_query(state.get("original_query") or "")
        )
        if not newspaper_name and q_ext.get("newspaper_name"):
            newspaper_name = q_ext["newspaper_name"]
        if not issue_date and q_ext.get("issue_date"):
            issue_date = q_ext["issue_date"]
        if not page_filter and q_ext.get("page_filter"):
            page_filter = str(q_ext["page_filter"])

        # 1. If photo_id or article_id not provided and no current query headline, try resolving from conversation history
        if not photo_id and not article_id and not target_headline:
            chat_history = state.get("chat_history") or []
            for turn in reversed(chat_history):
                # Check turn citations (only inherit if matching known newspaper/date filters)
                turn_citations = turn.get("citations") or []
                for cit in turn_citations:
                    if isinstance(cit, dict):
                        c_np = cit.get("newspaper_name") or ""
                        c_dt = cit.get("issue_date") or ""
                        np_ok = not newspaper_name or not c_np or (newspaper_name.lower() in c_np.lower() or c_np.lower() in newspaper_name.lower())
                        dt_ok = not issue_date or not c_dt or c_dt == issue_date

                        if np_ok and dt_ok:
                            if cit.get("photo_id") and not photo_id:
                                with contextlib.suppress(ValueError):
                                    photo_id = int(cit["photo_id"])
                            if cit.get("article_id") and not article_id:
                                with contextlib.suppress(ValueError):
                                    article_id = int(cit["article_id"])
                            if cit.get("headline") and not target_headline:
                                target_headline = str(cit["headline"]).strip()

                # Check turn attached asset (only inherit if matching known newspaper/date filters)
                att = turn.get("attachedAsset") or {}
                if isinstance(att, dict):
                    att_np = att.get("newspaperName") or att.get("newspaper_name") or ""
                    att_dt = att.get("issueDate") or att.get("issue_date") or ""
                    att_np_ok = not newspaper_name or not att_np or (newspaper_name.lower() in att_np.lower() or att_np.lower() in newspaper_name.lower())
                    att_dt_ok = not issue_date or not att_dt or att_dt == issue_date
                    if att_np_ok and att_dt_ok:
                        if att.get("photoId") and not photo_id:
                            with contextlib.suppress(ValueError):
                                photo_id = int(att["photoId"])
                        if att.get("articleId") and not article_id:
                            with contextlib.suppress(ValueError):
                                article_id = int(att["articleId"])
                        if att.get("headline") and not target_headline:
                            target_headline = str(att["headline"]).strip()

                # Check turn message text for inline citation
                c_text = str(turn.get("content", ""))
                cite_m = parse_inline_citation(c_text)
                if cite_m:
                    c_np = cite_m.get("newspaper_name") or ""
                    c_dt = cite_m.get("issue_date") or ""
                    np_ok = not newspaper_name or not c_np or (newspaper_name.lower() in c_np.lower() or c_np.lower() in newspaper_name.lower())
                    dt_ok = not issue_date or not c_dt or c_dt == issue_date

                    if np_ok and dt_ok:
                        if not newspaper_name and c_np:
                            newspaper_name = c_np
                        if not issue_date and c_dt:
                            issue_date = c_dt
                        if not page_filter and cite_m.get("page_number"):
                            page_filter = str(cite_m["page_number"])
                        if not target_headline and cite_m.get("headline"):
                            target_headline = cite_m["headline"]

                if photo_id or article_id or target_headline:
                    break

        items: list[dict[str, Any]] = []

        async with self._session_factory() as session:
            photos_to_process: list[Photo] = []

            # Strategy A: Explicit Photo ID provided or resolved
            if photo_id:
                stmt = (
                    select(Photo)
                    .where(Photo.id == int(photo_id))
                    .options(
                        selectinload(Photo.article)
                        .selectinload(Article.issue)
                        .selectinload(Issue.newspaper),
                        selectinload(Photo.article).selectinload(Article.article_pages),
                    )
                )
                res = await session.execute(stmt)
                single_photo = res.scalar_one_or_none()
                if single_photo:
                    # Validate compatibility with requested newspaper and date
                    is_compat = True
                    p_art = single_photo.article
                    if p_art and p_art.issue:
                        if newspaper_name and p_art.issue.newspaper:
                            np_m = (newspaper_name.lower() in p_art.issue.newspaper.name.lower() or p_art.issue.newspaper.name.lower() in newspaper_name.lower())
                            if not np_m:
                                is_compat = False
                        if issue_date and str(p_art.issue.issue_date) != issue_date:
                            is_compat = False

                    if is_compat:
                        photos_to_process.append(single_photo)
                    elif bool(
                        (state.get("attached_photo_id") and int(state["attached_photo_id"]) == single_photo.id)
                        or (args.get("photo_id") and int(args["photo_id"]) == single_photo.id)
                    ):
                        explicit_query_date = (
                            q_citation.get("issue_date")
                            or q_ext.get("issue_date")
                        )
                        explicit_query_np = (
                            q_citation.get("newspaper_name")
                            or q_ext.get("newspaper_name")
                        )
                        has_date_conflict = bool(
                            explicit_query_date
                            and p_art
                            and p_art.issue
                            and str(p_art.issue.issue_date) != explicit_query_date
                        )
                        has_np_conflict = bool(
                            explicit_query_np
                            and p_art
                            and p_art.issue
                            and p_art.issue.newspaper
                            and explicit_query_np.lower() not in p_art.issue.newspaper.name.lower()
                            and p_art.issue.newspaper.name.lower() not in explicit_query_np.lower()
                        )
                        if not has_date_conflict and not has_np_conflict:
                            # Stale conversation context mismatch - adopt photo's true issue date and publication
                            photos_to_process.append(single_photo)
                            if p_art and p_art.issue:
                                issue_date = str(p_art.issue.issue_date)
                                if p_art.issue.newspaper:
                                    newspaper_name = p_art.issue.newspaper.name
                        else:
                            logger.warning(
                                "Photo ID %s rejected for visual inspection due to explicit query entity mismatch (requested %s %s, photo %s %s)",
                                photo_id,
                                explicit_query_np,
                                explicit_query_date,
                                p_art.issue.newspaper.name if p_art and p_art.issue and p_art.issue.newspaper else "unknown",
                                p_art.issue.issue_date if p_art and p_art.issue else "unknown",
                            )
                            photo_id = None
                    else:
                        logger.warning("Photo ID %s rejected for visual inspection due to newspaper/date mismatch", photo_id)
                        photo_id = None

                    if photos_to_process and single_photo in photos_to_process:
                        # If this photo is part of an article, pull companion charts
                        if single_photo.article_id:
                            comp_stmt = (
                                select(Photo)
                                .where(
                                    Photo.article_id == single_photo.article_id,
                                    Photo.id != single_photo.id,
                                )
                                .order_by(Photo.id.asc())
                                .options(
                                    selectinload(Photo.article)
                                    .selectinload(Article.issue)
                                    .selectinload(Issue.newspaper),
                                    selectinload(Photo.article).selectinload(Article.article_pages),
                                )
                            )
                            comp_res = await session.execute(comp_stmt)
                            companion_photos = comp_res.scalars().all()
                            for cp in companion_photos:
                                if cp.visual_type in ("data_chart", "infographic", "table") and len(photos_to_process) < 6:
                                    photos_to_process.append(cp)

            # Strategy B: Target headline resolved from query citation or context, find article
            if target_headline and not photos_to_process:
                art_stmt = (
                    select(Article)
                    .join(Issue, Article.issue_id == Issue.id)
                    .join(Newspaper, Issue.newspaper_id == Newspaper.id)
                    .where(
                        (Article.headline == target_headline)
                        | (Article.headline.ilike(f"%{target_headline[:50]}%"))
                    )
                )
                if newspaper_name:
                    art_stmt = art_stmt.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
                if issue_date:
                    art_stmt = art_stmt.where(Issue.issue_date == issue_date)

                matched_art = (await session.execute(art_stmt.order_by(Article.id.desc()).limit(1))).scalars().first()
                if matched_art:
                    article_id = matched_art.id

            # Strategy C: Explicit Article ID provided or resolved
            if article_id and not photos_to_process:
                # Validate article_id compatibility with requested newspaper & issue_date
                chk_stmt = (
                    select(Article)
                    .where(Article.id == int(article_id))
                    .options(
                        selectinload(Article.issue).selectinload(Issue.newspaper),
                    )
                )
                chk_art = (await session.execute(chk_stmt)).scalar_one_or_none()
                is_compat = True
                if chk_art and chk_art.issue:
                    if newspaper_name and chk_art.issue.newspaper:
                        np_m = (newspaper_name.lower() in chk_art.issue.newspaper.name.lower() or chk_art.issue.newspaper.name.lower() in newspaper_name.lower())
                        if not np_m:
                            is_compat = False
                    if issue_date and str(chk_art.issue.issue_date) != issue_date:
                        is_compat = False

                is_explicit_article = bool(
                    (state.get("attached_article_id") and int(state["attached_article_id"]) == int(article_id))
                    or (args.get("article_id") and int(args["article_id"]) == int(article_id))
                )
                if not is_compat and is_explicit_article:
                    explicit_query_date = (
                        q_citation.get("issue_date")
                        or q_ext.get("issue_date")
                    )
                    explicit_query_np = (
                        q_citation.get("newspaper_name")
                        or q_ext.get("newspaper_name")
                    )
                    has_date_conflict = bool(
                        explicit_query_date
                        and chk_art
                        and chk_art.issue
                        and str(chk_art.issue.issue_date) != explicit_query_date
                    )
                    has_np_conflict = bool(
                        explicit_query_np
                        and chk_art
                        and chk_art.issue
                        and chk_art.issue.newspaper
                        and explicit_query_np.lower() not in chk_art.issue.newspaper.name.lower()
                        and chk_art.issue.newspaper.name.lower() not in explicit_query_np.lower()
                    )
                    if not has_date_conflict and not has_np_conflict:
                        is_compat = True
                        if chk_art and chk_art.issue:
                            issue_date = str(chk_art.issue.issue_date)
                            if chk_art.issue.newspaper:
                                newspaper_name = chk_art.issue.newspaper.name
                    else:
                        logger.warning(
                            "Article ID %s rejected for visual inspection due to explicit query entity mismatch (requested %s %s, article %s %s)",
                            article_id,
                            explicit_query_np,
                            explicit_query_date,
                            chk_art.issue.newspaper.name if chk_art and chk_art.issue and chk_art.issue.newspaper else "unknown",
                            chk_art.issue.issue_date if chk_art and chk_art.issue else "unknown",
                        )
                        article_id = None

                if not is_compat:
                    if article_id:
                        logger.warning(
                            "Article ID %s rejected for visual inspection due to newspaper/date mismatch: requested %s (%s)",
                            article_id,
                            newspaper_name,
                            issue_date,
                        )
                    article_id = None

            if article_id and not photos_to_process:
                stmt = (
                    select(Photo)
                    .where(Photo.article_id == int(article_id))
                    .order_by(Photo.id.asc())
                    .options(
                        selectinload(Photo.article)
                        .selectinload(Article.issue)
                        .selectinload(Issue.newspaper),
                        selectinload(Photo.article).selectinload(Article.article_pages),
                    )
                )
                res = await session.execute(stmt)
                all_photos = res.scalars().all()
                charts = [p for p in all_photos if p.visual_type in ("data_chart", "infographic", "table")]
                other = [p for p in all_photos if p not in charts]
                photos_to_process = (charts + other)[:6]

                # If article genuinely has 0 visual assets, return clear verification notice
                if not photos_to_process:
                    art_stmt = (
                        select(Article)
                        .where(Article.id == int(article_id))
                        .options(
                            selectinload(Article.issue).selectinload(Issue.newspaper),
                            selectinload(Article.article_pages),
                        )
                    )
                    art_obj = (await session.execute(art_stmt)).scalar_one_or_none()
                    if art_obj:
                        np_name = art_obj.issue.newspaper.name if art_obj.issue and art_obj.issue.newspaper else "Archive Publication"
                        iss_d = str(art_obj.issue.issue_date) if art_obj.issue else "Current"
                        pg = art_obj.article_pages[0].page_number if art_obj.article_pages else 1
                        items.append({
                            "article_id": art_obj.id,
                            "issue_id": art_obj.issue_id,
                            "headline": art_obj.headline or "Article",
                            "newspaper_name": np_name,
                            "issue_date": iss_d,
                            "pages": [pg],
                            "snippet": (
                                f"=== VISUAL ASSET INSPECTION RESULT ===\n"
                                f"Article: {art_obj.headline}\n"
                                f"Publication: {np_name} ({iss_d}), Page {pg}\n\n"
                                f"Visual Asset Audit: Verified that no infographics, data charts, graphs, or visual tables are attached to this article in the broadsheet layout."
                            ),
                            "prominence_score": 0.9,
                            "source_tool": "inspect_visual_asset",
                            "is_visual_asset": False,
                        })
                        return items, 1

            # Strategy D: Multi-criteria database search by metadata (newspaper, date, page, query)
            if not photos_to_process:
                cand_stmt = (
                    select(Article)
                    .join(Issue, Article.issue_id == Issue.id)
                    .join(Newspaper, Issue.newspaper_id == Newspaper.id)
                )
                if newspaper_name:
                    cand_stmt = cand_stmt.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
                if issue_date:
                    cand_stmt = cand_stmt.where(Issue.issue_date == issue_date)
                if page_filter and str(page_filter).strip().isdigit():
                    cand_stmt = cand_stmt.join(ArticlePage, ArticlePage.article_id == Article.id).where(
                        ArticlePage.page_number == int(str(page_filter).strip())
                    )
                if target_headline:
                    cand_stmt = cand_stmt.where(
                        (Article.headline == target_headline)
                        | (Article.headline.ilike(f"%{target_headline[:50]}%"))
                    )

                cand_res = await session.execute(cand_stmt.limit(30))
                candidates = cand_res.scalars().all()

                q_tokens = [w.lower() for w in re.findall(r"\w+", query_text) if len(w) > 2]
                scored_cands: list[tuple[int, Article]] = []
                for cand in candidates:
                    hl_lower = (cand.headline or "").lower()
                    score = sum(1 for tok in q_tokens if tok in hl_lower)
                    if score > 0 or target_headline:
                        scored_cands.append((score, cand))
                scored_cands.sort(key=lambda x: x[0], reverse=True)

                if scored_cands:
                    best_art = scored_cands[0][1]
                    p_stmt = (
                        select(Photo)
                        .where(Photo.article_id == best_art.id)
                        .order_by(Photo.id.asc())
                        .options(
                            selectinload(Photo.article)
                            .selectinload(Article.issue)
                            .selectinload(Issue.newspaper),
                            selectinload(Photo.article).selectinload(Article.article_pages),
                        )
                    )
                    p_res = await session.execute(p_stmt)
                    all_art_photos = p_res.scalars().all()
                    charts = [p for p in all_art_photos if p.visual_type in ("data_chart", "infographic", "table")]
                    other = [p for p in all_art_photos if p not in charts]
                    photos_to_process = (charts + other)[:6]

            # Strategy E: Fallback search on Photo captions and VLM descriptions
            if not photos_to_process and query_text:
                _photo_stopwords = {
                    "the", "and", "for", "any", "have", "with", "from", "that", "this",
                    "what", "does", "dated", "photo", "photos", "picture", "pictures",
                    "image", "images", "there",
                }
                q_tokens = [w.lower() for w in re.findall(r"\w+", query_text) if len(w) >= 3 and w.lower() not in _photo_stopwords]
                if q_tokens:
                    p_search_stmt = (
                        select(Photo)
                        .join(Article, Photo.article_id == Article.id)
                        .where(Photo.visual_type.in_(["data_chart", "infographic", "table", "photo"]))
                        .options(
                            selectinload(Photo.article)
                            .selectinload(Article.issue)
                            .selectinload(Issue.newspaper),
                            selectinload(Photo.article).selectinload(Article.article_pages),
                        )
                        .order_by(Photo.id.desc())
                        .limit(10)
                    )
                    # Scope by newspaper and date if specified to prevent cross-issue leakage
                    if newspaper_name or issue_date:
                        p_search_stmt = (
                            p_search_stmt.join(Issue, Article.issue_id == Issue.id)
                            .join(Newspaper, Issue.newspaper_id == Newspaper.id)
                        )
                        if newspaper_name:
                            p_search_stmt = p_search_stmt.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
                        if issue_date:
                            p_search_stmt = p_search_stmt.where(Issue.issue_date == issue_date)

                    p_search_res = await session.execute(p_search_stmt)
                    cand_photos = p_search_res.scalars().all()
                    scored_photos: list[tuple[int, Photo]] = []
                    for cp in cand_photos:
                        blob = f"{cp.caption or ''} {cp.vlm_description or ''} {(cp.article.headline if cp.article else '')}".lower()
                        sc = sum(1 for tok in q_tokens if tok in blob)
                        if sc > 0:
                            scored_photos.append((sc, cp))
                    scored_photos.sort(key=lambda x: x[0], reverse=True)
                    if scored_photos:
                        photos_to_process = [sp[1] for sp in scored_photos[:4]]

            if not photos_to_process:
                logger.warning("No photo found for inspect_visual_asset", extra={"tool_args": args})
                return [], 0

            # Process all resolved visual assets
            for target_photo in photos_to_process:
                # Check if VLM description is missing or generic fallback
                desc = (target_photo.vlm_description or "").strip()
                is_placeholder = (
                    not desc
                    or desc.startswith("Visual asset:")
                    or desc in (
                        "Editorial news photograph.",
                        "Visual asset: infographic from broadsheet.",
                        "Visual asset: table from broadsheet.",
                        "Visual asset: data_chart from broadsheet.",
                    )
                )

                if is_placeholder and target_photo.object_key:
                    try:
                        from app.core.config import get_settings
                        from app.storage.minio_store import MinioStore
                        from app.ingestion.visual_extractor import VisualDataExtractor

                        cfg = get_settings()
                        minio = MinioStore(cfg.minio)
                        crop_bytes = await minio.get(
                            bucket=cfg.minio.bucket_pages,
                            key=target_photo.object_key,
                        )
                        if crop_bytes:
                            extractor = VisualDataExtractor()
                            classification, extraction = await extractor.process_image_crop(
                                image_bytes=crop_bytes,
                                ocr_text=target_photo.caption or "",
                            )
                            if extraction:
                                parts = [extraction.summary]
                                if extraction.key_metrics:
                                    parts.append("\nKey Data Points & Metrics:\n• " + "\n• ".join(extraction.key_metrics))
                                if extraction.markdown_table:
                                    parts.append("\n" + extraction.markdown_table)
                                new_desc = "\n".join(parts)

                                target_photo.visual_type = classification.visual_type or target_photo.visual_type
                                target_photo.vlm_description = new_desc
                                session.add(target_photo)
                                await session.commit()
                                await session.refresh(target_photo)
                    except Exception as vlm_err:
                        logger.warning(
                            "On-demand visual extraction failed in inspect_visual_asset",
                            extra={"photo_id": target_photo.id, "error": str(vlm_err)},
                        )

                article = target_photo.article
                hl = article.headline if article else "Visual Intelligence Asset"
                issue = article.issue if article else None
                np_name = issue.newspaper.name if issue and issue.newspaper else (newspaper_name or state.get("active_newspaper_name") or "Archive Publication")
                iss_d = str(issue.issue_date) if issue else (issue_date or state.get("active_issue_date") or "Current")
                page_val = article.article_pages[0].page_number if (article and article.article_pages) else (int(page_filter) if str(page_filter).isdigit() else 1)

                v_type_clean = (target_photo.visual_type or "Infographic").replace("_", " ").title()
                vlm_content = target_photo.vlm_description or target_photo.caption or "Visual graphic from broadsheet."

                snippet = (
                    f"=== VISUAL DATA ASSET: {v_type_clean} (Asset #{target_photo.id}) ===\n"
                    f"Headline: {hl}\n"
                    f"Publication: {np_name} ({iss_d}), Page {page_val}\n"
                    f"Visual Type: {v_type_clean}\n"
                    f"Printed Caption: {target_photo.caption or 'None'}\n\n"
                    f"Extracted Infographic Data & Visual Table Content:\n{vlm_content}"
                )

                items.append(
                    {
                        "article_id": target_photo.article_id or (article.id if article else 0),
                        "issue_id": article.issue_id if article else 0,
                        "headline": hl,
                        "newspaper_name": np_name,
                        "issue_date": iss_d,
                        "pages": [page_val],
                        "bboxes": target_photo.bbox_json.get("bbox", []) if target_photo.bbox_json else [],
                        "snippet": snippet,
                        "prominence_score": 1.0,
                        "source_tool": "inspect_visual_asset",
                        "is_visual_asset": True,
                        "photo_id": target_photo.id,
                        "visual_type": target_photo.visual_type,
                        "image_url": f"/api/photos/{target_photo.id}/image",
                        "vlm_description": target_photo.vlm_description,
                        "caption": target_photo.caption,
                    }
                )

        return items, len(items)

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

        model_override = state.get("model_override")
        res = await self._tool_maker.generate_and_execute(
            query=query,
            context=context,
            model_override=model_override,
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
