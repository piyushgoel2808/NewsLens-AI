"""Tool Execution Engine for NewsLens-AI Agentic Workflow.

Provides unified dispatch, concurrent execution, parameter sanitization, and standardized
evidence formatting across all broadsheet retrieval tools with zero dynamic imports.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.state import AgentState, ToolExecutionRecord
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

        author_info = f" by {a['byline_author']}" if a.get("byline_author") else ""
        clean_a_hl = repair_text_ligatures(a.get("headline") or "")
        manifest_lines.append(
            f'{idx}. [{a.get("section", "General")}] "{clean_a_hl}" '
            f"({folio_info}{author_info}, {a.get('word_count', 0)} words)"
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
        p_str = f"Page {ex.get('page_number')} (PDF Page {ex.get('page_number')})"
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
    ) -> None:
        self._session_factory = session_factory
        self._hybrid_search = hybrid_search
        self._entity_search = entity_search
        self._timeline_builder = timeline_builder
        self._sql_analytics = sql_analytics
        self._coverage_analyzer = coverage_analyzer
        self._web_search = web_search

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
        name = call.get("tool_name")
        args = call.get("arguments", {})
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
                    "photos": hr.photos,
                    "has_visual_data": hr.has_visual_data,
                    "visual_type": hr.visual_type,
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
            items.append(
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
 
 
__all__ = [
    "ToolExecutor",
    "format_coverage_difference_snippet",
    "format_coverage_matrix_snippet",
    "format_issue_manifest",
]
