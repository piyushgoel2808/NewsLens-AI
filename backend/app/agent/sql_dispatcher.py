"""SQL Analytics Dispatcher and Evidence Formatter for NewsLens-AI.

Encapsulates relational SQL query dispatching and formatting across pre-compiled
analytical routines, separating database reporting from tool execution flow.
"""

from __future__ import annotations

from typing import Any

from app.agent.extractor import (
    is_archive_wide_newspaper_query,
)
from app.agent.sql_formatters import SQLCountsAndComparisonMixin
from app.agent.state import AgentState
from app.core.logging import get_logger
from app.retrieval.coverage_analyzer import CoverageAnalyzer
from app.retrieval.formatters import (
    format_issue_manifest,
)
from app.retrieval.sql_analytics import SQLAnalyticsEngine

logger = get_logger(__name__)


class SQLAnalyticsDispatcher(SQLCountsAndComparisonMixin):
    """Dispatches and formats relational SQL analytics queries."""

    def __init__(
        self,
        sql_analytics: SQLAnalyticsEngine,
        coverage_analyzer: CoverageAnalyzer | None = None,
    ) -> None:
        self._sql_analytics = sql_analytics
        self._coverage_analyzer = coverage_analyzer

    async def dispatch(
        self,
        args: dict[str, Any],
        state: AgentState,
        active_issue_id: int | None = None,
        active_newspaper_name: str | None = None,
        active_issue_date: str | None = None,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]] | None:
        """Dispatch pre-compiled SQL analytics routines.

        Returns (evidence_items, hits_count, context_updates) or None if analysis_type is unhandled.
        """
        analysis_type = args.get("analysis_type")
        if not analysis_type or str(analysis_type).lower() in ("none", "null", ""):
            q_text = str(args.get("query") or state.get("query") or "").lower()
            if any(w in q_text for w in ["advertisement", "ad count", "ads count", "number of ads"]):
                analysis_type = "count_advertisements"
            elif any(w in q_text for w in ["photo count", "photos count", "number of photos"]):
                analysis_type = "count_photos"
            elif any(w in q_text for w in [
                "number of issues", "total issues", "availability", "how many issues",
                "issue count", "count issues", "list all the newspaper", "list all newspapers",
                "available newspaper", "how many newspaper", "how many editions",
            ]):
                analysis_type = "count_issues"
            elif any(w in q_text for w in ["how many", "count", "number of", "total", "volume"]):
                if any(w in q_text for w in ["issue", "newspaper", "edition", "paper"]):
                    analysis_type = "count_issues"
                elif any(w in q_text for w in ["article", "story", "news"]):
                    analysis_type = "count_articles"
                else:
                    analysis_type = "count_issues"
            elif is_archive_wide_newspaper_query(q_text) or any(w in q_text for w in ["available", "inventory", "archive", "newspapers"]):
                analysis_type = "count_issues"
            else:
                analysis_type = "issue_summary"
            args["analysis_type"] = analysis_type

        items: list[dict[str, Any]] = []
        hits_count = 0
        context_updates: dict[str, Any] = {}

        if analysis_type == "entity_trends":
            items, hits_count = await self._exec_sql_entity_trends(args)

        elif analysis_type == "issue_summary":
            return await self._exec_sql_issue_summary(
                args, state, active_issue_id, active_newspaper_name, active_issue_date
            )

        elif analysis_type in ("count_advertisements", "count_ads", "advertisements", "ad_counts") or (
            analysis_type == "count_articles"
            and (args.get("article_type") == "advertisement" or "advertis" in str(args.get("section") or "").lower())
        ):
            items, hits_count = await self._exec_sql_count_ads(
                args, active_issue_id, active_newspaper_name, active_issue_date
            )

        elif analysis_type in ("count_issues", "issue_counts", "total_issues", "newspaper_availability", "check_availability"):
            items, hits_count = await self._exec_sql_count_issues(
                args, state, active_newspaper_name, active_issue_date
            )

        elif analysis_type == "count_articles":
            items, hits_count = await self._exec_sql_count_articles(args)

        elif analysis_type in ("photo_count_per_section", "count_photos", "photo_counts", "photos_by_section"):
            items, hits_count = await self._exec_sql_photo_counts(
                args, active_issue_id, active_newspaper_name, active_issue_date
            )

        elif analysis_type == "topic_distribution":
            items, hits_count = await self._exec_sql_topic_distribution()

        elif analysis_type == "frontpage_ratio":
            items, hits_count = await self._exec_sql_frontpage_ratio()

        elif analysis_type == "coverage_comparison":
            items, hits_count = await self._exec_sql_coverage_comparison(args, state)

        elif analysis_type == "coverage_difference":
            items, hits_count = await self._exec_sql_coverage_difference(args, active_issue_date)

        elif analysis_type in ("shared_coverage", "similar_articles", "common_stories", "shared_stories"):
            items, hits_count = await self._exec_sql_shared_coverage(args, active_issue_date)

        else:
            return None

        return items, hits_count, context_updates

    async def _exec_sql_entity_trends(
        self,
        args: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        trends = await self._sql_analytics.get_entity_mention_trends(entity_name=args.get("term", ""))
        hits_count = len(trends)
        trend_items = [
            f"{t['date']}: {t['article_count']} articles ({t['total_mentions']} mentions)"
            for t in trends[:5]
        ]
        summary_str = f"Mention Trends for '{args.get('term')}': " + ", ".join(trend_items)
        items = [
            {
                "article_id": 0,
                "headline": f"Statistical Trends: {args.get('term')}",
                "newspaper_name": "Aggregated Archive Analytics",
                "issue_date": trends[0]["date"] if trends else "Overview",
                "pages": [1],
                "snippet": summary_str,
                "source_tool": "sql_analytics",
            }
        ]
        return items, hits_count

    async def _exec_sql_issue_summary(
        self,
        args: dict[str, Any],
        state: AgentState,
        active_issue_id: int | None,
        active_newspaper_name: str | None,
        active_issue_date: str | None,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        page_filter = str(args["page_filter"]).strip() if args.get("page_filter") is not None else None
        d_from = args.get("date_from")
        d_to = args.get("date_to")
        has_multi_day_range = bool(d_from and d_to and d_from != d_to)
        has_single_date = bool(args.get("issue_date") or (d_from and d_to and d_from == d_to))
        has_date_range = bool(d_from or d_to or has_multi_day_range)
        is_archive_np = is_archive_wide_newspaper_query(state.get("query", "")) or is_archive_wide_newspaper_query(args.get("query", ""))

        # Redirect multi-day date range or archive-wide requests without a specific date or newspaper to count_issues
        if ((has_multi_day_range or is_archive_np) and not has_single_date and not args.get("newspaper_name")) or (
            has_multi_day_range and not args.get("newspaper_name")
        ):
            args["analysis_type"] = "count_issues"
            res = await self.dispatch(args, state, active_issue_id, active_newspaper_name, active_issue_date)
            return res if res is not None else ([], 0, {})

        is_comparative = (
            state.get("archetype") == "cross_newspaper_comparison"
            or any(w in str(state.get("query", "")).lower() for w in ["all available", "all newspaper", "across newspaper", "both newspaper", "different newspaper"])
        )
        date_mismatch = bool(args.get("issue_date") and active_issue_date and args.get("issue_date") != active_issue_date)
        target_all_on_date = bool(args.get("issue_date") and not args.get("newspaper_name"))
        inherit_history = (
            not is_comparative
            and not is_archive_np
            and not has_date_range
            and not date_mismatch
            and not target_all_on_date
        )

        np_arg = args.get("newspaper_name") or (active_newspaper_name if inherit_history else None)
        iss_d_arg = args.get("issue_date") or (active_issue_date if inherit_history else None)
        iss_id_arg = args.get("issue_id") or (active_issue_id if inherit_history else None)

        target_nps: list[str] = []
        if not np_arg or not page_filter or not iss_d_arg:
            from app.agent.extractor import extract_parameters_from_query
            extracted = extract_parameters_from_query(args.get("query") or state.get("query") or "")
            target_nps = extracted.get("target_newspapers", [])
            if not np_arg and extracted.get("newspaper_name") and len(target_nps) < 2:
                # If multiple newspapers are targeted (e.g. comparative queries), do NOT collapse
                # np_arg to only the first one so is_multi_issue_date fetches all targeted papers on that date
                np_arg = extracted["newspaper_name"]
            if not page_filter and extracted.get("page_filter"):
                page_filter = str(extracted["page_filter"]).strip()
            if not iss_d_arg and extracted.get("issue_date"):
                iss_d_arg = extracted["issue_date"]

        is_multi_issue_date = iss_d_arg and not np_arg and not iss_id_arg

        items: list[dict[str, Any]] = []
        hits_count = 0
        context_updates: dict[str, Any] = {}

        if is_multi_issue_date and iss_d_arg:
            date_issues = await self._sql_analytics.get_issues_by_date(iss_d_arg)
            if date_issues and target_nps and len(target_nps) >= 2:
                target_lower = {tn.lower() for tn in target_nps}
                date_issues = [
                    di for di in date_issues
                    if di.newspaper and any(tn in di.newspaper.name.lower() or di.newspaper.name.lower() in tn for tn in target_lower)
                ]
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
                    for a in summary.get("articles", [])[:30]:
                        art_id = a.get("id") or a.get("article_id")
                        if art_id:
                            items.append(
                                {
                                    "article_id": art_id,
                                    "headline": a.get("headline", ""),
                                    "newspaper_name": summary.get("newspaper", "Archive"),
                                    "issue_date": summary.get("issue_date", iss_d_arg),
                                    "pages": [a.get("page_number", 1)],
                                    "snippet": f"\"{a.get('headline')}\" published in {summary.get('newspaper', 'Archive')} on Page {a.get('page_number', 1)} ({summary.get('issue_date', iss_d_arg)}). Section: {a.get('section', 'General')}, Word count: {a.get('word_count', 0)}.",
                                    "prominence_score": 0.85,
                                    "source_tool": "sql_analytics_manifest",
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

        return items, hits_count, context_updates

