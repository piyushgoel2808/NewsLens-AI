"""SQL Analytics Dispatcher and Evidence Formatter for NewsLens-AI.

Encapsulates relational SQL query dispatching and formatting across pre-compiled
analytical routines, separating database reporting from tool execution flow.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    is_archive_wide_newspaper_query,
)
from app.agent.state import AgentState
from app.core.logging import get_logger
from app.retrieval.coverage_analyzer import CoverageAnalyzer
from app.retrieval.formatters import (
    format_coverage_difference_snippet,
    format_coverage_matrix_snippet,
    format_issue_manifest,
    format_shared_coverage_snippet,
)
from app.retrieval.sql_analytics import SQLAnalyticsEngine

logger = get_logger(__name__)


class SQLAnalyticsDispatcher:
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
            elif any(w in q_text for w in ["number of issues", "total issues", "availability"]):
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
        has_date_range = bool(d_from and d_to)
        is_archive_np = is_archive_wide_newspaper_query(state.get("query", "")) or is_archive_wide_newspaper_query(args.get("query", ""))

        # Redirect date-range or archive-wide requests without a specific newspaper to count_issues
        if (has_date_range or is_archive_np) and not args.get("newspaper_name"):
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

        if not np_arg or not page_filter or not iss_d_arg:
            from app.agent.extractor import extract_parameters_from_query
            extracted = extract_parameters_from_query(args.get("query") or state.get("query") or "")
            if not np_arg and extracted.get("newspaper_name"):
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

    async def _exec_sql_count_ads(
        self,
        args: dict[str, Any],
        active_issue_id: int | None,
        active_newspaper_name: str | None,
        active_issue_date: str | None,
    ) -> tuple[list[dict[str, Any]], int]:
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

        items: list[dict[str, Any]] = [
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
        ]

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

        return items, hits_count

    async def _exec_sql_count_issues(
        self,
        args: dict[str, Any],
        state: AgentState,
        active_newspaper_name: str | None,
        active_issue_date: str | None,
    ) -> tuple[list[dict[str, Any]], int]:
        d_from = args.get("date_from")
        d_to = args.get("date_to")
        q_low = str(state.get("query", "")).lower()
        q_arg = str(args.get("query", "")).lower()
        is_archive_wide = bool(
            is_archive_wide_newspaper_query(q_low)
            or is_archive_wide_newspaper_query(q_arg)
            or re.search(r"\b(?:no|number|count|how many|all|total)\s+(?:of\s+)?newspapers?\b", q_low)
            or any(w in q_low for w in ["all available", "all newspaper", "both newspaper", "across newspaper"])
        )
        has_date_in_q = bool(re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", f"{q_low} {q_arg}"))
        if is_archive_wide and not has_date_in_q and not args.get("issue_date") and not d_from and not d_to or d_from or d_to:
            iss_date = None
        else:
            iss_date = args.get("issue_date") or args.get("date") or (active_issue_date if not is_archive_wide else None)

        named_in_q = any(pat.search(q_low) or pat.search(q_arg) for pat, _ in _KNOWN_BRANDS_PATTERNS)
        if (is_archive_wide and not named_in_q) or args.get("newspaper_name") == "":
            np_name = None
        else:
            np_name = args.get("newspaper_name") or (active_newspaper_name if not is_archive_wide else None)

        iss_res = await self._sql_analytics.count_issues(
            newspaper_name=np_name,
            issue_date=iss_date,
            date_from=d_from,
            date_to=d_to,
        )
        c_val = iss_res.get("count", 0)
        hits_count = c_val
        filt_info = ", ".join(f"{k}: {v}" for k, v in iss_res.get("filters", {}).items() if v)
        if d_from and d_to:
            target_date_val = f"{d_from} to {d_to}"
        else:
            target_date_val = iss_res.get("filters", {}).get("issue_date") or iss_date or "Overview"

        if c_val > 0:
            matching_nps = iss_res.get("newspapers", [])
            nps_count = len(matching_nps)
            nps_str = ", ".join(matching_nps) or (np_name or "All Newspapers")
            issues_sample = ", ".join(
                f"{iss.get('newspaper')} ({iss.get('issue_date')})"
                for iss in iss_res.get("issues", [])[:5]
            )
            summary_str = (
                f"=== RELATIONAL ISSUE COUNT AUDIT ===\n"
                f"• Total Matching Issues: {c_val}\n"
                f"• Total Distinct Newspapers: {nps_count}\n"
                f"• Distinct Publication Count: {nps_count}\n"
                f"• Target Date / Range: {target_date_val}\n"
                f"• Newspaper(s): {nps_str}\n"
                f"• Active Filters: {filt_info or 'None'}\n"
                f"• Issues Found: {issues_sample}\n"
            )
            hl_text = f"Issue & Newspaper Count Analysis: {nps_count} newspapers ({c_val} issues) found for {target_date_val}"
        else:
            rng = iss_res.get("archive_range")
            rng_str = f"{rng['start']} to {rng['end']}" if rng else "Archive Range Available"
            all_nps = ", ".join(iss_res.get("archive_newspapers", [])[:10])
            summary_str = (
                f"=== RELATIONAL ISSUE COUNT AUDIT ===\n"
                f"• Target Date / Range: {target_date_val}\n"
                f"• Total Matching Issues: 0\n"
                f"• Total Distinct Newspapers: 0\n"
                f"• Newspaper Scope: {np_name or 'All Newspapers'}\n"
                f"• Verification Status: No newspaper issues are available in the archive for {target_date_val}.\n"
                f"• Archive Coverage Range: {rng_str}\n"
                f"• Available Publications in Archive: {all_nps or 'None'}\n"
            )
            hl_text = f"Archive Availability Audit: 0 issues found for {target_date_val}"

        items = [
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
                    "total_issues": c_val,
                    "distinct_newspapers_count": len(iss_res.get("newspapers", [])) if c_val > 0 else 0,
                    "target_date": target_date_val,
                    "newspapers": iss_res.get("newspapers", []),
                    "archive_range": iss_res.get("archive_range"),
                    "archive_newspapers": iss_res.get("archive_newspapers", []),
                    "filters": iss_res.get("filters", {}),
                },
            }
        ]
        return items, hits_count

    async def _exec_sql_count_articles(
        self,
        args: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
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

        items: list[dict[str, Any]] = [
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
        ]

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

        return items, hits_count

    async def _exec_sql_photo_counts(
        self,
        args: dict[str, Any],
        active_issue_id: int | None,
        active_newspaper_name: str | None,
        active_issue_date: str | None,
    ) -> tuple[list[dict[str, Any]], int]:
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
        items = [
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
        ]
        return items, hits_count

    async def _exec_sql_topic_distribution(self) -> tuple[list[dict[str, Any]], int]:
        topics_dist = await self._sql_analytics.get_topic_distribution()
        hits_count = len(topics_dist)
        top_lines = [
            f"• [{t['section']} / {t['article_type']}]: {t['count']} articles (avg prominence: {t['avg_prominence']})"
            for t in topics_dist[:10]
        ]
        summary_str = "=== TOPIC & SECTION DISTRIBUTION ===\n" + "\n".join(top_lines)
        items = [
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
        ]
        return items, hits_count

    async def _exec_sql_frontpage_ratio(self) -> tuple[list[dict[str, Any]], int]:
        ratio_res = await self._sql_analytics.get_frontpage_prominence_ratio()
        hits_count = ratio_res.get("total_articles", 0)
        summary_str = (
            f"=== FRONTPAGE PROMINENCE RATIO ===\n"
            f"• Total Articles: {ratio_res.get('total_articles')}\n"
            f"• Frontpage Articles (Page 1): {ratio_res.get('frontpage_articles')}\n"
            f"• Frontpage Ratio: {round(ratio_res.get('frontpage_ratio', 0) * 100, 2)}%\n"
        )
        items = [
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
        ]
        return items, hits_count

    async def _exec_sql_coverage_comparison(
        self,
        args: dict[str, Any],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self._coverage_analyzer:
            return [], 0

        cov_matrix = await self._coverage_analyzer.generate_coverage_matrix(
            query_or_event=args.get("query", state["query"]),
            target_date=args.get("target_date") or args.get("issue_date"),
        )
        items = [
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
        ]
        return items, cov_matrix.covered_count

    async def _exec_sql_coverage_difference(
        self,
        args: dict[str, Any],
        active_issue_date: str | None,
    ) -> tuple[list[dict[str, Any]], int]:
        src_np = str(args.get("newspaper_name") or args.get("source_newspaper") or "").strip()
        cmp_np = str(args.get("comparison_newspaper") or "").strip()
        iss_dt = args.get("issue_date") or args.get("target_date") or active_issue_date

        if not src_np or not cmp_np:
            return [
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
            ], 0

        diff_res = await self._sql_analytics.get_newspaper_coverage_difference(
            source_newspaper=src_np,
            comparison_newspaper=cmp_np,
            issue_date=iss_dt,
        )
        if "error" in diff_res:
            return [
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
            ], 0

        items = [
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
        ]
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
        return items, len(diff_res.get("exclusive_articles", []))

    async def _exec_sql_shared_coverage(
        self,
        args: dict[str, Any],
        active_issue_date: str | None,
    ) -> tuple[list[dict[str, Any]], int]:
        src_np = str(args.get("newspaper_name") or args.get("source_newspaper") or "").strip()
        cmp_np = str(args.get("comparison_newspaper") or "").strip()
        iss_dt = args.get("issue_date") or args.get("target_date") or active_issue_date

        if not src_np or not cmp_np:
            return [
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
            ], 0

        shared_res = await self._sql_analytics.get_newspaper_shared_coverage(
            newspaper_a=src_np,
            newspaper_b=cmp_np,
            issue_date=iss_dt,
        )
        if "error" in shared_res:
            return [
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
            ], 0

        shared_stories = shared_res.get("shared_stories", [])
        hits_count = len(shared_stories)

        # 1. Macro overview snippet (article_id: 0)
        items = [
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
        ]

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

        return items, hits_count


__all__ = ["SQLAnalyticsDispatcher"]
