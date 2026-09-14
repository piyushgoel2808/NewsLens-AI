"""Presentation and manifest formatting helpers for broadsheet retrieval evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.retrieval.sanitizer import repair_text_ligatures

if TYPE_CHECKING:
    from app.retrieval.coverage_analyzer import CoverageMatrix


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


__all__ = [
    "format_issue_manifest",
    "format_coverage_matrix_snippet",
    "format_coverage_difference_snippet",
    "format_shared_coverage_snippet",
]
