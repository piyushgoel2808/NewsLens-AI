"""Evidence Context Formatter and Prompt Builder for NewsLens-AI."""

from __future__ import annotations

import re
from typing import Any

from app.agent.taxonomy import detect_domain_from_query, score_evidence_item
from app.retrieval.sanitizer import repair_text_ligatures
from app.retrieval.sql_analytics import sanitize_headline

_CHUNK_TAG_REGEX = re.compile(
    r"\[(?:Newspaper|Page\(s\)|Exact Chunk Match|Visual Data Asset|Article Parent Context|📷\s*Attached Image/Photo):?.*?\]",
    re.IGNORECASE,
)


def clean_snippet(snip: str) -> str:
    """Strip retrieval artifacts and chunk tags from snippet text."""
    cleaned = _CHUNK_TAG_REGEX.sub("", snip).strip()
    cleaned = re.sub(r"^[\<\#\s\.\,\-]+", "", cleaned).strip()
    return repair_text_ligatures(cleaned)


def sanitize_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    """Sanitize headlines, bylines, and snippet text for an evidence item."""
    raw_hl = item.get("headline", "Untitled Article")
    sub_hl = item.get("subheadline")
    byline = item.get("byline_author")
    snip = item.get("snippet") or item.get("summary") or item.get("full_text") or ""

    hl, eff_byline = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
    item["headline"] = repair_text_ligatures(hl)
    if eff_byline:
        item["byline_author"] = eff_byline

    raw_text = (item.get("snippet") or item.get("full_text") or item.get("summary") or "").strip()
    item["snippet"] = repair_text_ligatures(raw_text)
    return item


def build_evidence_context(evidence_items: list[dict[str, Any]], query: str = "") -> str:
    """Format retrieved evidence documents into structured prompt context with strict token budgeting."""
    domain_name = detect_domain_from_query(query, evidence_items) if query else None

    if domain_name and evidence_items:
        sorted_evidence = sorted(
            evidence_items,
            key=lambda it: score_evidence_item(it, domain_name),
            reverse=True,
        )
        budgeted_items = sorted_evidence[:12]
    else:
        budgeted_items = evidence_items[:12] if evidence_items else []

    seen_keys: set[str] = set()
    context_blocks: list[str] = []

    for item in budgeted_items:
        sanitize_evidence_item(item)
        hl = item.get("headline", "Untitled Article")
        text = item.get("snippet", "")

        dedup_key = f"{hl.lower().strip()}_{text[:80].lower().strip()}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        is_manifest_or_matrix = (
            item.get("source_tool") in ("sql_analytics", "coverage_analysis", "inspect_visual_asset")
            or item.get("is_visual_asset")
            or "RELATIONAL ARCHIVE MANIFEST" in text
            or "COVERAGE RECONCILIATION MATRIX" in text
            or "VERIFIED EXCLUSIVE COVERAGE" in text
            or "VISUAL DATA ASSET:" in text
        )
        max_chars = 4500 if is_manifest_or_matrix else 1200
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ... [excerpt truncated for length]"

        idx = len(context_blocks) + 1
        is_web = bool(item.get("is_web") or item.get("source_tool") == "web_search")
        is_visual = bool(item.get("is_visual_asset") or item.get("source_tool") == "inspect_visual_asset")

        if is_web:
            url = item.get("url", "")
            src = item.get("newspaper_name", "Live Web")
            dt = item.get("issue_date", "Current")
            context_blocks.append(
                f"--- LIVE WEB EVIDENCE EXCERPT [{idx}] ---\n"
                f"Source: {src}\n"
                f"Title: {hl}\n"
                f"URL: {url}\n"
                f"Date: {dt}\n"
                f"Content:\n{text}\n"
            )
        elif is_visual:
            np_name = item.get("newspaper_name", "Unknown Publication")
            dt = item.get("issue_date", "Unknown Date")
            pages = item.get("pages", [1])
            page_val = int(pages[0]) if pages and pages[0] else 1
            evidence_tag = f'[Evidence: {np_name}, {dt}, Page {page_val}, Headline: "{hl}"]'
            v_pid = item.get("photo_id", "")
            v_type = item.get("visual_type", "infographic")
            context_blocks.append(
                f"--- VISUAL INFOGRAPHIC & DATA ASSET EXCERPT [{idx}] ---\n"
                f"{evidence_tag}\n"
                f"Publication: {np_name}\n"
                f"Date: {dt}\n"
                f"Page: {page_val}\n"
                f"Headline: {hl}\n"
                f"Visual Asset #{v_pid} ({v_type})\n"
                f"Content:\n{text}\n"
            )
        else:
            np_name = item.get("newspaper_name", "Unknown Publication")
            dt = item.get("issue_date", "Unknown Date")
            pages = item.get("pages", [1])
            page_val = int(pages[0]) if pages and pages[0] else 1
            evidence_tag = f'[Evidence: {np_name}, {dt}, Page {page_val}, Headline: "{hl}"]'

            photos = item.get("photos") or []
            photos_text = ""
            if photos:
                photo_lines = [
                    f"  * [{p.get('visual_type') or 'Photo'} {p_idx}] Caption: \"{p.get('caption') or 'No printed caption'}\""
                    + (f" | Visual Scene: {p['vlm_description']}" if p.get("vlm_description") else "")
                    for p_idx, p in enumerate(photos, 1)
                ]
                photos_text = "\nAttached Photos & Visual Elements:\n" + "\n".join(photo_lines) + "\n"

            context_blocks.append(
                f"--- ARCHIVE EVIDENCE EXCERPT [{idx}] ---\n"
                f"{evidence_tag}\n"
                f"Publication: {np_name}\n"
                f"Date: {dt}\n"
                f"Page: {page_val}\n"
                f"Headline: {hl}\n"
                f"Content:\n{text}\n"
                f"{photos_text}"
            )
    return "\n".join(context_blocks)


def build_synthesizer_user_prompt(
    query: str,
    archetype: str,
    evidence_items: list[dict[str, Any]],
    context: str,
) -> str:
    """Construct grounded synthesizer prompt with explicit publication boundaries."""
    verified_pubs = sorted(list({
        str(item.get("newspaper_name", "")).strip() for item in evidence_items
        if item.get("newspaper_name") and item.get("newspaper_name") not in (
            "Multi-Newspaper Audit", "Aggregated Archive Analytics", "Archive", "Unknown Publication", "Live Web"
        )
    }))
    verified_dates = sorted(list({
        str(item.get("issue_date", "")).strip() for item in evidence_items
        if item.get("issue_date") and str(item.get("issue_date")).strip() not in (
            "", "Overview", "Live Web", "Unknown Date"
        )
    }))

    pubs_note = f"Verified Available Publications for this Query: {', '.join(verified_pubs)}\n" if verified_pubs else ""
    dates_note = f"Verified Target Issue Date(s): {', '.join(verified_dates)}\n" if verified_dates else ""

    date_anchor_clause = f" on date(s): {', '.join(verified_dates)}" if verified_dates else ""
    isolation_rule = (
        f"STRICT PUBLICATION & DATE ISOLATION:\n"
        f"- You must ONLY report on and analyze the verified publications present in the current evidence ({', '.join(verified_pubs) or 'Current Evidence'}){date_anchor_clause}.\n"
        f"- Target Date Anchoring: All synthesized summaries, tables, and references must strictly reflect the verified date(s): {', '.join(verified_dates) or 'current query date'}. NEVER carry forward or conflate dates discussed in earlier conversation turns.\n"
        f"- NEVER mention, summarize, or cite articles from other publications or dates not in the current evidence.\n\n"
        if verified_pubs or verified_dates else ""
    )
    domain = detect_domain_from_query(query, evidence_items)
    domain_note = (
        f"TARGET DOMAIN FOCUS: {domain}\n"
        f"- Strictly filter your synthesis to {domain} topics, markets, figures, and developments.\n"
        f"- Discard any unrelated general news (sports, crime, local repairs) from the final response.\n\n"
        if domain else ""
    )

    return (
        f"User Research Query: {query}\n"
        f"Query Archetype: {archetype}\n"
        f"{pubs_note}"
        f"{dates_note}"
        f"{isolation_rule}"
        f"{domain_note}"
        f"Available Newspaper Evidence:\n"
        f"{context or 'No new search results—refer to conversation history if applicable.'}\n\n"
        f"Synthesize an insightful, highly-structured executive intelligence response."
    )


__all__ = [
    "build_evidence_context",
    "build_synthesizer_user_prompt",
    "clean_snippet",
    "sanitize_evidence_item",
]
