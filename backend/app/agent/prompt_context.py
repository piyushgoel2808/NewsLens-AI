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

    q_lower = (query or "").lower()
    quoted_hl_match = re.search(r"[\"“]([^\"”]{8,150})[\"”]", query or "")
    target_hl = quoted_hl_match.group(1).strip().lower() if quoted_hl_match else ""
    is_article_focus = bool(
        target_hl
        or re.search(
            r"\b(?:tell me about|explain|summ[ae]ri[sz]e|find|read|what does|describe|details? of|takeaways? from)\s+(?:about|on|for)?\s*(?:this|the|that|ir)?\s*(?:article|story|piece|it)\b",
            q_lower,
        )
        or re.search(r"\b(?:in\s+\d+\s+words?|brief\s+summary|key\s+takeaways?)\b", q_lower)
    ) and not any(w in q_lower for w in ["how many", "count", "list all", "catalog", "compare all"])

    if domain_name and evidence_items:
        sorted_evidence = sorted(
            evidence_items,
            key=lambda it: score_evidence_item(it, domain_name),
            reverse=True,
        )
    else:
        sorted_evidence = list(evidence_items or [])

    # If this query targets a specific article, prioritize the matching article and cap extraneous noise
    if is_article_focus and sorted_evidence:
        matching_idx = 0
        if target_hl:
            for i, it in enumerate(sorted_evidence):
                it_hl = (it.get("headline") or "").lower()
                if target_hl in it_hl or it_hl in target_hl:
                    matching_idx = i
                    break
        else:
            for i, it in enumerate(sorted_evidence):
                it_hl = (it.get("headline") or "").lower().strip()
                if it_hl and len(it_hl) > 10 and (it_hl in q_lower or any(part in q_lower for part in it_hl.split() if len(part) > 5)):
                    matching_idx = i
                    break
        target_item = sorted_evidence.pop(matching_idx)
        target_art_id = target_item.get("article_id")
        target_pages = set(target_item.get("pages") or [])
        related_items = [
            it for it in sorted_evidence
            if (target_art_id and it.get("article_id") == target_art_id)
            or (target_pages and set(it.get("pages") or []) & target_pages and not it.get("is_advertisement"))
        ]
        budgeted_items = [target_item] + (related_items[:2] if related_items else [])
    else:
        has_manifest_evidence = any(
            item.get("source_tool", "").startswith("sql_analytics")
            or "RELATIONAL ARCHIVE MANIFEST" in str(item.get("snippet", ""))
            or "Issue Manifest:" in str(item.get("headline", ""))
            for item in sorted_evidence
        )
        item_cap = 30 if has_manifest_evidence else 12
        budgeted_items = sorted_evidence[:item_cap] if sorted_evidence else []

    is_visual_query = any(
        w in q_lower
        for w in ["photo", "image", "picture", "infographic", "chart", "graphic", "visual"]
    )

    seen_keys: set[str] = set()
    context_blocks: list[str] = []

    for item in budgeted_items:
        sanitize_evidence_item(item)
        hl = item.get("headline", "Untitled Article")

        # Prefer rich parent article full-text for target article when available
        parent_txt = item.get("parent_article_text") or ""
        base_text = item.get("snippet", "")
        if is_article_focus and parent_txt and len(parent_txt) > len(base_text):
            text = parent_txt
        else:
            text = base_text

        dedup_key = f"{hl.lower().strip()}_{text[:80].lower().strip()}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        is_manifest_or_matrix = (
            item.get("source_tool") in ("sql_analytics", "coverage_analysis", "inspect_visual_asset", "dynamic_analysis")
            or item.get("is_visual_asset")
            or "RELATIONAL ARCHIVE MANIFEST" in text
            or "COVERAGE RECONCILIATION MATRIX" in text
            or "VERIFIED EXCLUSIVE COVERAGE" in text
            or "VERIFIED SHARED SYNDICATED WIRE COVERAGE" in text
            or "VISUAL DATA ASSET:" in text
            or "DYNAMIC ANALYTICAL" in text.upper()
        )
        if is_article_focus:
            max_chars = 7500
        elif is_manifest_or_matrix:
            max_chars = 4500
        else:
            max_chars = 1800

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
        elif bool(
            item.get("is_statistical_metric")
            or (item.get("source_tool") in ("dynamic_analysis", "sql_analytics") and item.get("article_id") == 0)
        ):
            np_name = item.get("newspaper_name") or "Archive Analytics"
            dt = item.get("issue_date") or ""
            dt_str = f"Date / Period: {dt}\n" if dt else ""
            context_blocks.append(
                f"--- ARCHIVE STATISTICAL ANALYTICS & METADATA EXCERPT [{idx}] ---\n"
                f"Source: Archive Statistical Analytics ({np_name})\n"
                f"{dt_str}"
                f"Calculated Finding & Scope:\n{text}\n"
                f"[Quantitative Archive Metric: State these verified figures directly. Do NOT cite as a printed newspaper story.]\n"
            )
        else:
            np_name = item.get("newspaper_name", "Unknown Publication")
            dt = item.get("issue_date", "Unknown Date")
            pages = item.get("pages", [1])
            page_val = int(pages[0]) if pages and pages[0] else 1
            sec_val = item.get("section")
            wc_val = item.get("word_count")
            extra_meta = []
            if sec_val:
                extra_meta.append(f"Section: {sec_val}")
            if wc_val:
                extra_meta.append(f"Word Count: {wc_val} words")
            meta_suffix = (", " + ", ".join(extra_meta)) if extra_meta else ""
            evidence_tag = f'[Evidence: {np_name}, {dt}, Page {page_val}{meta_suffix}, Headline: "{hl}"]'

            meta_lines = "".join(f"{m}\n" for m in extra_meta)

            photos = item.get("photos") or []
            photos_text = ""
            if photos:
                included_photos = photos[:1] if (is_article_focus and not is_visual_query) else photos[:3]
                photo_lines = [
                    f"  * [{p.get('visual_type') or 'Photo'} {p_idx}] Caption: \"{p.get('caption') or 'No printed caption'}\""
                    + (f" | Visual Scene: {p['vlm_description']}" if (p.get("vlm_description") and (not is_article_focus or is_visual_query)) else "")
                    for p_idx, p in enumerate(included_photos, 1)
                ]
                photos_text = "\nAttached Photos & Visual Elements:\n" + "\n".join(photo_lines) + "\n"

            context_blocks.append(
                f"--- ARCHIVE EVIDENCE EXCERPT [{idx}] ---\n"
                f"{evidence_tag}\n"
                f"Publication: {np_name}\n"
                f"Date: {dt}\n"
                f"Page: {page_val}\n"
                f"{meta_lines}"
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

    is_shared = any(
        w in query.lower()
        for w in [
            "similar", "shared", "common", "same article", "same articles",
            "same story", "same stories", "both newspaper", "both newspapers",
            "both paper", "both papers", "in both", "covered by both", "both carried",
            "syndicated", "wire stories", "wire story",
        ]
    ) or any(
        "VERIFIED SHARED SYNDICATED WIRE COVERAGE" in str(item.get("snippet", ""))
        or item.get("source_tool") == "sql_analytics_shared"
        for item in evidence_items
    )
    shared_note = (
        "SHARED WIRE COVERAGE DIRECTIVE:\n"
        "- The user is specifically requesting identical, similar, or shared syndicated wire stories covered by both newspapers.\n"
        "- Structure your response around the verified shared wire matches provided in the evidence.\n"
        "- Strictly adhere to verified wire pairings. NEVER fabricate false equivalence between unrelated local stories.\n\n"
        if is_shared else ""
    )

    return (
        f"User Research Query: {query}\n"
        f"Query Archetype: {archetype}\n"
        f"{pubs_note}"
        f"{dates_note}"
        f"{isolation_rule}"
        f"{domain_note}"
        f"{shared_note}"
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
