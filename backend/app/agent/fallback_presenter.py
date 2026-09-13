"""Deterministic Offline Report Presenter and Markdown Generator for NewsLens-AI."""

from __future__ import annotations

from typing import Any

from app.agent.prompt_context import clean_snippet, sanitize_evidence_item
from app.agent.taxonomy import DOMAIN_TAXONOMY, detect_domain_from_query, is_domain_match

EMPTY_EVIDENCE_RESPONSE = (
    "I could not find any evidence or articles matching this query in the database. "
    "Please try adjusting your search terms."
)


def has_valid_evidence(evidence_items: list[dict[str, Any]]) -> bool:
    """Check whether evidence contains non-empty grounded content."""
    if not evidence_items:
        return False
    for item in evidence_items:
        snip = item.get("snippet") or item.get("full_text") or item.get("summary") or ""
        hl = item.get("headline") or ""
        if len(snip.strip()) >= 5 or len(hl.strip()) >= 5:
            return True
    return False


def render_comparison_matrix(pub_groups: dict[str, list[dict[str, Any]]], domain: str) -> list[str]:
    """Render cross-newspaper comparison matrix table with zero-coverage handling."""
    lines = [
        f"### 📊 Cross-Newspaper {domain} Comparison Matrix\n",
        "| Publication | Issue Date | Coverage Focus | Key Verified Highlights |",
        "|---|---|---|---|",
    ]
    for pub, items in pub_groups.items():
        pub_reals = [
            it for it in items
            if not (it.get("headline") or "").startswith("Issue Manifest:")
            and "exact chunk match" not in (it.get("headline") or "").lower()
            and is_domain_match(it, domain)
        ]
        if pub_reals:
            lead_item = pub_reals[0]
            hl = lead_item.get("headline") or "General Reporting"
            if hl.startswith("Issue Manifest:"):
                hl = f"General {domain or 'News'} Archive Coverage"
            dt_val = lead_item.get("issue_date") or items[0].get("issue_date", "")
            lines.append(f"| **{pub}** | {dt_val} | {len(pub_reals)} verified item(s) | {hl} |")
        else:
            dt_val = items[0].get("issue_date", "")
            lines.append(
                f"| **{pub}** | {dt_val} | No standalone {domain} reporting | "
                f"Carried no standalone {domain} reporting in this edition |"
            )
    lines.append("\n### 📌 Key Verified Sector Highlights & Policies")
    return lines


def render_front_page_comparison(
    pub_groups: dict[str, list[dict[str, Any]]], domain: str | None
) -> list[str]:
    """Render front page headline comparisons across publications."""
    lines = ["### 📰 Front-Page (Page 1) Lead Stories Comparison"]
    for pub, items in pub_groups.items():
        pub_reals = [
            it for it in items
            if not (it.get("headline") or "").startswith("Issue Manifest:")
            and "exact chunk match" not in (it.get("headline") or "").lower()
        ]
        lead_item = pub_reals[0] if pub_reals else items[0]
        hl = lead_item.get("headline") or "Front-page report"
        if hl.startswith("Issue Manifest:"):
            hl = f"General {domain or 'News'} Archive Coverage"
        p_num = lead_item.get("pages", [1])[0]
        dt_val = lead_item.get("issue_date") or items[0].get("issue_date", "")
        lines.append(f"- **{pub}**: Front-page lead report '{hl}' [{pub}, {dt_val}, Page {p_num}, \"{hl}\"]")
    lines.append("\n### 📊 Section Distribution & Coverage Scale")
    return lines


def render_broadsheet_perspectives(
    pub_groups: dict[str, list[dict[str, Any]]], domain: str | None
) -> list[str]:
    """Render publication perspectives and zero-coverage notes."""
    lines = ["\n### 📰 Broadsheet Perspectives"]
    for pub, items in pub_groups.items():
        pub_reals = [
            it for it in items
            if not (it.get("headline") or "").startswith("Issue Manifest:")
            and "exact chunk match" not in (it.get("headline") or "").lower()
            and is_domain_match(it, domain)
        ]
        if pub_reals:
            top_hl = pub_reals[0].get("headline") or "Reporting"
            if top_hl.startswith("Issue Manifest:"):
                top_hl = f"General {domain or 'News'} Coverage"
            lines.append(f"- **{pub}**: Emphasized '{top_hl}' across {len(pub_reals)} related report(s).")
        else:
            lines.append(f"- **{pub}**: Carried no dedicated {domain} reports in this issue.")
    return lines


def render_explore_further(pub_groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Render explore further follow-up hints."""
    lines = ["\n### 🔍 Explore Further"]
    for pub in list(pub_groups.keys())[:2]:
        lines.append(f"> 💡 Explore: What was {pub}'s detailed coverage on this topic?")
    return lines


def generate_deterministic_summary(
    query: str,
    evidence_items: list[dict[str, Any]],
    archetype: str = "factual_lookup",
) -> str:
    """Structured deterministic grounded synthesis when all LLMs are offline."""
    if not has_valid_evidence(evidence_items):
        return EMPTY_EVIDENCE_RESPONSE

    # 1. Sanitize all evidence items and repair ligatures
    for item in evidence_items:
        sanitize_evidence_item(item)

    # 2. Domain & keyword filtering
    domain = detect_domain_from_query(query, evidence_items)
    query_words = {w.lower() for w in query.split() if len(w) > 3}
    if domain and domain in DOMAIN_TAXONOMY:
        query_words.update(DOMAIN_TAXONOMY[domain]["stems"])

    if archetype == "cross_newspaper_comparison":
        relevant_items = evidence_items
    else:
        relevant_items = [
            item for item in evidence_items
            if not query_words or any(
                qw in (
                    (item.get("headline") or "")
                    + " "
                    + (item.get("snippet") or "")
                    + " "
                    + (item.get("summary") or "")
                ).lower()
                for qw in query_words
            )
        ]

    filtered_evidence = relevant_items if relevant_items else evidence_items

    # 3. Partition real news articles vs manifests
    real_articles = [
        item for item in filtered_evidence
        if not (item.get("headline") or "").startswith("Issue Manifest:")
        and "exact chunk match" not in (item.get("headline") or "").lower()
        and item.get("newspaper_name") not in ("Multi-Newspaper Audit", "Aggregated Archive Analytics", "Archive Analytics", "Statistical Engine")
        and not item.get("is_statistical_metric")
    ]

    # If evidence consists solely of statistical/analytical calculations, return direct quantitative finding
    if not real_articles and any(it.get("is_statistical_metric") or it.get("source_tool") in ("dynamic_analysis", "sql_analytics") for it in filtered_evidence):
        stat_item = filtered_evidence[0]
        stat_snip = clean_snippet(stat_item.get("snippet") or stat_item.get("summary") or "")
        lines = [
            "### ⚡ Direct Finding\n",
            stat_snip,
            "\n### 📊 Key Computed Metrics",
        ]
        meta = stat_item.get("metadata") or {}
        if meta:
            for k, v in meta.items():
                k_clean = k.replace("_", " ").title()
                lines.append(f"- **{k_clean}**: {v}")
        else:
            lines.append(f"- **Verified Archive Metadata**: {stat_snip}")
        return "\n".join(lines)

    first = (real_articles or filtered_evidence)[0]
    first_np = first.get("newspaper_name", "Daily News")
    first_dt = first.get("issue_date", "")

    pub_groups: dict[str, list[dict[str, Any]]] = {}
    for item in filtered_evidence:
        np_name = item.get("newspaper_name", "Archive")
        if np_name not in ("Multi-Newspaper Audit", "Aggregated Archive Analytics", "Archive"):
            pub_groups.setdefault(np_name, []).append(item)

    if not pub_groups:
        for item in filtered_evidence[:6]:
            np_name = item.get("newspaper_name", "Archive")
            pub_groups.setdefault(np_name, []).append(item)

    lines: list[str] = []

    # 4. Render sections by archetype
    if archetype == "cross_newspaper_comparison":
        shared_manifest = next((it for it in filtered_evidence if "VERIFIED SHARED SYNDICATED WIRE COVERAGE" in (it.get("snippet") or "")), None)
        if shared_manifest:
            lines.append("### ⚡ Executive Summary: Shared Syndicated Coverage\n")
            lines.append(clean_snippet(shared_manifest.get("snippet") or "") + "\n")
            lines.extend(render_broadsheet_perspectives(pub_groups, domain))
            lines.extend(render_explore_further(pub_groups))
            return "\n".join(lines)

        summary_desc = (
            f"Comparative analysis across verified broadsheet archives covering {domain or 'regional news'} developments."
        )
        lines.append(
            f"### ⚡ Executive Summary: {domain} Intelligence" if domain else "### ⚡ Executive Summary: Broadsheet Edition Comparison"
        )
        lines.append(f"{summary_desc}\n")

        if domain:
            lines.extend(render_comparison_matrix(pub_groups, domain))
        else:
            lines.extend(render_front_page_comparison(pub_groups, domain))
    else:
        lines.append("### ⚡ Executive Summary")
        lines.append(
            f"Archival broadsheet reporting covering {domain or 'regional news'} was documented across "
            f"regional publications, led by *{first_np}* ({first_dt}).\n"
        )
        lines.append("### 📌 Key Verified Facts & Highlights")

    # Prioritize real news articles for facts and highlights
    domain_reals = [a for a in real_articles if is_domain_match(a, domain)]
    display_facts = domain_reals[:6] if domain_reals else real_articles[:6] if real_articles else filtered_evidence[:6]
    for item in display_facts:
        np_name = item.get("newspaper_name", "Archive")
        dt = item.get("issue_date", "")
        pages = item.get("pages", [1])
        page_str = f"Page {pages[0]}" if pages else "Page 1"
        hl = item.get("headline", "Untitled")
        clean_snip = clean_snippet(item.get("snippet") or item.get("summary") or "")
        lines.append(f'- **{hl}**: {clean_snip[:180]}... [{np_name}, {dt}, {page_str}, "{hl}"]')

    lines.extend(render_broadsheet_perspectives(pub_groups, domain))
    lines.extend(render_explore_further(pub_groups))

    return "\n".join(lines)


__all__ = [
    "EMPTY_EVIDENCE_RESPONSE",
    "generate_deterministic_summary",
    "has_valid_evidence",
    "render_broadsheet_perspectives",
    "render_comparison_matrix",
    "render_explore_further",
    "render_front_page_comparison",
]
