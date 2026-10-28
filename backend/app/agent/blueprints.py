"""Dynamic Answer Blueprint Registry and Intent-Proportional Blueprint Compiler for NewsLens-AI.

Constructs query-adaptive, parameter-aware AnswerBlueprint objects rather than forcing
rigid, static pre-baked templates that ignore query parameters, domains, and page scopes.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from app.agent.decision_layer import (
    is_availability_query,
    is_macro_edition_query,
    is_ordinal_lead_story_query,
)
from app.agent.extractor import extract_parameters_from_query
from app.agent.models import AnswerBlueprint, SectionSpec
from app.agent.synthesizer_prompts import compile_structure_from_blueprint
from app.agent.taxonomy import DOMAIN_TAXONOMY, detect_domain_from_query

DEFAULT_BLUEPRINTS: dict[str, AnswerBlueprint] = {
    "archive_availability": AnswerBlueprint(
        user_intent="archive_availability",
        overall_tone="concise_atomic",
        target_word_count=90,
        sections=[
            SectionSpec(
                title="### ⚡ Availability Status",
                format_type="narrative",
                content_focus="Direct, authoritative statement stating whether newspaper issues exist in the archive for the queried date or publication.",
                target_length="1 to 2 crisp sentences",
            ),
            SectionSpec(
                title="### 📋 Archive Scope & Available Coverage",
                format_type="bullet_list",
                content_focus="Compact list of available newspapers on that date, or if none, the verified archive date range and available publications.",
                target_length="Compact bullet points",
            ),
        ],
        prohibited_elements=[
            "speculative corporate strategy or publication planning advice",
            "fake future collaboration suggestions",
            "claiming positive availability when count is zero",
            "claiming 1 or more newspapers when count is zero",
            "artificial explore further questions",
            "conversational filler",
        ],
    ),
    "scalar_count_metric": AnswerBlueprint(
        user_intent="scalar_count_metric",
        overall_tone="concise_atomic",
        target_word_count=80,
        sections=[
            SectionSpec(
                title="### ⚡ Direct Finding",
                format_type="narrative",
                content_focus="Direct, authoritative answer with the verified number, publication, and date scope.",
                target_length="1 to 2 sentences",
            ),
            SectionSpec(
                title="### 📊 Key Computed Metrics",
                format_type="metric_card",
                content_focus="Compact metric summary of verified archive counts and active query filters.",
                target_length="Compact metric card or bullet list",
            ),
        ],
        prohibited_elements=[
            "empty or single-row article catalog tables",
            "fake sector highlights",
            "artificial explore further questions",
            "conversational filler",
        ],
    ),
    "single_article_summary": AnswerBlueprint(
        user_intent="single_article_summary",
        overall_tone="executive_brief",
        sections=[
            SectionSpec(
                title="### ⚡ Executive Summary",
                format_type="narrative",
                content_focus="Main news development, operational details, key figures, and significance directly from the article text.",
                target_length="1-2 concise paragraphs",
            ),
            SectionSpec(
                title="### 📌 Key Takeaways & Operational Highlights",
                format_type="bullet_list",
                content_focus="Key verified operational facts, statistics, personnel counts, dates, and locations.",
                target_length="3 to 4 concise bullets",
            ),
            SectionSpec(
                title="### 🔍 Explore Further",
                format_type="bullet_list",
                content_focus="Specific journalistic follow-up angles or broader implications.",
                target_length="2 concise follow-up prompts",
            ),
        ],
        prohibited_elements=[
            "robotic catalog tables",
            "article breakdown lists",
            "metadata inventories",
            "unrelated visual element descriptions",
            "conversational filler",
        ],
    ),
    "ordinal_lead_story": AnswerBlueprint(
        user_intent="ordinal_lead_story",
        overall_tone="executive_brief",
        target_word_count=180,
        sections=[
            SectionSpec(
                title="### ⚡ Lead Story Identification",
                format_type="narrative",
                content_focus="Identify the requested ordinal/lead story on the page by highest prominence score, stating its exact headline and core development.",
                target_length="1 to 2 concise paragraphs",
            ),
            SectionSpec(
                title="### 📌 Key Verified Facts & Page Context",
                format_type="bullet_list",
                content_focus="Specific verified facts, figures, and placement context from that story.",
                target_length="2 to 4 concise bullets",
            ),
        ],
        prohibited_elements=[
            "multi-row article catalog tables when only a single ordinal story was requested",
            "claiming the page only has 1 article when multiple articles exist on the page",
            "conversational filler",
        ],
    ),
    "macro_summary": AnswerBlueprint(
        user_intent="macro_edition_summary",
        overall_tone="authoritative_journalistic",
        sections=[
            SectionSpec(
                title="### ⚡ Executive Summary: Edition Overview",
                format_type="narrative",
                content_focus="High-level editorial synthesis of the newspaper issue's overarching news agenda, scale, and dominant themes.",
                target_length="1 to 2 paragraphs",
            ),
            SectionSpec(
                title="### 📰 Front-Page (Page 1) Lead Stories",
                format_type="bullet_list",
                content_focus="Top banner headlines and lead front-page developments with inline citations.",
                target_length="3 to 5 bullets",
            ),
            SectionSpec(
                title="### 📌 Section & Domain Highlights",
                format_type="bullet_list",
                content_focus="Notable reporting across Business, Politics, National, International, and Sports sections.",
                target_length="4 to 6 bullets",
            ),
            SectionSpec(
                title="### 🔍 Explore Further",
                format_type="bullet_list",
                content_focus="Follow-up deep-dives into specific sections or lead stories.",
                target_length="2 to 3 concise prompts",
            ),
        ],
        prohibited_elements=[
            "reducing a whole-newspaper summary to a bare scalar count",
            "conversational filler",
        ],
    ),
    "cross_newspaper_comparison": AnswerBlueprint(
        user_intent="cross_newspaper_tabular_comparison",
        overall_tone="analytical_comparison",
        sections=[
            SectionSpec(
                title="### ⚡ Executive Summary: Broadsheet Comparison",
                format_type="narrative",
                content_focus="Overarching synthesis of how the publications framed the issue, common threads, and key differences.",
                target_length="2 to 3 crisp paragraphs",
            ),
            SectionSpec(
                title="### 📊 Cross-Newspaper Comparison Matrix",
                format_type="markdown_table",
                content_focus="Structured comparative table across publications with verified headlines, findings, and angles.",
                target_length="1 row per publication edition",
            ),
            SectionSpec(
                title="### 📌 Key Verified Highlights & Policies",
                format_type="bullet_list",
                content_focus="Specific sector decisions, statistics, policy moves, or quotes with strict inline citations.",
                target_length="3 to 5 bullet points",
            ),
            SectionSpec(
                title="### 🎯 Editorial Framing & Divergence",
                format_type="bullet_list",
                content_focus="Comparative breakdown of editorial tone, prominence, and regional focus between papers.",
                target_length="1 bullet point per publication",
            ),
            SectionSpec(
                title="### 🔍 Explore Further",
                format_type="bullet_list",
                content_focus="Comparative follow-up angles or unresolved developments.",
                target_length="2 to 3 concise prompts",
            ),
        ],
        table_columns=["Publication", "Issue Date", "Top Headline & Page", "Core Findings", "Editorial Angle"],
        prohibited_elements=[
            "substituting stories from unrelated sections",
            "false equivalence across unrelated local articles",
            "conversational filler",
        ],
    ),
    "article_catalog": AnswerBlueprint(
        user_intent="comprehensive_article_catalog",
        overall_tone="authoritative_journalistic",
        sections=[
            SectionSpec(
                title="### ⚡ Executive Summary: Article Catalog Scope",
                format_type="narrative",
                content_focus="Scope of catalog: total articles identified, publications, dates, and topical coverage.",
                target_length="1 to 2 sentences",
            ),
            SectionSpec(
                title="### 📋 Comprehensive Articles Catalog",
                format_type="markdown_table",
                content_focus="Markdown table listing matching articles with verified headlines, pages, sections, and bylines.",
                target_length="Full table of manifest articles",
            ),
            SectionSpec(
                title="### 📌 Key Featured Stories & Highlights",
                format_type="bullet_list",
                content_focus="Notable stories, key developments, and verified highlights with inline citations.",
                target_length="3 to 5 bullets",
            ),
            SectionSpec(
                title="### 🔍 Explore Further",
                format_type="bullet_list",
                content_focus="Suggested follow-up deep-dives into specific listed stories.",
                target_length="2 to 3 prompts",
            ),
        ],
        table_columns=["#", "Publication", "Issue Date", "Page", "Section", "Headline", "Author / Byline"],
        prohibited_elements=[
            "displaying tool names as headlines",
            "displaying doctors or authors as article headlines",
            "conversational filler",
        ],
    ),
    "thematic_timeline": AnswerBlueprint(
        user_intent="chronological_timeline",
        overall_tone="authoritative_journalistic",
        sections=[
            SectionSpec(
                title="### ⚡ Executive Summary: Chronological Progression",
                format_type="narrative",
                content_focus="Overarching arc, beginning, key inflection points, and latest status.",
                target_length="1 to 2 paragraphs",
            ),
            SectionSpec(
                title="### 📅 Milestone Timeline & Event Progression",
                format_type="timeline",
                content_focus="Dated chronological sequence of milestones with key figures, actions, and citations.",
                target_length="Chronological sequence of dated milestones",
            ),
            SectionSpec(
                title="### 📈 Thematic Trajectory & Broadsheet Evolution",
                format_type="narrative",
                content_focus="Shifts in broadsheet coverage, sentiment, and editorial stance over time.",
                target_length="1 to 2 paragraphs",
            ),
            SectionSpec(
                title="### 🔍 Explore Further",
                format_type="bullet_list",
                content_focus="Follow-up timeline angles.",
                target_length="2 concise prompts",
            ),
        ],
        prohibited_elements=["inventing ungrounded dates", "conversational filler"],
    ),
    "entity_deep_dive": AnswerBlueprint(
        user_intent="entity_profile",
        overall_tone="authoritative_journalistic",
        sections=[
            SectionSpec(
                title="### ⚡ Executive Profile Summary",
                format_type="narrative",
                content_focus="Overarching entity significance, core roles, and primary news footprint.",
                target_length="1 to 2 concise paragraphs",
            ),
            SectionSpec(
                title="### 📌 Key Corporate Moves, Policies & Activities",
                format_type="bullet_list",
                content_focus="Verified actions, public statements, initiatives, and legal/regulatory developments.",
                target_length="3 to 5 bullets",
            ),
            SectionSpec(
                title="### 📰 Broadsheet Sentiment & Media Scrutiny",
                format_type="narrative",
                content_focus="Media portrayal, tone, and editorial scrutiny across broadsheets.",
                target_length="1 to 2 paragraphs",
            ),
            SectionSpec(
                title="### 🔍 Explore Further",
                format_type="bullet_list",
                content_focus="Targeted entity follow-up questions.",
                target_length="2 prompts",
            ),
        ],
        prohibited_elements=["fictional biography", "conversational filler"],
    ),
    "factual_lookup": AnswerBlueprint(
        user_intent="targeted_factual_lookup",
        overall_tone="authoritative_journalistic",
        sections=[
            SectionSpec(
                title="### ⚡ Direct Finding & Factual Context",
                format_type="narrative",
                content_focus="Direct answer to the user's specific question, citing authoritative reporting.",
                target_length="1 to 2 concise paragraphs",
            ),
            SectionSpec(
                title="### 📌 Key Verified Details & Quotes",
                format_type="bullet_list",
                content_focus="Specific facts, statistics, direct quotes, and operational specifics.",
                target_length="3 to 4 bullets",
            ),
            SectionSpec(
                title="### 🔍 Related Angles & Coverage Context",
                format_type="bullet_list",
                content_focus="Wider reporting context, related angles, and follow-up paths.",
                target_length="2 concise prompts",
            ),
        ],
        prohibited_elements=["speculative ungrounded claims", "conversational filler"],
    ),
    "quantitative_trend": AnswerBlueprint(
        user_intent="scalar_count_metric",
        overall_tone="concise_atomic",
        target_word_count=100,
        sections=[
            SectionSpec(
                title="### ⚡ Direct Finding",
                format_type="narrative",
                content_focus="Direct authoritative answer with verified number, publication, and date scope.",
                target_length="1 to 2 sentences",
            ),
            SectionSpec(
                title="### 📊 Key Computed Metrics",
                format_type="metric_card",
                content_focus="Compact metric summary of verified archive counts and active query filters.",
                target_length="Compact bullet list or metric card",
            ),
        ],
        prohibited_elements=[
            "empty catalog tables",
            "fake sector highlights",
            "artificial explore further questions",
            "conversational filler",
        ],
    ),
    "analytical_computation": AnswerBlueprint(
        user_intent="statistical_computation",
        overall_tone="analytical_comparison",
        sections=[
            SectionSpec(
                title="### ⚡ Computed Result",
                format_type="narrative",
                content_focus="Direct computed answer with exact verified number and methodology.",
                target_length="1 to 2 sentences",
            ),
            SectionSpec(
                title="### 📊 Analytical Breakdown",
                format_type="metric_card",
                content_focus="Supporting statistics, breakdowns, or intermediate computed values.",
                target_length="Compact metric card",
            ),
        ],
        prohibited_elements=["narrative speculation", "conversational filler"],
    ),
}


def build_heuristic_answer_blueprint(
    query: str,
    archetype: str,
    parameters: dict[str, Any] | None = None,
) -> AnswerBlueprint:
    """Dynamically construct a query-adaptive, parameter-aware AnswerBlueprint."""
    q_lower = (query or "").lower().strip()
    params = parameters if parameters is not None else extract_parameters_from_query(query)
    domain = detect_domain_from_query(query)

    target_word_count: int | None = None
    wc_match = re.search(r"\b(?:in|under|around|within|max(?:imum)?|less\s+than|at\s+most)\s+(\d+)\s+words?\b", q_lower)
    if wc_match:
        with contextlib.suppress(ValueError):
            target_word_count = int(wc_match.group(1))

    # 1. Single-Article Deep Dive / Word-Constrained Summary
    quoted_hl = re.search(r"[\"“]([^\"”]{8,150})[\"”]", query)
    is_target_art = bool(
        quoted_hl
        or re.search(
            r"\b(?:tell me about|explain|summ[ae]ri[sz]e|find|read|what does|describe|details? of|takeaways? from)\s+(?:about|on|for)?\s*(?:this|the|that|ir)?\s*(?:article|story|piece|it)\b",
            q_lower,
        )
        or re.search(r"\b(?:in\s+\d+\s+words?|brief\s+summary|key\s+takeaways?)\b", q_lower)
    ) and not any(w in q_lower for w in ["how many", "count", "list all", "catalog", "compare all"]) and not is_macro_edition_query(query)

    if is_target_art:
        bp = DEFAULT_BLUEPRINTS["single_article_summary"].model_copy(deep=True)
        hl_label = quoted_hl.group(1).strip() if quoted_hl else (params.get("headline") or "")
        if hl_label:
            bp.sections[0].title = f"### ⚡ Executive Summary: {hl_label}"
        if target_word_count:
            bp.target_word_count = target_word_count
            bp.sections[0].target_length = f"{target_word_count} words maximum"
            if target_word_count <= 150 and len(bp.sections) >= 3:
                # Drop trailing Explore Further section on tight word-count budgets so content isn't squeezed
                bp.sections = bp.sections[:2]
        return bp

    # 2. Archive Availability Audit
    if is_availability_query(query):
        bp = DEFAULT_BLUEPRINTS["archive_availability"].model_copy(deep=True)
        bp.target_word_count = target_word_count or 90
        return bp

    # 3. Ordinal / Lead Story Identification on a Specific Page ("What is the 1st news on page 6?")
    if is_ordinal_lead_story_query(query):
        bp = DEFAULT_BLUEPRINTS["ordinal_lead_story"].model_copy(deep=True)
        p_num = params.get("page_filter")
        np_name = params.get("newspaper_name")
        scope_suffix = f" on Page {p_num}" if p_num else ""
        if np_name:
            scope_suffix += f" ({np_name})"
        bp.sections[0].title = f"### ⚡ Lead Story{scope_suffix}"
        if target_word_count:
            bp.target_word_count = target_word_count
        return bp

    # 4. Macro Edition Summary ("Summarize the whole newspaper / today's edition")
    if archetype == "macro_summary" or is_macro_edition_query(query):
        bp = DEFAULT_BLUEPRINTS["macro_summary"].model_copy(deep=True)
        np_name = params.get("newspaper_name")
        iss_dt = params.get("issue_date")
        if np_name or iss_dt:
            label = " — ".join(x for x in (np_name, iss_dt) if x)
            bp.sections[0].title = f"### ⚡ Executive Summary: {label} Edition Overview"
        if target_word_count:
            bp.target_word_count = target_word_count
        return bp

    # 5. Scalar Counts & Quantitative Volume Queries
    is_scalar_or_count = bool(
        re.search(
            r"\b(how many|no of|number of|count of\s+(?:issues|newspapers|pages|articles|ads|advertisements|photos)|total issues|total pages|total articles|count issues|count pages|count advertisements|ad count)\b",
            q_lower,
        )
    ) and not any(w in q_lower for w in ["word count", "average length", "character count", "ratio", "percentage", "correlat"])
    if is_scalar_or_count or (archetype == "quantitative_trend" and not any(w in q_lower for w in ["table", "list all", "catalog", "compare"])):
        bp = DEFAULT_BLUEPRINTS["scalar_count_metric"].model_copy(deep=True)
        bp.target_word_count = target_word_count or 80
        return bp

    # 6. Statistical / Analytical Computation
    if archetype == "analytical_computation" or any(w in q_lower for w in ["average word count", "word count", "average length", "calculate the average", "pearson correlation"]):
        bp = DEFAULT_BLUEPRINTS["analytical_computation"].model_copy(deep=True)
        if target_word_count:
            bp.target_word_count = target_word_count
        return bp

    # 7. Dynamic Cross-Newspaper, Shared-Wire, Differential, & Cross-Page Comparisons
    if archetype == "cross_newspaper_comparison" or any(w in q_lower for w in ["in a table", "as a table", "table comparing", "tabular format", "comparison table", "in table format", "markdown table", "table of"]):
        target_nps = params.get("target_newspapers") or []
        target_pages = params.get("target_pages") or []
        is_shared = bool(params.get("is_shared"))
        is_diff = bool(params.get("is_differential"))

        # 7a. Shared Syndicated Wire Coverage Comparison
        if is_shared:
            np_a = target_nps[0] if len(target_nps) >= 1 else "Newspaper A"
            np_b = target_nps[1] if len(target_nps) >= 2 else "Newspaper B"
            bp = AnswerBlueprint(
                user_intent="shared_wire_comparison",
                overall_tone="analytical_comparison",
                target_word_count=target_word_count,
                sections=[
                    SectionSpec(
                        title="### ⚡ Executive Summary: Shared Syndicated Coverage",
                        format_type="narrative",
                        content_focus=f"Synthesize the scale and thematic overlap of shared syndicated wire stories between {np_a} and {np_b}.",
                        target_length="2 to 3 authoritative sentences",
                    ),
                    SectionSpec(
                        title="### 📰 Verified Shared Coverage Matrix",
                        format_type="markdown_table",
                        content_focus="Map verified shared wire stories across both newspapers using strictly verified matches.",
                        target_length="1 row per verified shared wire story",
                    ),
                    SectionSpec(
                        title="### 🎯 Regional Framing & Placement Divergence",
                        format_type="bullet_list",
                        content_focus="Compare page placement, prominence, headline framing, and editorial depth across both papers.",
                        target_length="2 to 4 analytical bullet points",
                    ),
                    SectionSpec(
                        title="### 🔍 Explore Further",
                        format_type="bullet_list",
                        content_focus="Comparative follow-up prompts.",
                        target_length="2 concise prompts",
                    ),
                ],
                table_columns=[
                    "#",
                    "Shared Wire Story / Event",
                    f"{np_a} Headline (Page, Section)",
                    f"{np_b} Headline (Page, Section)",
                    "Overlap Basis & Match Confidence",
                ],
                prohibited_elements=[
                    "pairing unrelated local stories as false equivalents",
                    "conversational filler",
                ],
            )
            return bp

        # 7b. Exclusive / Differential Coverage Comparison ("in Paper A but not in Paper B")
        if is_diff and len(target_nps) >= 2:
            src_np, comp_np = target_nps[0], target_nps[1]
            bp = AnswerBlueprint(
                user_intent="coverage_difference_audit",
                overall_tone="analytical_comparison",
                target_word_count=target_word_count,
                sections=[
                    SectionSpec(
                        title=f"### ⚡ Executive Summary: Exclusive Coverage in {src_np}",
                        format_type="narrative",
                        content_focus=f"Overview of stories published by {src_np} that were omitted by {comp_np}.",
                        target_length="1 to 2 paragraphs",
                    ),
                    SectionSpec(
                        title=f"### 📋 Exclusive Stories in {src_np} (Omitted by {comp_np})",
                        format_type="markdown_table",
                        content_focus=f"Structured table of exclusive {src_np} stories with page, section, and core development.",
                        target_length="Full table of verified exclusive stories",
                    ),
                    SectionSpec(
                        title="### 🎯 Editorial Priorities & Omission Analysis",
                        format_type="bullet_list",
                        content_focus=f"Key thematic reasons and regional focus differences explaining why {src_np} covered these stories.",
                        target_length="3 to 4 bullets",
                    ),
                ],
                table_columns=["#", "Publication", "Page", "Section", "Exclusive Headline", "Core Finding"],
                prohibited_elements=["including stories covered by both newspapers", "conversational filler"],
            )
            return bp

        # 7c. Cross-Page Comparison within a Single Newspaper ("Compare page 1 and page 3 of The Goan")
        if len(target_pages) >= 2 and len(target_nps) <= 1:
            pages_label = ", ".join(f"Page {p}" for p in target_pages)
            np_label = f" ({target_nps[0]})" if target_nps else ""
            bp = AnswerBlueprint(
                user_intent="cross_page_comparison",
                overall_tone="analytical_comparison",
                target_word_count=target_word_count,
                sections=[
                    SectionSpec(
                        title=f"### ⚡ Executive Summary: Cross-Page Comparison ({pages_label}){np_label}",
                        format_type="narrative",
                        content_focus=f"Synthesize how news focus, prominence, and sections differ across {pages_label}.",
                        target_length="1 to 2 paragraphs",
                    ),
                    SectionSpec(
                        title=f"### 📊 Cross-Page Comparison Matrix ({pages_label})",
                        format_type="markdown_table",
                        content_focus=f"Structured comparison across {pages_label} showing lead headlines, sections, and core developments.",
                        target_length="1 or more rows per requested page",
                    ),
                    SectionSpec(
                        title="### 📌 Key Verified Stories by Page",
                        format_type="bullet_list",
                        content_focus="Notable stories from each requested page with strict inline citations.",
                        target_length="3 to 6 bullets",
                    ),
                ],
                table_columns=["Page", "Section", "Top Headline", "Core Findings", "Editorial Focus"],
                prohibited_elements=["omitting any of the requested pages", "conversational filler"],
            )
            return bp

        # 7d. Domain-Specific or Edition-Wide Cross-Newspaper Comparison
        bp = DEFAULT_BLUEPRINTS["cross_newspaper_comparison"].model_copy(deep=True)
        if domain:
            metric_col = DOMAIN_TAXONOMY.get(domain, {}).get("metric_col", "Core Findings")
            bp.sections[0].title = f"### ⚡ Executive Summary: {domain} Intelligence"
            bp.sections[0].content_focus = f"Overarching synthesis of {domain} developments, market/policy triggers, and broadsheet takeaways."
            bp.sections[1].title = f"### 📊 Cross-Newspaper {domain} Comparison Matrix"
            bp.sections[2].title = "### 📌 Key Verified Sector Highlights & Policies"
            bp.sections[3].title = "### 📰 Broadsheet Editorial Framing & Divergence"
            bp.table_columns = ["Publication", "Issue Date", f"Top {domain} Headline", metric_col, "Editorial Angle / Focus"]
        if target_word_count:
            bp.target_word_count = target_word_count
        return bp

    # 8. Dynamic Article Catalog (Page-Scoped or Domain-Scoped)
    if archetype == "article_catalog":
        bp = DEFAULT_BLUEPRINTS["article_catalog"].model_copy(deep=True)
        p_filt = params.get("page_filter")
        np_name = params.get("newspaper_name")
        display_scope = domain or (f"Page {p_filt}" if p_filt else "Archive")
        bp.sections[0].title = f"### ⚡ Executive Summary: {display_scope} Article Catalog"
        bp.sections[1].title = f"### 📋 Comprehensive {display_scope} Articles Catalog"
        if np_name and p_filt:
            bp.user_intent = "page_article_catalog"
        if target_word_count:
            bp.target_word_count = target_word_count
        return bp

    key = archetype if archetype in DEFAULT_BLUEPRINTS else "factual_lookup"
    bp = DEFAULT_BLUEPRINTS[key].model_copy(deep=True)
    if target_word_count is not None:
        bp.target_word_count = target_word_count
    return bp


__all__ = [
    "DEFAULT_BLUEPRINTS",
    "build_heuristic_answer_blueprint",
    "compile_structure_from_blueprint",
]
