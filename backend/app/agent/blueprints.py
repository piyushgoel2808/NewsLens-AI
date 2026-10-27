"""Declarative Answer Blueprint Registry and Heuristic Blueprint Builder for NewsLens-AI.

Separates declarative AnswerBlueprint definitions and heuristic blueprint construction
from the runtime QueryPlanner orchestration while preserving backward-compatible re-exports.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from app.agent.decision_layer import is_availability_query
from app.agent.models import AnswerBlueprint, SectionSpec
from app.agent.synthesizer_prompts import compile_structure_from_blueprint

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

DEFAULT_BLUEPRINTS["macro_summary"] = DEFAULT_BLUEPRINTS["quantitative_trend"]


def build_heuristic_answer_blueprint(
    query: str,
    archetype: str,
    parameters: dict[str, Any] | None = None,
) -> AnswerBlueprint:
    """Construct a tailored deterministic AnswerBlueprint based on user query and archetype."""
    q_lower = query.lower().strip()
    target_word_count: int | None = None
    wc_match = re.search(r"\b(?:in|under|around|within|max(?:imum)?)\s+(\d+)\s+words?\b", q_lower)
    if wc_match:
        with contextlib.suppress(ValueError):
            target_word_count = int(wc_match.group(1))

    quoted_hl = re.search(r"[\"“]([^\"”]{8,150})[\"”]", query)
    is_target_art = bool(
        quoted_hl
        or re.search(
            r"\b(?:tell me about|explain|summ[ae]ri[sz]e|find|read|what does|describe|details? of|takeaways? from)\s+(?:about|on|for)?\s*(?:this|the|that|ir)?\s*(?:article|story|piece|it)\b",
            q_lower,
        )
        or re.search(r"\b(?:in\s+\d+\s+words?|brief\s+summary|key\s+takeaways?)\b", q_lower)
    ) and not any(w in q_lower for w in ["how many", "count", "list all", "catalog", "compare all"])
    if is_target_art:
        bp = DEFAULT_BLUEPRINTS["single_article_summary"].model_copy(deep=True)
        if quoted_hl:
            bp.sections[0].title = f"### ⚡ Executive Summary: {quoted_hl.group(1).strip()}"
        if target_word_count:
            bp.target_word_count = target_word_count
            bp.sections[0].target_length = f"{target_word_count} words maximum"
        return bp

    if is_availability_query(query):
        bp = DEFAULT_BLUEPRINTS["archive_availability"].model_copy(deep=True)
        bp.target_word_count = target_word_count or 90
        return bp

    is_scalar_or_count = bool(
        re.search(
            r"\b(how many|no of|number of|count of\s+(?:issues|newspapers|pages|articles|ads|advertisements|photos)|total issues|total pages|total articles|count issues|count pages|count advertisements|ad count)\b",
            q_lower,
        )
    ) and not any(w in q_lower for w in ["word count", "average length", "character count", "ratio", "percentage", "correlat"])
    if is_scalar_or_count or (archetype == "quantitative_trend" and not any(w in q_lower for w in ["table", "list all", "catalog"])):
        bp = DEFAULT_BLUEPRINTS["scalar_count_metric"].model_copy(deep=True)
        bp.target_word_count = target_word_count or 80
        return bp

    if archetype == "analytical_computation" or any(w in q_lower for w in ["average word count", "word count", "average length", "calculate the average", "pearson correlation"]):
        bp = DEFAULT_BLUEPRINTS["analytical_computation"].model_copy(deep=True)
        if target_word_count:
            bp.target_word_count = target_word_count
        return bp

    if archetype == "cross_newspaper_comparison" or any(w in q_lower for w in ["in a table", "as a table", "table comparing", "tabular format", "comparison table", "in table format", "markdown table", "table of"]):
        bp = DEFAULT_BLUEPRINTS["cross_newspaper_comparison"].model_copy(deep=True)
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
