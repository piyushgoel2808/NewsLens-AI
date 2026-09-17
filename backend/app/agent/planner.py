"""Query Planner: True Agentic Direct Tool Planning and High-Cohesion Orchestration.

Coordinates LLM-driven structured tool sequence planning with single-pass deterministic
heuristic fallback, parameter reconciliation, and seamless multi-provider failover.
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

from app.agent.archive_context import (
    STATIC_BROADSHEET_SCHEMA,
    get_known_publications,
)
from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    _SECTION_PATTERNS,
    _build_targeted_web_query,
    build_targeted_web_query,
    extract_parameters_from_query,
    is_archive_wide_newspaper_query,
)
from app.agent.models import (
    AgentPlan,
    AnswerBlueprint,
    ExtractedToolArguments,
    PlannedToolCall,
    PlanResult,
    QueryArchetype,
    QueryPlan,
    SectionSpec,
    ToolCallSpec,
    ToolName,
)
from app.agent.state import ToolExecutionRecord
from app.agent.tool_factory import (
    build_coverage_analysis_tool,
    build_dynamic_analysis_tool,
    build_entity_search_tool,
    build_hybrid_search_tool,
    build_inspect_visual_asset_tool,
    build_sql_coverage_comparison_tool,
    build_sql_difference_tool,
    build_sql_shared_coverage_tool,
    build_sql_summary_tool,
    build_timeline_tool,
    build_web_search_tool,
    reconcile_and_sanitize_arguments,
)
from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry

logger = get_logger(__name__)


MATH_AGGREGATE_PATTERNS: tuple[str, ...] = (
    r"\b(?:avg|average|mean|median)\b",
    r"\b(?:length|word count|word length|longest|shortest)\s+of\s+articles?\b",
    r"\b(?:article\s+length|article\s+word\s+count)\b",
    r"\b(?:correlation|regression|variance|standard deviation|percentile|quantile|histogram|moving average|pearson|spearman|gini)\b",
    r"\b(?:distribution of\s+(?:words?|lengths?|sizes?)|word count distribution|length distribution)\b",
    r"\b(?:ratio of\s+.*to|proportion of\s+.*to)\b",
)


def is_dynamic_analysis_permitted(query: str) -> bool:
    """Verify whether dynamic_analysis is appropriate for this query.

    Dynamic analysis is permitted for ANY query requiring calculations, aggregations,
    averages, lengths, word counts, distributions, metrics, or statistical analysis across broadsheet tables.
    It is ONLY disallowed when the query is purely an article reading/summarization request where word
    count refers to an answer length constraint (e.g. 'summarize the article in 100 words' or 'explain in 2 sentences').
    """
    q_low = query.lower()
    has_math = any(re.search(pat, q_low) for pat in MATH_AGGREGATE_PATTERNS)
    if has_math:
        return True
    is_answer_length_constraint = bool(
        re.search(
            r"\b(?:summarize|summary|overview|explain|tell me about|write an essay|in brief)\b"
            r".*\b(?:in|under|around|within|less than)\s+\d+\s+(?:words|sentences|paragraphs)\b",
            q_low,
        )
    )
    return not is_answer_length_constraint


# ---------------------------------------------------------------------------
# Planner System Prompt
# ---------------------------------------------------------------------------

_PROMPT_PREAMBLE = """You are the expert Query Planner for NewsLens-AI, an agentic intelligence system over Indian broadsheet newspapers.
Analyze the user's query, understand their underlying intent, produce step-by-step reasoning against the database schema, and directly schedule the optimal ordered sequence of 1 to 3 tool calls.
"""

_PROMPT_BODY = """### 🛠️ STRICT OPERATIONAL TOOL CONTRACTS & BOUNDARIES

1. `sql_analytics`: Relational system of record for FIXED, PRE-COMPILED routines.
   - Arguments: {"analysis_type": "issue_summary" | "count_articles" | "count_advertisements" | "count_photos" | "count_issues" | "coverage_difference" | "shared_coverage", "newspaper_name": str, "comparison_newspaper": str, "issue_date": "YYYY-MM-DD", "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD", "category_filter": str, "page_filter": str, "query": str}
   - STRICT ENUM REQUIREMENT: `analysis_type` MUST be one of these 7 exact literals:
     * `count_issues`: Issue availability, publication rosters, and issue volume (e.g. 'list available newspapers in September 2026', 'is any newspaper available on 2026-04-28', 'total issues of a publication').
     * `count_articles`: Total article count for a newspaper or date range.
     * `count_advertisements`: Advertisement and commercial notices count.
     * `count_photos`: Photo and visual assets count.
     * `issue_summary`: Manifest catalog / section breakdown of a single broadsheet issue.
     * `coverage_difference`: Exclusive stories published by one newspaper but omitted by another.
     * `shared_coverage`: Common/syndicated wire stories published across two newspapers.
   - ⚠️ NEGATIVE CONSTRAINT: `sql_analytics` CANNOT generate custom SQL, CANNOT calculate averages, CANNOT compute word lengths, ratios, medians, or custom groupings. Calling `sql_analytics` for anything other than these 7 fixed enum values is a fatal tool error!

2. `dynamic_analysis`: LLM-synthesized custom Python & SQL analysis engine.
   - Arguments: {"query": str, "analysis_description": str}
   - POSITIVE SCOPE: Use for ANY question requiring calculations, mathematical aggregations, averages (e.g. 'what is the average length of articles in The Times dated 2026-08-01', 'average word count of editorials'), distributions (e.g. 'distribution of word counts'), ratios, author statistics, correlations, or multi-table SQL groupings across the broadsheet schema.
   - ⚠️ NEGATIVE CONSTRAINT: NEVER schedule `dynamic_analysis` for text summarization, reading articles, quotes, or narrative inquiries (e.g. 'summarize article in 100 words' uses `hybrid_search`, where word count is an answer length constraint, not a database calculation).

3. `hybrid_search`: Dense vector + BM25 keyword search for factual answers, quotes, and specific events.
   - Arguments: {"query": str, "newspaper_name": str, "date_from": str, "date_to": str, "page_filter": str, "category_filter": str, "top_k": int}
   - Use for: Point-in-time facts, quotes, event details, or targeted content.

4. `timeline_builder`: Chronological evolution and milestone articles.
   - Arguments: {"query": str, "limit": int}
   - Use for: Evolution over time, trajectories, and multi-date developments.

5. `entity_search`: Multi-hop entity network search and profiles.
   - Arguments: {"entity_name": str, "top_k": int}
   - Use for: Deep profiles of specific people or corporations.

6. `coverage_analysis`: Unreported news and negative coverage audit.
   - Arguments: {"query": str, "target_date": str}
   - Use for: Identifying what a newspaper omitted or missed across the archive.

7. `web_search`: Live internet search.
   - Arguments: {"query": str, "num_results": int}
   - Use for: Real-time current events outside the archive.

8. `inspect_visual_asset`: Deep multimodal visual inspection, numerical table extraction, and chart axis reading from broadsheet visual crops.
   - Arguments: {"photo_id": int, "article_id": int, "query": str, "newspaper_name": str, "issue_date": str, "page_filter": str}
   - Use for: Extracting specific numbers, data tables, infographic graphics, charts, and captions from an attached, cited, or inquired broadsheet visual asset, or checking if an article/page has infographics or graphs.

### 📚 FEW-SHOT EXAMPLES
Query: "What is the average length of articles in Financial Times on 2026-05-10?"
Output: {"thought_process": "Calculating the average article length or word count requires database aggregation beyond the fixed count tools in sql_analytics. Schedule dynamic_analysis.", "archetype": "analytical_computation", "tool_calls": [{"tool_name": "dynamic_analysis", "arguments": {"query": "What is the average length of articles in Financial Times on 2026-05-10?", "analysis_description": "Compute the average article length and word count distribution in Financial Times on 2026-05-10"}, "purpose": "Synthesize and execute dynamic Python/SQL tool to calculate average article length"}]}

Query: "List distinct newspaper names available in July 2026"
Output: {"thought_process": "User is requesting a roster of distinct newspapers available across July 2026. Schedule sql_analytics count_issues across the July date range without publication constraints.", "archetype": "quantitative_trend", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"analysis_type": "count_issues", "date_from": "2026-07-01", "date_to": "2026-07-31"}, "purpose": "Retrieve distinct newspapers and issue counts in July 2026"}]}

Query: "What happened regarding the high-speed rail corridor on page 3?"
Output: {"thought_process": "Factual question about high-speed rail corridor on page 3. Direct hybrid search bounded to page 3.", "archetype": "factual_lookup", "tool_calls": [{"tool_name": "hybrid_search", "arguments": {"query": "high-speed rail corridor", "page_filter": "3", "top_k": 6}, "purpose": "Search page 3 for high-speed rail reporting"}]}

Query: "Calculate the Pearson correlation between daily article counts in The Indian Express and The Hindu over June 2026"
Output: {"thought_process": "Mathematical correlation between daily article volumes across two newspapers. Requires statistical computation beyond sql_analytics.", "archetype": "analytical_computation", "tool_calls": [{"tool_name": "dynamic_analysis", "arguments": {"query": "Pearson correlation between daily article counts in The Indian Express and The Hindu over June 2026", "analysis_description": "Compute Pearson correlation between daily article counts of The Indian Express and The Hindu in June 2026"}, "purpose": "Synthesize and execute dynamic Python correlation tool"}]}

Query: "What are the exact figures and data points shown in this infographic?" (Attached asset photo_id: 1042)
Output: {"thought_process": "User is asking about specific figures and content in an attached infographic. Schedule inspect_visual_asset.", "archetype": "factual_lookup", "tool_calls": [{"tool_name": "inspect_visual_asset", "arguments": {"photo_id": 1042, "query": "figures and data points"}, "purpose": "Transcribe and analyze infographic visual crop"}]}

Query: "Does the renewable energy article on page 4 have any infographics or graphs with it?"
Output: {"thought_process": "Inquiring about visual assets, charts, or infographics attached to an article on page 4.", "archetype": "factual_lookup", "tool_calls": [{"tool_name": "inspect_visual_asset", "arguments": {"query": "renewable energy solar wind power", "page_filter": "4"}, "purpose": "Inspect visual charts and infographics on page 4"}, {"tool_name": "hybrid_search", "arguments": {"query": "renewable energy solar wind power", "page_filter": "4", "top_k": 4}, "purpose": "Retrieve article textual context"}]}

Query: "How many total issues of The Guardian are in the archive?"
Output: {"thought_process": "Quantitative count of newspaper issues. Schedule sql_analytics to retrieve archive count.", "archetype": "quantitative_trend", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"newspaper_name": "The Guardian", "analysis_type": "count_issues"}, "purpose": "Count total issues of The Guardian"}]}

Query: "Is any newspaper available for dated 2024-04-15?"
Output: {"thought_process": "Relational archive availability inquiry for 2024-04-15. Schedule sql_analytics count_issues with exact date.", "archetype": "quantitative_trend", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"analysis_type": "count_issues", "issue_date": "2024-04-15"}, "purpose": "Check archive newspaper availability for 2024-04-15"}], "answer_blueprint": {"user_intent": "archive_availability", "overall_tone": "concise_atomic", "target_word_count": 90, "sections": [{"title": "### ⚡ Availability Status", "format_type": "narrative", "content_focus": "Direct authoritative statement stating whether newspaper issues exist in the archive for the queried date", "target_length": "1 to 2 crisp sentences"}, {"title": "### 📋 Archive Scope & Available Coverage", "format_type": "bullet_list", "content_focus": "Compact list of available newspapers on that date, or if none, the verified archive date range and available publications", "target_length": "Compact bullet points"}], "prohibited_elements": ["speculative corporate strategy or publication planning advice", "fake future collaboration suggestions", "claiming positive availability when count is zero", "conversational filler"]}}

Query: "How many advertisements are there in The Times of India on 2026-06-15?"
Output: {"thought_process": "Quantitative count of advertisements in The Times of India on 2026-06-15. Schedule sql_analytics count_advertisements.", "archetype": "quantitative_trend", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"newspaper_name": "The Times of India", "issue_date": "2026-06-15", "analysis_type": "count_advertisements"}, "purpose": "Count total advertisements in The Times of India on 2026-06-15"}]}

Query: "List all business and finance news in The Economic Times on 2026-05-20"
Output: {"thought_process": "Relational catalog query for business articles in The Economic Times.", "archetype": "article_catalog", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"newspaper_name": "The Economic Times", "issue_date": "2026-05-20", "category_filter": "Business", "analysis_type": "issue_summary"}, "purpose": "Retrieve complete manifest of Business articles"}]}

Query: "Compare all available newspapers dated 2026-05-20 on healthcare policy"
Output: {"thought_process": "Cross-newspaper domain comparison on healthcare policy. First fetch SQL manifest for all newspapers on that date, then retrieve comparative excerpts.", "archetype": "cross_newspaper_comparison", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"issue_date": "2026-05-20", "category_filter": "Health", "analysis_type": "issue_summary", "query": "healthcare policy"}, "purpose": "Fetch complete manifest of healthcare articles across all newspapers"}, {"tool_name": "hybrid_search", "arguments": {"query": "healthcare policy hospitals insurance", "category_filter": "Health", "date_from": "2026-05-20", "date_to": "2026-05-20", "top_k": 12}, "purpose": "Retrieve comparative excerpts across broadsheet editions"}]}

Query: "Give me the similar articles from The Indian Express and The Hindu on 2026-05-20"
Output: {"thought_process": "User wants shared/similar syndicated wire stories between The Indian Express and The Hindu. Schedule shared_coverage analysis.", "archetype": "cross_newspaper_comparison", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"newspaper_name": "The Indian Express", "comparison_newspaper": "The Hindu", "issue_date": "2026-05-20", "analysis_type": "shared_coverage"}, "purpose": "Identify verified shared syndicated wire coverage between The Indian Express and The Hindu"}, {"tool_name": "hybrid_search", "arguments": {"query": "national world syndicated wire news", "date_from": "2026-05-20", "date_to": "2026-05-20", "top_k": 8}, "purpose": "Retrieve corroborating shared article texts"}]}

Query: "List the news that are in The Goan dated 2026-08-01 but not in The Morning Standard"
Output: {"thought_process": "User is requesting stories exclusive to The Goan that were omitted or not reported by The Morning Standard. Schedule coverage_difference analysis.", "archetype": "cross_newspaper_comparison", "tool_calls": [{"tool_name": "sql_analytics", "arguments": {"newspaper_name": "The Goan", "comparison_newspaper": "The Morning Standard", "issue_date": "2026-08-01", "analysis_type": "coverage_difference"}, "purpose": "Identify exclusive coverage in The Goan omitted by The Morning Standard"}, {"tool_name": "hybrid_search", "arguments": {"query": "exclusive news reporting", "newspaper_name": "The Goan", "date_from": "2026-08-01", "date_to": "2026-08-01", "top_k": 8}, "purpose": "Retrieve supporting article texts"}]}

Query: "Find key news and coverage regarding Sawant or Pramod Sawant in The Goan"
Output: {"thought_process": "User is searching for key news and coverage profile of a specific entity (Pramod Sawant) in The Goan. Schedule entity_search.", "archetype": "factual_lookup", "tool_calls": [{"tool_name": "entity_search", "arguments": {"entity_name": "Pramod Sawant", "newspaper_name": "The Goan", "top_k": 10}, "purpose": "Search archive for entity occurrences and profile of Pramod Sawant"}]}

Query: "Did The Morning Standard report on power tariff on 2026-08-01?"
Output: {"thought_process": "Auditing single-newspaper coverage or omission of a specific topic/event on a given date. Schedule coverage_analysis.", "archetype": "factual_lookup", "tool_calls": [{"tool_name": "coverage_analysis", "arguments": {"query": "power tariff", "target_date": "2026-08-01", "newspaper_name": "The Morning Standard"}, "purpose": "Audit coverage presence for The Morning Standard regarding power tariff"}]}

### ⚡ REASONING & OUTPUT INSTRUCTIONS
- Keep internal chain-of-thought concise (<80 words).
- CRITICAL DATE RESTRAINT: NEVER invent or hallucinate date ranges (e.g. "2020-01-01" to "2022-12-31") or historical years when the user query does NOT specify any dates! If the query contains no dates, leave `date_from`, `date_to`, `issue_date`, and `target_date` empty or omitted so the retrieval tools search across the entire broadsheet archive.
- ARCHETYPE SELECTION:
  * For queries citing specific statements, article quotes, headlines, or factual claims without explicit multi-newspaper comparative keywords, choose `factual_lookup` and schedule targeted `hybrid_search`.
  * For counting, frequencies, volume, or metadata questions (e.g. "how many issues", "number of pages", "count of articles", "total editions"), choose `quantitative_trend` or `factual_lookup`. NEVER select `article_catalog` for scalar counts!
  * Select `article_catalog` ONLY when the user explicitly asks to list, enumerate, or browse multiple distinct articles (e.g. "list all articles", "show catalog of health news").
  * Only select `cross_newspaper_comparison` when the user explicitly asks to compare across publications (e.g. "compare newspapers", "across editions", "coverage differences").
- You MUST respond with a valid JSON object matching the required schema. Return only the JSON object, with no markdown fences or conversational text.
- DYNAMIC ANSWER BLUEPRINT (OPTIONAL):
  You may optionally include an `answer_blueprint` object in your JSON response to design the exact sections and layout of the final answer:
  {
    "user_intent": "concise interpretation of query intent",
    "overall_tone": "authoritative_journalistic" | "executive_brief" | "analytical_comparison" | "concise_atomic",
    "target_word_count": 150 (or null if unspecified),
    "sections": [
      {"title": "### ⚡ Executive Summary", "format_type": "narrative", "content_focus": "Core news event", "target_length": "120-150 words"}
    ],
    "table_columns": ["Publication", "Headline", "Focus"] (or null if no table),
    "prohibited_elements": ["robotic catalog tables", "conversational filler"]
  }
"""

PLANNER_SYSTEM_PROMPT = f"{_PROMPT_PREAMBLE}\n{STATIC_BROADSHEET_SCHEMA}\n\n{_PROMPT_BODY}"



# ---------------------------------------------------------------------------
# Declarative Answer Blueprint Registry & Builder
# ---------------------------------------------------------------------------

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
}


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

    # Single-article focus check
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

    # Archive availability query
    if bool(re.search(r"\b(is\s+(?:any\s+)?newspaper\s+available|are\s+there\s+(?:any\s+)?newspapers|is\s+there\s+an?\s+issue|papers?\s+available|newspapers?\s+available|check\s+availability|issues?\s+available|edition\s+available|available\s+for\s+dated?|issues?\s+for\s+dated?|paper\s+for\s+dated?)\b", q_lower)):
        bp = DEFAULT_BLUEPRINTS["archive_availability"].model_copy(deep=True)
        bp.target_word_count = target_word_count or 90
        return bp

    # Scalar / count query
    is_scalar_or_count = bool(re.search(r"\b(how many|no of|number of|count of|total issues|total pages|total articles|count issues|count pages|count advertisements|ad count)\b", q_lower))
    if is_scalar_or_count or (archetype in ("quantitative_trend", "analytical_computation") and not any(w in q_lower for w in ["table", "list all", "catalog"])):
        bp = DEFAULT_BLUEPRINTS["scalar_count_metric"].model_copy(deep=True)
        bp.target_word_count = target_word_count or 80
        return bp

    # Cross-newspaper comparison or explicit table requested
    if archetype == "cross_newspaper_comparison" or any(w in q_lower for w in ["in a table", "as a table", "table comparing", "tabular format", "comparison table", "in table format", "markdown table", "table of"]):
        bp = DEFAULT_BLUEPRINTS["cross_newspaper_comparison"].model_copy(deep=True)
        if target_word_count:
            bp.target_word_count = target_word_count
        return bp

    # Lookup from declarative registry
    key = archetype if archetype in DEFAULT_BLUEPRINTS else "factual_lookup"
    bp = DEFAULT_BLUEPRINTS[key].model_copy(deep=True)
    if target_word_count is not None:
        bp.target_word_count = target_word_count
    return bp


# ---------------------------------------------------------------------------
# Backward-Compatible Helper
# ---------------------------------------------------------------------------

def resolve_tool_sequence(archetype: str, query: str, **kwargs: Any) -> list[PlannedToolCall]:
    """Backward-compatible helper delegating to QueryPlanner deterministic heuristic planner."""
    planner = QueryPlanner()
    res = planner._plan_query_heuristic(query)
    return res.tool_calls


# ---------------------------------------------------------------------------
# QueryPlanner Class
# ---------------------------------------------------------------------------

class QueryPlanner:
    """Agentic Query Planner with True Direct Tool Calling and Lean Deterministic Fallback."""

    def __init__(self, provider: ChatModelProvider | None = None) -> None:
        self._provider = provider

    def _get_provider_candidates(self, model_override: str | None = None) -> list[ChatModelProvider]:
        """Resolve LLM provider candidates for planning failover."""
        if self._provider is not None and not model_override:
            return [self._provider]

        candidates: list[ChatModelProvider] = []
        seen_keys: set[str] = set()

        try:
            reg = get_registry()
            primary = reg.get_chat_provider(model_override) if model_override else reg.get_provider("query_planner")
            if isinstance(primary, ChatModelProvider):
                candidates.append(primary)
                seen_keys.add(f"{getattr(primary, 'provider_name', '')}:{getattr(primary, '_model', '')}")
        except Exception as e:
            logger.warning("Could not resolve primary provider for QueryPlanner", extra={"error": str(e)})

        try:
            reg = get_registry()
            for k in reg.get_chat_failover_candidates():
                with contextlib.suppress(Exception):
                    p = reg.get_chat_provider(k)
                    if p:
                        ident = f"{getattr(p, 'provider_name', '')}:{getattr(p, '_model', '')}"
                        if ident not in seen_keys:
                            seen_keys.add(ident)
                            candidates.append(p)
        except Exception:
            pass

        return candidates

    async def plan_query_async(
        self,
        query: str,
        enable_web_search: bool = False,
        model_override: str | None = None,
        archive_context: str | None = None,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
    ) -> PlanResult:
        """Plan query using true direct agentic tool calling with graceful heuristic fallback."""
        providers = self._get_provider_candidates(model_override)
        for provider in providers:
            try:
                q_lower = query.lower()
                is_visual_query = bool(
                    attached_photo_id
                    or any(
                        w in q_lower
                        for w in [
                            "infographic", "data chart", "chart", "diagram", "table",
                            "graph", "visual", "figure", "photograph", "photo",
                            "caption", "picture", "image", "illustration", "map",
                        ]
                    )
                )
                user_content = f"Analyze and plan the following broadsheet research query:\nQuery: \"{query}\"\n"
                if archive_context:
                    user_content += f"\nACTIVE ARCHIVE STATE:\n{archive_context}\n"
                if active_issue_date or active_newspapers or attached_article_id:
                    ctx_lines = []
                    if active_issue_date:
                        ctx_lines.append(f"Active Date: {active_issue_date}")
                    if active_newspapers:
                        ctx_lines.append(f"Active Newspapers: {', '.join(active_newspapers)}")
                    if attached_article_id:
                        ctx_lines.append(f"Referenced Article ID: {attached_article_id}")
                    user_content += "\nCONVERSATION WORKING CONTEXT:\n" + "\n".join(ctx_lines) + "\n"
                if attached_photo_id or (attached_article_id and is_visual_query):
                    user_content += f"\nATTACHED VISUAL ASSET CONTEXT:\nAttached Photo ID: {attached_photo_id or 'None'}\nAttached Article ID: {attached_article_id or 'None'}\n"
                user_content += "\nSchedule the exact tool calls needed to gather evidence for this query."

                messages = [
                    Message(role="system", content=PLANNER_SYSTEM_PROMPT),
                    Message(role="user", content=user_content),
                ]
                resp = await provider.complete(
                    messages=messages,
                    response_schema=AgentPlan.model_json_schema(),
                    temperature=0.0,
                    max_tokens=2048,
                )

                parsed_dict = resp.parsed if isinstance(resp.parsed, dict) else self._parse_json_plan(resp.text)
                if parsed_dict:
                    plan_obj = AgentPlan.model_validate(parsed_dict)
                    return self._build_plan_from_structured_model(
                        query=query,
                        plan_obj=plan_obj,
                        enable_web_search=enable_web_search,
                        active_issue_date=active_issue_date,
                        active_newspapers=active_newspapers,
                        attached_article_id=attached_article_id,
                        attached_photo_id=attached_photo_id,
                    )
                logger.warning(
                    "Provider returned unparseable plan output, attempting failover candidate",
                    extra={"provider": getattr(provider, "provider_name", ""), "raw": (resp.text or "")[:150]},
                )
            except Exception as ex:
                logger.warning(
                    "LLM Agentic Planning attempt failed on provider, trying failover candidate",
                    extra={"query": query[:50], "provider": getattr(provider, "provider_name", ""), "error": str(ex)},
                )

        # Fallback to lean deterministic router
        return self._plan_query_heuristic(
            query,
            enable_web_search=enable_web_search,
            active_issue_date=active_issue_date,
            active_newspapers=active_newspapers,
            attached_article_id=attached_article_id,
            attached_photo_id=attached_photo_id,
        )

    def plan_query(
        self,
        query: str,
        enable_web_search: bool = False,
        archive_context: str | None = None,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
    ) -> PlanResult:
        """Synchronous planning interface."""
        return self._plan_query_heuristic(
            query,
            enable_web_search=enable_web_search,
            active_issue_date=active_issue_date,
            active_newspapers=active_newspapers,
            attached_article_id=attached_article_id,
            attached_photo_id=attached_photo_id,
        )

    def classify_archetype(self, query: str) -> tuple[str, str]:
        """Classify query archetype synchronously."""
        res = self._plan_query_heuristic(query)
        return res.archetype, res.reasoning

    @staticmethod
    def _parse_json_plan(text: str) -> dict[str, Any] | None:
        """Extract and parse JSON from text, stripping reasoning tags and markdown fences."""
        if not text:
            return None
        cleaned = re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^<(thought|think)>.*?</\1>", "", cleaned, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = text.strip()

        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        with contextlib.suppress(Exception):
            res = json.loads(cleaned)
            if isinstance(res, dict):
                return res

        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            with contextlib.suppress(Exception):
                res = json.loads(match.group(1))
                if isinstance(res, dict):
                    return res
        return None

    def _build_plan_from_structured_model(
        self,
        query: str,
        plan_obj: AgentPlan,
        enable_web_search: bool = False,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
    ) -> PlanResult:
        """Translate AgentPlan into PlanResult, executing direct tool calls or legacy adapter."""
        extracted = extract_parameters_from_query(query)
        if attached_article_id is not None:
            extracted["attached_article_id"] = attached_article_id
        if attached_photo_id is not None:
            extracted["attached_photo_id"] = attached_photo_id
        tool_calls: list[PlannedToolCall] = []

        archetype = plan_obj.archetype
        if archetype == "macro_summary":
            archetype = "quantitative_trend"
        elif archetype == "negative_coverage_audit":
            archetype = "cross_newspaper_comparison"

        # 1. Direct tool calling (Option 2)
        if plan_obj.tool_calls:
            for spec in plan_obj.tool_calls:
                args = reconcile_and_sanitize_arguments(
                    spec.tool_name,
                    spec.arguments,
                    extracted,
                    query,
                    active_issue_date=active_issue_date,
                    active_newspapers=active_newspapers,
                )
                tool_calls.append(PlannedToolCall(tool_name=spec.tool_name, arguments=args, purpose=spec.purpose))

        # 2. Legacy adapter: if mock/legacy caller provided primary_tool or arguments without tool_calls
        elif plan_obj.primary_tool or plan_obj.arguments:
            raw_args: dict[str, Any] = {}
            if isinstance(plan_obj.arguments, dict):
                raw_args = dict(plan_obj.arguments)
            elif hasattr(plan_obj.arguments, "model_dump"):
                raw_args = plan_obj.arguments.model_dump(exclude_none=True)

            primary_tool_name = plan_obj.primary_tool or "hybrid_search"
            args = reconcile_and_sanitize_arguments(
                primary_tool_name,
                raw_args,
                extracted,
                query,
                active_issue_date=active_issue_date,
                active_newspapers=active_newspapers,
            )

            if archetype == "cross_newspaper_comparison":
                target_dt = args.get("issue_date") or extracted.get("issue_date")
                comp_np = args.get("comparison_newspaper") or extracted.get("comparison_newspaper")
                src_np = args.get("newspaper_name") or extracted.get("newspaper_name")
                is_diff = (
                    extracted.get("is_differential", False)
                    or plan_obj.archetype == "negative_coverage_audit"
                    or args.get("analysis_type") == "coverage_difference"
                )
                is_shared = (
                    extracted.get("is_shared", False)
                    or args.get("analysis_type") in ("shared_coverage", "similar_articles", "common_stories")
                    or any(w in query.lower() for w in ["similar", "shared", "common", "same article", "same stories", "both"])
                )

                if src_np and comp_np and is_diff:
                    tool_calls.append(build_sql_difference_tool(src_np, comp_np, args.get("query", query), target_dt))
                    tool_calls.append(build_hybrid_search_tool(args.get("query", query), newspaper_name=src_np, top_k=10, purpose=f"Articles from {src_np}"))
                elif src_np and comp_np and is_shared:
                    tool_calls.append(build_sql_shared_coverage_tool(src_np, comp_np, issue_date=target_dt, query=args.get("query", query)))
                    tool_calls.append(build_hybrid_search_tool(args.get("query", query), date_from=target_dt, date_to=target_dt, top_k=10, purpose=f"Shared coverage between {src_np} and {comp_np}"))
                elif not src_np and target_dt:
                    tool_calls.append(build_sql_summary_tool(
                        analysis_type="issue_summary",
                        newspaper_name=None,
                        issue_date=target_dt,
                        category_filter=args.get("category_filter"),
                        query=args.get("query", query),
                        purpose="SQL article manifest across all newspapers",
                    ))
                    tool_calls.append(build_hybrid_search_tool(
                        query=args.get("query", query),
                        date_from=target_dt,
                        date_to=target_dt,
                        category_filter=args.get("category_filter"),
                        top_k=12,
                        purpose="Comparative article excerpts",
                    ))
                    if not args.get("category_filter"):
                        cov_target_dt = target_dt or args.get("target_date")
                        tool_calls.append(build_coverage_analysis_tool(args.get("query", query), target_date=cov_target_dt))
                elif primary_tool_name == "coverage_analysis":
                    cov_target_dt = target_dt or args.get("target_date")
                    cov_np = src_np or args.get("newspaper_name")
                    tool_calls.append(build_coverage_analysis_tool(args.get("query", query), target_date=cov_target_dt, newspaper_name=cov_np))
                else:
                    tool_calls.append(build_sql_summary_tool(
                        analysis_type="issue_summary",
                        newspaper_name=src_np,
                        issue_date=target_dt,
                        category_filter=args.get("category_filter"),
                        query=args.get("query", query),
                        purpose="SQL article manifest",
                    ))
                    tool_calls.append(build_hybrid_search_tool(
                        query=args.get("query", query),
                        newspaper_name=src_np,
                        date_from=target_dt,
                        date_to=target_dt,
                        category_filter=args.get("category_filter"),
                        top_k=12,
                        purpose="Comparative article excerpts",
                    ))
                    if not src_np and target_dt and not args.get("category_filter"):
                        tool_calls.append(build_coverage_analysis_tool(args.get("query", query), target_date=target_dt))
            else:
                tool_calls.append(PlannedToolCall(tool_name=primary_tool_name, arguments=args, purpose=f"Execute {primary_tool_name}"))
                if getattr(plan_obj, "include_secondary_hybrid_search", False) and primary_tool_name != "hybrid_search":
                    tool_calls.append(build_hybrid_search_tool(getattr(plan_obj, "secondary_search_query", None) or query, top_k=6, purpose="Corroborating search"))

        # 3. Fallback if no tool calls produced
        if not tool_calls:
            heur = self._plan_query_heuristic(
                query,
                enable_web_search=enable_web_search,
                active_issue_date=active_issue_date,
                active_newspapers=active_newspapers,
                attached_article_id=attached_article_id,
                attached_photo_id=attached_photo_id,
            )
            tool_calls = heur.tool_calls

        q_lower = query.lower()
        is_visual_query = bool(
            attached_photo_id
            or any(
                w in q_lower
                for w in [
                    "infographic", "data chart", "chart", "diagram", "table",
                    "graph", "visual", "figure", "photograph", "photo",
                    "caption", "picture", "image", "illustration", "map",
                ]
            )
        )

        # Guardrail: If inspect_visual_asset was emitted for non-visual text query without an attached photo, strip it
        if not is_visual_query and not attached_photo_id:
            tool_calls = [t for t in tool_calls if t.tool_name != "inspect_visual_asset"]
        # Guardrail: dynamic_analysis is for database calculations, aggregations, averages, and statistics.
        # Only strip it if the query was a narrative reading/summarization request with an answer length constraint.
        if not is_dynamic_analysis_permitted(query):
            tool_calls = [t for t in tool_calls if t.tool_name != "dynamic_analysis"]

        # Guardrail: Entity deep dive intent routing
        is_entity_query = any(w in q_lower for w in ["everything about", "all mentions of", "profile the coverage", "profile of", "coverage regarding", "news regarding", "mentions of", "involvement of", "role of", "tracked entity"]) or bool(re.search(r"\b(?:mentions?\s+of|coverage\s+regarding|news\s+regarding|involvement\s+of|role\s+of|profile\s+of|profile\s+the\s+coverage\s+of|everything\s+about)\b", q_lower))
        if is_entity_query and not any(t.tool_name == "entity_search" for t in tool_calls):
            clean_ent = re.sub(
                r"(?i)^(?:find\s+(?:key\s+)?(?:news|coverage|reports?)\s+(?:and\s+(?:news|coverage)\s+)?(?:regarding|about|on)|everything\s+about|all\s+mentions\s+of|mentions\s+of|profile\s+the\s+coverage\s+of|profile\s+of|coverage\s+regarding|news\s+regarding|involvement\s+of|role\s+of)\s*",
                "",
                query,
            ).strip("?:!.,\"' ") or query
            for pat, _ in _KNOWN_BRANDS_PATTERNS:
                clean_ent = pat.sub("", clean_ent).strip("?:!.,\"' ")
            clean_ent = re.sub(r"(?i)\s+in\s*$", "", clean_ent).strip("?:!.,\"' ")
            ent_np = (active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name")
            if clean_ent:
                tool_calls.insert(0, build_entity_search_tool(
                    entity_name=clean_ent,
                    top_k=10,
                    newspaper_name=ent_np,
                    purpose=f"Profile entity '{clean_ent}'",
                ))

        # Guardrail: Single/Multi-newspaper coverage presence audit
        is_coverage_audit = bool(re.search(r"\b(?:did\s+.+\s+(?:report\s+on|cover|publish|carry)|was\s+.+\s+(?:reported|covered|published|carried)\s+(?:in|by)|(?:coverage|reporting)\s+(?:of|on)\s+.+\s+(?:in|by)|check\s+(?:if|whether)\s+.+\s+(?:reported|covered))\b", q_lower))
        if is_coverage_audit and not any(t.tool_name == "coverage_analysis" for t in tool_calls):
            cov_np = (active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name")
            cov_dt = active_issue_date or extracted.get("issue_date") or extracted.get("date_from")
            topic_clean = re.sub(r"(?i)\b(?:did|was|were|is|are|has|have)\s+(?:the\s+)?[\w\s]+\s+(?:report\s+on|cover|publish|carry|mention)\s*", "", query)
            topic_clean = re.sub(r"(?i)\b(?:on|dated?)\s+\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", topic_clean).strip("?:!.,\"' ") or query
            tool_calls.insert(0, build_coverage_analysis_tool(
                query=topic_clean,
                target_date=cov_dt,
                newspaper_name=cov_np,
                purpose=f"Audit coverage presence for {cov_np or 'archive'}",
            ))

        if not any(t.tool_name in ("hybrid_search", "sql_analytics", "dynamic_analysis", "timeline", "entity_search", "coverage_analysis") for t in tool_calls):
            tool_calls.append(build_hybrid_search_tool(
                query=query,
                newspaper_name=(active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name"),
                date_from=active_issue_date or extracted.get("issue_date"),
                date_to=active_issue_date or extracted.get("issue_date"),
                top_k=8,
                purpose=f"Search broadsheet reporting on {query[:50]}",
            ))

        if attached_photo_id and not any(t.tool_name == "inspect_visual_asset" for t in tool_calls):
            np_target = (active_newspapers[0] if active_newspapers else None) or extracted.get("newspaper_name") or ""
            dt_target = active_issue_date or extracted.get("issue_date") or ""
            tool_calls.insert(0, build_inspect_visual_asset_tool(
                photo_id=attached_photo_id,
                article_id=attached_article_id,
                query=query,
                newspaper_name=np_target,
                issue_date=dt_target,
                purpose="Inspect attached visual asset and transcribe metadata",
            ))

        if enable_web_search and not any(t.tool_name == "web_search" for t in tool_calls):
            tool_calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))

        # Deduplicate identical tool calls
        deduped_calls: list[PlannedToolCall] = []
        seen_call_keys: set[str] = set()
        for t in tool_calls:
            c_key = f"{t.tool_name}:{sorted((k, str(v)) for k, v in t.arguments.items())}"
            if c_key not in seen_call_keys:
                seen_call_keys.add(c_key)
                deduped_calls.append(t)
        tool_calls = deduped_calls

        blueprint = getattr(plan_obj, "answer_blueprint", None)
        if blueprint is None:
            blueprint = self._build_heuristic_answer_blueprint(query, archetype, extracted)

        return PlanResult(
            archetype=archetype,
            reasoning=plan_obj.thought_process or f"Structured agentic plan ({archetype})",
            tool_calls=tool_calls,
            answer_blueprint=blueprint,
        )

    async def replan_with_feedback_async(
        self,
        query: str,
        previous_plan: list[dict[str, Any]] | list[PlannedToolCall],
        tool_executions: list[dict[str, Any]] | list[ToolExecutionRecord],
        gap_diagnosis: str | None = None,
        evidence_summary: str | None = None,
        model_override: str | None = None,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
    ) -> PlanResult:
        """Adaptive closed-loop re-planner: inspects gap diagnosis and previous executions to generate targeted corrective tool calls."""
        q_lower = query.lower()

        extracted = extract_parameters_from_query(query)

        # Build set of tried calls for strict anti-repetition
        tried_calls: list[tuple[str, dict[str, Any]]] = []
        for t in tool_executions:
            t_name = t.get("tool_name") if isinstance(t, dict) else getattr(t, "tool_name", "")
            t_inp = t.get("tool_input", {}) if isinstance(t, dict) else getattr(t, "tool_input", {})
            if t_name:
                tried_calls.append((t_name, dict(t_inp)))

        exec_lines = []
        for t_name, t_inp in tried_calls:
            exec_lines.append(f"- {t_name}: {json.dumps(t_inp, default=str)}")
        tried_summary = "\n".join(exec_lines) if exec_lines else "None"

        prompt = (
            f"You are the expert Adaptive Query Planner for NewsLens-AI, an intelligence platform over Indian broadsheet newspapers.\n"
            f"The previous retrieval plan was judged INSUFFICIENT to answer the user's query.\n\n"
            f"User Query: \"{query}\"\n"
            f"Evaluator Gap Diagnosis: \"{gap_diagnosis or 'No grounded results returned.'}\"\n"
            f"Previously Executed Tools (Returned Deficient/Zero Results):\n{tried_summary}\n\n"
            f"Your task: Formulate 1 to 2 targeted CORRECTIVE tool calls to retrieve the missing information.\n"
            f"CRITICAL RULES:\n"
            f"1. ANTI-REPETITION: NEVER emit a tool call with the exact same tool_name and arguments that was already tried.\n"
            f"2. If search keywords were too specific or narrow, broaden or rephrase them.\n"
            f"3. If a date filter or newspaper filter resulted in 0 hits, remove or relax the filter to search the whole archive.\n"
            f"4. If an aggregate, count, or manifest was missing, schedule `sql_analytics`.\n"
            f"5. Available Tools:\n"
            f"   - `hybrid_search`: {{\"query\": str, \"newspaper_name\": str, \"date_from\": str, \"date_to\": str, \"page_filter\": str, \"top_k\": int}}\n"
            f"   - `sql_analytics`: {{\"analysis_type\": \"issue_summary\"|\"count_articles\"|\"count_advertisements\"|\"count_photos\"|\"count_issues\"|\"coverage_difference\"|\"shared_coverage\", \"newspaper_name\": str, \"comparison_newspaper\": str, \"issue_date\": str, \"query\": str}}\n"
            f"   - `timeline_builder`: {{\"query\": str, \"limit\": int}}\n"
            f"   - `entity_search`: {{\"entity_name\": str, \"top_k\": int}}\n"
            f"   - `inspect_visual_asset`: {{\"photo_id\": int, \"article_id\": int, \"query\": str, \"page_filter\": str}}\n"
            f"6. CRITICAL: NEVER schedule `dynamic_analysis` for text summarization, explanation, or standard article search.\n\n"
            f"Output a JSON object matching this schema:\n"
            f"{{\n"
            f'  "thought_process": string (<60 words explaining the recovery strategy),\n'
            f'  "archetype": "factual_lookup" | "quantitative_trend" | "cross_newspaper_comparison" | "article_catalog",\n'
            f'  "tool_calls": [\n'
            f'    {{"tool_name": string, "arguments": object, "purpose": string}}\n'
            f'  ]\n'
            f"}}"
        )

        candidates = self._get_provider_candidates(model_override)

        is_dyn_allowed = is_dynamic_analysis_permitted(query)

        planned_calls: list[PlannedToolCall] = []
        archetype = "factual_lookup"
        reasoning = "Adaptive re-planning"

        for prov in candidates:
            try:
                resp = await prov.complete(
                    messages=[
                        Message(role="system", content="You are a precise adaptive query replanner. Respond ONLY in valid JSON."),
                        Message(role="user", content=prompt),
                    ],
                    max_tokens=400,
                    temperature=0.0,
                )
                raw_text = (resp.text or "").strip()
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    archetype = data.get("archetype", "factual_lookup")
                    reasoning = data.get("thought_process", "Targeted recovery re-plan")
                    raw_calls = data.get("tool_calls", [])

                    for rc in raw_calls:
                        t_name = rc.get("tool_name", "")
                        if not t_name:
                            continue
                        # Never allow dynamic_analysis if query was a narrative reading length constraint
                        if t_name == "dynamic_analysis" and not is_dyn_allowed:
                            continue

                        raw_args = rc.get("arguments", {})
                        san_args = reconcile_and_sanitize_arguments(
                            tool_name=t_name,
                            args=raw_args,
                            extracted=extracted,
                            query=query,
                            active_issue_date=active_issue_date,
                            active_newspapers=active_newspapers,
                        )
                        purpose = rc.get("purpose") or f"Recovery {t_name}"

                        # Anti-repetition check against previously attempted calls
                        is_repeated = False
                        for prev_name, prev_args in tried_calls:
                            if prev_name == t_name and (
                                all(str(san_args.get(k, "")) == str(prev_args.get(k, "")) for k in ("query", "newspaper_name", "date_from", "date_to", "analysis_type"))
                                or all(str(raw_args.get(k, "")) == str(prev_args.get(k, "")) for k in ("query", "newspaper_name", "date_from", "date_to", "analysis_type"))
                            ):
                                is_repeated = True
                                break

                        if is_repeated:
                            # Modify to prevent identical repeated failure
                            if t_name == "hybrid_search":
                                san_args.pop("date_from", None)
                                san_args.pop("date_to", None)
                                san_args["top_k"] = max(8, int(san_args.get("top_k", 6)) + 4)
                            elif t_name == "sql_analytics":
                                san_args.pop("issue_date", None)
                                san_args.pop("date_from", None)
                                san_args.pop("date_to", None)
                        elif t_name == "hybrid_search" and "top_k" not in san_args:
                            san_args["top_k"] = 8

                        planned_calls.append(PlannedToolCall(
                            tool_name=t_name,
                            arguments=san_args,
                            purpose=purpose,
                        ))

                    if planned_calls:
                        break
            except Exception as e:
                logger.debug("Adaptive replanner candidate failed", extra={"error": str(e)})
                continue

        # Deterministic fallback if LLM replanner didn't produce calls
        if not planned_calls:
            diag = (gap_diagnosis or "").lower()
            params = extracted
            newspaper = params.get("newspaper_name")

            if "missing_newspaper_coverage" in diag or "missing articles from" in diag:
                # Find which newspaper is missing
                for np_name in get_known_publications():
                    if np_name.lower() in diag:
                        planned_calls.append(build_hybrid_search_tool(
                            query=query,
                            newspaper_name=np_name,
                            top_k=8,
                            purpose=f"Recover missing coverage from {np_name}",
                        ))
            elif "missing_quantitative_payload" in diag or any(w in q_lower for w in ["advertisement", "ad count", "ads count"]):
                is_ad = any(w in q_lower for w in ["advertisement", "ad", "commercial"])
                analysis_type = "count_advertisements" if is_ad else "count_articles"
                planned_calls.append(build_sql_summary_tool(
                    analysis_type=analysis_type,
                    newspaper_name=newspaper,
                    query=query,
                    purpose=f"Recover quantitative aggregate ({analysis_type})",
                ))
            else:
                # Broaden search: hybrid search with no date constraints and expanded top_k
                planned_calls.append(build_hybrid_search_tool(
                    query=query,
                    top_k=8,
                    purpose="Broadened archive retrieval fallback",
                ))

        # Deduplicate identical tool calls
        deduped: list[PlannedToolCall] = []
        seen_keys: set[str] = set()
        for t in planned_calls:
            c_key = f"{t.tool_name}:{sorted((k, str(v)) for k, v in t.arguments.items())}"
            if c_key not in seen_keys:
                seen_keys.add(c_key)
                deduped.append(t)

        blueprint = self._build_heuristic_answer_blueprint(query, archetype)
        return PlanResult(
            archetype=archetype,
            reasoning=reasoning,
            tool_calls=deduped,
            answer_blueprint=blueprint,
        )

    def _plan_query_heuristic(
        self,
        query: str,
        enable_web_search: bool = False,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
    ) -> PlanResult:
        """Clean, deterministic single-pass intent routing for offline fallback."""
        q_lower = query.lower().strip()
        params = extract_parameters_from_query(query)

        is_archive_np = is_archive_wide_newspaper_query(query) or bool(
            re.search(r"\b(?:no|number|count|how many|all|total|which|list)\s+(?:of\s+)?newspapers?\b", q_lower)
            or any(w in q_lower for w in ["all available", "all newspaper", "both newspaper", "across newspaper"])
        )
        raw_np = params.get("newspaper_name")
        newspaper = raw_np if raw_np and (not is_archive_np or raw_np.lower() in q_lower) else None
        issue_id = params.get("issue_id") if newspaper else None

        has_explicit_range = bool(params.get("date_from") and params.get("date_to"))
        issue_date = params.get("issue_date") or (None if has_explicit_range else active_issue_date)
        date_from = params.get("date_from") or issue_date
        date_to = params.get("date_to") or issue_date
        target_dates = params.get("target_dates") or ([issue_date] if issue_date else [])
        category = params.get("category_filter")
        is_diff = params.get("is_differential", False)
        comp_newspaper = params.get("comparison_newspaper")

        page_filter = params.get("page_filter")
        if not page_filter:
            if re.search(r"\b(?:front[\s-]*page|cover[\s-]*page|page\s*(?:1|one))\b", q_lower):
                page_filter = "1"
            else:
                p_match = re.search(r"\b(?:page|pg|p\.?)\s*(\d{1,3})\b", q_lower)
                page_filter = p_match.group(1) if p_match else None

        # Detect single article reading / summarization intent
        is_target_art = bool(
            re.search(r"[\"“][^\"”]{8,150}[\"”]", query)
            or re.search(r"\b(?:tell me about|explain|summ[ae]ri[sz]e|find|read|what does|describe|details? of|takeaways? from)\s+(?:about|on|for)?\s*(?:this|the|that|an?|ir)?\s*(?:[\w-]+\s+){0,4}(?:article|story|piece|it|report)\b", q_lower)
            or re.search(r"\b(?:in\s+(?:under\s+|less\s+than\s+|at\s+most\s+)?\d+\s+words?|under\s+\d+\s+words?|brief\s+summary|key\s+takeaways?)\b", q_lower)
        ) and not any(w in q_lower for w in ["how many", "count", "list all", "catalog", "compare all"])

        # 0. Visual Asset / Infographic Inspection
        is_visual = any(w in q_lower for w in ["infographic", "data chart", "chart", "diagram", "table", "graph", "visual", "figure", "photograph", "photo", "caption", "picture", "image"])
        has_vis_trigger = any(w in q_lower for w in ["this", "the infographic", "attached", "chart", "diagram", "table", "above", "shown", "it have", "have any", "has any", "with it", "there any", "what does the photo", "show the photo"])
        if attached_photo_id or (attached_article_id and is_visual and has_vis_trigger) or (is_visual and has_vis_trigger):
            calls = [
                build_inspect_visual_asset_tool(photo_id=attached_photo_id, article_id=attached_article_id, query=query, newspaper_name=newspaper or "", issue_date=issue_date or "", page_filter=page_filter or "", purpose="Inspect visual crop and transcribe numerical data table"),
                build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, page_filter=page_filter, top_k=4, purpose="Contextual article evidence"),
            ]
            if enable_web_search:
                calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))
            return PlanResult(archetype="factual_lookup", reasoning="Deterministic visual inspection plan (factual_lookup)", tool_calls=calls)

        # 1. Timeline / Chronological Trajectory
        if (
            any(w in q_lower for w in ["timeline", "chronology", "chronological", "evolution", "over time", "history of", "progression", "trace back", "trace it back", "where it began", "from the beginning", "how it unfolded", "how it started"])
            or bool(re.search(r"\b(?:trace\s+(?:it\s+)?back|where\s+it\s+began|how\s+it\s+(?:unfolded|started)|from\s+the\s+beginning)\b", q_lower))
        ):
            archetype = "thematic_timeline"
            calls = [build_timeline_tool(query=query, limit=25), build_hybrid_search_tool(query=query, top_k=8, purpose="Retrieve anchor articles")]

        # 2. Entity Deep Dive
        elif any(w in q_lower for w in ["everything about", "all mentions of", "profile the coverage", "profile of", "coverage regarding", "news regarding", "mentions of", "involvement of", "role of", "tracked entity"]) or bool(re.search(r"\b(?:mentions?\s+of|coverage\s+regarding|news\s+regarding|involvement\s+of|role\s+of|profile\s+of|profile\s+the\s+coverage\s+of|everything\s+about)\b", q_lower)):
            archetype = "entity_deep_dive"
            clean_ent = re.sub(
                r"(?i)^(?:find\s+(?:key\s+)?(?:news|coverage|reports?)\s+(?:and\s+(?:news|coverage)\s+)?(?:regarding|about|on)|everything\s+about|all\s+mentions\s+of|mentions\s+of|profile\s+the\s+coverage\s+of|profile\s+of|coverage\s+regarding|news\s+regarding|involvement\s+of|role\s+of)\s*",
                "",
                query,
            ).strip("?:!.,\"' ") or query
            for pat, _ in _KNOWN_BRANDS_PATTERNS:
                clean_ent = pat.sub("", clean_ent).strip("?:!.,\"' ")
            clean_ent = re.sub(r"(?i)\s+in\s*$", "", clean_ent).strip("?:!.,\"' ")
            calls = [
                build_entity_search_tool(entity_name=clean_ent, top_k=10),
                build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, top_k=8, purpose="Semantic context"),
            ]

        # 3. Analytical Computation / Statistical Analysis (Dynamic Tool)
        elif any(re.search(pat, q_lower) for pat in MATH_AGGREGATE_PATTERNS) and is_dynamic_analysis_permitted(query):
            archetype = "analytical_computation"
            calls = [
                build_dynamic_analysis_tool(
                    query=query,
                    analysis_description=f"Statistical computation: {query}",
                    newspaper_name=newspaper,
                    issue_date=issue_date or date_from,
                    purpose="Execute custom analytical computation",
                ),
            ]
            # If query is targeting an issue or page metrics, also schedule fast sql_analytics summary
            # which computes avg_word_count, total_words, etc. in <5ms as robust ground truth
            if (issue_date or date_from) and (newspaper or page_filter):
                calls.append(
                    build_sql_summary_tool(
                        analysis_type="issue_summary",
                        newspaper_name=newspaper,
                        issue_date=issue_date if not has_explicit_range else None,
                        date_from=date_from if has_explicit_range else None,
                        date_to=date_to if has_explicit_range else None,
                        page_filter=page_filter,
                        query=query,
                        purpose="Ground-truth broadsheet issue summary and page metrics",
                    )
                )
            calls.append(
                build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, page_filter=page_filter, top_k=4, purpose="Contextual evidence for analytical results")
            )

        # 4. Single Newspaper Multi-Issue Comparison
        elif newspaper and not comp_newspaper and len(target_dates) >= 2:
            archetype = "quantitative_trend"
            calls = [build_sql_summary_tool(newspaper_name=newspaper, issue_date=dt, query=query, purpose=f"Manifest for {newspaper} on {dt}") for dt in target_dates]
            calls.append(build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, top_k=10, purpose="Multi-issue articles"))

        # 5. Cross-Newspaper Comparison
        elif comp_newspaper is not None or is_diff or any(w in q_lower for w in ["across newspapers", "across different papers", "all available", "all the available", "all newspapers", "both newspapers", "different papers", "different newspapers"]) or bool(re.search(r"\b(?:compa[a-z]*|contrast[a-z]*|diff[a-z]*|versus|vs\.?)\b", q_lower)):
            archetype = "cross_newspaper_comparison"
            target_dt = issue_date or date_from
            is_shared = params.get("is_shared", False) or any(w in q_lower for w in ["similar", "shared", "common", "same article", "same stories", "both"])
            if newspaper and comp_newspaper and is_diff:
                calls = [build_sql_difference_tool(newspaper, comp_newspaper, query, target_dt), build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=target_dt, date_to=target_dt, top_k=10, purpose=f"Retrieve articles from {newspaper}")]
            elif newspaper and comp_newspaper and is_shared:
                calls = [build_sql_shared_coverage_tool(newspaper, comp_newspaper, issue_date=target_dt, query=query), build_hybrid_search_tool(query=query, date_from=target_dt, date_to=target_dt, top_k=10, purpose=f"Shared coverage between {newspaper} and {comp_newspaper}")]
            elif newspaper and comp_newspaper:
                calls = [build_sql_summary_tool(newspaper_name=np_name, issue_date=target_dt, category_filter=category, query=query, purpose=f"Retrieve manifest for {np_name}") for np_name in (newspaper, comp_newspaper)]
                calls.append(build_hybrid_search_tool(query=query, date_from=target_dt, date_to=target_dt, category_filter=category, top_k=12, purpose=f"Comparative articles across {newspaper} and {comp_newspaper}"))
            elif not target_dt:
                calls = [build_hybrid_search_tool(query=query, category_filter=category, top_k=12, purpose="Comparative articles across broadsheet editions")]
                if any(w in q_lower for w in ["omit", "miss", "exclusive", "gap", "audit", "coverage analysis", "unreported"]):
                    calls.append(build_coverage_analysis_tool(query=query, purpose="Archive-wide coverage comparison"))
            else:
                calls = [
                    build_sql_summary_tool(issue_date=target_dt, category_filter=category, query=query, purpose=f"Manifest across all newspapers on {target_dt}"),
                    build_hybrid_search_tool(query=query, date_from=target_dt, date_to=target_dt, page_filter="1" if not newspaper and not category else None, category_filter=category, top_k=12, purpose="Diverse articles across editions"),
                ]
                if not category or any(w in q_lower for w in ["omit", "miss", "exclusive", "gap", "audit"]):
                    calls.append(build_coverage_analysis_tool(query=query, target_date=target_dt))
                if not category and not newspaper:
                    calls.append(build_sql_coverage_comparison_tool(target_date=target_dt, query=query))

        # 5b. Single Newspaper Coverage Audit / Presence Check
        elif newspaper and not comp_newspaper and bool(re.search(r"\b(?:did\s+.+\s+(?:report\s+on|cover|publish|carry)|was\s+.+\s+(?:reported|covered|published|carried)\s+(?:in|by)|(?:coverage|reporting)\s+(?:of|on)\s+.+\s+(?:in|by)|check\s+(?:if|whether)\s+.+\s+(?:reported|covered))\b", q_lower)):
            archetype = "coverage_audit"
            target_dt = issue_date or date_from
            topic_clean = re.sub(r"(?i)\b(?:did|was|were|is|are|has|have)\s+(?:the\s+)?[\w\s]+\s+(?:report\s+on|cover|publish|carry|mention)\s*", "", query)
            topic_clean = re.sub(r"(?i)\b(?:on|dated?)\s+\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "", topic_clean).strip("?:!.,\"' ") or query
            calls = [
                build_coverage_analysis_tool(query=topic_clean, target_date=target_dt, newspaper_name=newspaper, purpose=f"Audit coverage presence for {newspaper}"),
                build_hybrid_search_tool(query=topic_clean, newspaper_name=newspaper, date_from=date_from, date_to=date_to, top_k=6, purpose=f"Search candidate articles in {newspaper}"),
            ]

        # 6. Quantitative Trend / Article Catalog / Issue Manifest / Counts / Availability / Ordinal Page Queries
        elif (not is_target_art) and (
            is_archive_np
            or (page_filter and any(w in q_lower for w in ["article", "story", "stories", "no of", "how many", "list", "news", "headline", "headlines", "lead", "top", "main", "1st", "first", "2nd", "second", "3rd", "third"]))
            or any(w in q_lower for w in [
                "how many", "count", "number of", "no of", "frequency", "trend", "distribution", "volume", "statistics", "summarize", "overview", "whole", "entire", "all articles", "list", "manifest", "today's paper", "edition", "what articles", "articles on", "is any newspaper available", "are there any newspapers", "available for dated", "newspaper available", "newspapers available", "paper available", "issues available",
            ])
            or (category and any(w in q_lower for w in ["news", "articles", "stories", "headlines"]))
        ):
            q_no_np = re.sub(r"\bnewspapers?\b", "", q_lower)
            has_article_words = bool(re.search(r"\b(articles?|story|stories|news)\b", q_no_np))
            is_ordinal_page = bool(
                page_filter and re.search(r"\b(?:1st|first|2nd|second|3rd|third|lead|top|main|head)\s+(?:news|story|article|headline|item)\b", q_lower)
            )
            is_availability = bool(re.search(r"\b(is\s+(?:any\s+)?newspaper\s+available|are\s+there\s+(?:any\s+)?newspapers|is\s+there\s+an?\s+issue|papers?\s+available|newspapers?\s+available|check\s+availability|issues?\s+available|edition\s+available|available\s+for\s+dated?|issues?\s+for\s+dated?|paper\s+for\s+dated?)\b", q_lower))
            is_count = is_availability or (any(w in q_lower for w in ["how many", "total articles", "number of articles", "count of articles", "no of", "count of", "number of issues", "total issues", "no of newspaper", "how many issues", "count of pages", "number of pages"]) and not page_filter)
            is_whole_or_count = is_availability or (bool(page_filter) and not is_ordinal_page) or is_count or any(w in q_lower for w in ["today's paper", "edition", "whole", "entire", "overview", "summarize", "how many", "count of", "number of", "no of", "distribution", "frequency", "trend", "statistics", "volume"])
            is_catalog = (not is_availability) and (is_ordinal_page or any(w in q_lower for w in ["list", "catalog", "manifest", "all articles", "all news", "all stories"]) or bool(category))
            if is_ordinal_page:
                archetype = "article_catalog"
            elif is_archive_np and not has_article_words:
                archetype = "quantitative_trend"
            else:
                archetype = "article_catalog" if is_catalog and not is_whole_or_count else "quantitative_trend"

            is_ad = any(w in q_lower for w in ["advertisement", "advertisements", "ad count", "ads count", "number of ads", "how many ads", "commercials", "notices and ads", "commercial notices"])
            is_photo = is_count and any(w in q_lower for w in ["photo", "photos", "picture", "pictures", "image", "images"])

            if is_archive_np and not any(pat.search(query) for pat, _ in _KNOWN_BRANDS_PATTERNS):
                newspaper = None

            if is_ad:
                atype = "count_advertisements"
            elif is_photo:
                atype = "count_photos"
            elif is_availability or (is_archive_np and not has_article_words) or (is_count and ("issue" in q_lower or "newspaper" in q_lower) and not has_article_words):
                atype = "count_issues"
            elif is_count:
                atype = "count_articles"
            else:
                atype = "issue_summary"

            calls = [build_sql_summary_tool(analysis_type=atype, newspaper_name=newspaper, issue_date=issue_date, date_from=date_from, date_to=date_to, issue_id=issue_id, page_filter=page_filter, category_filter=category, query=query)]
            if page_filter and (is_ordinal_page or any(w in q_lower for w in ["list", "articles on", "stories on", "all articles", "news", "lead", "top", "first", "1st"])):
                calls.append(build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, page_filter=page_filter, top_k=6, purpose=f"Articles on Page {page_filter}"))

        # 7. Factual Lookup (Default)
        else:
            archetype = "factual_lookup"
            calls = [build_hybrid_search_tool(query=query, newspaper_name=newspaper, date_from=date_from, date_to=date_to, page_filter=page_filter, top_k=4 if is_target_art else 6, purpose="Search for factual evidence")]

        if enable_web_search:
            calls.append(build_web_search_tool(build_targeted_web_query(query), num_results=5))

        blueprint = self._build_heuristic_answer_blueprint(query, archetype, params)
        return PlanResult(
            archetype=archetype,
            reasoning=f"Deterministic heuristic plan ({archetype})",
            tool_calls=calls,
            answer_blueprint=blueprint,
        )

    _build_heuristic_answer_blueprint = staticmethod(build_heuristic_answer_blueprint)


# ---------------------------------------------------------------------------
# Public Re-Exports for 100% Backward Compatibility
# ---------------------------------------------------------------------------

__all__ = [
    "ToolName",
    "QueryArchetype",
    "PlannedToolCall",
    "PlanResult",
    "ToolCallSpec",
    "AgentPlan",
    "QueryPlan",
    "ExtractedToolArguments",
    "QueryPlanner",
    "extract_parameters_from_query",
    "_KNOWN_BRANDS_PATTERNS",
    "_SECTION_PATTERNS",
    "build_targeted_web_query",
    "_build_targeted_web_query",
    "resolve_tool_sequence",
    "build_heuristic_answer_blueprint",
    "is_dynamic_analysis_permitted",
    "MATH_AGGREGATE_PATTERNS",
]
