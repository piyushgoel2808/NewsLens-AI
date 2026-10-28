"""Explicit Decision-Making & Tool Selection Policy Layer for NewsLens-AI.

Acts as the single canonical authority for mapping user query intent + extracted ground-truth
parameters to the 8 retrieval tools and query archetypes, preventing downstream tool drift
and centralizing all query-intent predicates across the RAG pipeline.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.extractor import (
    extract_parameters_from_query,
    is_archive_wide_newspaper_query,
)
from app.agent.models import ToolDecision

_VISUAL_KEYWORDS: tuple[str, ...] = (
    "infographic",
    "data chart",
    "chart",
    "diagram",
    "table",
    "graph",
    "visual",
    "figure",
    "photograph",
    "photo",
    "caption",
    "picture",
    "image",
    "illustration",
    "map",
)

MATH_AGGREGATE_PATTERNS: tuple[str, ...] = (
    r"\b(?:avg|average|mean|median)\b",
    r"\b(?:length|word count|word length|longest|shortest)\s+of\s+articles?\b",
    r"\b(?:article\s+length|article\s+word\s+count)\b",
    r"\b(?:correlation|regression|variance|standard deviation|percentile|quantile|histogram|moving average|pearson|spearman|gini)\b",
    r"\b(?:distribution of\s+(?:words?|lengths?|sizes?)|word count distribution|length distribution)\b",
    r"\b(?:ratio of\s+.*to|proportion of\s+.*to)\b",
)

_TIMELINE_PATTERNS = re.compile(
    r"\b(timeline|chronolog\w*|evolution|trajectory|progression|history\s+of|over\s+time|developments?\s+over|"
    r"trace\s+(?:it\s+)?back|where\s+it\s+began|how\s+it\s+(?:unfolded|started)|from\s+the\s+beginning)\b",
    re.IGNORECASE,
)

_ENTITY_DEEP_DIVE_PATTERNS = re.compile(
    r"\b(everything\s+about|all\s+mentions\s+of|profile\s+the\s+coverage\s+of|profile\s+of|who\s+is\s+[A-Z]|"
    r"mentions?\s+of|coverage\s+regarding|news\s+regarding|involvement\s+of|role\s+of|tracked\s+entity)\b",
    re.IGNORECASE,
)

_ORDINAL_LEAD_STORY_PATTERNS = re.compile(
    r"\b(?:1st|first|2nd|second|3rd|third|lead|top|main|head)\s+(?:news|story|article|headline|item)\b",
    re.IGNORECASE,
)

_MACRO_EDITION_PATTERNS = re.compile(
    r"\b(?:summarize\s+(?:the\s+)?(?:whole|entire|full)\s+(?:newspaper|issue|edition|paper)|"
    r"summarize\s+(?:today'?s\s+)?(?:issue|edition|paper)|"
    r"overview\s+of\s+(?:the\s+)?(?:whole\s+)?(?:newspaper|issue|edition)|"
    r"whole\s+newspaper\s+summary)\b",
    re.IGNORECASE,
)

_LISTING_CATALOG_PATTERNS = re.compile(
    r"\b(list\s+all|show\s+all|all\s+articles|all\s+news|all\s+headlines|all\s+stories|"
    r"what\s+articles\s+are\s+on|what\s+news\s+are\s+on|articles\s+on\s+page|news\s+on\s+page|"
    r"headings\s+on\s+page|show\s+me\s+all\s+headlines|what\s+are\s+all\s+the\s+articles)\b",
    re.IGNORECASE,
)

_COVERAGE_AUDIT_PATTERNS = re.compile(
    r"\b(?:did\s+.+\s+(?:report\s+on|cover|publish|carry)|was\s+.+\s+(?:reported|covered|published|carried)\s+(?:in|by)|"
    r"(?:coverage|reporting)\s+(?:of|on)\s+.+\s+(?:in|by)|check\s+(?:if|whether)\s+.+\s+(?:reported|covered))\b",
    re.IGNORECASE,
)

_AVAILABILITY_PATTERNS = re.compile(
    r"\b(is\s+(?:any\s+)?newspaper\s+available|are\s+there\s+(?:any\s+)?newspapers|is\s+there\s+an?\s+issue|"
    r"papers?\s+available|newspapers?\s+available|check\s+availability|issues?\s+available|edition\s+available|"
    r"available\s+for\s+dated?|issues?\s+for\s+dated?|paper\s+for\s+dated?|available\s+on\b)\b",
    re.IGNORECASE,
)

_COUNT_PATTERNS = re.compile(
    r"\b(how\s+many|count\s+of|number\s+of|no\s+of|total\s+number\s+of|total\s+issues|total\s+articles|"
    r"total\s+advertisements|total\s+photos|frequency\s+trend|distribution\s+of\s+articles\s+across\s+topics|"
    r"volume\s+of|trend\s+of|distribution\s+of|total\s+newspapers|photos?\s+in\s+each|photos?\s+per|articles?\s+per|"
    r"list\s+distinct|which\s+newspapers|what\s+newspapers|available\s+newspapers|newspaper\s+names|list\s+newspapers)\b",
    re.IGNORECASE,
)

_ANALYTICAL_PATTERNS = re.compile(
    r"\b(correlation|regression|variance|standard deviation|percentile|median word count|"
    r"histogram|distribution of word|distribution of|moving average|pearson|spearman|gini|"
    r"avg length|average length|average word count|avg word count|length of articles|word count statistics|"
    r"longest article|shortest article|article length|article word count)\b",
    re.IGNORECASE,
)

_COMPARATIVE_PATTERNS = re.compile(
    r"\b(compare|comapare|compa[a-z]*|contrast[a-z]*|diff[a-z]*|versus|vs\.?|differences?\s+between|"
    r"across\s+newspapers|across\s+different\s+papers|different\s+papers\s+cover|both\s+newspapers|"
    r"all\s+available\s+newspaper|all\s+the\s+available|similar\s+articles|shared\s+articles|omissions\s+across)\b",
    re.IGNORECASE,
)

_GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings|who\s+are\s+you|what\s+can\s+you\s+do|help|namaste)[!?.]*$",
    re.IGNORECASE,
)


def is_dynamic_analysis_permitted(query: str) -> bool:
    """Return True only when query requires mathematical/statistical computation rather than a summary word cap."""
    q_low = (query or "").lower()
    if any(re.search(pat, q_low) for pat in MATH_AGGREGATE_PATTERNS):
        return True
    is_answer_length_constraint = bool(
        re.search(
            r"\b(?:summarize|summary|overview|explain|tell me about|write an essay|in brief)\b"
            r".*\b(?:in|under|around|within|less than)\s+\d+\s+(?:words|sentences|paragraphs)\b",
            q_low,
        )
    )
    return not is_answer_length_constraint


def is_availability_query(query: str) -> bool:
    """Return True when query checks whether issues/newspapers are available on a date."""
    return bool(_AVAILABILITY_PATTERNS.search((query or "").lower()))


def is_quantitative_query(query: str) -> bool:
    """Return True when query asks for counts, totals, distributions, or archive-wide publication lists."""
    q = query or ""
    return bool(_COUNT_PATTERNS.search(q.lower()) or is_archive_wide_newspaper_query(q))


def is_analytical_query(query: str) -> bool:
    """Return True when query asks for statistical/mathematical computation (average length, correlation, variance)."""
    q = query or ""
    return bool(_ANALYTICAL_PATTERNS.search(q.lower()) and is_dynamic_analysis_permitted(q))


def is_comparative_query(query: str, target_newspapers: list[str] | None = None) -> bool:
    """Return True when query compares multiple newspapers, editions, or pages."""
    if target_newspapers and len(target_newspapers) >= 2:
        return True
    return bool(_COMPARATIVE_PATTERNS.search((query or "").lower()))


def is_volume_query(query: str) -> bool:
    """Return True when query asks for total issue/newspaper volume across the archive."""
    q_low = (query or "").lower()
    return bool(
        re.search(r"\b(?:how\s+many|total|count\s+of|number\s+of|volume\s+of)\s+.*(?:issues?|newspapers?|editions?|papers?)\b", q_low)
        or ("issue" in q_low and any(w in q_low for w in ("how many", "total", "count", "number of", "list all")))
        or is_archive_wide_newspaper_query(query or "")
    )


def is_visual_query(query: str, attached_photo_id: int | None = None) -> bool:
    """Return True when query references visual assets, charts, infographics, or an attached photo."""
    if attached_photo_id is not None:
        return True
    q_low = (query or "").lower()
    return any(w in q_low for w in _VISUAL_KEYWORDS)


def is_ordinal_lead_story_query(query: str) -> bool:
    """Return True when query asks for the 1st/lead/top story on a page rather than a full catalog table."""
    q_low = (query or "").lower()
    if any(w in q_low for w in ("list all", "all articles", "all news", "all stories", "compare")):
        return False
    return bool(_ORDINAL_LEAD_STORY_PATTERNS.search(q_low))


def is_macro_edition_query(query: str) -> bool:
    """Return True when query asks to summarize an entire newspaper issue/edition."""
    return bool(_MACRO_EDITION_PATTERNS.search((query or "").lower()))


class ToolDecisionPolicy:
    """Deterministic-first decision engine for selecting query archetypes and retrieval tools."""

    is_dynamic_analysis_permitted = staticmethod(is_dynamic_analysis_permitted)
    is_availability_query = staticmethod(is_availability_query)
    is_quantitative_query = staticmethod(is_quantitative_query)
    is_analytical_query = staticmethod(is_analytical_query)
    is_comparative_query = staticmethod(is_comparative_query)
    is_volume_query = staticmethod(is_volume_query)
    is_visual_query = staticmethod(is_visual_query)
    is_ordinal_lead_story_query = staticmethod(is_ordinal_lead_story_query)
    is_macro_edition_query = staticmethod(is_macro_edition_query)

    @classmethod
    def decide(
        cls,
        query: str,
        extracted: dict[str, Any] | None = None,
        active_issue_date: str | None = None,
        active_newspapers: list[str] | None = None,
        attached_article_id: int | None = None,
        attached_photo_id: int | None = None,
        enable_web_search: bool = False,
    ) -> ToolDecision:
        """Evaluate query and context to produce an authoritative ToolDecision."""
        q_clean = (query or "").strip()
        q_low = q_clean.lower()
        if _GREETING_PATTERN.match(q_clean):
            return ToolDecision(
                intent="conversational_greeting",
                archetype="conversational_meta_query",
                selected_tools=[],
                is_deterministic=True,
                confidence=1.0,
                rationale="User submitted a conversational greeting or capability query.",
            )

        params = extracted if extracted is not None else extract_parameters_from_query(q_clean)

        target_nps = params.get("target_newspapers") or []
        target_dates = params.get("target_dates") or []
        target_pages = params.get("target_pages") or []

        # 1. Statistical / Mathematical Computation (dynamic_analysis)
        if any(re.search(pat, q_low) for pat in MATH_AGGREGATE_PATTERNS) and is_dynamic_analysis_permitted(q_clean):
            return ToolDecision(
                intent="statistical_computation",
                archetype="analytical_computation",
                selected_tools=["dynamic_analysis"],
                is_deterministic=True,
                confidence=0.95,
                rationale="Query requests statistical/mathematical aggregation requiring dynamic Python/SQL computation.",
            )

        # 2. Visual Asset / Photo / Infographic Inspection
        is_photo_count = bool(
            re.search(r"\b(?:how\s+many|count\s+of|number\s+of|total)\s+(?:photos?|images?|pictures?)\b", q_low)
        )
        if not is_photo_count and (
            attached_photo_id is not None
            or any(
                re.search(rf"\b{re.escape(w)}\b", q_low)
                for w in ("infographic", "chart", "diagram", "graph", "photograph", "photo", "picture", "image")
            )
        ):
            tools = ["inspect_visual_asset"]
            if not attached_photo_id:
                tools.append("hybrid_search")
            return ToolDecision(
                intent="visual_asset_inspection",
                archetype="factual_lookup",
                selected_tools=tools,
                is_deterministic=bool(attached_photo_id),
                confidence=0.92,
                rationale="Query targets visual asset, photograph, or infographic inspection.",
            )

        # 3. Cross-Newspaper or Multi-Date / Multi-Page Comparison
        if params.get("is_differential") and len(target_nps) >= 2:
            return ToolDecision(
                intent="coverage_difference",
                archetype="cross_newspaper_comparison",
                selected_tools=["sql_analytics", "hybrid_search"],
                recommended_analysis_type="coverage_difference",
                is_deterministic=True,
                confidence=0.95,
                rationale="Differential coverage query between two named publications.",
            )

        if params.get("is_shared") and len(target_nps) >= 2:
            return ToolDecision(
                intent="shared_coverage",
                archetype="cross_newspaper_comparison",
                selected_tools=["sql_analytics", "hybrid_search"],
                recommended_analysis_type="shared_coverage",
                is_deterministic=True,
                confidence=0.95,
                rationale="Shared/syndicated coverage query between two named publications.",
            )

        if len(target_pages) >= 2 and len(target_nps) <= 1 and _COMPARATIVE_PATTERNS.search(q_low):
            return ToolDecision(
                intent="cross_page_comparison",
                archetype="cross_newspaper_comparison",
                selected_tools=["sql_analytics", "hybrid_search"],
                recommended_analysis_type="issue_summary",
                is_deterministic=True,
                confidence=0.92,
                rationale="Cross-page comparative analysis within a broadsheet edition.",
            )

        if _COMPARATIVE_PATTERNS.search(q_low) or len(target_nps) >= 2:
            if len(target_nps) == 1 and len(target_dates) >= 2:
                return ToolDecision(
                    intent="single_newspaper_multi_date_comparison",
                    archetype="quantitative_trend",
                    selected_tools=["sql_analytics", "hybrid_search"],
                    recommended_analysis_type="issue_summary",
                    is_deterministic=True,
                    confidence=0.92,
                    rationale="Single publication compared across multiple dates.",
                )
            return ToolDecision(
                intent="cross_newspaper_comparison",
                archetype="cross_newspaper_comparison",
                selected_tools=["sql_analytics", "hybrid_search"],
                recommended_analysis_type="issue_summary",
                is_deterministic=False,
                confidence=0.90,
                rationale="Cross-newspaper comparative analysis.",
            )

        # 3b. Single-Newspaper Coverage Presence Audit ("Did X report on Y?")
        if len(target_nps) == 1 and _COVERAGE_AUDIT_PATTERNS.search(q_low):
            return ToolDecision(
                intent="coverage_presence_audit",
                archetype="factual_lookup",
                selected_tools=["coverage_analysis", "hybrid_search"],
                is_deterministic=True,
                confidence=0.92,
                rationale="Single-newspaper topic presence or omission audit.",
            )

        # 4. Chronological Timeline / Narrative Trajectory
        if _TIMELINE_PATTERNS.search(q_low):
            return ToolDecision(
                intent="thematic_timeline",
                archetype="thematic_timeline",
                selected_tools=["timeline_builder", "hybrid_search"],
                is_deterministic=True,
                confidence=0.92,
                rationale="Chronological evolution or timeline query.",
            )

        # 5. Quantitative Counts & Archive Availability
        if _AVAILABILITY_PATTERNS.search(q_low):
            return ToolDecision(
                intent="archive_availability",
                archetype="quantitative_trend",
                selected_tools=["sql_analytics"],
                recommended_analysis_type="count_issues",
                is_deterministic=True,
                confidence=0.95,
                rationale="Archive availability audit for date or publication.",
            )

        if _COUNT_PATTERNS.search(q_low) or is_archive_wide_newspaper_query(q_clean):
            if any(w in q_low for w in ("advertisement", "ads", "ad count", "commercial notice")):
                a_type = "count_advertisements"
            elif any(w in q_low for w in ("photo", "image", "picture", "infographic")):
                a_type = "count_photos"
            elif target_pages:
                a_type = "issue_summary"
            elif any(w in q_low for w in ("issue", "newspaper", "publication", "edition", "dailies", "paper")) and "article" not in q_low:
                a_type = "count_issues"
            elif "article" in q_low:
                a_type = "count_articles"
            else:
                a_type = "count_issues"
            return ToolDecision(
                intent="quantitative_count",
                archetype="quantitative_trend",
                selected_tools=["sql_analytics"],
                recommended_analysis_type=a_type,
                is_deterministic=True,
                confidence=0.92,
                rationale=f"Quantitative count or archive roster query ({a_type}).",
            )

        # 6a. Ordinal / Lead Story on a Page ("What is the 1st news on page 6?")
        if is_ordinal_lead_story_query(q_clean):
            return ToolDecision(
                intent="ordinal_lead_story",
                archetype="factual_lookup",
                selected_tools=["sql_analytics", "hybrid_search"],
                recommended_analysis_type="issue_summary",
                is_deterministic=True,
                confidence=0.92,
                rationale="Ordinal or lead story identification on a specific page using prominence score ranking.",
            )

        # 6b. Macro Edition Summary ("Summarize the whole newspaper")
        if is_macro_edition_query(q_clean):
            return ToolDecision(
                intent="macro_edition_summary",
                archetype="macro_summary",
                selected_tools=["sql_analytics", "hybrid_search"],
                recommended_analysis_type="issue_summary",
                is_deterministic=True,
                confidence=0.90,
                rationale="Whole-edition executive briefing combining relational manifest and lead story excerpts.",
            )

        # 6c. Page-Scoped or Issue-Wide Article Catalog / Listing
        if _LISTING_CATALOG_PATTERNS.search(q_low):
            tools = ["sql_analytics"]
            if target_pages:
                tools.append("hybrid_search")
            return ToolDecision(
                intent="page_article_catalog" if target_pages else "issue_manifest",
                archetype="article_catalog",
                selected_tools=tools,
                recommended_analysis_type="issue_summary",
                is_deterministic=True,
                confidence=0.90,
                rationale="Article catalog or issue manifest listing request.",
            )

        # 7. Entity Deep Dive
        if _ENTITY_DEEP_DIVE_PATTERNS.search(q_low):
            return ToolDecision(
                intent="entity_profile",
                archetype="entity_deep_dive",
                selected_tools=["entity_search", "hybrid_search"],
                is_deterministic=False,
                confidence=0.88,
                rationale="Entity deep-dive profile request.",
            )

        # 8. Default: Factual Lookup via Hybrid Search
        tools = ["hybrid_search"]
        if enable_web_search:
            tools.append("web_search")
        return ToolDecision(
            intent="factual_lookup",
            archetype="factual_lookup",
            selected_tools=tools,
            is_deterministic=False,
            confidence=0.85,
            rationale="Point-in-time factual lookup or article explanation via hybrid search.",
        )


__all__ = [
    "MATH_AGGREGATE_PATTERNS",
    "ToolDecisionPolicy",
    "is_analytical_query",
    "is_availability_query",
    "is_comparative_query",
    "is_dynamic_analysis_permitted",
    "is_macro_edition_query",
    "is_ordinal_lead_story_query",
    "is_quantitative_query",
    "is_visual_query",
    "is_volume_query",
]
