"""Corrective RAG (CRAG) Evaluator for NewsLens-AI Agentic Workflow.

Grades retrieved broadsheet evidence for semantic groundedness, preserves vector search hits
against over-aggressive lexical pruning, and triggers corrective retrieval fallbacks when needed.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    extract_parameters_from_query,
    is_archive_wide_newspaper_query,
)
from app.agent.state import AgentState, ToolExecutionRecord
from app.agent.tool_maker import ToolMaker
from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry
from app.retrieval.entity_filter import EntitySearchEngine
from app.retrieval.sql_analytics import SQLAnalyticsEngine, normalize_date_to_iso
from app.retrieval.web_search import WebSearchEngine

logger = get_logger(__name__)

_QUANT_QUERY_PATTERN = re.compile(
    r"\b(how many|count of|number of|no of|volume of|trend of|distribution of|"
    r"total|total issues|total articles|total photos|total newspapers|"
    r"percentage|ratio|average|mean|median|"
    r"correlation|variance|distribution|frequency|breakdown|photos? in each|"
    r"photos? per|articles? per|word count|how much|"
    r"distinct|list distinct|which newspapers|what newspapers|available newspapers|"
    r"newspaper names|list newspapers)\b",
    re.IGNORECASE,
)

_ANALYTICAL_QUERY_PATTERN = re.compile(
    r"\b(correlation|regression|variance|standard deviation|percentile|median word count|"
    r"histogram|distribution of word|distribution of|moving average|pearson|spearman|gini|"
    r"avg length|average length|average word count|avg word count|length of articles|word count statistics|"
    r"longest article|shortest article|article length|article word count)\b",
    re.IGNORECASE,
)


@dataclass
class EvaluationVerdict:
    """Multi-dimensional qualitative evaluation of retrieved evidence against query intent."""

    is_sufficient: bool
    quality_score: float  # 0.0 - 1.0
    gap_reason: str | None = None
    detected_gaps: list[str] = field(default_factory=list)
    recommended_action: str = "proceed_to_synthesis"  # "proceed_to_synthesis" | "replan_static_tools" | "synthesize_dynamic_tool"
    corrective_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize verdict to a dictionary for state storage."""
        return {
            "is_sufficient": self.is_sufficient,
            "quality_score": self.quality_score,
            "gap_reason": self.gap_reason,
            "detected_gaps": self.detected_gaps,
            "recommended_action": self.recommended_action,
            "corrective_hints": self.corrective_hints,
        }


def _has_quantitative_payload(evidence: list[dict[str, Any]]) -> bool:
    """Verify if retrieved evidence contains structured aggregate numbers, tables, or metric dictionaries."""
    for item in evidence:
        meta = item.get("metadata")
        if meta and isinstance(meta, dict) and any(
            isinstance(v, (int, float, dict, list))
            and bool(v)
            and not (isinstance(v, float) and (v != v or str(v).lower() == "nan"))
            for v in meta.values()
        ):
            return True
        snip = item.get("snippet") or item.get("summary") or ""
        # Check for Markdown table
        if re.search(r"\|.*\|.*\|\s*\n\|[\s\-:]+\|", snip):
            return True
        # Check for aggregate tool record with non-zero article_id or summary counts
        if (
            (item.get("source_tool", "").startswith("sql_analytics") or item.get("source_tool") == "dynamic_analysis")
            and item.get("article_id") == 0
        ):
            if re.search(
                r"\b(?:total|count|breakdown|photos?|articles?|advertisements?|ads?|issues?|average|avg|mean|median|length|words?|word\s+count|distribution|ratio|correlation|variance|std)(?:\s+[a-zA-Z]+)*:\s*\d+(?:\.\d+)?\b",
                snip,
                re.I,
            ):
                return True
            if re.search(
                r"\b\d+(?:\.\d+)?\s*(?:articles?|photos?|advertisements?|ads?|issues?|words?|chars?)\s+(?:found|average|avg|mean|per\s+article|each)\b",
                snip,
                re.I,
            ):
                return True
    return False

# Stop-words to exclude from core query terms during lexical grading
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "tell", "what", "which", "who",
    "whom", "this", "that", "these", "those", "have", "has", "had",
    "give", "details", "information", "report", "news", "articles", "anything",
    "something", "know", "find", "show", "read", "say", "said",
})


def _stem(word: str) -> str:
    """Stem word token using common suffix rules."""
    w = word.lower()
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


def _score_relevance(item: dict[str, Any], query_tokens: list[str]) -> float:
    """Compute lexical relevance score between query tokens and evidence text."""
    if not query_tokens:
        return 1.0
    hl = (item.get("headline") or "").lower()
    snip = (item.get("snippet") or item.get("full_text") or item.get("summary") or "").lower()
    combined = f"{hl} {snip}"
    corpus_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", combined))
    corpus_stems = {_stem(w) for w in corpus_words}

    matches = 0.0
    for qt in query_tokens:
        stem = _stem(qt)
        if qt in hl or stem in hl:
            matches += 2.0
        elif qt in combined or stem in corpus_stems:
            matches += 1.0

    return matches / max(1.0, float(len(query_tokens)))


def is_structural_or_relevant_evidence(
    item: dict[str, Any],
    archetype: str,
    target_date: str | None = None,
) -> bool:
    """Determine whether evidence is structural or has a high-confidence semantic floor.

    Protects relational manifests, matrices, and high-confidence dense vector search hits
    (prominence_score >= 0.65) from naive lexical stem pruning, while ensuring that if an
    explicit target date was requested, items from completely unrelated dates or unfiltered
    data without auditing target_date are not falsely given full protection.
    """
    snip = str(item.get("snippet") or "")
    hl = str(item.get("headline") or "")
    if (
        snip.startswith("⚠️")
        or "Issue Summary Error:" in hl
        or "error" in hl.lower()
        or "error" in snip.lower()
        or bool(re.search(r"\b(?:nan|null)\s*(?:words?|articles?|issues?|pages?|%|\b)", snip, re.I))
        or "is nan" in snip.lower()
    ):
        return False

    if target_date:
        norm_target = normalize_date_to_iso(target_date)
        it_date = normalize_date_to_iso(item.get("issue_date"))
        it_snip = item.get("snippet", "")
        it_meta = item.get("metadata") or {}
        meta_target = normalize_date_to_iso(it_meta.get("target_date")) or normalize_date_to_iso(
            (it_meta.get("filters") or {}).get("issue_date")
        )
        if it_date and it_date != norm_target and it_date != "Overview":
            return False
        if (
            it_date == "Overview"
            and norm_target
            and norm_target not in it_snip
            and meta_target != norm_target
        ):
            return False

    src = item.get("source_tool", "")
    return (
        archetype in ("cross_newspaper_comparison", "quantitative_trend", "article_catalog", "analytical_computation")
        or src.startswith("sql_analytics")
        or src in ("coverage_analysis", "dynamic_analysis")
        or item.get("article_id") == 0
        or float(item.get("prominence_score", 0.0)) >= 0.65
    )


class EvidenceEvaluator:
    """Grades retrieval relevance and orchestrates Corrective RAG (CRAG) fallback retrieval."""

    def __init__(
        self,
        entity_search: EntitySearchEngine,
        web_search: WebSearchEngine,
        tool_maker: ToolMaker | None = None,
        sql_analytics: SQLAnalyticsEngine | None = None,
        provider: ChatModelProvider | None = None,
    ) -> None:
        self._entity_search = entity_search
        self._web_search = web_search
        self._tool_maker = tool_maker
        self._sql_analytics = sql_analytics
        self._provider = provider

    def filter_evidence(
        self,
        evidence: list[dict[str, Any]],
        query: str,
        archetype: str,
    ) -> list[dict[str, Any]]:
        """Retain evidence items that have positive relevance to query tokens OR are protected structural/semantic hits."""
        raw_query_words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower())
        query_tokens = [w for w in raw_query_words if w not in _STOP_WORDS]
        params = extract_parameters_from_query(query)
        target_dates = params.get("target_dates") or []
        target_date = params.get("issue_date") or (target_dates[0] if target_dates else None)

        filtered = [
            item for item in evidence
            if is_structural_or_relevant_evidence(item, archetype, target_date=target_date) or _score_relevance(item, query_tokens) > 0.0
        ]
        filtered.sort(
            key=lambda it: (1.0 if is_structural_or_relevant_evidence(it, archetype, target_date=target_date) else _score_relevance(it, query_tokens)),
            reverse=True,
        )
        return filtered

    def audit_evidence_sufficiency(
        self,
        evidence: list[dict[str, Any]],
        state: AgentState,
    ) -> EvaluationVerdict:
        """Audit retrieval quality across completeness, comparative balance, and quantitative sufficiency."""
        query = state.get("query", "")
        archetype = state.get("archetype", "factual_lookup")
        detected_gaps: list[str] = []
        is_analytical = bool(_ANALYTICAL_QUERY_PATTERN.search(query))

        # 0. Empty, stub, or error-only evidence
        is_quant_query = bool(_QUANT_QUERY_PATTERN.search(query)) or is_archive_wide_newspaper_query(query)
        is_availability = bool(
            re.search(
                r"\b(is\s+(?:any\s+)?newspaper\s+available|are\s+there\s+(?:any\s+)?newspapers|is\s+there\s+an?\s+issue|"
                r"papers?\s+available|newspapers?\s+available|check\s+availability|issues?\s+available|edition\s+available|"
                r"available\s+for\s+dated?|issues?\s+for\s+dated?|paper\s+for\s+dated?|available\s+on\b)\b",
                query.lower(),
            )
        )
        valid_evidence = [
            item for item in evidence
            if not str(item.get("snippet") or "").startswith("⚠️")
            and "Issue Summary Error:" not in str(item.get("headline") or "")
            and not ("error" in str(item.get("headline") or "").lower() and item.get("article_id") == 0)
            and not bool(re.search(r"\b(?:nan|null)\s*(?:words?|articles?|issues?|pages?|%|\b)", str(item.get("snippet") or ""), re.I))
            and "is nan" not in str(item.get("snippet") or "").lower()
        ]
        if not valid_evidence or not any(len((item.get("snippet") or item.get("summary") or "").strip()) >= 20 for item in valid_evidence):
            # Distinguish legitimate factual archive misses from engine crashes in CRAG
            if is_availability and any(
                "not in archive" in str(it.get("snippet", "")).lower()
                or "no issue found" in str(it.get("snippet", "")).lower()
                or "0 issues found" in str(it.get("snippet", "")).lower()
                for it in evidence
            ):
                return EvaluationVerdict(
                    is_sufficient=True,
                    quality_score=0.85,
                    gap_reason=None,
                    detected_gaps=[],
                    recommended_action="proceed_to_synthesis",
                )
            gap_type = "tool_execution_error_gap" if evidence else "empty_retrieval"
            return EvaluationVerdict(
                is_sufficient=False,
                quality_score=0.0,
                gap_reason="Static retrieval returned 0 grounded evidence records, encountered tool execution errors, or returned invalid NaN calculations.",
                detected_gaps=[gap_type],
                recommended_action="synthesize_dynamic_tool" if (is_analytical or is_quant_query) else "replan_static_tools",
            )

        # 1. Comparative / Cross-Newspaper Balance Audit
        params = extract_parameters_from_query(query)
        np_source = params.get("newspaper_name") or state.get("active_newspaper_name")
        np_comp = params.get("comparison_newspaper")
        target_nps = [np.strip() for np in [np_source, np_comp] if np and np.strip()]

        is_cross_np = (
            archetype == "cross_newspaper_comparison"
            or params.get("is_differential")
            or params.get("is_shared")
            or len(target_nps) >= 2
        )

        if is_cross_np and len(target_nps) >= 2:
            present_nps: set[str] = set()
            for item in evidence:
                item_np = (item.get("newspaper_name") or "").lower()
                for req in target_nps:
                    if req.lower() in item_np or item_np in req.lower():
                        present_nps.add(req)

            missing_nps = [req for req in target_nps if req not in present_nps]
            if missing_nps:
                detected_gaps.append(f"missing_newspaper_coverage:{','.join(missing_nps)}")
                gap_msg = (
                    f"Comparative query requested coverage across {', '.join(target_nps)}, "
                    f"but retrieved evidence is completely missing articles from: {', '.join(missing_nps)}."
                )
                return EvaluationVerdict(
                    is_sufficient=False,
                    quality_score=0.35,
                    gap_reason=gap_msg,
                    detected_gaps=detected_gaps,
                    recommended_action="replan_static_tools",
                    corrective_hints={"missing_newspapers": missing_nps},
                )

        # 1.5. Temporal / Date Alignment Audit
        target_dates = params.get("target_dates") or []
        target_date = params.get("issue_date") or (target_dates[0] if target_dates else None)
        norm_target = normalize_date_to_iso(target_date) if target_date else None
        if norm_target:
            matches_target_date = False
            for item in evidence:
                it_date = normalize_date_to_iso(item.get("issue_date"))
                it_snip = item.get("snippet", "") + " " + item.get("headline", "")
                it_meta = item.get("metadata") or {}
                meta_target = normalize_date_to_iso(it_meta.get("target_date")) or normalize_date_to_iso(
                    (it_meta.get("filters") or {}).get("issue_date")
                )
                if (
                    (it_date and it_date == norm_target)
                    or (meta_target and meta_target == norm_target)
                    or (norm_target in it_snip)
                    or (target_date and target_date in it_snip)
                ):
                    matches_target_date = True
                    break

            if not matches_target_date:
                detected_gaps.append(f"temporal_mismatch_missing_date:{norm_target}")
                gap_msg = (
                    f"Query explicitly requested broadsheet information for date {norm_target}, "
                    f"but retrieved evidence contains data from different dates or unfiltered archives without auditing this date."
                )
                return EvaluationVerdict(
                    is_sufficient=False,
                    quality_score=0.30,
                    gap_reason=gap_msg,
                    detected_gaps=detected_gaps,
                    recommended_action="replan_static_tools",
                    corrective_hints={"issue_date": norm_target},
                )

        # 2. Quantitative / Aggregate Metric Audit
        is_quant_query = bool(_QUANT_QUERY_PATTERN.search(query))
        is_availability = bool(
            re.search(
                r"\b(is\s+(?:any\s+)?newspaper\s+available|are\s+there\s+(?:any\s+)?newspapers|is\s+there\s+an?\s+issue|"
                r"papers?\s+available|newspapers?\s+available|check\s+availability|issues?\s+available|edition\s+available|"
                r"available\s+for\s+dated?|issues?\s+for\s+dated?|paper\s+for\s+dated?)\b",
                query.lower(),
            )
        )
        if is_quant_query:
            has_quant = _has_quantitative_payload(evidence)
            if not has_quant:
                detected_gaps.append("missing_quantitative_payload")
                gap_msg = (
                    "Query requested quantitative aggregates, counts, or statistical distributions, "
                    "but retrieved evidence contains only unaggregated text snippets without numerical metrics or tables."
                )
                return EvaluationVerdict(
                    is_sufficient=False,
                    quality_score=0.40,
                    gap_reason=gap_msg,
                    detected_gaps=detected_gaps,
                    recommended_action="synthesize_dynamic_tool" if (is_analytical or is_quant_query) else "replan_static_tools",
                    corrective_hints={"missing_quantitative": True},
                )

            # Check if quantitative query returned all zeros when not an explicit availability check
            if not is_availability:
                has_positive = False
                for it in evidence:
                    m = it.get("metadata") or {}
                    if isinstance(m, dict):
                        for _k, v in m.items():
                            if isinstance(v, (int, float)) and v > 0:
                                has_positive = True
                                break
                    if has_positive:
                        break
                    sn = (it.get("snippet") or "") + " " + (it.get("headline") or "")
                    if re.search(r"\|.*\|.*\|\s*\n\|[\s\-:]+\|", sn):
                        has_positive = True
                        break
                    if re.search(r"\b(?:total|count|breakdown|photos?|articles?|advertisements?|ads?|issues?|average|avg|mean|median|length|words?|word\s+count|distribution|ratio|correlation|variance|std)(?:\s+[a-zA-Z]+){0,2}\s*[:=]\s*(?:\*\*)?([1-9]\d*(?:\.\d+)?|0\.\d+)\b", sn, re.I):
                        has_positive = True
                        break
                    if re.search(r"\b(?:average|avg|mean|median|length|word\s*count|words?|total|count)\s*[:=]\s*(?:\*\*)?([1-9]\d*(?:\.\d+)?)\b", sn, re.I):
                        has_positive = True
                        break
                    if re.search(r"\b([1-9]\d*(?:\.\d+)?)\s*(?:articles?|photos?|advertisements?|ads?|issues?|words?|chars?)\b", sn, re.I):
                        has_positive = True
                        break

                if not has_positive:
                    detected_gaps.append("zero_count_aggregate_gap")
                    gap_msg = (
                        "Static archive retrieval returned 0 matching records or failed to compute aggregated counts for this quantitative query. "
                        "Dynamic database tool synthesis required to inspect schema and compute exact aggregate results."
                    )
                    return EvaluationVerdict(
                        is_sufficient=False,
                        quality_score=0.25,
                        gap_reason=gap_msg,
                        detected_gaps=detected_gaps,
                        recommended_action="synthesize_dynamic_tool",
                        corrective_hints={"require_dynamic_tool": True},
                    )

            # Temporal range scope audit: query requested a range/month, but evidence only audited a single day
            req_from = params.get("date_from")
            req_to = params.get("date_to")
            if req_from and req_to and req_from != req_to:
                range_audited = False
                for it in evidence:
                    m = it.get("metadata") or {}
                    filt = m.get("filters") or {}
                    sn = (it.get("snippet") or "") + " " + (it.get("headline") or "")
                    if (filt.get("date_from") and filt.get("date_to")) or (req_from in sn and req_to in sn):
                        range_audited = True
                        break
                if not range_audited:
                    detected_gaps.append(f"temporal_range_scope_mismatch:{req_from}_to_{req_to}")
                    gap_msg = (
                        f"Query requested coverage across date range {req_from} to {req_to}, "
                        f"but retrieved evidence only audited a single day without covering the full range."
                    )
                    return EvaluationVerdict(
                        is_sufficient=False,
                        quality_score=0.30,
                        gap_reason=gap_msg,
                        detected_gaps=detected_gaps,
                        recommended_action="synthesize_dynamic_tool",
                        corrective_hints={"date_from": req_from, "date_to": req_to},
                    )

            # Archive-wide publication scope audit: query asked for total/distinct newspapers across archive, but evidence narrowed to single publication
            is_archive_np = is_archive_wide_newspaper_query(query)
            named_in_q = any(pat.search(query) for pat, _ in _KNOWN_BRANDS_PATTERNS)
            if is_archive_np and not named_in_q:
                only_single_np = False
                for it in evidence:
                    m = it.get("metadata") or {}
                    filt = m.get("filters") or {}
                    np_filtered = filt.get("newspaper_name") or it.get("newspaper_name")
                    if np_filtered and np_filtered not in ("Archive", "All Newspapers", None, "") and not m.get("distinct_newspapers_count"):
                        only_single_np = True
                        break
                if only_single_np:
                    detected_gaps.append("archive_newspaper_scope_mismatch")
                    gap_msg = (
                        "Query requested archive-wide newspaper availability or count, but retrieved evidence was narrowed to a single publication."
                    )
                    return EvaluationVerdict(
                        is_sufficient=False,
                        quality_score=0.30,
                        gap_reason=gap_msg,
                        detected_gaps=detected_gaps,
                        recommended_action="synthesize_dynamic_tool",
                        corrective_hints={"is_archive_wide": True},
                    )

            # Publication-wide issue volume audit: query asked for total issues of a newspaper without specifying a date,
            # but evidence only summarized a single day's issue manifest instead of an archive-wide issue count
            is_issue_vol_q = bool(
                re.search(r"\b(?:how\s+many|total|count\s+of|number\s+of|volume\s+of)\s+.*(?:issues?|newspapers?|editions?|papers?)\b", query.lower())
                or ("issue" in query.lower() and any(w in query.lower() for w in ["how many", "total", "count", "number of"]))
            )
            has_explicit_date_in_q = bool(params.get("issue_date") or params.get("target_dates") or re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", query))
            if is_issue_vol_q and not has_explicit_date_in_q:
                # Evidence is insufficient if it is merely a single-issue daily manifest rather than a relational issue count audit
                has_issue_count_audit = any(
                    "=== RELATIONAL ISSUE COUNT" in str(it.get("snippet", ""))
                    or "Total Matching Issues:" in str(it.get("snippet", ""))
                    or (it.get("metadata", {}).get("total_issues") is not None and it.get("metadata", {}).get("target_date") == "Overview")
                    for it in evidence
                )
                if not has_issue_count_audit:
                    detected_gaps.append("single_issue_scope_mismatch")
                    gap_msg = (
                        "Query requested total issues of a publication across the archive, but retrieved evidence only returned a single day's issue manifest. "
                        "Dynamic tool synthesis required to inspect schema and compute exact aggregate counts across all issues."
                    )
                    return EvaluationVerdict(
                        is_sufficient=False,
                        quality_score=0.30,
                        gap_reason=gap_msg,
                        detected_gaps=detected_gaps,
                        recommended_action="synthesize_dynamic_tool",
                        corrective_hints={"analysis_type": "count_issues", "is_archive_wide": True},
                    )

        # 3. Semantic Relevance Floor Audit
        raw_query_words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower())
        query_tokens = [w for w in raw_query_words if w not in _STOP_WORDS]
        scores: list[float] = []
        for it in evidence:
            if is_structural_or_relevant_evidence(it, archetype, target_date=norm_target):
                scores.append(1.0)
            else:
                lex_score = _score_relevance(it, query_tokens)
                rr_score = float(it.get("rerank_score") or 0.0)
                rrf_score = float(it.get("rrf_score") or 0.0)
                # Boost confidence if neural reranker or vector search confirmed strong semantic match
                if rr_score >= 0.30 or rrf_score >= 0.015:
                    scores.append(max(lex_score, 0.85))
                else:
                    scores.append(lex_score)

        avg_score = sum(scores) / max(1, len(scores))
        max_score = max(scores) if scores else 0.0

        if max_score < 0.20 and avg_score < 0.15:
            detected_gaps.append("low_semantic_relevance")
            gap_msg = f"Retrieved evidence has weak semantic relevance (max: {max_score:.2f}, avg: {avg_score:.2f}) to query terms: {query_tokens[:5]}."
            return EvaluationVerdict(
                is_sufficient=False,
                quality_score=round(avg_score, 2),
                gap_reason=gap_msg,
                detected_gaps=detected_gaps,
                recommended_action="synthesize_dynamic_tool" if is_analytical else "replan_static_tools",
            )

        return EvaluationVerdict(
            is_sufficient=True,
            quality_score=round(max(0.70, avg_score), 2),
            gap_reason=None,
            detected_gaps=[],
            recommended_action="proceed_to_synthesis",
            corrective_hints={},
        )

    async def evaluate_evidence_async(
        self,
        evidence: list[dict[str, Any]],
        state: AgentState,
        model_override: str | None = None,
    ) -> EvaluationVerdict:
        """Reflexive hybrid evaluation: fast-floor check followed by LLM-as-Judge for borderline/empty evidence."""
        archetype = state.get("archetype", "factual_lookup")
        if archetype in ("clarification_needed", "conversational_meta_query"):
            return EvaluationVerdict(
                is_sufficient=True,
                quality_score=1.0,
                gap_reason=None,
                detected_gaps=[],
                recommended_action="proceed_to_synthesis",
            )

        # 1. Fast-Floor Heuristic Evaluation (0ms overhead)
        heuristic_verdict = self.audit_evidence_sufficiency(evidence, state)

        # High-confidence pass: if heuristic audit passes with high quality, proceed immediately
        if heuristic_verdict.is_sufficient and heuristic_verdict.quality_score >= 0.70:
            return heuristic_verdict

        # Attached asset grounding pass
        att_art = state.get("attached_article_id")
        if att_art and any(it.get("article_id") == att_art for it in evidence):
            return EvaluationVerdict(
                is_sufficient=True,
                quality_score=0.95,
                gap_reason=None,
                detected_gaps=[],
                recommended_action="proceed_to_synthesis",
            )
        att_pho = state.get("attached_photo_id")
        if att_pho and any(it.get("photo_id") == att_pho for it in evidence):
            return EvaluationVerdict(
                is_sufficient=True,
                quality_score=0.95,
                gap_reason=None,
                detected_gaps=[],
                recommended_action="proceed_to_synthesis",
            )

        # 2. Reflexive LLM-as-Judge (Borderline / Deficient Evidence)
        judge_verdict = await self._evaluate_with_llm_judge(evidence, state, model_override)
        if judge_verdict is not None:
            return judge_verdict

        # 3. Fallback to heuristic verdict if LLM judge is unavailable
        return heuristic_verdict

    async def _evaluate_with_llm_judge(
        self,
        evidence: list[dict[str, Any]],
        state: AgentState,
        model_override: str | None = None,
    ) -> EvaluationVerdict | None:
        """Use lightweight LLM to diagnose retrieval sufficiency and route recovery action."""
        query = state.get("query", "")
        archetype = state.get("archetype", "factual_lookup")
        tool_execs = state.get("tool_executions", [])

        # Format attempted tools summary
        exec_lines = []
        for t in tool_execs:
            t_name = t.get("tool_name", "")
            t_in = t.get("tool_input", {})
            t_cnt = t.get("results_count", 0)
            exec_lines.append(f"- {t_name}({t_in}): returned {t_cnt} records")
        tools_summary = "\n".join(exec_lines) if exec_lines else "None"

        # Format compact evidence snippets (top 5 items, max 250 chars)
        snip_lines = []
        for idx, it in enumerate(evidence[:5], 1):
            hl = it.get("headline") or "Untitled"
            np = it.get("newspaper_name") or "Unknown"
            dt = it.get("issue_date") or ""
            snip = (it.get("snippet") or it.get("summary") or "")[:250].strip()
            snip_lines.append(f"{idx}. [{np} - {dt}] \"{hl}\": {snip}")
        evidence_summary = "\n".join(snip_lines) if snip_lines else "No grounded evidence retrieved."

        prompt = (
            f"You are the expert Retrieval Judge for NewsLens-AI, an intelligence platform over broadsheet newspapers.\n"
            f"Assess whether the retrieved evidence is SUFFICIENT to formulate an accurate and complete answer.\n\n"
            f"User Query: \"{query}\"\n"
            f"Query Archetype: {archetype}\n"
            f"Attempted Tools:\n{tools_summary}\n\n"
            f"Retrieved Evidence:\n{evidence_summary}\n\n"
            f"Available Tool Types:\n"
            f"- `hybrid_search`: Dense vector + keyword text retrieval for articles.\n"
            f"- `sql_analytics`: Relational manifests, article counts, ad counts, photo counts, issue counts, coverage diffs.\n"
            f"- `timeline_builder`: Chronological story evolution.\n"
            f"- `entity_search`: Entity network profiles.\n"
            f"- `dynamic_analysis`: Custom Python/SQL script execution (for multi-table statistical math like Pearson correlation, variance, percentiles, or custom database aggregations and counts across date ranges when static tools fail or return 0 records. NEVER for plain text summarization).\n\n"
            f"Evaluation Criteria:\n"
            f"1. Is the retrieved evidence sufficient to answer the user's specific question?\n"
            f"2. If NOT sufficient, diagnose the exact gap (e.g. wrong date, overly narrow search terms, missing comparison publication, or missing quantitative counts).\n"
            f"3. Select the best recovery action:\n"
            f"   - 'proceed_to_synthesis': Evidence is adequate or best-effort answer can be generated.\n"
            f"   - 'replan_static_tools': Missing articles, wrong dates/sections, or missing counts/manifests that standard archive tools can retrieve.\n"
            f"   - 'synthesize_dynamic_tool': Query requires complex statistical computation (correlation, regression, variance) or custom aggregations/counts across tables where static tools fail or yield 0 records.\n\n"
            f"Respond ONLY with a valid JSON object matching this schema:\n"
            f"{{\n"
            f'  "is_sufficient": boolean,\n'
            f'  "quality_score": float (0.0 to 1.0),\n'
            f'  "gap_diagnosis": string or null,\n'
            f'  "recommended_action": "proceed_to_synthesis" | "replan_static_tools" | "synthesize_dynamic_tool",\n'
            f'  "corrective_hints": {{"suggested_tool": string or null, "suggested_query": string or null, "missing_newspaper": string or null}}\n'
            f"}}"
        )

        candidates: list[ChatModelProvider] = []
        if self._provider is not None:
            candidates.append(self._provider)
        else:
            try:
                reg = get_registry()
                if model_override:
                    with contextlib.suppress(Exception):
                        candidates.append(reg.get_chat_provider(model_override))
                with contextlib.suppress(Exception):
                    p = reg.get_provider("evaluator")
                    if p is not None and hasattr(p, "complete") and p not in candidates:
                        candidates.append(p)
                with contextlib.suppress(Exception):
                    p = reg.get_provider("query_planner")
                    if p is not None and hasattr(p, "complete") and p not in candidates:
                        candidates.append(p)
                for k in reg.get_chat_failover_candidates():
                    with contextlib.suppress(Exception):
                        p = reg.get_chat_provider(k)
                        if p is not None and hasattr(p, "complete") and p not in candidates:
                            candidates.append(p)
            except Exception as e:
                logger.warning("Could not resolve candidate models for LLM Judge", extra={"error": str(e)})

        for prov in candidates:
            try:
                resp = await prov.complete(
                    messages=[
                        Message(role="system", content="You are a strict retrieval quality judge. Respond ONLY in valid JSON."),
                        Message(role="user", content=prompt),
                    ],
                    max_tokens=350,
                    temperature=0.0,
                    thinking_budget=0,
                )
                raw_text = (resp.text or "").strip()
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    is_suff = bool(data.get("is_sufficient", False))
                    q_score = float(data.get("quality_score", 0.0 if not is_suff else 0.8))
                    action = str(data.get("recommended_action", "proceed_to_synthesis" if is_suff else "replan_static_tools"))
                    if action not in ("proceed_to_synthesis", "replan_static_tools", "synthesize_dynamic_tool"):
                        action = "proceed_to_synthesis" if is_suff else "replan_static_tools"

                    # Safeguard: Allow synthesize_dynamic_tool for analytical queries AND quantitative queries
                    if action == "synthesize_dynamic_tool" and not (
                        _ANALYTICAL_QUERY_PATTERN.search(query) or _QUANT_QUERY_PATTERN.search(query)
                    ):
                        action = "replan_static_tools"

                    gap_diag = data.get("gap_diagnosis")
                    hints = data.get("corrective_hints", {})
                    gaps = ["llm_diagnosed_gap"] if (not is_suff and gap_diag) else []

                    return EvaluationVerdict(
                        is_sufficient=is_suff,
                        quality_score=round(max(0.0, min(1.0, q_score)), 2),
                        gap_reason=gap_diag,
                        detected_gaps=gaps,
                        recommended_action=action,
                        corrective_hints=hints if isinstance(hints, dict) else {},
                    )
            except Exception as e:
                logger.debug("LLM Judge invocation failed on candidate provider", extra={"error": str(e)})
                continue

        return None

    async def evaluate_and_fallback(
        self,
        evidence: list[dict[str, Any]],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], list[ToolExecutionRecord]]:
        """Filter evidence, audit sufficiency, and execute corrective dynamic tool fallback if needed."""
        archetype = state.get("archetype", "factual_lookup")
        tool_records: list[ToolExecutionRecord] = []

        # Skip fallback evaluation for meta queries or user clarification states
        if archetype in ("clarification_needed", "conversational_meta_query"):
            return evidence, tool_records

        filtered_evidence = self.filter_evidence(evidence, state.get("query", ""), archetype)
        verdict = self.audit_evidence_sufficiency(filtered_evidence, state)

        if not verdict.is_sufficient:
            logger.info(
                "CRAG triggered: Retrieval evaluation below sufficiency threshold",
                extra={
                    "quality_score": verdict.quality_score,
                    "gap_reason": verdict.gap_reason,
                    "gaps": verdict.detected_gaps,
                },
            )
            fallback_items: list[dict[str, Any]] = list(filtered_evidence)

            # 1. Fallback: Dynamic Tool Synthesis (Database-aware custom Python code)
            if self._tool_maker:
                t_dyn = time.monotonic()
                context: dict[str, Any] = {
                    "available_newspapers": [],
                    "available_dates": [],
                    "categories": [],
                }
                if self._sql_analytics:
                    with contextlib.suppress(Exception):
                        meta = await self._sql_analytics.get_archive_metadata()
                        context["available_newspapers"] = meta.get("newspapers", [])
                        context["available_dates"] = list(meta.get("available_dates", {}).keys())
                        context["categories"] = meta.get("categories", [])

                dyn_res = await self._tool_maker.generate_and_execute(
                    query=state.get("query", ""),
                    context=context,
                    model_override=state.get("model_override"),
                    attempted_tools=state.get("tool_executions", []),
                    gap_diagnosis=verdict.gap_reason,
                )
                if dyn_res.success and dyn_res.evidence_items:
                    # Prepend dynamic synthesis items to prioritize complete computed findings
                    fallback_items = dyn_res.evidence_items + fallback_items
                    dur_ms = round((time.monotonic() - t_dyn) * 1000)
                    tool_records.append(
                        ToolExecutionRecord(
                            tool_name="crag_dynamic_tool_fallback",
                            tool_input={
                                "query": state.get("query", ""),
                                "gap_diagnosis": verdict.gap_reason,
                            },
                            results_count=len(dyn_res.evidence_items),
                            execution_time_ms=dur_ms,
                        )
                    )
                    return fallback_items, tool_records

            # 2. Fallback: Entity Search if dynamic tool did not produce items and an entity candidate is detected
            if not fallback_items:
                query_terms = [w.strip() for w in state.get("query", "").split() if len(w) > 3 and w[0].isupper()]
                if query_terms:
                    t_ent = time.monotonic()
                    ent_target = query_terms[0]
                    ent_res = await self._entity_search.search_by_entity(
                        entity_name=ent_target,
                        top_k=5,
                    )
                    if ent_res:
                        for er in ent_res:
                            fallback_items.append(
                                {
                                    "article_id": er.article_id,
                                    "issue_id": er.issue_id,
                                    "headline": er.headline,
                                    "newspaper_name": er.newspaper_name,
                                    "issue_date": er.issue_date,
                                    "pages": er.pages,
                                    "bboxes": er.bboxes,
                                    "snippet": f"[CRAG Entity Fallback - {er.entity_name}]: {er.summary}",
                                    "prominence_score": er.prominence_score,
                                    "source_tool": "crag_entity_fallback",
                                }
                            )
                        dur_ms = round((time.monotonic() - t_ent) * 1000)
                        tool_records.append(
                            ToolExecutionRecord(
                                tool_name="crag_entity_fallback",
                                tool_input={"entity_name": ent_target},
                                results_count=len(ent_res),
                                execution_time_ms=dur_ms,
                            )
                        )

            # 3. Fallback: Live Web Search if web search is enabled and archive yielded no items
            if not fallback_items and state.get("enable_web_search", False):
                t_web = time.monotonic()
                web_res = await self._web_search.search(
                    query=state.get("query", ""),
                    num_results=4,
                )
                if web_res:
                    for wr in web_res:
                        fallback_items.append(
                            {
                                "article_id": 0,
                                "headline": wr.title,
                                "newspaper_name": wr.source,
                                "issue_date": wr.published_date or "Live Web",
                                "pages": [1],
                                "snippet": f"[CRAG Web Fallback]: {wr.snippet}",
                                "url": wr.url,
                                "is_web": True,
                                "prominence_score": 0.8,
                                "source_tool": "crag_web_fallback",
                            }
                        )
                    dur_ms = round((time.monotonic() - t_web) * 1000)
                    tool_records.append(
                        ToolExecutionRecord(
                            tool_name="crag_web_fallback",
                            tool_input={"query": state.get("query", "")},
                            results_count=len(web_res),
                            execution_time_ms=dur_ms,
                        )
                    )

            return fallback_items, tool_records

        return filtered_evidence, tool_records


__all__ = [
    "EvaluationVerdict",
    "EvidenceEvaluator",
    "_has_quantitative_payload",
    "is_structural_or_relevant_evidence",
]
