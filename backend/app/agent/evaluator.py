"""Corrective RAG (CRAG) Evaluator for NewsLens-AI Agentic Workflow.

Grades retrieved broadsheet evidence for semantic groundedness, preserves vector search hits
against over-aggressive lexical pruning, and triggers corrective retrieval fallbacks when needed.
"""

from __future__ import annotations

import contextlib
import re
import time
from typing import Any

from app.agent.state import AgentState, ToolExecutionRecord
from app.agent.tool_maker import ToolMaker
from app.core.logging import get_logger
from app.retrieval.entity_filter import EntitySearchEngine
from app.retrieval.sql_analytics import SQLAnalyticsEngine
from app.retrieval.web_search import WebSearchEngine

logger = get_logger(__name__)

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


def is_structural_or_relevant_evidence(item: dict[str, Any], archetype: str) -> bool:
    """Determine whether evidence is structural or has a high-confidence semantic floor.

    Protects relational manifests, matrices, and high-confidence dense vector search hits
    (prominence_score >= 0.65) from naive lexical stem pruning.
    """
    return (
        archetype in ("cross_newspaper_comparison", "quantitative_trend", "article_catalog", "analytical_computation")
        or item.get("source_tool") in ("sql_analytics", "coverage_analysis", "dynamic_analysis")
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
    ) -> None:
        self._entity_search = entity_search
        self._web_search = web_search
        self._tool_maker = tool_maker
        self._sql_analytics = sql_analytics

    async def evaluate_and_fallback(
        self,
        evidence: list[dict[str, Any]],
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], list[ToolExecutionRecord]]:
        """Filter evidence, verify groundedness, and execute corrective retrieval if needed."""
        archetype = state.get("archetype", "factual_lookup")
        tool_records: list[ToolExecutionRecord] = []

        # Skip fallback evaluation for meta queries or user clarification states
        if archetype in ("clarification_needed", "conversational_meta_query"):
            return evidence, tool_records

        raw_query_words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", state.get("query", "").lower())
        query_tokens = [w for w in raw_query_words if w not in _STOP_WORDS]

        # Retain evidence items that have positive relevance to query tokens OR are protected structural/semantic hits
        filtered_evidence = [
            item for item in evidence
            if is_structural_or_relevant_evidence(item, archetype) or _score_relevance(item, query_tokens) > 0.0
        ]
        filtered_evidence.sort(
            key=lambda it: (1.0 if is_structural_or_relevant_evidence(it, archetype) else _score_relevance(it, query_tokens)),
            reverse=True,
        )

        has_grounded_content = bool(
            filtered_evidence and any(len((item.get("snippet") or "").strip()) >= 20 for item in filtered_evidence)
        )

        if not has_grounded_content:
            logger.info("CRAG triggered: 0 high-confidence articles retrieved, attempting corrective fallback")
            fallback_items: list[dict[str, Any]] = list(filtered_evidence)

            # 1. Fallback: Dynamic Tool Synthesis (Database-aware custom Python code)
            # When static tools produced 0 valid evidence items, synthesize an on-demand SQL/Python tool to directly query the archive.
            if self._tool_maker and not fallback_items:
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
                )
                if dyn_res.success and dyn_res.evidence_items:
                    fallback_items.extend(dyn_res.evidence_items)
                    dur_ms = round((time.monotonic() - t_dyn) * 1000)
                    tool_records.append(
                        ToolExecutionRecord(
                            tool_name="crag_dynamic_tool_fallback",
                            tool_input={"query": state.get("query", "")},
                            results_count=len(dyn_res.evidence_items),
                            execution_time_ms=dur_ms,
                        )
                    )

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

            # 2. Fallback: Live Web Search if web search is enabled and archive yielded no items
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
    "EvidenceEvaluator",
    "is_structural_or_relevant_evidence",
]
