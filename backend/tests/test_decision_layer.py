"""Unit tests for the explicit ToolDecisionPolicy layer and RAG retrieval fixes."""

from __future__ import annotations

import pytest
from qdrant_client.http import models as qmodels

from app.agent.decision_layer import ToolDecisionPolicy
from app.core.config import QdrantSettings
from app.retrieval.reranker import CrossEncoderReranker
from app.storage.mysql_fulltext import _clean_fulltext_query
from app.storage.qdrant_store import QdrantStore


class TestToolDecisionPolicy:
    """Validate explicit decision-making layer across all 8 canonical tools."""

    def test_decide_statistical_computation(self) -> None:
        dec = ToolDecisionPolicy.decide(
            "What is the average word count of sports articles in Hindustan Times on 2026-09-03?"
        )
        assert dec.intent == "statistical_computation"
        assert dec.archetype == "analytical_computation"
        assert dec.selected_tools == ["dynamic_analysis"]
        assert dec.is_deterministic is True

    def test_decide_summary_with_word_limit_does_not_trigger_dynamic_analysis(self) -> None:
        dec = ToolDecisionPolicy.decide(
            'Summarize the article "Adani leads rush for 37,500 cr coal gasification plan" in 100 words'
        )
        assert dec.archetype == "factual_lookup"
        assert "dynamic_analysis" not in dec.selected_tools
        assert "hybrid_search" in dec.selected_tools

    def test_decide_visual_asset_inspection(self) -> None:
        dec = ToolDecisionPolicy.decide(
            "What are the exact figures shown in this infographic?",
            attached_photo_id=1042,
        )
        assert dec.intent == "visual_asset_inspection"
        assert dec.selected_tools == ["inspect_visual_asset"]
        assert dec.is_deterministic is True

    def test_decide_differential_and_shared_coverage(self) -> None:
        diff_dec = ToolDecisionPolicy.decide(
            "List the news that are in The Goan dated 2026-08-01 but not in The Morning Standard"
        )
        assert diff_dec.intent == "coverage_difference"
        assert diff_dec.archetype == "cross_newspaper_comparison"
        assert diff_dec.recommended_analysis_type == "coverage_difference"

        shared_dec = ToolDecisionPolicy.decide(
            "Give me the similar articles from The Indian Express and The Hindu on 2026-05-20"
        )
        assert shared_dec.intent == "shared_coverage"
        assert shared_dec.archetype == "cross_newspaper_comparison"
        assert shared_dec.recommended_analysis_type == "shared_coverage"

    def test_decide_archive_availability_and_counts(self) -> None:
        avail_dec = ToolDecisionPolicy.decide("Is any newspaper available for dated 2026-04-15?")
        assert avail_dec.intent == "archive_availability"
        assert avail_dec.archetype == "quantitative_trend"
        assert avail_dec.recommended_analysis_type == "count_issues"

        ad_dec = ToolDecisionPolicy.decide("How many advertisements are in Mint on 2026-08-01?")
        assert ad_dec.intent == "quantitative_count"
        assert ad_dec.recommended_analysis_type == "count_advertisements"

    def test_decide_page_catalog(self) -> None:
        cat_dec = ToolDecisionPolicy.decide("List all news on page 5 of Hindustan Times on 2026-09-03")
        assert cat_dec.intent == "page_article_catalog"
        assert cat_dec.archetype == "article_catalog"
        assert cat_dec.recommended_analysis_type == "issue_summary"
        assert "sql_analytics" in cat_dec.selected_tools


class TestRetrievalFixes:
    """Validate Qdrant KEYWORD date range expansion, FULLTEXT query cleaning, and Reranker calibration."""

    def test_qdrant_keyword_date_range_uses_match_any(self) -> None:
        store = QdrantStore(QdrantSettings())
        flt = store._build_filter({"issue_date": {"gte": "2026-08-01", "lte": "2026-08-03"}})
        assert flt is not None
        assert flt.must is not None
        assert len(flt.must) == 1
        cond = flt.must[0]
        assert isinstance(cond, qmodels.FieldCondition)
        assert isinstance(cond.match, qmodels.MatchAny)
        assert cond.match.any == ["2026-08-01", "2026-08-02", "2026-08-03"]

    def test_clean_fulltext_query_preserves_quoted_headline_and_strips_meta(self) -> None:
        q1 = 'Explain article "Adani leads rush for 37,500 cr coal gasification plan" in Hindustan Times dated 2026-09-09'
        assert _clean_fulltext_query(q1) == "Adani leads rush for 37,500 cr coal gasification plan"

        q2 = "What happened on page 5 dated 2026-08-01 regarding semiconductor subsidies?"
        cleaned = _clean_fulltext_query(q2)
        assert "page 5" not in cleaned.lower()
        assert "2026-08-01" not in cleaned
        assert "semiconductor subsidies" in cleaned

    def test_reranker_logit_calibration_and_default_model(self) -> None:
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # Probability scores in [0, 1] are preserved verbatim
        assert CrossEncoderReranker._calibrate_score(0.92) == pytest.approx(0.92)
        # Raw positive/negative logits outside [0, 1] are mapped cleanly into (0, 1) via sigmoid
        assert 0.85 < CrossEncoderReranker._calibrate_score(2.5) < 1.0
        assert 0.0 < CrossEncoderReranker._calibrate_score(-2.5) < 0.15
