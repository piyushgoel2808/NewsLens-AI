"""Comprehensive test suite for Reranker, 3-Tier Coverage Engine, and IR Evaluation Metrics."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.evaluation.metrics import (
    compute_citation_precision,
    compute_citation_recall,
    compute_coverage_f1,
    compute_faithfulness,
    compute_faithfulness_llm_judge,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from app.models.newspaper import Issue, Newspaper
from app.retrieval.coverage_analyzer import CoverageAnalyzer, CoverageStatus
from app.retrieval.hybrid_search import HybridSearchResult
from app.retrieval.reranker import CrossEncoderReranker, HeuristicReranker


class TestIRMetrics:
    """Test deterministic and statistical IR and generation evaluation metrics."""

    def test_recall_and_precision_at_k(self) -> None:
        retrieved = ["art_1", "art_2", "art_3", "art_4", "art_5"]
        ground_truth = ["art_2", "art_4", "art_6"]

        # At k=1: retrieved=["art_1"], hits=0
        assert compute_recall_at_k(retrieved, ground_truth, k=1) == 0.0
        assert compute_precision_at_k(retrieved, ground_truth, k=1) == 0.0

        # At k=2: retrieved=["art_1", "art_2"], hits=1 / 3
        assert compute_recall_at_k(retrieved, ground_truth, k=2) == pytest.approx(1 / 3)
        assert compute_precision_at_k(retrieved, ground_truth, k=2) == pytest.approx(1 / 2)

        # At k=5: retrieved=["art_1".."art_5"], hits=["art_2", "art_4"] -> 2 / 3
        assert compute_recall_at_k(retrieved, ground_truth, k=5) == pytest.approx(2 / 3)
        assert compute_precision_at_k(retrieved, ground_truth, k=5) == pytest.approx(2 / 5)

    def test_mrr_and_ndcg_at_k(self) -> None:
        retrieved = ["art_10", "art_20", "art_30"]
        ground_truth = ["art_20"]

        # First hit is at rank 2 -> MRR = 1/2 = 0.5
        assert compute_mrr(retrieved, ground_truth) == 0.5

        # NDCG@3 with rank 2 match
        ndcg = compute_ndcg_at_k(retrieved, ground_truth, k=3)
        assert ndcg > 0.0
        assert ndcg < 1.0

    def test_citation_precision_and_recall(self) -> None:
        cited = ["art_1", "art_2", "art_3"]
        ground_truth = ["art_1", "art_2"]

        assert compute_citation_precision(cited, ground_truth) == pytest.approx(2 / 3)
        assert compute_citation_recall(cited, ground_truth) == 1.0

    def test_faithfulness_lexical(self) -> None:
        context = [
            "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent on Thursday.",
            "Inflation projections remained steady around 4.5 percent.",
        ]
        grounded_answer = "The RBI kept the repo rate at 6.5 percent and maintained inflation projections at 4.5 percent."
        hallucinated_answer = "The Federal Reserve lowered interest rates by 50 basis points amidst economic uncertainty."

        assert compute_faithfulness(grounded_answer, context) >= 0.50
        assert compute_faithfulness(hallucinated_answer, context) <= 0.20

    @pytest.mark.asyncio
    async def test_faithfulness_llm_judge(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=MagicMock(text="0.95"))

        context = ["Sensex gained 400 points led by banking and IT stocks."]
        answer = "The stock market rallied with Sensex gaining 400 points."

        score = await compute_faithfulness_llm_judge(answer, context, llm_provider=mock_llm)
        assert score == 0.95

    def test_coverage_f1(self) -> None:
        pred_covered = {"Mint", "Business Standard"}
        gt_covered = {"Mint", "Economic Times"}
        pred_omitted = {"Economic Times"}
        gt_omitted = {"Business Standard"}

        f1 = compute_coverage_f1(pred_covered, gt_covered, pred_omitted, gt_omitted)
        assert 0.0 <= f1 <= 1.0


class TestCrossEncoderReranker:
    """Test two-stage neural reranking cascade."""

    @pytest.mark.asyncio
    async def test_heuristic_reranker_fallback(self) -> None:
        reranker = HeuristicReranker()
        query = "RBI repo rate monetary policy"
        candidates = [
            {"id": 1, "headline": "Sports: India wins cricket match", "snippet": "cricket series finale", "rrf_score": 0.05},
            {"id": 2, "headline": "RBI monetary policy: Repo rate unchanged", "snippet": "The central bank decided to hold the policy repo rate.", "rrf_score": 0.02},
        ]

        reranked = reranker.rerank(query, candidates, top_k=2)
        assert len(reranked) == 2
        # Candidate 2 has high overlap with query keywords -> should be ranked first
        assert reranked[0]["id"] == 2
        assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

    @pytest.mark.asyncio
    async def test_cross_encoder_mock_prediction(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.15, 0.92])
        reranker._model = mock_model

        candidates = [
            {"id": 1, "headline": "Unrelated topic", "snippet": "foo bar", "rrf_score": 0.04},
            {"id": 2, "headline": "Defense budget allocation", "snippet": "Rajnath Singh announces defense package", "rrf_score": 0.01},
        ]

        reranked = await reranker.rerank("defense budget allocation", candidates, top_k=2)
        assert reranked[0]["id"] == 2
        assert reranked[0]["rerank_score"] == 0.92


class TestCoverageAnalyzer:
    """Test 3-Tier Negative Coverage Engine and Invariant Enforcement."""

    @pytest.mark.asyncio
    async def test_tier1_incomplete_ingestion_returns_processing_error(self) -> None:
        mock_db = AsyncMock()
        mock_session_factory = MagicMock(return_value=mock_db)
        mock_db.__aenter__.return_value = mock_db

        # Mock issue with status != 'completed' (e.g. 'pending' or 'failed')
        mock_issue = Issue(
            id=10,
            newspaper_id=1,
            issue_date=datetime.date(2026, 8, 20),
            ingestion_status="pending",
        )
        mock_res = MagicMock()
        mock_res.scalars.return_value.all.return_value = [mock_issue]
        mock_db.execute = AsyncMock(return_value=mock_res)

        mock_search = MagicMock()
        mock_np = Newspaper(id=1, name="Mint")

        analyzer = CoverageAnalyzer(
            session_factory=mock_session_factory,
            hybrid_search_engine=mock_search,
        )

        rep = await analyzer.analyze_newspaper_coverage(
            newspaper=mock_np,
            query_or_event="Railway Modernization Package",
            target_date="2026-08-20",
        )

        # Invariant check: Incomplete ingestion must NEVER return NOT_FOUND
        assert rep.status == CoverageStatus.PROCESSING_ERROR
        assert "not fully ingested" in rep.audit_notes.lower()

    @pytest.mark.asyncio
    async def test_tier2_completed_issue_with_matches_returns_covered(self) -> None:
        mock_db = AsyncMock()
        mock_session_factory = MagicMock(return_value=mock_db)
        mock_db.__aenter__.return_value = mock_db

        mock_issue = Issue(
            id=10,
            newspaper_id=1,
            issue_date=datetime.date(2026, 8, 20),
            ingestion_status="completed",
        )
        mock_res = MagicMock()
        mock_res.scalars.return_value.all.return_value = [mock_issue]
        mock_db.execute = AsyncMock(return_value=mock_res)

        # Mock hybrid search returning strong match
        mock_search = MagicMock()
        mock_hit = HybridSearchResult(
            article_id=42,
            headline="Cabinet approves railway modernisation plan",
            subheadline=None,
            byline_author="Staff Reporter",
            section="Economy",
            article_type="news",
            prominence_score=85.0,
            rrf_score=0.035,
            vector_rank=1,
            keyword_rank=1,
            snippet="The Union cabinet on Wednesday approved...",
            newspaper_name="Mint",
            issue_date="2026-08-20",
            pages=[1, 2],
            rerank_score=0.88,
        )
        mock_search.search = AsyncMock(return_value=[mock_hit])
        mock_np = Newspaper(id=1, name="Mint")

        analyzer = CoverageAnalyzer(
            session_factory=mock_session_factory,
            hybrid_search_engine=mock_search,
        )

        rep = await analyzer.analyze_newspaper_coverage(
            newspaper=mock_np,
            query_or_event="Cabinet railway modernization",
            target_date="2026-08-20",
        )

        assert rep.status == CoverageStatus.COVERED
        assert rep.matched_article_ids == [42]
        assert rep.top_score == 0.88

    @pytest.mark.asyncio
    async def test_tier3_completed_issue_with_zero_matches_returns_not_found(self) -> None:
        mock_db = AsyncMock()
        mock_session_factory = MagicMock(return_value=mock_db)
        mock_db.__aenter__.return_value = mock_db

        mock_issue = Issue(
            id=10,
            newspaper_id=1,
            issue_date=datetime.date(2026, 8, 20),
            ingestion_status="completed",
        )
        mock_res = MagicMock()
        mock_res.scalars.return_value.all.return_value = [mock_issue]
        mock_db.execute = AsyncMock(return_value=mock_res)

        # Mock hybrid search returning 0 hits
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=[])
        mock_np = Newspaper(id=1, name="Business Standard")

        analyzer = CoverageAnalyzer(
            session_factory=mock_session_factory,
            hybrid_search_engine=mock_search,
        )

        rep = await analyzer.analyze_newspaper_coverage(
            newspaper=mock_np,
            query_or_event="Unrelated local event",
            target_date="2026-08-20",
        )

        # Issue is completed and 0 hits -> confirmed omission
        assert rep.status == CoverageStatus.NOT_FOUND
        assert "confirmed omission" in rep.audit_notes.lower()
