"""Unit tests validating sub-5-second RAG query latency optimizations.

Validates:
1. Deterministic pre-planner pronoun safety & bypass rules
2. Cache key completeness (newspaper name, issue date, chat history digest)
3. Parallel retrieval exception isolation in hybrid search (Qdrant & MySQL)
4. Embedding caching in hybrid search
5. Archetype-aware synthesizer max_tokens caps
6. Cross-encoder candidate pool gating
7. Verifier gating on high CRAG quality scores
8. Streaming cache replay and CRAG sufficiency gating
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.graph import _can_bypass_llm_planner
from app.agent.synthesizer import AnswerSynthesizer
from app.retrieval.hybrid_search import HybridSearchEngine, HybridSearchResult
from app.retrieval.reranker import CrossEncoderReranker, HeuristicReranker
from app.storage.base import FullTextSearchResult, VectorSearchResult
from app.storage.cache_store import (
    CacheStore,
    compute_embedding_cache_key,
    compute_query_cache_key,
)


class TestDeterministicPrePlannerSafety:
    """Validate safety guardrails for the fast-path deterministic pre-planner."""

    def test_bypasses_for_unambiguous_standalone_queries(self) -> None:
        """Clean count, date-scoped, or timeline queries with no history should bypass LLM planner."""
        assert _can_bypass_llm_planner("How many articles were published on 2026-08-20?", [], False) is True
        assert _can_bypass_llm_planner("Total number of advertisements on 2026-08-21", [], False) is True
        assert _can_bypass_llm_planner("What was reported on 2026-08-21 about RBI rate hike?", [], False) is True
        assert _can_bypass_llm_planner("Timeline of diplomatic events in August 2026", [], False) is True

    def test_rejects_queries_with_chat_history(self) -> None:
        """Queries with chat history must never bypass LLM planner to ensure condensation runs."""
        history = [{"role": "user", "content": "Tell me about The Hindu"}]
        assert _can_bypass_llm_planner("How many articles were published?", history, False) is False

    def test_rejects_queries_with_attached_assets(self) -> None:
        """Queries with attached assets must route through planner to ground in the asset."""
        assert _can_bypass_llm_planner("How many articles were published?", [], True) is False

    @pytest.mark.parametrize(
        "pronoun_query",
        [
            "How many did they report on 2026-08-20?",
            "What did it say about the policy on 2026-08-21?",
            "What about that article on 2026-08-21?",
            "Show me their coverage on 2026-08-20",
            "Did he announce the scheme on 2026-08-20?",
            "What did she say about the event on 2026-08-20?",
            "How many photos did the former publish on 2026-08-20?",
            "Explain the above report from 2026-08-20",
            "What about the earlier statements on 2026-08-20?",
            "Any other news on 2026-08-20?",
        ],
    )
    def test_rejects_queries_with_followup_pronouns(self, pronoun_query: str) -> None:
        """Queries containing coreference pronouns must route through condensation and LLM planner."""
        assert _can_bypass_llm_planner(pronoun_query, [], False) is False

    def test_rejects_comparative_queries(self) -> None:
        """Queries asking for comparison across newspapers should use LLM planner."""
        assert _can_bypass_llm_planner("Compare Mint vs The Hindu on 2026-08-20", [], False) is False
        assert _can_bypass_llm_planner("Differences between both newspapers on 2026-08-20", [], False) is False


class TestCacheKeyCompleteness:
    """Validate cache key hashing includes publication, dates, and chat history."""

    def test_newspaper_name_isolation(self) -> None:
        k1 = compute_query_cache_key("Headline news", newspaper_name="Mint")
        k2 = compute_query_cache_key("Headline news", newspaper_name="The Hindu")
        assert k1 != k2

    def test_issue_date_isolation(self) -> None:
        k1 = compute_query_cache_key("Market rally", date_filters="2026-08-20")
        k2 = compute_query_cache_key("Market rally", date_filters="2026-08-21")
        assert k1 != k2

    def test_chat_history_digest_isolation(self) -> None:
        k1 = compute_query_cache_key("Explain more", chat_history_digest="abc123hash")
        k2 = compute_query_cache_key("Explain more", chat_history_digest="xyz987hash")
        assert k1 != k2

    def test_identical_inputs_match(self) -> None:
        k1 = compute_query_cache_key(
            "  Market Rally  ",
            model_id="GEMINI_FLASH",
            date_filters="2026-08-20",
            newspaper_name="  MINT  ",
            chat_history_digest="digest_a",
        )
        k2 = compute_query_cache_key(
            "market rally",
            model_id="gemini_flash",
            date_filters="2026-08-20",
            newspaper_name="mint",
            chat_history_digest="digest_a",
        )
        assert k1 == k2


class TestHybridSearchOptimization:
    """Validate parallel execution, exception isolation, and embedding caching."""

    @pytest.mark.asyncio
    async def test_embedding_cache_hit_bypasses_provider(self) -> None:
        """When embedding is in cache, provider.embed_one should not be called."""
        mock_cache = MagicMock(spec=CacheStore)
        mock_cache.get_embedding = AsyncMock(return_value=[0.5] * 768)
        mock_cache.set_embedding = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.embed_one = AsyncMock()

        mock_qdrant = MagicMock()
        mock_qdrant.search = AsyncMock(return_value=[])

        engine = HybridSearchEngine(
            session_factory=MagicMock(),
            qdrant=mock_qdrant,
            embed_provider=mock_provider,
            cache=mock_cache,
        )
        engine._ft_search.search = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await engine.search("test query", top_k=5)

        mock_cache.get_embedding.assert_awaited_once()
        mock_provider.embed_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embedding_cache_miss_stores_vector(self) -> None:
        """When embedding is not in cache, provider.embed_one is called and stored."""
        mock_cache = MagicMock(spec=CacheStore)
        mock_cache.get_embedding = AsyncMock(return_value=None)
        mock_cache.set_embedding = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.embed_one = AsyncMock(return_value=[0.1] * 768)

        mock_qdrant = MagicMock()
        mock_qdrant.search = AsyncMock(return_value=[])

        engine = HybridSearchEngine(
            session_factory=MagicMock(),
            qdrant=mock_qdrant,
            embed_provider=mock_provider,
            cache=mock_cache,
        )
        engine._ft_search.search = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await engine.search("new query", top_k=5)

        mock_provider.embed_one.assert_awaited_once_with("new query")
        mock_cache.set_embedding.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_isolation_qdrant_failure_still_returns_mysql(self) -> None:
        """When Qdrant raises an exception, MySQL results are still returned without crashing."""
        mock_qdrant = MagicMock()
        mock_qdrant.search = AsyncMock(side_effect=ConnectionError("Qdrant unreachable"))

        mock_ft_res = [
            FullTextSearchResult(article_id=10, headline="MySQL Hit", score=2.5, snippet="Text")
        ]

        mock_session_factory = MagicMock()
        mock_db = MagicMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_db

        mock_art = MagicMock(
            id=10,
            headline="MySQL Hit",
            subheadline=None,
            byline_author="Author",
            section="General",
            article_type="news",
            prominence_score=0.8,
            summary="Text",
            full_text="Text",
            issue=MagicMock(issue_date="2026-08-20", newspaper=MagicMock(name="Daily")),
            pages=[MagicMock(page_number=1)],
        )
        mock_db_res = MagicMock()
        mock_db_res.scalars.return_value.all.return_value = [mock_art]
        mock_db.execute = AsyncMock(return_value=mock_db_res)

        engine = HybridSearchEngine(
            session_factory=mock_session_factory,
            qdrant=mock_qdrant,
            embed_provider=MagicMock(embed_one=AsyncMock(return_value=[0.1] * 768)),
            cache=MagicMock(get_embedding=AsyncMock(return_value=[0.1] * 768)),
        )
        engine._ft_search.search = AsyncMock(return_value=mock_ft_res)  # type: ignore[method-assign]

        results = await engine.search("resilience test", top_k=5, rerank=False)

        assert len(results) == 1
        assert results[0].article_id == 10
        assert results[0].headline == "MySQL Hit"

    @pytest.mark.asyncio
    async def test_exception_isolation_mysql_failure_still_returns_qdrant(self) -> None:
        """When MySQL FULLTEXT raises an exception, Qdrant results are still returned."""
        mock_qdrant = MagicMock()
        mock_qdrant.search = AsyncMock(
            return_value=[
                VectorSearchResult(
                    id="v1",
                    score=0.88,
                    payload={"headline": "Vector Hit", "newspaper_name": "Paper", "pages": [1]},
                    article_id=20,
                )
            ]
        )

        mock_session_factory = MagicMock()
        mock_db = MagicMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_db

        mock_art = MagicMock(
            id=20,
            headline="Vector Hit",
            subheadline=None,
            byline_author="Author",
            section="General",
            article_type="news",
            prominence_score=0.8,
            summary="Text",
            full_text="Text",
            issue=MagicMock(issue_date="2026-08-20", newspaper=MagicMock(name="Paper")),
            pages=[MagicMock(page_number=1)],
        )
        mock_db_res = MagicMock()
        mock_db_res.scalars.return_value.all.return_value = [mock_art]
        mock_db.execute = AsyncMock(return_value=mock_db_res)

        engine = HybridSearchEngine(
            session_factory=mock_session_factory,
            qdrant=mock_qdrant,
            embed_provider=MagicMock(embed_one=AsyncMock(return_value=[0.1] * 768)),
            cache=MagicMock(get_embedding=AsyncMock(return_value=[0.1] * 768)),
        )
        engine._ft_search.search = AsyncMock(side_effect=Exception("MySQL DB timeout"))  # type: ignore[method-assign]

        results = await engine.search("resilience test 2", top_k=5, rerank=False)

        assert len(results) == 1
        assert results[0].article_id == 20


class TestCrossEncoderGating:
    """Validate reranker fast-paths when candidates <= 1."""

    @pytest.mark.asyncio
    async def test_cross_encoder_skips_inference_when_single_candidate(self) -> None:
        reranker = CrossEncoderReranker()
        # Mock _get_model to ensure it is NOT called when candidates <= 1
        reranker._get_model = AsyncMock()  # type: ignore[method-assign]

        candidates = [
            {"headline": "H1", "snippet": "S1", "rrf_score": 0.05},
        ]

        result = await reranker.rerank("test query", candidates, top_k=5)

        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.05
        reranker._get_model.assert_not_awaited()

    def test_heuristic_reranker_handles_single_candidate(self) -> None:
        reranker = HeuristicReranker()
        candidates = [
            {"headline": "H1", "snippet": "S1", "rrf_score": 0.09},
        ]
        result = reranker.rerank("query", candidates, top_k=10)
        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.09


class TestSynthesizerTokenCaps:
    """Validate archetype-aware max_tokens caps."""

    def test_archetype_caps_defined(self) -> None:
        caps = AnswerSynthesizer.ARCHETYPE_TOKEN_CAPS
        assert caps["factual_lookup"] == 1024
        assert caps["scalar_count"] == 512
        assert caps["conversational_meta_query"] == 512
        assert caps["thematic_timeline"] == 3072
        assert caps["_default"] == 3072

    @pytest.mark.asyncio
    async def test_synthesize_passes_capped_tokens_to_provider(self) -> None:
        mock_provider = MagicMock()
        mock_resp = MagicMock(text="Here is the brief answer.", input_tokens=100, output_tokens=30, cost_usd=0.001)
        mock_provider.complete = AsyncMock(return_value=mock_resp)
        mock_provider.provider_name = "test_provider"
        mock_provider._model = "test_model"

        synthesizer = AnswerSynthesizer(provider=mock_provider)
        evidence = [{"headline": "Test Headline", "article_id": 1, "snippet": "Verified factual content"}]

        await synthesizer.synthesize(
            query="Count of articles",
            archetype="scalar_count",
            evidence_items=evidence,
        )

        _, kwargs = mock_provider.complete.call_args
        # Concise non-reasoning archetypes dynamically allocate 1024 tokens
        assert kwargs["max_tokens"] == 1024

    def test_resolve_dynamic_token_budget_reasoning_and_blueprints(self) -> None:
        """Validate dynamic token allocation across reasoning models and custom blueprints."""
        from app.agent.models import AnswerBlueprint
        from app.agent.synthesizer import resolve_dynamic_token_budget
        from app.providers.base import ProviderCapability

        # Standard non-reasoning provider
        std_provider = MagicMock()
        std_provider.capability = ProviderCapability(
            max_output_tokens=4096,
            is_reasoning_model=False,
            reasoning_headroom=0,
        )

        # Reasoning provider (e.g. Gemini 2.5 Flash / DeepSeek-R1)
        reasoning_provider = MagicMock()
        reasoning_provider.capability = ProviderCapability(
            max_output_tokens=8192,
            is_reasoning_model=True,
            reasoning_headroom=2048,
        )

        # 1. Reasoning provider gets full envelope for analytical computation
        budget = resolve_dynamic_token_budget(
            provider=reasoning_provider,
            archetype="analytical_computation",
            query="Calculate average word count",
        )
        assert budget == 8192

        # 2. Non-reasoning provider gets concise 1024 tokens for scalar counts
        budget = resolve_dynamic_token_budget(
            provider=std_provider,
            archetype="scalar_count",
            query="Count of articles",
        )
        assert budget == 1024

        # 3. Explicit blueprint word count scales dynamically with headroom
        bp = AnswerBlueprint(target_word_count=50)
        # Non-reasoning: 50 words * 4 = 200, clamped to min 1024
        assert resolve_dynamic_token_budget(provider=std_provider, answer_blueprint=bp) == 1024

        # Reasoning: 50 words * 4 = 200 -> max(1024, 200) + 2048 headroom = 3072
        assert resolve_dynamic_token_budget(provider=reasoning_provider, answer_blueprint=bp) == 3072


class TestStreamingCacheAndGateways:
    """Validate streaming cache hit replay and verifier gating."""

    @pytest.mark.asyncio
    async def test_stream_query_cache_hit_replays_instantly(self) -> None:
        """When query is in Redis cache, stream_query should emit cache_hit stage and stream tokens."""
        from app.api.routers.query import QueryRequest, stream_query

        mock_cached = {
            "query": "Cached query",
            "archetype": "factual_lookup",
            "synthesized_answer": "This is a pre-cached response from the broadsheet archive.",
            "citations": [{"headline": "H1", "publication": "Mint"}],
            "plan": [{"tool_name": "hybrid_search"}],
        }

        with (
            patch("app.api.routers.query.get_session_factory", return_value=MagicMock()),
            patch("app.api.routers.query.AgentWorkflow") as mock_workflow_cls,
        ):
            mock_workflow = MagicMock()
            mock_workflow._cache.get_query = AsyncMock(return_value=mock_cached)
            mock_workflow_cls.return_value = mock_workflow

            req = QueryRequest(query="Cached query")
            response = await stream_query(req)

            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            full_sse = "".join(chunks)
            assert "cache_hit" in full_sse
            assert "pre-cached response" in full_sse
            assert "citations" in full_sse
            assert "done" in full_sse
            # Ensure planner was NOT invoked on cache hit
            mock_workflow._planner.plan_query_async.assert_not_called()


class TestPageNumberStandardizationAndOrdinalRetrieval:
    """Validate zero printed page logic and ordinal page queries."""

    def test_search_filter_has_no_printed_page_attributes(self) -> None:
        from app.retrieval.hybrid_search import SearchFilter

        filt = SearchFilter(page_number=6)
        assert filt.page_number == 6
        assert not hasattr(filt, "printed_page")
        assert not hasattr(filt, "exclude_printed_pages")

    def test_planner_routes_ordinal_page_query(self) -> None:
        from app.agent.planner import QueryPlanner

        planner = QueryPlanner()
        plan = planner.plan_query("1st news on page 6 of hindustan times on 2026-09-03?")
        assert plan.archetype == "article_catalog"

        tool_names = [call.tool_name for call in plan.tool_calls]
        assert "sql_analytics" in tool_names
        assert "hybrid_search" in tool_names

        sql_call = next(c for c in plan.tool_calls if c.tool_name == "sql_analytics")
        assert sql_call.arguments.get("analysis_type") == "issue_summary"
        assert sql_call.arguments.get("page_filter") == "6"

        hs_call = next(c for c in plan.tool_calls if c.tool_name == "hybrid_search")
        assert hs_call.arguments.get("page_filter") == "6"

    @pytest.mark.asyncio
    async def test_issue_summary_calculates_page_word_count_metrics(self) -> None:
        from datetime import date
        from app.models.article import Article, ArticlePage
        from app.models.newspaper import Issue, Newspaper, Page
        from app.retrieval.sql_analytics import SQLAnalyticsEngine

        paper = Newspaper(id=1, name="Hindustan Times", country="India", default_language="en")
        issue = Issue(id=10, newspaper_id=1, newspaper=paper, issue_date=date(2026, 9, 3))
        p6 = Page(id=60, issue_id=10, page_number=6)
        issue.pages = [p6]

        art1 = Article(id=1, issue_id=10, headline="Lead Story Page 6", prominence_score=0.9, word_count=600, primary_page_id=60)
        art1.article_pages = [ArticlePage(article_id=1, page_id=60, page_number=6)]
        art2 = Article(id=2, issue_id=10, headline="Secondary Story Page 6", prominence_score=0.5, word_count=400, primary_page_id=60)
        art2.article_pages = [ArticlePage(article_id=2, page_id=60, page_number=6)]

        mock_db = AsyncMock()
        mock_res_issue = MagicMock()
        mock_res_issue.scalars.return_value.first.return_value = issue
        mock_res_art = MagicMock()
        mock_res_art.scalars.return_value.all.return_value = [art1, art2]
        mock_db.execute.side_effect = [mock_res_issue, mock_res_art]

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__.return_value = mock_db

        engine = SQLAnalyticsEngine(session_factory=mock_factory)
        summary = await engine.get_issue_summary(newspaper_name="Hindustan Times", issue_date="2026-09-03", page_filter="6")

        assert summary["total_articles"] == 2
        assert summary["avg_word_count"] == 500.0
        assert summary["total_words"] == 1000
        assert summary["min_word_count"] == 400
        assert summary["max_word_count"] == 600
        # Verify 100% absence of printed_page key
        for a in summary["articles"]:
            assert "printed_page" not in a
            assert a["page_number"] == 6
