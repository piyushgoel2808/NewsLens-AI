"""Unit tests for WebSearchEngine and Dual-Mode Grounding."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.planner import QueryPlanner
from app.agent.synthesizer import AnswerSynthesizer
from app.retrieval.web_search import WebSearchEngine, WebSearchResult


class TestWebSearchEngine:
    """Unit tests for WebSearchEngine retrieval and fallbacks."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_list(self) -> None:
        engine = WebSearchEngine()
        results = await engine.search("   ")
        assert results == []

    @pytest.mark.asyncio
    async def test_serper_search_mock(self) -> None:
        engine = WebSearchEngine(serper_api_key="test-serper-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Telecom AGR Dues Supreme Court Ruling",
                    "link": "https://www.reuters.com/business/telecom-agr",
                    "snippet": "The Supreme Court dismissed the plea for re-computation...",
                    "date": "2026-08-20",
                }
            ]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            results = await engine.search("telecom agr dues ruling")

            assert len(results) == 1
            assert isinstance(results[0], WebSearchResult)
            assert results[0].title == "Telecom AGR Dues Supreme Court Ruling"
            assert results[0].url == "https://www.reuters.com/business/telecom-agr"
            assert "Supreme Court dismissed" in results[0].snippet
            assert results[0].source == "www.reuters.com"

    @pytest.mark.asyncio
    async def test_tavily_search_mock(self) -> None:
        engine = WebSearchEngine(tavily_api_key="test-tavily-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Tata Power expands clean energy portfolio",
                    "url": "https://economictimes.indiatimes.com/industry/energy",
                    "content": "Tata Power announced major solar and transmission investments...",
                    "published_date": "2026-08-15",
                }
            ]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            results = await engine.search("tata power clean energy")

            assert len(results) == 1
            assert results[0].title == "Tata Power expands clean energy portfolio"
            assert results[0].url == "https://economictimes.indiatimes.com/industry/energy"
            assert results[0].source == "economictimes.indiatimes.com"

    @pytest.mark.asyncio
    async def test_duckduckgo_fallback_search(self) -> None:
        engine = WebSearchEngine()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            '<div class="result__body">'
            '<a class="result__url" href="https://example.com/article"></a>'
            '<a class="result__snippet">Sample snippet for live web context.</a>'
            '</div>'
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            results = await engine.search("sample news topic")

            assert len(results) >= 1
            assert results[0].url == "https://example.com/article"
            assert results[0].snippet == "Sample snippet for live web context."


class TestDualModePlannerAndSynthesizer:
    """Unit tests for planner and synthesizer handling of live web grounding."""

    def test_planner_omits_web_search_when_disabled(self) -> None:
        planner = QueryPlanner()
        plan_res = planner.plan_query("Tata Power Odisha investment", enable_web_search=False)
        tool_names = [t.tool_name for t in plan_res.tool_calls]
        assert "web_search" not in tool_names

    def test_planner_includes_web_search_when_enabled(self) -> None:
        planner = QueryPlanner()
        plan_res = planner.plan_query("Tata Power Odisha investment", enable_web_search=True)
        tool_names = [t.tool_name for t in plan_res.tool_calls]
        assert "web_search" in tool_names

    def test_synthesizer_context_builder_separates_web_and_archive(self) -> None:
        synthesizer = AnswerSynthesizer()
        evidence_items = [
            {
                "newspaper_name": "Mint",
                "issue_date": "2026-08-01",
                "pages": [3],
                "headline": "Tata Power Nuclear Deal",
                "snippet": "Tata Power signed an MoU with Odisha government.",
                "source_tool": "hybrid_search",
                "is_web": False,
            },
            {
                "newspaper_name": "Reuters",
                "issue_date": "2026-08-23",
                "pages": [1],
                "headline": "Global Energy Outlook",
                "snippet": "Renewable investments rise across South Asia.",
                "url": "https://www.reuters.com/business/energy",
                "source_tool": "web_search",
                "is_web": True,
            },
        ]

        context = synthesizer._build_evidence_context(evidence_items)
        assert "--- ARCHIVE EVIDENCE EXCERPT [1] ---" in context
        assert "--- LIVE WEB EVIDENCE EXCERPT [2] ---" in context
        assert "URL: https://www.reuters.com/business/energy" in context

    def test_synthesizer_citation_extractor_differentiates_badges(self) -> None:
        synthesizer = AnswerSynthesizer()
        evidence_items = [
            {
                "newspaper_name": "Mint",
                "issue_date": "2026-08-01",
                "pages": [3],
                "headline": "Tata Power Deal",
                "article_id": 42,
                "issue_id": 7,
                "bboxes": [{"x0": 10, "y0": 20, "x1": 100, "y1": 200}],
                "is_web": False,
            },
            {
                "newspaper_name": "Reuters",
                "headline": "Live Web Story",
                "url": "https://reuters.com/story",
                "is_web": True,
                "source_tool": "web_search",
            },
        ]

        citations = synthesizer.extract_citations("text", evidence_items)
        assert len(citations) == 2

        # Newspaper citation
        np_cit = [c for c in citations if not c.get("is_web")][0]
        assert np_cit["newspaper_name"] == "Mint"
        assert np_cit["page_number"] == 3
        assert np_cit["issue_id"] == 7
        assert np_cit["source_type"] == "newspaper"
        assert np_cit["url"] is None

        # Web citation
        web_cit = [c for c in citations if c.get("is_web")][0]
        assert web_cit["url"] == "https://reuters.com/story"
        assert web_cit["source_type"] == "web"
        assert web_cit["is_web"] is True

    def test_synthesizer_citation_pruning_excludes_unreferenced_sources(self) -> None:
        synthesizer = AnswerSynthesizer()
        evidence_items = [
            {
                "newspaper_name": "Mint",
                "issue_date": "2026-08-01",
                "pages": [3],
                "headline": "Tata Power Nuclear Deal",
                "article_id": 101,
                "issue_id": 1,
                "bboxes": [],
                "is_web": False,
            },
            {
                "newspaper_name": "Daily Commercial",
                "issue_date": "2026-08-01",
                "pages": [8],
                "headline": "Classified Ad Mattress Sale",
                "article_id": 999,
                "issue_id": 1,
                "bboxes": [],
                "is_web": False,
            },
            {
                "newspaper_name": "Reuters",
                "headline": "Global Energy Transition",
                "url": "https://reuters.com/energy",
                "is_web": True,
                "source_tool": "web_search",
            },
        ]

        # Text only cites Mint and Reuters, ignores the classified ad
        synthesized_text = (
            "### Executive Summary\n"
            "Tata Power signed an agreement with the state government "
            "[Mint, 2026-08-01, Page 3, \"Tata Power Nuclear Deal\"].\n"
            "Further analysis on clean energy can be found at "
            "[Web: Global Energy Transition](https://reuters.com/energy)."
        )

        citations = synthesizer.extract_citations(synthesized_text, evidence_items)
        assert len(citations) == 2
        headlines = [c["headline"] for c in citations]
        assert "Tata Power Nuclear Deal" in headlines
        assert "Global Energy Transition" in headlines
        assert "Classified Ad Mattress Sale" not in headlines

    def test_planner_targeted_web_query_builder(self) -> None:
        from app.agent.planner import _build_targeted_web_query

        q1 = "Can you please tell me about the latest Telecom AGR Dues ruling?"
        assert _build_targeted_web_query(q1) == "latest Telecom AGR Dues ruling"

        q2 = "What happened with Tata Power in Odisha?"
        assert _build_targeted_web_query(q2) == "Tata Power in Odisha"

        q3 = "Find news about Supreme Court judgment"
        assert _build_targeted_web_query(q3) == "Supreme Court judgment"

    def test_model_registry_get_chat_provider_resolution(self) -> None:
        from app.providers.registry import get_registry

        registry = get_registry()
        # Default answerer
        provider = registry.get_chat_provider()
        assert provider is not None

        # Alias resolution
        gemini_provider = registry.get_chat_provider("gemini_flash")
        assert gemini_provider is not None
        p_name = gemini_provider.provider_name.lower()
        assert "gemini" in p_name or "ollama" in p_name

