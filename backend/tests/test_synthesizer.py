"""Unit tests for AnswerSynthesizer structured generation and conversational memory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.synthesizer import AnswerSynthesizer, parse_thought_and_answer
from app.providers.base import ModelResponse


class TestAnswerSynthesizer:
    """Test suite for structured answer synthesis and conversational context."""

    def test_parse_thought_and_answer_standard_tags(self) -> None:
        raw = (
            "<think>Analyzing Tata Power reports...</think>"
            "### ⚡ Executive Summary\nTata Power announced expansion."
        )
        thought, ans = parse_thought_and_answer(raw)
        assert thought == "Analyzing Tata Power reports..."
        assert "Executive Summary" in ans

    def test_parse_thought_and_answer_unclosed_tag_draft_recovery(self) -> None:
        raw = (
            "<think>We need to compare Mint and BS.\n"
            "### ⚡ Executive Summary\nTata Power will invest."
        )
        thought, ans = parse_thought_and_answer(raw)
        assert thought == "We need to compare Mint and BS."
        assert "Executive Summary" in ans

    def test_deterministic_summary_produces_structured_tiers(self) -> None:
        synth = AnswerSynthesizer()
        evidence = [
            {
                "headline": "Tata Power Clean Energy Bet",
                "newspaper_name": "Mint",
                "issue_date": "2026-08-01",
                "pages": [3],
                "snippet": "Tata Power is allocating $1.2B for nuclear and solar capacity.",
                "article_id": 1,
            },
            {
                "headline": "Odisha Power Grid Overhaul",
                "newspaper_name": "Business Standard",
                "issue_date": "2026-08-01",
                "pages": [5],
                "snippet": "State regulators greenlit green corridor expansion.",
                "article_id": 2,
            },
        ]
        summary = synth._generate_deterministic_summary("Tata Power expansion", evidence)
        assert "### ⚡ Executive Summary" in summary
        assert "### 📌 Key Verified Facts & Highlights" in summary
        assert "### 📰 Broadsheet Perspectives" in summary
        assert "### 🔍 Explore Further" in summary
        assert "Mint" in summary
        assert "Business Standard" in summary
        assert "> 💡 Explore:" in summary

    @pytest.mark.asyncio
    async def test_synthesize_with_conversational_history(self) -> None:
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(
            return_value=ModelResponse(
                text=(
                    "### ⚡ Executive Summary\n"
                    "The news was reported in Mint on August 1, 2026 (Page 3).\n\n"
                    "### 📌 Key Verified Facts & Highlights\n"
                    "- Reported by Mint [Mint, 2026-08-01, Page 3, \"Tata Power Bet\"]"
                ),
                input_tokens=100,
                output_tokens=50,
            )
        )
        mock_provider.provider_name = "mock"
        mock_provider.model_name = "test"

        synth = AnswerSynthesizer(provider=mock_provider)
        history = [
            {"role": "user", "content": "What did Tata Power announce?"},
            {
                "role": "assistant",
                "content": (
                    "### ⚡ Executive Summary\n"
                    "Tata Power announced a 2800 MW expansion "
                    "[Mint, 2026-08-01, Page 3, \"Tata Power Bet\"]."
                ),
            },
        ]

        ans, citations, _ = await synth.synthesize(
            query="Which newspaper was this from and what was the date?",
            archetype="conversational_meta_query",
            evidence_items=[],
            chat_history=history,
        )

        assert "Mint" in ans
        assert "2026-08-01" in ans
        call_messages = mock_provider.complete.call_args.kwargs["messages"]
        assert len(call_messages) >= 3
        assert any("2800 MW" in m.content for m in call_messages if isinstance(m.content, str))

    @pytest.mark.asyncio
    async def test_empty_evidence_hard_stop_prevents_hallucination(self) -> None:
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(
            return_value=ModelResponse(
                text="Prince Harry visited India yesterday.",
                input_tokens=100,
                output_tokens=50,
            )
        )
        synth = AnswerSynthesizer(provider=mock_provider)

        # Empty evidence with a standard factual query
        ans, citations, cost = await synth.synthesize(
            query="Tell me about Prince Harry",
            archetype="factual_lookup",
            evidence_items=[],
            chat_history=[],
        )

        assert "I could not find any evidence or articles matching this query in the database" in ans
        assert not mock_provider.complete.called
        assert len(citations) == 0
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_empty_evidence_streaming_hard_stop(self) -> None:
        mock_provider = MagicMock()
        synth = AnswerSynthesizer(provider=mock_provider)

        chunks = []
        async for chunk in synth.synthesize_stream(
            query="Tell me about Prince Harry",
            archetype="factual_lookup",
            evidence_items=[],
            chat_history=[],
        ):
            chunks.append(chunk)

        full_stream_text = "".join(chunks)
        assert "I could not find any evidence or articles matching this query in the database" in full_stream_text

    def test_evidence_context_budgeting_truncates(self) -> None:
        synth = AnswerSynthesizer()
        large_evidence = [
            {
                "headline": f"Story {i}",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "pages": [i],
                "snippet": "Very long story text " * 150,  # ~3000 chars each
                "article_id": i,
            }
            for i in range(25)  # 25 items
        ]
        context = synth._build_evidence_context(large_evidence)
        # Should be capped at 12 items
        assert "ARCHIVE EVIDENCE EXCERPT [12]" in context
        assert "ARCHIVE EVIDENCE EXCERPT [13]" not in context
        # Check snippet truncation
        assert "[excerpt truncated for length]" in context

    def test_parse_thought_and_answer_strips_memo_headers(self) -> None:
        raw_output = (
            "EXECUTIVE INTELLIGENCE BRIEFING\n"
            "Date: October 26, 2023 (Current Analysis) Subject: Governance Transparency\n\n"
            "### ⚡ Executive Summary\nThe Goa government is facing scrutiny."
        )
        _, cleaned = parse_thought_and_answer(raw_output)
        assert "October 26, 2023" not in cleaned
        assert cleaned.startswith("### ⚡ Executive Summary")

    def test_conversation_history_publication_isolation(self) -> None:
        synthesizer = AnswerSynthesizer()
        evidence = [
            {
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "headline": "Beware! AI-enabled traffic challans go live",
                "snippet": "AI traffic challans in Goa started today.",
                "pages": [1],
                "source_tool": "sql_analytics",
            },
            {
                "newspaper_name": "The Morning Standard",
                "issue_date": "2026-08-01",
                "headline": "Rapid rise of boxer Ankush",
                "snippet": "Boxer Ankush from Haryana defeated his opponent.",
                "pages": [12],
                "source_tool": "hybrid_search",
            },
        ]
        context = synthesizer._build_evidence_context(evidence)
        prompt = synthesizer._build_synthesizer_user_prompt(
            query="comapare all the available newspaper dated 1/8/2026",
            archetype="cross_newspaper_comparison",
            evidence_items=evidence,
            context=context,
        )

        assert "The Goan" in prompt
        assert "The Morning Standard" in prompt
        assert "STRICT PUBLICATION & DATE ISOLATION:" in prompt
        assert "You must ONLY report on and analyze the verified publications present in the current evidence (The Goan, The Morning Standard)" in prompt

    def test_extract_active_issue_guardrails_prevent_leakage(self) -> None:
        from app.agent.condenser import extract_active_issue_from_history

        # Past chat history discussed The New York Times on August 26
        history = [
            {
                "role": "user",
                "content": "Tell me about LIV Golf in THE NEW YORK TIMES dated 2026-08-26",
            },
            {
                "role": "assistant",
                "content": "The New York Times reported on LIV Golf...",
            },
        ]

        # Case 1: Cross-newspaper comparison on 2026-08-01 -> MUST NOT inherit The New York Times
        ctx_compare = extract_active_issue_from_history(
            history,
            current_query="comapare all the available newspaper dated 1/8/2026",
        )
        assert ctx_compare.get("newspaper_name") is None
        assert ctx_compare.get("issue_id") is None

        # Case 2: New date provided (2026-08-01) -> MUST NOT inherit from 2026-08-26
        ctx_date = extract_active_issue_from_history(
            history,
            current_query="What happened on 2026-08-01?",
        )
        assert ctx_date.get("newspaper_name") is None

        # Case 3: Follow-up question without new date or comparative intent -> CAN inherit context
        ctx_followup = extract_active_issue_from_history(
            history,
            current_query="What else did it say on page 3?",
        )
        assert ctx_followup.get("newspaper_name") == "The New York Times"
        assert ctx_followup.get("issue_date") == "2026-08-26"

    def test_evidence_context_includes_attached_photos(self) -> None:
        """Verify that attached photos with VLM descriptions are formatted into evidence context."""
        synth = AnswerSynthesizer(provider=MagicMock())
        evidence_items = [
            {
                "article_id": 41914,
                "headline": "Novak Djokovic's US Open campaign ends in pain",
                "newspaper_name": "The Goan",
                "issue_date": "2026-09-01",
                "pages": [14],
                "snippet": "Novak Djokovic grinded through physical and internal ailments on Sunday night.",
                "photos": [
                    {
                        "id": 8094,
                        "caption": "Djokovic in despair",
                        "visual_type": "photo",
                        "vlm_description": "A male tennis player, identifiable as Novak Djokovic, displays visible frustration with his head in hands on a tennis court.",
                    }
                ],
            }
        ]
        context = synth._build_evidence_context(evidence_items)
        assert "Attached Photos & Visual Elements:" in context
        assert "Djokovic in despair" in context
        assert "A male tennis player, identifiable as Novak Djokovic" in context
        assert "The Goan" in context

    def test_dynamic_prompt_domain_comparison(self) -> None:
        """Verify dynamic prompt builder adapts to domain-specific cross-newspaper comparison."""
        synth = AnswerSynthesizer(provider=MagicMock())
        query = "COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on economics or finance related news"
        prompt = synth._build_synthesizer_system_prompt(
            archetype="cross_newspaper_comparison",
            query=query,
            evidence_items=[],
        )
        assert "Executive Summary: Economics & Finance Intelligence" in prompt
        assert "Cross-Newspaper Economics & Finance Comparison Matrix" in prompt
        assert "Key Verified Sector Highlights & Policies" in prompt
        assert "STRICT DOMAIN PURITY MANDATE" in prompt

    def test_dynamic_prompt_macro_broadsheet_comparison(self) -> None:
        """Verify dynamic prompt builder adapts to whole-edition comparison without domain."""
        synth = AnswerSynthesizer(provider=MagicMock())
        query = "compare all the available newspaper dated 1/8/2026"
        prompt = synth._build_synthesizer_system_prompt(
            archetype="cross_newspaper_comparison",
            query=query,
            evidence_items=[],
        )
        assert "Executive Summary: Broadsheet Edition Overview" in prompt
        assert "Front-Page (Page 1) Lead Stories Comparison" in prompt
        assert "Section Distribution & Coverage Scale" in prompt
        assert "Exclusive Stories & Distinct Agendas" in prompt

    def test_dynamic_prompt_thematic_timeline(self) -> None:
        """Verify dynamic prompt builder adapts to chronological timeline inquiries."""
        synth = AnswerSynthesizer(provider=MagicMock())
        query = "Provide a timeline of the stock market crash in October"
        prompt = synth._build_synthesizer_system_prompt(
            archetype="thematic_timeline",
            query=query,
            evidence_items=[],
        )
        assert "Executive Summary: Chronological Progression" in prompt
        assert "Milestone Timeline & Event Progression" in prompt
        assert "Thematic Trajectory & Broadsheet Evolution" in prompt

    def test_deterministic_summary_cross_newspaper_domain(self) -> None:
        """Verify deterministic summary produces matrix and sector highlights for domain comparisons."""
        synth = AnswerSynthesizer()
        evidence = [
            {
                "headline": "Floating Solar Project Launched",
                "newspaper_name": "The Morning Standard",
                "issue_date": "2026-08-01",
                "pages": [1],
                "snippet": "Govt approves Rs 5,070 cr floating solar scheme.",
                "article_id": 1,
            },
            {
                "headline": "Seabed Mineral Discovery",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "pages": [3],
                "snippet": "Crucial rare earth minerals found in EEZ seabed.",
                "article_id": 2,
            },
        ]
        summary = synth._generate_deterministic_summary(
            "COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on economics or finance related news",
            evidence,
            archetype="cross_newspaper_comparison",
        )
        assert "Executive Summary: Economics & Finance Intelligence" in summary
        assert "Cross-Newspaper Economics & Finance Comparison Matrix" in summary
        assert "The Morning Standard" in summary
        assert "The Goan" in summary
        assert "Floating Solar Project Launched" in summary

    def test_dynamic_prompt_article_catalog(self) -> None:
        """Verify dynamic prompt builder adapts to article_catalog inquiries."""
        synth = AnswerSynthesizer(provider=MagicMock())
        query = "list all their health news"
        prompt = synth._build_synthesizer_system_prompt(
            archetype="article_catalog",
            query=query,
            evidence_items=[],
        )
        assert "Executive Summary: Health & Medicine Article Catalog" in prompt
        assert "Comprehensive Health & Medicine Articles Catalog" in prompt
        assert "| # | Publication | Issue Date | Page | Section | Headline | Author / Byline |" in prompt
        assert "ANTI-REPETITION CONSTRAINT" in prompt

    def test_domain_adaptive_comparison_column_headers(self) -> None:
        """Verify comparison table columns adapt to domain (finance metrics vs health medical focus)."""
        synth = AnswerSynthesizer(provider=MagicMock())

        # 1. Finance query
        prompt_fin = synth._build_synthesizer_system_prompt(
            archetype="cross_newspaper_comparison",
            query="compare coverage on stock markets and economy",
            evidence_items=[],
        )
        assert "Key Figures & Metrics" in prompt_fin

        # 2. Health query
        prompt_health = synth._build_synthesizer_system_prompt(
            archetype="cross_newspaper_comparison",
            query="compare coverage on hospital treatments and medicine",
            evidence_items=[],
        )
        assert "Key Findings & Medical Focus" in prompt_health

    def test_sanitize_headline_integration_in_synthesizer(self) -> None:
        """Verify doctor name as headline is sanitized into a clean topic in citations and context."""
        synth = AnswerSynthesizer()
        evidence = [
            {
                "article_id": 101,
                "headline": "Dr. Smriti Naswa Singh",
                "subheadline": "Skin Cancer Prevention Tips",
                "byline_author": "Dr. Smriti Naswa Singh",
                "newspaper_name": "The Daily Health",
                "issue_date": "2026-08-01",
                "pages": [4],
                "snippet": "Protecting your skin from UV rays is critical in summer.",
            }
        ]

        # 1. Check citations
        citations = synth.extract_citations("Skin Cancer Prevention Tips page 4", evidence)
        assert len(citations) == 1
        assert citations[0]["headline"] == "Skin Cancer Prevention Tips"

        # 2. Check evidence context
        context = synth._build_evidence_context(evidence)
        assert "Skin Cancer Prevention Tips" in context





