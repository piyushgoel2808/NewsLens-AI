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

    def test_sanitize_headline_with_chunk_markers_in_snippet(self) -> None:
        """Verify sanitize_headline strips chunk match and visual markers before using snippet."""
        from app.retrieval.sql_analytics import sanitize_headline

        hl, byline = sanitize_headline(
            headline="Dr. Smriti Naswa Singh",
            subheadline="",
            byline_author="Dr. Smriti Naswa Singh",
            snippet="[Exact Chunk Match]:\n[Visual Data Asset: Infographic]\nProtecting your skin from UV rays is critical in summer.",
        )
        assert hl != "[Exact Chunk Match]:"
        assert "[Visual Data Asset" not in hl
        assert "Protecting your skin from UV rays is critical in summer" in hl
        assert "Dr. Smriti Naswa Singh" in (byline or "")

    @pytest.mark.asyncio
    async def test_deterministic_summary_preserves_cross_newspaper_archetype(self) -> None:
        """Verify fallback deterministic summary preserves cross_newspaper_comparison matrix format."""
        from unittest.mock import AsyncMock, MagicMock
        mock_p = MagicMock()
        mock_p.provider_name = "mock"
        mock_p._model = "mock_model"
        mock_p.complete = AsyncMock(side_effect=RuntimeError("Provider offline"))
        synth = AnswerSynthesizer(provider=mock_p)
        evidence = [
            {
                "article_id": 1,
                "headline": "New Hospital Wing Opens in Panaji",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "pages": [3],
                "snippet": "Health Minister inaugurated the modern 200-bed facility.",
            },
            {
                "article_id": 2,
                "headline": "Pharma Sector Subsidies Announced",
                "newspaper_name": "The Morning Standard",
                "issue_date": "2026-08-01",
                "pages": [5],
                "snippet": "New government incentives aim to boost domestic drug production.",
            },
        ]
        text, citations, _ = await synth.synthesize(
            query="COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on health related news",
            archetype="cross_newspaper_comparison",
            evidence_items=evidence,
        )
        # Must produce comparison matrix, NOT "Key broadsheet reporting regarding..." single-newspaper template
        assert "Cross-Newspaper Health & Medicine Comparison Matrix" in text or "Comparison Matrix" in text
        assert "| **The Goan** |" in text
        assert "| **The Morning Standard** |" in text
        assert "Key broadsheet reporting regarding" not in text

    def test_domain_terms_health_matches_individual_medical_tokens(self) -> None:
        """Verify individual terms like 'hospital', 'doctor', 'pharma' score high for Health domain."""
        synth = AnswerSynthesizer()
        evidence = [
            {
                "article_id": 1,
                "headline": "New Orthopedic Clinic Launched",
                "snippet": "Local doctors and surgeons established a new patient care facility.",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
            },
            {
                "article_id": 2,
                "headline": "City Pothole Repair Work Underway",
                "snippet": "Road engineers filled potholes on the highway.",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
            },
        ]
        context = synth._build_evidence_context(
            evidence,
            query="COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on health related news",
        )
        assert "Orthopedic Clinic" in context

    def test_deterministic_summary_excludes_issue_manifest_and_cleans_chunks(self) -> None:
        """Verify _generate_deterministic_summary suppresses manifests from facts/perspectives and cleans chunk tags."""
        synth = AnswerSynthesizer()
        evidence = [
            {
                "article_id": 1,
                "headline": "Issue Manifest: The Goan (2026-08-01)",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "pages": [1],
                "snippet": "=== RELATIONAL ARCHIVE MANIFEST FOR The Goan (2026-08-01) ===",
            },
            {
                "article_id": 2,
                "headline": "Issue Manifest: The Morning Standard (2026-08-01)",
                "newspaper_name": "The Morning Standard",
                "issue_date": "2026-08-01",
                "pages": [1],
                "snippet": "=== RELATIONAL ARCHIVE MANIFEST FOR The Morning Standard (2026-08-01) ===",
            },
            {
                "article_id": 3,
                "headline": "Cholangitis (Bile duct infection)",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "pages": [11],
                "snippet": "[Exact Chunk Match]: [Visual Data Asset: Infographic] The infographic illustrates liver and gallbladder anatomy.",
            },
            {
                "article_id": 4,
                "headline": "Pharma Subsidies Approved",
                "newspaper_name": "The Morning Standard",
                "issue_date": "2026-08-01",
                "pages": [4],
                "snippet": "[Exact Chunk Match]: [📷 Attached Image/Photo: Subsidies] Ministry announced new medicine incentives.",
            },
        ]
        text = synth._generate_deterministic_summary(
            query="COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on health related news",
            evidence_items=evidence,
            archetype="cross_newspaper_comparison",
        )

        assert "Comparison Matrix" in text
        # Manifests should never appear as headlines or facts in user-facing brief
        assert "Issue Manifest:" not in text
        assert "[Exact Chunk Match]" not in text
        assert "[Visual Data Asset" not in text
        assert "[📷" not in text
        # Real articles must be highlighted
        assert "Cholangitis" in text
        assert "Pharma Subsidies Approved" in text
        assert "Emphasized 'Cholangitis (Bile duct infection)'" in text
        assert "Emphasized 'Pharma Subsidies Approved'" in text

    def test_deterministic_summary_zero_coverage_publication_handling(self) -> None:
        """Verify publication with 0 domain articles is marked as having no standalone reporting, not filled with noise."""
        synth = AnswerSynthesizer()
        evidence = [
            {
                "article_id": 1,
                "headline": "Lifestyle changes for healthy heart",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "pages": [3],
                "snippet": "Doctors emphasize diet and exercise for cardiovascular wellness.",
            },
            {
                "article_id": 2,
                "headline": "WHEN: August 8, 9 pm WHERE: Studio XO, Noida",
                "newspaper_name": "The Morning Standard",
                "issue_date": "2026-08-01",
                "pages": [2],
                "snippet": "Live concert tickets are on sale.",
            },
            {
                "article_id": 3,
                "headline": "Cases still pending in designated courts",
                "newspaper_name": "The Morning Standard",
                "issue_date": "2026-08-01",
                "pages": [5],
                "snippet": "Judicial bench reviewed trial delays.",
            },
        ]
        text = synth._generate_deterministic_summary(
            query="COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on health related news",
            evidence_items=evidence,
            archetype="cross_newspaper_comparison",
        )

        assert "The Goan" in text
        assert "Lifestyle changes for healthy heart" in text
        # Studio XO and Cases pending must be rejected by domain purity
        assert "Studio XO" not in text
        assert "Cases still pending" not in text
        # The Morning Standard should explicitly state zero coverage
        assert "No standalone Health & Medicine reporting" in text
        assert "Carried no standalone Health & Medicine reporting in this edition" in text
        assert "- **The Morning Standard**: Carried no dedicated Health & Medicine reports in this issue." in text
        # Executive summary must never copy the user's raw prompt
        assert "COMPARE ALL THE NEWSPAPER" not in text

    def test_system_prompt_excludes_hardcoded_tax_example(self) -> None:
        """Verify prompt does not contain hardcoded Section 80C CBDT tax citation example."""
        synth = AnswerSynthesizer()
        prompt_comp = synth._build_synthesizer_system_prompt(
            archetype="cross_newspaper_comparison",
            query="COMPARE ALL THE NEWSPAPER AVALABLE DATED 1/8/2026 on health related news",
        )
        assert "Section 80C" not in prompt_comp
        assert "Direct Tax Compliance" not in prompt_comp
        assert "Over 4.5 lakh taxpayers" not in prompt_comp

        prompt_cat = synth._build_synthesizer_system_prompt(
            archetype="article_catalog",
            query="list all health articles",
        )
        assert "Section 80C" not in prompt_cat
        assert "Direct Tax Compliance" not in prompt_cat
        assert "Over 4.5 lakh taxpayers" not in prompt_cat

    def test_dynamic_prompt_scalar_or_count_sizing(self) -> None:
        """Verify queries asking for counts or metadata select DIRECT SCALAR & QUANTITATIVE FINDING structure."""
        synth = AnswerSynthesizer()
        # Query asking for count of issues
        prompt_scalar = synth._build_synthesizer_system_prompt(
            archetype="article_catalog",
            query="no of newspaper in the Goan issues ?",
            evidence_items=[
                {
                    "article_id": 0,
                    "headline": "Analytical Computation: no of newspaper in the Goan issues ?",
                    "newspaper_name": "Archive Analytics",
                    "snippet": "The Goan issues contain 8 issues and 1349 articles.",
                    "source_tool": "dynamic_analysis",
                    "is_statistical_metric": True,
                }
            ],
        )
        assert "DIRECT SCALAR & QUANTITATIVE FINDING" in prompt_scalar
        assert "Direct Finding" in prompt_scalar
        assert "Key Computed Metrics" in prompt_scalar
        assert "Comprehensive Archive Articles Catalog" not in prompt_scalar

    def test_deterministic_summary_scalar_statistical_metadata(self) -> None:
        """Verify deterministic summary produces direct quantitative finding without fake articles or explore prompts."""
        synth = AnswerSynthesizer()
        evidence = [
            {
                "article_id": 0,
                "headline": "Analytical Computation: no of newspaper in the Goan issues ?",
                "newspaper_name": "Archive Analytics",
                "snippet": "The Goan issues contain 8 issues and 1349 articles.",
                "source_tool": "dynamic_analysis",
                "is_statistical_metric": True,
                "metadata": {"total_issues": 8, "total_articles": 1349},
            }
        ]
        text = synth._generate_deterministic_summary(
            query="no of newspaper in the Goan issues ?",
            evidence_items=evidence,
            archetype="article_catalog",
        )
        assert "### ⚡ Direct Finding" in text
        assert "The Goan issues contain 8 issues and 1349 articles." in text
        assert "### 📊 Key Computed Metrics" in text
        assert "- **Total Issues**: 8" in text
        assert "Explore Further" not in text
        assert "Statistical Engine" not in text

    def test_synthesizer_system_prompt_mandates_metric_absence_hard_stop(self) -> None:
        """Verify synthesizer prompt explicitly mandates refusal to invent missing statistical metrics."""
        from app.agent.synthesizer import AnswerSynthesizer, COMMON_ANALYTICAL_GUIDELINES

        assert "QUANTITATIVE & STATISTICAL METRIC ABSENCE HARD-STOP" in COMMON_ANALYTICAL_GUIDELINES
        assert "could not be computed or is unavailable" in COMMON_ANALYTICAL_GUIDELINES
        assert "STRICTLY AND ABSOLUTELY FORBIDDEN from estimating, guessing, fabricating" in COMMON_ANALYTICAL_GUIDELINES

        synth = AnswerSynthesizer()
        prompt = synth._build_synthesizer_system_prompt(
            archetype="analytical_computation",
            query="What is the variance and standard deviation of word counts?",
        )
        assert "QUANTITATIVE & STATISTICAL METRIC ABSENCE HARD-STOP" in prompt

    def test_synthesizer_shared_coverage_layout_and_guardrail(self) -> None:
        """Verify synthesizer prompt builds 3-part layout and anti-hallucination guardrail for shared wire coverage."""
        synth = AnswerSynthesizer()
        prompt = synth._build_synthesizer_system_prompt(
            archetype="cross_newspaper_comparison",
            query="give me the similar articles from newspaper of the Goan and the morning standard both dated 1/8/2026",
        )

        assert "Executive Summary: Shared Syndicated Coverage" in prompt
        assert "Verified Shared Coverage Matrix" in prompt
        assert "Regional Framing & Placement Divergence" in prompt
        assert "STRICT SIMILARITY & SHARED STORY INTEGRITY" in prompt
        assert "ABSOLUTE PROHIBITION ON FALSE EQUIVALENCE" in prompt

    def test_synthesizer_shared_article_citations_extracted(self) -> None:
        """Verify individual article records from shared coverage yield clickable AgentCitations."""
        synth = AnswerSynthesizer()
        evidence = [
            {
                "article_id": 0,
                "headline": "Verified Shared Wire Coverage: The Goan & The Morning Standard",
                "newspaper_name": "The Goan & The Morning Standard",
                "issue_date": "2026-08-01",
                "snippet": "Macro shared overview",
                "source_tool": "sql_analytics",
            },
            {
                "article_id": 501,
                "headline": "SC stays stray animal compensation order",
                "newspaper_name": "The Goan",
                "issue_date": "2026-08-01",
                "pages": [5],
                "snippet": "Supreme Court stayed compensation order.",
                "source_tool": "sql_analytics_shared",
            },
            {
                "article_id": 702,
                "headline": "Apex court puts hold on stray animal compensation",
                "newspaper_name": "The Morning Standard",
                "issue_date": "2026-08-01",
                "pages": [7],
                "snippet": "Apex court puts hold on compensation order.",
                "source_tool": "sql_analytics_shared",
            },
        ]

        text = (
            "### ⚡ Executive Summary: Shared Syndicated Coverage\n"
            "Both papers carried coverage of the stray animal order.\n\n"
            "### 📰 Verified Shared Coverage Matrix\n"
            "| # | Story | The Goan | The Morning Standard |\n"
            "| 1 | Stray animal order | SC stays stray animal compensation order (Page 5) | Apex court puts hold on stray animal compensation (Page 7) |\n"
        )

        citations = synth.extract_citations(text, evidence)
        assert len(citations) == 2
        art_ids = {c["article_id"] for c in citations}
        assert 501 in art_ids
        assert 702 in art_ids
        assert 0 not in art_ids  # Macro item (article_id: 0) must be excluded








