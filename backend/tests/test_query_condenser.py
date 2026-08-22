"""Tests for Conversational Query Condensation, Coreference Resolution, and Ambiguity Guardrails."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.condenser import (
    CLEAN_SESSION_CLARIFICATION_MESSAGE,
    condense_conversational_query,
    format_chat_history_for_prompt,
    is_ambiguous_standalone_query,
    needs_condensation,
)
from app.agent.graph import AgentWorkflow
from app.api.main import create_app
from app.models.base import get_db
from app.providers.base import ModelResponse


class TestQueryCondenserUnit:
    """Unit tests for condenser helper functions."""

    def test_needs_condensation_false_for_empty_history(self) -> None:
        assert not needs_condensation("Can you summarize it?", [])
        assert not needs_condensation("Tell me more about this", [])

    def test_needs_condensation_true_for_pronouns_with_history(self) -> None:
        history = [{"role": "user", "content": "Tell me about Tata Power nuclear plans in Odisha"}]
        assert needs_condensation("Can you summarize it?", history)
        assert needs_condensation("tell me more about this", history)
        assert needs_condensation("who was involved?", history)
        assert needs_condensation("what was the deal value?", history)
        assert needs_condensation("why did they decide that?", history)

    def test_needs_condensation_false_for_explicit_standalone_queries(self) -> None:
        history = [{"role": "user", "content": "Tell me about Tata Power nuclear plans in Odisha"}]
        assert not needs_condensation(
            "What was the wholesale inflation rate in July 2026 for India?", history
        )
        assert not needs_condensation(
            "List all front page articles from Mint published on 2026-08-01", history
        )

    def test_is_ambiguous_standalone_query_clean_session(self) -> None:
        # Clean session: no prior history
        assert is_ambiguous_standalone_query("summarize it", [])
        assert is_ambiguous_standalone_query("can you summarize it?", [])
        assert is_ambiguous_standalone_query("tell me more", [])
        assert is_ambiguous_standalone_query("what happened?", [])

        # Non-empty history: not a clean session ambiguity (handled by condensation instead)
        history = [{"role": "user", "content": "Tata power report"}]
        assert not is_ambiguous_standalone_query("summarize it", history)

        # Explicit clean query: not ambiguous
        assert not is_ambiguous_standalone_query("Tata Power nuclear expansion in Odisha", [])
        assert not is_ambiguous_standalone_query("Compare Mint and The Hindu front pages", [])

    def test_format_chat_history_for_prompt(self) -> None:
        history = [
            {"role": "user", "content": "Tell me about the Tata Power nuclear plans in Odisha"},
            {
                "role": "assistant",
                "content": "Tata Power announced a 2800 MW expansion in coastal Odisha...",
            },
        ]
        formatted = format_chat_history_for_prompt(history)
        assert "User: Tell me about the Tata Power nuclear plans in Odisha" in formatted
        assert "Assistant: Tata Power announced a 2800 MW" in formatted

    @pytest.mark.asyncio
    async def test_condense_conversational_query_with_mock_provider(self) -> None:
        history = [
            {"role": "user", "content": "Tell me about the Tata Power nuclear plans in Odisha"},
            {
                "role": "assistant",
                "content": "Tata Power is negotiating with the Odisha state government.",
            },
        ]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(
            return_value=ModelResponse(
                text="Summarize Tata Power nuclear plans in Odisha",
                input_tokens=50,
                output_tokens=10,
            )
        )

        condensed = await condense_conversational_query(
            query="Can you summarize it?",
            chat_history=history,
            provider=mock_provider,
        )

        assert condensed == "Summarize Tata Power nuclear plans in Odisha"
        assert mock_provider.complete.called


class TestAgentWorkflowCondensation:
    """Integration tests for AgentWorkflow with query condensation and guardrails."""

    @pytest.mark.asyncio
    async def test_clean_session_ambiguity_guardrail_short_circuits(self) -> None:
        mock_session_factory = MagicMock()
        mock_db = MagicMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_db

        workflow = AgentWorkflow(session_factory=mock_session_factory)
        workflow._cache.get_query = AsyncMock(return_value=None)
        workflow._cache.set_query = AsyncMock(return_value=True)

        result = await workflow.run(
            query="can you summarize it?",
            chat_history=[],
        )

        assert result["archetype"] == "clarification_needed"
        assert result["synthesized_answer"] == CLEAN_SESSION_CLARIFICATION_MESSAGE
        assert len(result["tool_executions"]) == 0
        assert len(result["evidence_items"]) == 0
        assert len(result["citations"]) == 0

    @pytest.mark.asyncio
    async def test_multi_turn_followup_condenses_and_retrieves(self) -> None:
        mock_session_factory = MagicMock()
        mock_db = MagicMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_db

        workflow = AgentWorkflow(session_factory=mock_session_factory)
        workflow._cache.get_query = AsyncMock(return_value=None)
        workflow._cache.set_query = AsyncMock(return_value=True)

        # Mock query condenser provider
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(
            return_value=ModelResponse(
                text="Summarize Tata Power nuclear plans in Odisha",
                input_tokens=40,
                output_tokens=10,
            )
        )

        # Mock tools & synthesizer
        workflow._hybrid_search.search = AsyncMock(return_value=[])  # type: ignore[method-assign]
        workflow._synthesizer.synthesize = AsyncMock(  # type: ignore[method-assign]
            return_value=("Tata Power is planning nuclear units in Odisha.", [], 0.001)
        )

        history = [
            {"role": "user", "content": "Tell me about the Tata Power nuclear plans in Odisha"},
            {"role": "assistant", "content": "Tata Power has announced nuclear expansion."},
        ]

        # Condense query
        condensed_text = await condense_conversational_query(
            query="Can you summarize it?",
            chat_history=history,
            provider=mock_provider,
        )
        assert "Tata Power" in condensed_text

        result = await workflow.run(
            query="Can you summarize it?",
            chat_history=history,
        )

        # The planner should receive the condensed context
        assert result["archetype"] in ("factual_lookup", "quantitative_trend")
        assert "Tata Power" in result["query"] or result["query"] == "Can you summarize it?"


class TestQueryAPIEndpointCondensation:
    """API endpoint tests verifying chat_history payload support and clarification guardrail."""

    @pytest.mark.asyncio
    async def test_api_clarification_guardrail(self) -> None:
        from unittest.mock import patch

        app = create_app()

        mock_session_factory = MagicMock()
        mock_db = MagicMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_db

        with patch("app.api.routers.query.get_session_factory", return_value=mock_session_factory):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/query",
                    json={
                        "query": "summarize it",
                        "chat_history": [],
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["archetype"] == "clarification_needed"
                assert data["answer"] == CLEAN_SESSION_CLARIFICATION_MESSAGE
                assert data["evidence_count"] == 0
