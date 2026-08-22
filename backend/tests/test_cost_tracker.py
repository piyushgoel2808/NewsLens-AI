"""Tests for Token Cost Accountant and Budget Guardrails."""

from __future__ import annotations

import pytest

from app.core.cost_tracker import (
    BudgetExceededError,
    calculate_cost_usd,
    record_usage_and_cost,
    resolve_model_price,
    validate_query_budget,
)


def test_resolve_model_prices() -> None:
    """Verify official pricing rates resolution."""
    # Anthropic
    anthropic_p = resolve_model_price("anthropic", "claude-3-5-sonnet")
    assert anthropic_p.input_usd_per_million == 3.00
    assert anthropic_p.output_usd_per_million == 15.00

    # OpenAI
    openai_p = resolve_model_price("openai", "gpt-4o")
    assert openai_p.input_usd_per_million == 2.50
    assert openai_p.output_usd_per_million == 10.00

    # Groq
    groq_p = resolve_model_price("groq", "llama-3.3-70b-versatile")
    assert groq_p.input_usd_per_million == 0.59
    assert groq_p.output_usd_per_million == 0.79

    # Local zero cost
    local_p = resolve_model_price("ollama", "llama3.2:3b")
    assert local_p.input_usd_per_million == 0.0
    assert local_p.output_usd_per_million == 0.0


def test_calculate_cost_usd() -> None:
    """Verify exact USD calculation."""
    # 1,000 input tokens + 200 output tokens on Claude 3.5 Sonnet
    # (1000/1e6 * 3.00) + (200/1e6 * 15.00) = 0.003 + 0.003 = 0.006
    cost = calculate_cost_usd(
        provider="anthropic",
        model="claude-3-5-sonnet",
        input_tokens=1000,
        output_tokens=200,
    )
    assert cost == 0.006

    # Zero cost for Ollama
    local_cost = calculate_cost_usd(
        provider="ollama",
        model="llama3.1:70b",
        input_tokens=10000,
        output_tokens=2000,
    )
    assert local_cost == 0.0


def test_record_usage_and_cost() -> None:
    """Verify recording usage returns the calculated cost."""
    cost = record_usage_and_cost("openai", "gpt-4o", 2000, 500)
    assert cost > 0.0


def test_validate_query_budget() -> None:
    """Verify budget guardrail catches queries exceeding max cost limit."""
    # Normal query within budget
    validate_query_budget(estimated_cost_usd=0.05, max_budget_usd=0.50)

    # Over budget query
    with pytest.raises(BudgetExceededError) as exc_info:
        validate_query_budget(estimated_cost_usd=0.85, max_budget_usd=0.50)
    assert "Query budget breached" in str(exc_info.value)
