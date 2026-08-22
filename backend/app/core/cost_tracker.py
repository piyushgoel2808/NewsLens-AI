"""Token Cost Accountant & Budget Guardrails.

Provides pricing catalog across Anthropic, OpenAI, Groq, and zero-cost local providers.
Enforces per-query budget guardrails and exports real-time Prometheus cost counters.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.metrics import record_llm_tokens_and_cost

logger = get_logger(__name__)


class BudgetExceededError(Exception):
    """Raised when an agent query exceeds the configured monetary or token limit."""


@dataclass(frozen=True)
class ModelPrice:
    input_usd_per_million: float
    output_usd_per_million: float


# Official pricing rates (USD per million tokens)
PRICING_CATALOG: dict[str, ModelPrice] = {
    # Anthropic
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-7-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00),
    "claude-3-opus": ModelPrice(15.00, 75.00),
    # OpenAI
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "text-embedding-3-large": ModelPrice(0.13, 0.0),
    "text-embedding-3-small": ModelPrice(0.02, 0.0),
    # Groq
    "llama-3.3-70b-versatile": ModelPrice(0.59, 0.79),
    "llama-3.1-8b-instant": ModelPrice(0.05, 0.08),
    "mixtral-8x7b-32768": ModelPrice(0.24, 0.24),
}

# Maximum default budget per single query ($0.50)
DEFAULT_MAX_QUERY_BUDGET_USD = 0.50


def resolve_model_price(provider: str, model: str) -> ModelPrice:
    """Resolve the pricing tier for a given provider and model name."""
    p_lower = (provider or "").lower()
    m_lower = (model or "").lower()

    # Local providers are always zero-cost
    if p_lower in ("ollama", "local_sentence_transformers", "tesseract", "docling", "local"):
        return ModelPrice(0.0, 0.0)

    for pattern, price in PRICING_CATALOG.items():
        if pattern in m_lower:
            return price

    # Default fallback pricing for cloud models if unlisted ($1.00 / $3.00 per MTok)
    if p_lower in ("anthropic", "openai", "groq"):
        return ModelPrice(1.00, 3.00)

    return ModelPrice(0.0, 0.0)


def calculate_cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate the exact USD expenditure for a completion."""
    price = resolve_model_price(provider, model)
    cost = (input_tokens / 1_000_000.0 * price.input_usd_per_million) + (
        output_tokens / 1_000_000.0 * price.output_usd_per_million
    )
    return round(cost, 6)


def record_usage_and_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute cost, update Prometheus counters, and log metrics."""
    cost = calculate_cost_usd(provider, model, input_tokens, output_tokens)
    record_llm_tokens_and_cost(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )
    return cost


def validate_query_budget(
    estimated_cost_usd: float,
    max_budget_usd: float | None = None,
) -> None:
    """Validate that query cost does not breach the budget ceiling."""
    limit = max_budget_usd or DEFAULT_MAX_QUERY_BUDGET_USD
    if estimated_cost_usd > limit:
        raise BudgetExceededError(
            f"Query budget breached: estimated cost ${estimated_cost_usd:.4f} "
            f"exceeds max limit of ${limit:.2f}"
        )
