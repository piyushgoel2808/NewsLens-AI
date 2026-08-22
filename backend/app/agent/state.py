"""LangGraph Agent State definition for NewsLens-AI."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentCitation(TypedDict, total=False):
    """A verified source citation grounding an agent response."""

    newspaper_name: str
    issue_date: str
    page_number: int
    headline: str
    article_id: int
    snippet: str
    issue_id: int
    bboxes: list[dict[str, Any]]


class ToolExecutionRecord(TypedDict):
    """Record of an individual tool invocation."""

    tool_name: str
    tool_input: dict[str, Any]
    results_count: int
    execution_time_ms: int


class AgentState(TypedDict):
    """Complete state container for the LangGraph agentic workflow."""

    query: str
    original_query: str | None
    chat_history: list[dict[str, Any]]
    archetype: str  # e.g. 'factual_lookup', 'thematic_timeline', 'entity_deep_dive'
    plan: list[dict[str, Any]]
    tool_executions: list[ToolExecutionRecord]
    evidence_items: list[dict[str, Any]]
    synthesized_answer: str
    citations: list[AgentCitation]
    cost_usd: float
    latency_ms: int
    user_id: str | None
    model_override: str | None
    error: str | None
