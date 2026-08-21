"""FastAPI router for Agentic RAG Queries and Plan Inspection."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import AgentWorkflow
from app.agent.planner import QueryPlanner
from app.models.base import get_db, get_session_factory
from app.models.query import QueryLog

router = APIRouter(prefix="/api/query", tags=["query"])


class QueryRequest(BaseModel):
    """Request payload for an agentic query."""

    query: str = Field(..., description="The user's research query", min_length=2)
    user_id: str | None = Field(None, description="Optional identifier of the user")


class QueryResponse(BaseModel):
    """Response payload containing synthesized answer, citations, and execution telemetry."""

    query: str
    archetype: str
    answer: str
    citations: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    tool_executions: list[dict[str, Any]]
    evidence_count: int
    latency_ms: int
    cost_usd: float


@router.post("", response_model=QueryResponse, summary="Execute full agentic RAG query pipeline")
async def execute_query(
    request: QueryRequest,
) -> QueryResponse:
    """Execute the multi-stage LangGraph query workflow."""
    factory = get_session_factory()
    workflow = AgentWorkflow(session_factory=factory)
    result = await workflow.run(query=request.query, user_id=request.user_id)

    citations_list: list[dict[str, Any]] = [dict(c) for c in result.get("citations", [])]
    tools_list: list[dict[str, Any]] = [dict(t) for t in result.get("tool_executions", [])]

    return QueryResponse(
        query=result["query"],
        archetype=result["archetype"],
        answer=result["synthesized_answer"],
        citations=citations_list,
        plan=result.get("plan", []),
        tool_executions=tools_list,
        evidence_count=len(result.get("evidence_items", [])),
        latency_ms=result.get("latency_ms", 0),
        cost_usd=result.get("cost_usd", 0.0),
    )


@router.post("/plan", summary="Inspect query archetype classification and tool execution plan")
async def inspect_plan(
    request: QueryRequest,
) -> dict[str, Any]:
    """Classify archetype and generate execution plan without executing tools."""
    planner = QueryPlanner()
    plan_res = planner.plan_query(request.query)

    return {
        "query": request.query,
        "archetype": plan_res.archetype,
        "reasoning": plan_res.reasoning,
        "planned_tools": [
            {
                "tool_name": c.tool_name,
                "arguments": c.arguments,
                "purpose": c.purpose,
            }
            for c in plan_res.tool_calls
        ],
    }


@router.get("/history", summary="Fetch past queries and executions from audit log")
async def get_query_history(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List historical agentic queries with metrics and citations."""
    stmt = select(QueryLog).order_by(desc(QueryLog.created_at)).limit(limit)
    res = await db.execute(stmt)
    records = res.scalars().all()

    history_items: list[dict[str, Any]] = []
    for r in records:
        cit_count = 0
        if isinstance(r.citations_json, dict):
            cit_count = len(r.citations_json.get("citations", []))
        elif isinstance(r.citations_json, list):
            cit_count = len(r.citations_json)

        history_items.append(
            {
                "id": r.id,
                "query_text": r.query_text,
                "query_type": r.query_type,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "citations_count": cit_count,
            }
        )

    return history_items
