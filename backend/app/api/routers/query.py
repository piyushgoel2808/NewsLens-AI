import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.condenser import (
    CLEAN_SESSION_CLARIFICATION_MESSAGE,
    condense_conversational_query,
    is_ambiguous_standalone_query,
    needs_condensation,
)
from app.agent.graph import AgentWorkflow
from app.agent.planner import QueryPlanner
from app.models.base import get_db, get_session_factory
from app.models.query import QueryLog

router = APIRouter(prefix="/api/query", tags=["query"])


class QueryRequest(BaseModel):
    """Request payload for an agentic query."""

    query: str = Field(..., description="The user's research query", min_length=2)
    user_id: str | None = Field(None, description="Optional identifier of the user")
    chat_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent turns for coreference resolution and follow-up condensation",
    )
    model_override: str | None = Field(
        None,
        description="Optional model provider override (e.g. groq_llama, ollama_chat)",
    )


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
    result = await workflow.run(
        query=request.query,
        chat_history=request.chat_history,
        user_id=request.user_id,
        model_override=request.model_override,
    )

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


@router.post(
    "/stream",
    summary="Stream agent execution steps and tokens via Server-Sent Events (SSE)",
)
async def stream_query(
    request: QueryRequest,
) -> StreamingResponse:
    """Stream real-time agent execution progress, token deltas, and citations."""
    factory = get_session_factory()
    workflow = AgentWorkflow(session_factory=factory)

    async def event_generator() -> AsyncIterator[str]:
        t0 = time.monotonic()
        query = request.query
        chat_history = request.chat_history or []

        # 0. Ambiguity Guardrail for Clean Sessions
        if is_ambiguous_standalone_query(query, chat_history):
            yield f"event: stage\ndata: {json.dumps({'stage': 'completed'})}\n\n"
            token_payload = json.dumps({"delta": CLEAN_SESSION_CLARIFICATION_MESSAGE})
            yield f"event: token\ndata: {token_payload}\n\n"
            yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
            done_payload = json.dumps(
                {
                    "latency_ms": round((time.monotonic() - t0) * 1000),
                    "cost_usd": 0.0,
                    "evidence_count": 0,
                }
            )
            yield f"event: done\ndata: {done_payload}\n\n"
            return

        # 1. Query Condensation & Coreference Resolution
        if chat_history and needs_condensation(query, chat_history):
            yield f"event: stage\ndata: {json.dumps({'stage': 'condensing_query'})}\n\n"
            condensed = await condense_conversational_query(
                query=query,
                chat_history=chat_history,
                model_override=request.model_override,
            )
            query = condensed
            yield f"event: query_condensed\ndata: {json.dumps({'condensed_query': condensed})}\n\n"

        # 2. Planning Stage
        yield f"event: stage\ndata: {json.dumps({'stage': 'planning'})}\n\n"
        plan_res = workflow._planner.plan_query(query)
        planned_calls = [
            {"tool_name": c.tool_name, "arguments": c.arguments, "purpose": c.purpose}
            for c in plan_res.tool_calls
        ]
        plan_data = json.dumps({"archetype": plan_res.archetype, "plan": planned_calls})
        yield f"event: plan\ndata: {plan_data}\n\n"

        # 3. Tool Execution Stage
        yield f"event: stage\ndata: {json.dumps({'stage': 'tool_execution'})}\n\n"
        tool_state = await workflow._execute_tools_node(
            {
                "query": query,
                "original_query": request.query,
                "chat_history": chat_history,
                "archetype": plan_res.archetype,
                "plan": planned_calls,
                "tool_executions": [],
                "evidence_items": [],
                "synthesized_answer": "",
                "citations": [],
                "cost_usd": 0.0,
                "latency_ms": 0,
                "user_id": request.user_id,
                "model_override": request.model_override,
                "error": None,
            }
        )
        evidence = tool_state.get("evidence_items", [])
        tool_records = [dict(t) for t in tool_state.get("tool_executions", [])]
        tool_data = json.dumps({"evidence_count": len(evidence), "tools": tool_records})
        yield f"event: tool_results\ndata: {tool_data}\n\n"

        # 4. Synthesis & Reasoning Stage
        yield f"event: stage\ndata: {json.dumps({'stage': 'synthesizing'})}\n\n"
        in_think = False
        raw_buffer = ""
        think_chunks: list[str] = []
        answer_chunks: list[str] = []
        think_start_time: float | None = None

        async for chunk in workflow._synthesizer.synthesize_stream(
            query=query,
            archetype=plan_res.archetype,
            evidence_items=evidence,
            model_override=request.model_override,
        ):
            raw_buffer += chunk

            while raw_buffer:
                if not in_think:
                    if "<think>" in raw_buffer:
                        pre, post = raw_buffer.split("<think>", 1)
                        if pre:
                            answer_chunks.append(pre)
                            yield f"event: token\ndata: {json.dumps({'delta': pre})}\n\n"
                        in_think = True
                        think_start_time = time.monotonic()
                        yield f"event: stage\ndata: {json.dumps({'stage': 'thinking'})}\n\n"
                        raw_buffer = post
                    else:
                        is_partial = any(
                            raw_buffer.endswith("<think>"[:i])
                            for i in range(1, len("<think>"))
                        )
                        if is_partial:
                            break
                        answer_chunks.append(raw_buffer)
                        yield f"event: token\ndata: {json.dumps({'delta': raw_buffer})}\n\n"
                        raw_buffer = ""
                else:
                    if "</think>" in raw_buffer:
                        thought_piece, post = raw_buffer.split("</think>", 1)
                        if thought_piece:
                            think_chunks.append(thought_piece)
                            th_payload = json.dumps({"delta": thought_piece})
                            yield f"event: thought\ndata: {th_payload}\n\n"
                        in_think = False
                        t_dur = round(time.monotonic() - (think_start_time or time.monotonic()), 1)
                        full_th = "".join(think_chunks).strip()
                        done_th = json.dumps({"thought": full_th, "duration_sec": t_dur})
                        yield f"event: thought_done\ndata: {done_th}\n\n"
                        yield f"event: stage\ndata: {json.dumps({'stage': 'synthesizing'})}\n\n"
                        raw_buffer = post.lstrip("\n ")
                    else:
                        is_partial = any(
                            raw_buffer.endswith("</think>"[:i])
                            for i in range(1, len("</think>"))
                        )
                        if is_partial:
                            break
                        think_chunks.append(raw_buffer)
                        yield f"event: thought\ndata: {json.dumps({'delta': raw_buffer})}\n\n"
                        raw_buffer = ""

        # Flush any remaining buffer
        if raw_buffer:
            if in_think:
                think_chunks.append(raw_buffer)
                yield f"event: thought\ndata: {json.dumps({'delta': raw_buffer})}\n\n"
                t_dur = round(time.monotonic() - (think_start_time or time.monotonic()), 1)
                full_th = "".join(think_chunks).strip()
                done_th = json.dumps({"thought": full_th, "duration_sec": t_dur})
                yield f"event: thought_done\ndata: {done_th}\n\n"
            else:
                answer_chunks.append(raw_buffer)
                yield f"event: token\ndata: {json.dumps({'delta': raw_buffer})}\n\n"

        full_answer = "".join(answer_chunks).strip()
        citations = workflow._synthesizer.extract_citations(full_answer, evidence)
        citations_list = [dict(c) for c in citations]
        yield f"event: citations\ndata: {json.dumps({'citations': citations_list})}\n\n"

        latency_ms = round((time.monotonic() - t0) * 1000)

        # 4. Save Query Log in DB
        async with factory() as db:
            log_record = QueryLog(
                user_id=request.user_id,
                query_text=query,
                query_type=plan_res.archetype,
                plan_json={"plan": planned_calls},
                tool_calls_json={"tools": tool_records},
                answer_text=full_answer,
                citations_json={"citations": citations_list},
                latency_ms=latency_ms,
                cost_usd=0.0,
            )
            db.add(log_record)
            await db.commit()

        yield f"event: done\ndata: {json.dumps({'latency_ms': latency_ms, 'cost_usd': 0.0})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
