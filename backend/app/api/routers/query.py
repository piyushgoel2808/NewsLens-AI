import asyncio
import json
import re
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
    extract_active_issue_from_history,
    is_ambiguous_standalone_query,
    is_in_context_meta_query,
    needs_condensation,
    parse_inline_citation,
    resolve_attached_asset_context,
)
from app.agent.extractor import extract_parameters_from_query
from app.agent.graph import AgentWorkflow
from app.agent.planner import QueryPlanner
from app.agent.synthesizer import parse_thought_and_answer
from app.models.base import get_db, get_session_factory
from app.models.query import QueryLog
from app.retrieval.timeline_builder import NarrativeTrajectoryResponse, TimelineBuilder

REASONING_START_REGEX = re.compile(
    r"^\s*(?:<think>|Here'?s a thinking process:?|Thinking Process:?|Thought:?)",
    re.IGNORECASE,
)

ANSWER_TRANSITION_REGEX = re.compile(
    r"\n\s*(?:#{1,4}\s+|Draft:\s*\n|Executive Summary:?\s*\n|Summary:\s*\n)",
    re.IGNORECASE,
)

router = APIRouter(prefix="/api/query", tags=["query"])


class QueryRequest(BaseModel):
    """Request payload for an agentic query."""

    query: str = Field(..., description="The user's research query", min_length=2)
    user_id: str | None = Field(None, description="Optional identifier of the user")
    chat_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent turns for coreference resolution and follow-up condensation",
    )
    model: str | None = Field(
        None,
        description="Optional model provider selection (e.g. gemini_flash, groq_llama)",
    )
    model_override: str | None = Field(
        None,
        description="Optional model provider override (e.g. groq_llama, ollama_chat)",
    )
    enable_web_search: bool = Field(
        False,
        description="Enable live internet search to complement newspaper archives",
    )
    attached_article_id: int | None = Field(
        None,
        description="Optional article ID currently active in user's reader workspace",
    )
    attached_photo_id: int | None = Field(
        None,
        description="Optional photo or infographic ID attached to the query",
    )
    attached_issue_date: str | None = Field(
        None,
        description="Optional issue date associated with the attached asset or reader context (YYYY-MM-DD)",
    )
    attached_newspaper_name: str | None = Field(
        None,
        description="Optional newspaper name associated with the attached asset or reader context",
    )

    @property
    def effective_model(self) -> str | None:
        """Return the user-selected model identifier."""
        return self.model or self.model_override


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
    attached_ctx = await resolve_attached_asset_context(
        session_factory=factory,
        attached_photo_id=request.attached_photo_id,
        attached_article_id=request.attached_article_id,
        attached_issue_date=request.attached_issue_date,
        attached_newspaper_name=request.attached_newspaper_name,
    )
    res_date = attached_ctx.get("issue_date")
    res_np = attached_ctx.get("newspaper_name")
    res_art_id = attached_ctx.get("article_id")
    result = await workflow.run(
        query=request.query,
        chat_history=request.chat_history,
        user_id=request.user_id,
        model_override=request.effective_model,
        enable_web_search=request.enable_web_search,
        attached_article_id=res_art_id or request.attached_article_id,
        attached_photo_id=request.attached_photo_id,
        attached_issue_date=res_date,
        attached_newspaper_name=res_np,
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
        effective_model = request.effective_model

        # 0. Ambiguity Guardrail for Clean Sessions
        has_attached = bool(request.attached_photo_id or request.attached_article_id)
        if is_ambiguous_standalone_query(query, chat_history, has_attached_asset=has_attached):
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

        # 1. Query Condensation, Coreference Resolution, & In-Context Meta Queries
        is_meta = is_in_context_meta_query(query, chat_history) if chat_history else False

        if is_meta:
            effective_archetype = "conversational_meta_query"
            planned_calls = []
            plan_data = json.dumps({"archetype": effective_archetype, "plan": []})
            yield f"event: plan\ndata: {plan_data}\n\n"
            evidence: list[dict[str, Any]] = []
            tool_data = json.dumps({"evidence_count": 0, "tools": []})
            yield f"event: tool_results\ndata: {tool_data}\n\n"
        else:
            attached_ctx = await resolve_attached_asset_context(
                session_factory=workflow._session_factory,
                attached_photo_id=request.attached_photo_id,
                attached_article_id=request.attached_article_id,
                attached_issue_date=request.attached_issue_date,
                attached_newspaper_name=request.attached_newspaper_name,
            )
            res_date = attached_ctx.get("issue_date")
            res_np = attached_ctx.get("newspaper_name")
            res_art_id = attached_ctx.get("article_id")
            res_iss_id = attached_ctx.get("issue_id")
            q_params = extract_parameters_from_query(query) if query else {}
            q_cite = parse_inline_citation(query or "")
            explicit_q_date = q_params.get("issue_date") or q_cite.get("issue_date")
            explicit_q_np = q_params.get("newspaper_name") or q_cite.get("newspaper_name")

            has_date_conflict = bool(explicit_q_date and res_date and explicit_q_date != res_date)
            has_np_conflict = bool(
                explicit_q_np
                and res_np
                and explicit_q_np.lower() not in res_np.lower()
                and res_np.lower() not in explicit_q_np.lower()
            )

            eff_attached_article_id = (res_art_id or request.attached_article_id) if not has_date_conflict and not has_np_conflict else None
            eff_attached_photo_id = request.attached_photo_id if not has_date_conflict and not has_np_conflict else None

            active_ctx = extract_active_issue_from_history(
                chat_history,
                current_query=query,
                attached_photo_id=eff_attached_photo_id,
                attached_article_id=eff_attached_article_id,
                attached_issue_date=res_date if not has_date_conflict else None,
                attached_newspaper_name=res_np if not has_np_conflict else None,
                attached_headline=attached_ctx.get("headline") if not has_date_conflict and not has_np_conflict else None,
            )
            if not has_date_conflict and not has_np_conflict:
                if res_date:
                    active_ctx["issue_date"] = res_date
                if res_np:
                    active_ctx["newspaper_name"] = res_np
                if res_iss_id:
                    active_ctx["issue_id"] = res_iss_id
                if eff_attached_article_id:
                    active_ctx["article_id"] = eff_attached_article_id
                if eff_attached_photo_id:
                    active_ctx["photo_id"] = eff_attached_photo_id
                if attached_ctx.get("headline"):
                    active_ctx["headline"] = attached_ctx["headline"]
            else:
                if explicit_q_date:
                    active_ctx["issue_date"] = explicit_q_date
                if explicit_q_np:
                    active_ctx["newspaper_name"] = explicit_q_np

            has_attached_asset = bool(
                (eff_attached_photo_id or eff_attached_article_id)
                and not has_date_conflict
                and not has_np_conflict
            )
            is_followup = bool(
                (chat_history or has_attached_asset)
                and needs_condensation(query, chat_history, attached_asset=attached_ctx)
            )

            if is_followup:
                yield f"event: stage\ndata: {json.dumps({'stage': 'condensing_query'})}\n\n"
                condensed = await condense_conversational_query(
                    query=query,
                    chat_history=chat_history,
                    model_override=effective_model,
                    active_issue_id=active_ctx.get("issue_id"),
                    active_newspaper_name=active_ctx.get("newspaper_name"),
                    active_issue_date=active_ctx.get("issue_date"),
                    attached_asset=attached_ctx,
                )
                query = condensed
                cond_data = json.dumps({"condensed_query": condensed})
                yield f"event: query_condensed\ndata: {cond_data}\n\n"

            # 2. Planning Stage
            # Only pass active_issue_date and active_newspapers to planner if an asset is attached,
            # or if this is an active follow-up query. Do NOT constrain standalone text queries.
            planner_active_date = active_ctx.get("issue_date") if (has_attached_asset or is_followup) else None
            planner_active_nps = (
                [active_ctx.get("newspaper_name")]
                if ((has_attached_asset or is_followup) and active_ctx.get("newspaper_name"))
                else None
            )

            yield f"event: stage\ndata: {json.dumps({'stage': 'planning'})}\n\n"
            plan_res = await workflow._planner.plan_query_async(
                query,
                enable_web_search=request.enable_web_search,
                model_override=effective_model,
                active_issue_date=planner_active_date,
                active_newspapers=planner_active_nps,
                attached_article_id=eff_attached_article_id,
                attached_photo_id=eff_attached_photo_id,
            )
            effective_archetype = plan_res.archetype
            planned_calls = [
                {"tool_name": c.tool_name, "arguments": c.arguments, "purpose": c.purpose}
                for c in plan_res.tool_calls
            ]
            plan_data = json.dumps({"archetype": effective_archetype, "plan": planned_calls})
            yield f"event: plan\ndata: {plan_data}\n\n"

            # 3. Tool Execution Stage
            if any(c.get("tool_name") == "dynamic_analysis" for c in planned_calls):
                yield f"event: stage\ndata: {json.dumps({'stage': 'generating_analysis_tool'})}\n\n"
            elif any(c.get("tool_name") == "inspect_visual_asset" for c in planned_calls):
                yield f"event: stage\ndata: {json.dumps({'stage': 'inspecting_visual_asset'})}\n\n"
            elif any(c.get("tool_name") == "web_search" for c in planned_calls):
                yield f"event: stage\ndata: {json.dumps({'stage': 'web_search'})}\n\n"
            else:
                yield f"event: stage\ndata: {json.dumps({'stage': 'tool_execution'})}\n\n"

            tool_state = await workflow._execute_tools_node(
                {
                    "query": query,
                    "original_query": request.query,
                    "chat_history": chat_history,
                    "archetype": effective_archetype,
                    "plan": planned_calls,
                    "tool_executions": [],
                    "evidence_items": [],
                    "synthesized_answer": "",
                    "citations": [],
                    "cost_usd": 0.0,
                    "latency_ms": 0,
                    "user_id": request.user_id,
                    "model_override": effective_model,
                    "enable_web_search": request.enable_web_search,
                    "web_search_results": [],
                    "active_issue_id": active_ctx.get("issue_id"),
                    "active_newspaper_name": active_ctx.get("newspaper_name"),
                    "active_issue_date": active_ctx.get("issue_date"),
                    "attached_article_id": eff_attached_article_id,
                    "attached_photo_id": eff_attached_photo_id,
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
        is_plain_think = False
        raw_buffer = ""
        think_chunks: list[str] = []
        answer_chunks: list[str] = []
        think_start_time: float | None = None

        async for chunk in workflow._synthesizer.synthesize_stream(
            query=query,
            archetype=effective_archetype,
            evidence_items=evidence,
            model_override=effective_model,
            chat_history=chat_history,
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
                        is_plain_think = False
                        think_start_time = time.monotonic()
                        yield f"event: stage\ndata: {json.dumps({'stage': 'thinking'})}\n\n"
                        raw_buffer = post
                    elif not answer_chunks and not think_chunks and REASONING_START_REGEX.match(raw_buffer):
                        in_think = True
                        is_plain_think = True
                        think_start_time = time.monotonic()
                        yield f"event: stage\ndata: {json.dumps({'stage': 'thinking'})}\n\n"
                    elif not answer_chunks and not think_chunks and len(raw_buffer.strip()) < 28 and any(
                        p.startswith(raw_buffer.strip().lower())
                        for p in ["here's a thinking process:", "thinking process:", "<think>"]
                    ):
                        break
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
                    if not is_plain_think and "</think>" in raw_buffer:
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
                    elif (match := ANSWER_TRANSITION_REGEX.search(raw_buffer)):
                        split_idx = match.start()
                        thought_piece = raw_buffer[:split_idx]
                        post = raw_buffer[split_idx:].lstrip("\n ")
                        post = re.sub(r"^draft:\s*\n*", "", post, flags=re.IGNORECASE)
                        if thought_piece:
                            think_chunks.append(thought_piece)
                            th_payload = json.dumps({"delta": thought_piece})
                            yield f"event: thought\ndata: {th_payload}\n\n"
                        in_think = False
                        is_plain_think = False
                        t_dur = round(time.monotonic() - (think_start_time or time.monotonic()), 1)
                        full_th = "".join(think_chunks).strip()
                        done_th = json.dumps({"thought": full_th, "duration_sec": t_dur})
                        yield f"event: thought_done\ndata: {done_th}\n\n"
                        yield f"event: stage\ndata: {json.dumps({'stage': 'synthesizing'})}\n\n"
                        raw_buffer = post
                    else:
                        if not is_plain_think:
                            is_partial = any(
                                raw_buffer.endswith("</think>"[:i])
                                for i in range(1, len("</think>"))
                            )
                            if is_partial:
                                break
                        else:
                            if raw_buffer.endswith("\n") or any(
                                raw_buffer.endswith(h)
                                for h in ["\n#", "\n##", "\n###", "\nDraft", "\nSummary", "\nExecutive"]
                            ):
                                break
                        think_chunks.append(raw_buffer)
                        yield f"event: thought\ndata: {json.dumps({'delta': raw_buffer})}\n\n"
                        raw_buffer = ""

        # Flush any remaining buffer
        if raw_buffer:
            if in_think:
                think_chunks.append(raw_buffer)
                th_payload = json.dumps({"delta": raw_buffer})
                yield f"event: thought\ndata: {th_payload}\n\n"
            else:
                answer_chunks.append(raw_buffer)
                yield f"event: token\ndata: {json.dumps({'delta': raw_buffer})}\n\n"

        if in_think and think_chunks:
            t_dur = round(time.monotonic() - (think_start_time or t0), 1)
            full_th = "".join(think_chunks).strip()
            done_th = json.dumps({"thought": full_th, "duration_sec": t_dur})
            yield f"event: thought_done\ndata: {done_th}\n\n"

        # Robust recovery for reasoning models where answer may be trapped inside <think> or scratchpad
        full_thought_final = "".join(think_chunks).strip()
        if not answer_chunks:
            combined_raw = f"<think>\n{full_thought_final}\n</think>" if full_thought_final else ""
            parsed_thought, parsed_ans = parse_thought_and_answer(combined_raw)
            t_dur = round(time.monotonic() - (think_start_time or t0), 1)
            if parsed_ans.strip():
                full_thought = parsed_thought or full_thought_final
                full_answer = parsed_ans
                done_th = json.dumps({"thought": full_thought, "duration_sec": t_dur})
                yield f"event: thought_done\ndata: {done_th}\n\n"
                yield f"event: stage\ndata: {json.dumps({'stage': 'synthesizing'})}\n\n"
                yield f"event: token\ndata: {json.dumps({'delta': full_answer})}\n\n"
            else:
                # If cannot separate or model only produced scratchpad / repetition loop,
                # synthesize clean deterministic grounded brief so UI is never blank or corrupted
                fallback_summary = workflow._synthesizer._generate_deterministic_summary(
                    query, evidence, archetype=effective_archetype
                )
                full_thought = full_thought_final
                done_th = json.dumps({"thought": full_thought, "duration_sec": t_dur})
                yield f"event: thought_done\ndata: {done_th}\n\n"
                yield f"event: stage\ndata: {json.dumps({'stage': 'synthesizing'})}\n\n"
                for word in fallback_summary.split(" "):
                    yield f"event: token\ndata: {json.dumps({'delta': word + ' '})}\n\n"
                full_answer = fallback_summary
        else:
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
    plan_res = await planner.plan_query_async(
        request.query,
        enable_web_search=request.enable_web_search,
        model_override=request.effective_model,
    )

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


class TimelineQueryRequest(BaseModel):
    """Request payload for narrative trajectory timeline generation."""

    query: str = Field(
        ...,
        min_length=2,
        description="Storyline topic to trace chronologically across broadsheets",
    )
    issue_ids: list[int] | None = Field(None, description="Optional issue IDs filter")
    model: str | None = Field(None, description="Optional model provider selection")
    model_override: str | None = Field(None, description="Optional model override")
    use_cache: bool = Field(True, description="Enable Redis response caching")

    @property
    def effective_model(self) -> str | None:
        """Return user-selected model identifier."""
        return self.model or self.model_override


@router.post(
    "/timeline",
    response_model=NarrativeTrajectoryResponse,
    summary="Generate cross-newspaper narrative trajectory and discrepancy analysis",
)
async def generate_timeline(
    request: TimelineQueryRequest,
) -> NarrativeTrajectoryResponse:
    """Construct a multi-perspective chronological story trajectory."""
    factory = get_session_factory()
    builder = TimelineBuilder(session_factory=factory)
    return await builder.build_narrative_trajectory(
        query=request.query,
        issue_ids=request.issue_ids,
        model_override=request.effective_model,
        use_cache=request.use_cache,
    )


@router.post(
    "/timeline/stream",
    summary="Stream timeline generation progress, milestones, and audit discrepancies via SSE",
)
async def stream_timeline(
    request: TimelineQueryRequest,
) -> StreamingResponse:
    """Stream real-time progress events and synthesized narrative milestones."""
    factory = get_session_factory()
    builder = TimelineBuilder(session_factory=factory)

    async def event_generator() -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        async def progress_hook(stage: str, data: dict[str, Any]) -> None:
            await queue.put((stage, data))

        async def run_builder() -> None:
            try:
                res = await builder.build_narrative_trajectory(
                    query=request.query,
                    issue_ids=request.issue_ids,
                    model_override=request.effective_model,
                    use_cache=request.use_cache,
                    on_progress=progress_hook,
                )
                await queue.put(("result", res.model_dump()))
            except Exception as err:
                await queue.put(("error", {"error": str(err)}))
            finally:
                await queue.put(("__DONE__", {}))

        task = asyncio.create_task(run_builder())

        while True:
            event_type, payload = await queue.get()
            if event_type == "__DONE__":
                break
            if event_type == "error":
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                break
            elif event_type == "result":
                yield f"event: result\ndata: {json.dumps(payload)}\n\n"
                yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"
            else:
                yield f"event: stage\ndata: {json.dumps({'stage': event_type, **payload})}\n\n"

        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/timeline/suggestions",
    summary="Get dynamic story trajectory topic suggestions from indexed articles and entities",
)
async def get_timeline_suggestions(
    limit: int = Query(6, ge=1, le=20, description="Max suggestions to return"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Dynamically generate timeline suggestions from prominent recent indexed headlines."""
    from app.models.article import Article
    from app.models.entity import Entity

    suggestions: list[str] = []

    # 1. Fetch prominent headlines from latest indexed articles
    stmt = (
        select(Article.headline)
        .where(
            Article.headline.is_not(None),
            Article.article_type.in_(["news", "editorial", "general"]),
        )
        .order_by(desc(Article.id))
        .limit(30)
    )
    res = await db.execute(stmt)
    raw_headlines = [h.strip() for h in res.scalars().all() if h and len(h.strip()) > 10]

    for h in raw_headlines:
        # Simplify long headlines for clean timeline query chips
        clean_h = h.split(":")[0].split(" - ")[0].split(" | ")[0].strip()
        if clean_h and clean_h not in suggestions and len(clean_h) <= 75:
            suggestions.append(clean_h)
        if len(suggestions) >= limit:
            break

    # 2. Fetch recurring entities/topics if we still need more suggestions
    if len(suggestions) < limit:
        entity_stmt = select(Entity.name).limit(10)
        ent_res = await db.execute(entity_stmt)
        for ent_name in ent_res.scalars().all():
            if ent_name and ent_name not in suggestions:
                suggestions.append(ent_name)
            if len(suggestions) >= limit:
                break

    # 3. Fallback defaults if archive is completely empty
    if not suggestions:
        suggestions = [
            "RBI monetary policy and interest rate trajectory",
            "Semiconductor manufacturing investments and subsidies",
            "Green hydrogen and renewable energy roadmap",
            "Foreign institutional investor capital flows",
        ]

    return {"suggestions": suggestions[:limit]}

