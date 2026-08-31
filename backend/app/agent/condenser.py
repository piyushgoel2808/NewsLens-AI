"""Conversational query condensation and coreference resolution engine."""

from __future__ import annotations

import contextlib
import re
from typing import Any

from app.core.logging import get_logger
from app.providers.base import ChatModelProvider, Message
from app.providers.registry import get_registry

logger = get_logger(__name__)

AMBIGUOUS_PRONOUNS_PATTERN = re.compile(
    r"\b(it|this|that|these|those|they|them|he|him|she|her|its|the article|the news|"
    r"the story|the company|the deal|the report|the issue|the paper|the event|the incident|"
    r"summarize it|more about this|tell me more|who was involved|what else|why did that happen|"
    r"what happened next|elaborate|explain it|give more details)\b",
    re.IGNORECASE,
)

GENERIC_FOLLOWUP_SHORT_PATTERN = re.compile(
    r"^(can you |please )?(summarize|summarise|explain|elaborate|expand|tell me more|more details|"
    r"what about it|who was involved|why|how|what happened|what else)( it| this| that| them)?\??$",
    re.IGNORECASE,
)

CLEAN_SESSION_CLARIFICATION_MESSAGE = (
    "Please specify which article, topic, or newspaper issue you would like me to summarize."
)


IN_CONTEXT_META_QUERY_PATTERN = re.compile(
    r"\b(which newspaper|what newspaper|what was the date|which date|what date|who wrote|"
    r"what source|which source|which paper|what paper|who is the author|give me the date|"
    r"from which news|from which paper|where was this published|when was this published|"
    r"who reported this|what are the sources|show sources|list the citations|what page was that|"
    r"which edition|who published this)\b",
    re.IGNORECASE,
)


def is_in_context_meta_query(query: str, chat_history: list[dict[str, Any]]) -> bool:
    """Detect if the user is asking directly about previous turn's sources, date, or metadata."""
    if not chat_history:
        return False
    q_clean = query.strip()
    return bool(IN_CONTEXT_META_QUERY_PATTERN.search(q_clean))


def needs_condensation(query: str, chat_history: list[dict[str, Any]]) -> bool:
    """Determine whether the query contains coreferences or follow-up ambiguity."""
    if not chat_history:
        return False

    if is_in_context_meta_query(query, chat_history):
        return False

    q_clean = query.strip()
    words = q_clean.split()

    # Very short follow-up phrases (< 6 words) or queries matching ambiguous pronouns
    if len(words) <= 6 and GENERIC_FOLLOWUP_SHORT_PATTERN.search(q_clean):
        return True

    return bool(AMBIGUOUS_PRONOUNS_PATTERN.search(q_clean))


def is_ambiguous_standalone_query(query: str, chat_history: list[dict[str, Any]]) -> bool:
    """Detect ungrounded ambiguous queries on clean sessions (e.g. 'summarize it' on turn 1)."""
    if chat_history and len(chat_history) > 0:
        return False

    q_clean = query.strip()
    words = q_clean.split()

    # Query has <= 6 words and matches follow-up patterns or pronouns without named entities
    if len(words) <= 6:
        if GENERIC_FOLLOWUP_SHORT_PATTERN.search(q_clean):
            return True
        if len(words) < 4 and AMBIGUOUS_PRONOUNS_PATTERN.search(q_clean):
            return True
        if q_clean.lower().strip("?.! ") in {
            "summarize",
            "summarise",
            "summarize it",
            "tell me more",
            "explain it",
            "what happened",
            "who is it",
            "who was involved",
            "details",
            "give details",
        }:
            return True

    return False


def extract_active_issue_from_history(
    chat_history: list[dict[str, Any]],
    current_query: str | None = None,
) -> dict[str, Any]:
    """Scan chat history for previously mentioned newspaper names, issue IDs, and dates.

    Safeguards against context leakage:
    - If current_query targets a cross-newspaper comparison or 'all newspapers', do not inherit a single newspaper.
    - If current_query specifies its own date or date range, do not inherit stale issue_id/newspaper from a different date.
    - If current_query specifies its own newspaper brand, do not inherit a different newspaper brand.
    """
    res: dict[str, Any] = {}
    if not chat_history:
        return res

    from app.agent.planner import _KNOWN_BRANDS_PATTERNS, extract_parameters_from_query

    q_lower = (current_query or "").lower()
    is_cross_newspaper = bool(
        re.search(r"\b(?:compa[a-z]*|contrast[a-z]*|diff(?:erence[s]?|ering)?|versus|vs\.?)\b", q_lower)
        or any(w in q_lower for w in ["all available", "all newspaper", "both newspaper", "across newspaper", "different newspaper"])
    )
    current_params = extract_parameters_from_query(current_query) if current_query else {}
    current_date = current_params.get("issue_date")
    current_np = current_params.get("newspaper_name")

    for turn in reversed(chat_history):
        content = str(turn.get("content", ""))
        params = extract_parameters_from_query(content)
        if params.get("newspaper_name") and not res.get("newspaper_name"):
            res["newspaper_name"] = params["newspaper_name"]
        if params.get("comparison_newspaper") and not res.get("comparison_newspaper"):
            res["comparison_newspaper"] = params["comparison_newspaper"]
        if params.get("target_newspapers") and not res.get("target_newspapers"):
            res["target_newspapers"] = params["target_newspapers"]
        if params.get("is_differential") and not res.get("is_differential"):
            res["is_differential"] = params["is_differential"]
        if params.get("issue_id") and not res.get("issue_id"):
            res["issue_id"] = params["issue_id"]
        if params.get("issue_date") and not res.get("issue_date"):
            res["issue_date"] = params["issue_date"]

        # 1. Inspect executive summary headers: e.g. "The Goan newspaper issue 94 (2026-08-02)"
        summary_m = re.search(
            r"([A-Za-z0-9\s\.\'\-]+?)\s+(?:newspaper\s+)?issue\s+(\d+)(?:\s*\((\d{4}-\d{2}-\d{2})\))?",
            content,
            re.IGNORECASE,
        )
        if summary_m:
            detected_pub = summary_m.group(1).strip().replace("⚡", "").replace("EXECUTIVE SUMMARY", "").strip()
            detected_pub = re.sub(r"^[\s\*\#\-\:]+", "", detected_pub).strip()
            if detected_pub and len(detected_pub) >= 3 and not res.get("newspaper_name"):
                for pat, brand in _KNOWN_BRANDS_PATTERNS:
                    if pat.search(detected_pub):
                        res["newspaper_name"] = brand
                        break
                if not res.get("newspaper_name") and len(detected_pub.split()) <= 4:
                    res["newspaper_name"] = detected_pub
            if summary_m.group(2) and not res.get("issue_id"):
                with contextlib.suppress(ValueError):
                    res["issue_id"] = int(summary_m.group(2))
            if summary_m.group(3) and not res.get("issue_date"):
                res["issue_date"] = summary_m.group(3)

        # 2. Inspect all known brand patterns across turn text
        if not res.get("newspaper_name"):
            for pat, brand in _KNOWN_BRANDS_PATTERNS:
                if pat.search(content):
                    res["newspaper_name"] = brand
                    break

        if res.get("newspaper_name") and res.get("issue_date"):
            break

    # Guardrail 1: If current query is cross-newspaper comparison, do NOT constrain to a single newspaper
    if is_cross_newspaper:
        res.pop("newspaper_name", None)
        res.pop("issue_id", None)

    # Guardrail 2: If current query explicitly provides its own date, invalidate stale context if from different date
    if current_date and res.get("issue_date") and res["issue_date"] != current_date:
        res.pop("newspaper_name", None)
        res.pop("issue_id", None)
        res.pop("issue_date", None)

    # Guardrail 3: If current query explicitly specifies its own newspaper, discard history newspaper
    if current_np:
        res.pop("newspaper_name", None)
        res.pop("issue_id", None)

    return res


def format_chat_history_for_prompt(chat_history: list[dict[str, Any]], max_turns: int = 5) -> str:
    """Format recent turns of chat history into clean dialog text with citations preserved."""
    recent = chat_history[-max_turns * 2 :]
    lines: list[str] = []
    for turn in recent:
        role = str(turn.get("role", "user")).capitalize()
        content = str(turn.get("content", "")).strip()
        # Keep up to 1000 chars per assistant turn to preserve citations and source names
        if role.lower() == "assistant" and len(content) > 1000:
            content = content[:1000] + "..."
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


_UNSET = object()


async def condense_conversational_query(
    query: str,
    chat_history: list[dict[str, Any]],
    provider: ChatModelProvider | None | Any = _UNSET,
    model_override: str | None = None,
    active_issue_id: int | None = None,
    active_newspaper_name: str | None = None,
    active_issue_date: str | None = None,
) -> str:
    """Rewrite a conversational follow-up query into a standalone search query with entities."""
    if not chat_history or not needs_condensation(query, chat_history):
        return query

    active_ctx = extract_active_issue_from_history(chat_history)
    np_name = active_newspaper_name or active_ctx.get("newspaper_name")
    iss_id = active_issue_id or active_ctx.get("issue_id")
    iss_date = active_issue_date or active_ctx.get("issue_date")

    # Resolve LLM provider (prefer lightweight/fast model like groq_llama or query_planner)
    resolved_provider: ChatModelProvider | None
    if provider is _UNSET:
        try:
            registry = get_registry()
            # Try to resolve lightweight query condenser provider first, falling back to query_planner
            try:
                resolved_provider = registry.get_chat_provider(model_override or "query_condenser")
            except Exception:
                resolved_provider = registry.get_chat_provider(model_override or "query_planner")
        except Exception as reg_err:
            logger.warning(
                "Could not obtain provider for query condensation",
                extra={"error": str(reg_err)},
            )
            resolved_provider = None
    else:
        resolved_provider = provider

    formatted_history = format_chat_history_for_prompt(chat_history)
    comp_np = active_ctx.get("comparison_newspaper")
    is_diff = active_ctx.get("is_differential")
    ctx_hint = ""
    if comp_np and np_name and is_diff:
        ctx_hint = f" Active Differential Comparison: {np_name} vs {comp_np} (Stories in {np_name} but not in {comp_np})" + (f" dated {iss_date}" if iss_date else "") + "."
    elif comp_np and np_name:
        ctx_hint = f" Active Publication Comparison: {np_name} vs {comp_np}" + (f" dated {iss_date}" if iss_date else "") + "."
    elif np_name or iss_id:
        ctx_hint = " Active Publication: " + (f"{np_name} " if np_name else "") + (f"(Issue {iss_id})" if iss_id else "") + (f" dated {iss_date}" if iss_date else "") + "."

    prompt = (
        f"Chat History:\n{formatted_history}\n\n"
        f"Latest User Query: {query}\n\n"
        f"Context Clue:{ctx_hint}\n"
        "Rewritten Standalone Query:"
    )

    if resolved_provider is not None:
        try:
            messages = [
                Message(
                    role="system",
                    content=(
                        "You are an expert search query reformulator. "
                        "Given the chat history and latest user query, rewrite the latest query into "
                        "a single, standalone sentence that contains all necessary context (entities, dates, page numbers). "
                        "Replace relative pronouns ('its', 'this', 'the newspaper', 'it') with the explicit "
                        "newspaper brand, issue ID, and date from the conversation context. "
                        "Output ONLY the rewritten query text. Do NOT add reasoning, bullets, quotes, or preambles."
                    ),
                ),
                Message(role="user", content=prompt),
            ]
            resp = await resolved_provider.complete(
                messages=messages,
                max_tokens=128,
                temperature=0.0,
            )
            rewritten = resp.text.strip()

            # Clean up any quotes or prefixes
            rewritten = re.sub(
                r'^(?:Standalone Query:|Rewritten Standalone Query:|Rewritten Query:|Query:|"|\'|`|\*|\-)\s*',
                "",
                rewritten,
                flags=re.IGNORECASE,
            ).strip()
            rewritten = re.sub(r'("|\'|`)$', "", rewritten).strip()

            # Verify that rewritten query is clean and resolved relative pronouns
            has_unresolved_pronoun = bool(re.search(r"\b(its|it|this\s+paper)\b", rewritten, re.I))
            is_echo = rewritten.lower() == query.lower()
            is_valid = len(rewritten) >= 3 and not (is_echo and (np_name or iss_id)) and not (has_unresolved_pronoun and (np_name or iss_id))

            if is_valid and "\n" not in rewritten and not rewritten.startswith("*"):
                logger.info(
                    "Conversational query condensed",
                    extra={"original_query": query, "condensed_query": rewritten},
                )
                return rewritten
        except Exception as e:
            logger.warning(
                "Conversational query condensation failed, falling back to heuristic resolution",
                extra={"error": str(e), "query": query},
            )

    # Deterministic Heuristic Coreference Fallback:
    if comp_np and np_name and is_diff:
        dt_suffix = f" dated {iss_date}" if iss_date else ""
        q_resolved = f"{query} in {np_name} but not in {comp_np}{dt_suffix}"
        logger.info(
            "Conversational query resolved via deterministic differential context heuristic",
            extra={"original_query": query, "resolved_query": q_resolved},
        )
        return q_resolved

    if comp_np and np_name:
        dt_suffix = f" dated {iss_date}" if iss_date else ""
        q_resolved = f"{query} comparing {np_name} and {comp_np}{dt_suffix}"
        logger.info(
            "Conversational query resolved via deterministic comparison context heuristic",
            extra={"original_query": query, "resolved_query": q_resolved},
        )
        return q_resolved

    if np_name or iss_id:
        q_resolved = query
        parts = []
        if np_name:
            parts.append(np_name)
        if iss_id:
            parts.append(f"issue {iss_id}")
        if iss_date:
            parts.append(f"dated {iss_date}")
        pub_desc = " ".join(parts)

        # Replace pronouns: e.g. "list all its sports related news" -> "list all sports related news from The Goan issue 94 dated 2026-08-02"
        if re.search(r"\b(?:all\s+)?its\b", q_resolved, re.I):
            q_resolved = re.sub(r"\b(?:all\s+)?its\s+(.+)$", rf"all \1 from {pub_desc}", q_resolved, flags=re.I)
            if pub_desc not in q_resolved:
                q_resolved = re.sub(r"\bits\b", f"{pub_desc}'s", q_resolved, flags=re.I)

        q_resolved = re.sub(r"\b(?:this\s+paper's|this\s+newspaper's|the\s+paper's|the\s+newspaper's)\b", f"{pub_desc}'s", q_resolved, flags=re.I)
        q_resolved = re.sub(r"\b(?:in\s+its|in\s+this\s+paper|in\s+this\s+newspaper|in\s+the\s+paper|in\s+the\s+newspaper|in\s+this\s+issue|in\s+the\s+issue)\b", f"in {pub_desc}", q_resolved, flags=re.I)
        q_resolved = re.sub(r"\b(?:from\s+its|from\s+this\s+paper|from\s+this\s+newspaper|from\s+the\s+paper|from\s+the\s+newspaper|from\s+this\s+issue|from\s+the\s+issue)\b", f"from {pub_desc}", q_resolved, flags=re.I)

        if pub_desc not in q_resolved:
            q_resolved = f"{q_resolved} from {pub_desc}"

        logger.info(
            "Conversational query resolved via deterministic active context heuristic",
            extra={"original_query": query, "resolved_query": q_resolved},
        )
        return q_resolved

    return query

