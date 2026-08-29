"""Conversational query condensation and coreference resolution engine."""

from __future__ import annotations

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


async def condense_conversational_query(
    query: str,
    chat_history: list[dict[str, Any]],
    provider: ChatModelProvider | None = None,
    model_override: str | None = None,
) -> str:
    """Rewrite a conversational follow-up query into a standalone search query with entities."""
    if not chat_history or not needs_condensation(query, chat_history):
        return query

    # Resolve LLM provider (prefer lightweight/fast model like groq_llama or query_planner)
    if provider is None:
        try:
            registry = get_registry()
            # Try to resolve lightweight query condenser provider first, falling back to query_planner
            try:
                provider = registry.get_chat_provider(model_override or "query_condenser")
            except Exception:
                provider = registry.get_chat_provider(model_override or "query_planner")
        except Exception as reg_err:
            logger.warning(
                "Could not obtain provider for query condensation",
                extra={"error": str(reg_err)},
            )
            provider = None

    if provider is None:
        return query

    formatted_history = format_chat_history_for_prompt(chat_history)
    prompt = (
        f"Chat History:\n{formatted_history}\n\n"
        f"Latest User Query: {query}\n\n"
        "Rewritten Standalone Query:"
    )

    try:
        messages = [
            Message(
                role="system",
                content=(
                    "Given the chat history and the latest user query, rewrite the latest query into "
                    "a single, standalone sentence that contains all necessary context (entities, dates, page numbers). "
                    "Do not answer it, just rewrite it."
                ),
            ),
            Message(role="user", content=prompt),
        ]
        resp = await provider.complete(
            messages=messages,
            max_tokens=128,
            temperature=0.0,
        )
        rewritten = resp.text.strip()

        # Clean up any quotes or prefixes
        rewritten = re.sub(
            r'^(?:Standalone Query:|Rewritten Standalone Query:|Rewritten Query:|Query:|"|\'|`)\s*',
            "",
            rewritten,
            flags=re.IGNORECASE,
        ).strip()
        rewritten = re.sub(r'("|\'|`)$', "", rewritten).strip()

        if rewritten and len(rewritten) >= 3:
            logger.info(
                "Conversational query condensed",
                extra={"original_query": query, "condensed_query": rewritten},
            )
            return rewritten
    except Exception as e:
        logger.warning(
            "Conversational query condensation failed, falling back to original query",
            extra={"error": str(e), "query": query},
        )

    return query
