"""Conversational query condensation and coreference resolution engine."""

from __future__ import annotations

import re
from typing import Any

from app.agent.extractor import (
    extract_parameters_from_query,
)
from app.agent.history_context import (
    AMBIGUOUS_PRONOUNS_PATTERN,
    CLEAN_SESSION_CLARIFICATION_MESSAGE,
    GENERIC_FOLLOWUP_SHORT_PATTERN,
    IN_CONTEXT_META_QUERY_PATTERN,
    extract_active_issue_from_history,
    format_chat_history_for_prompt,
    is_ambiguous_standalone_query,
    is_in_context_meta_query,
    needs_condensation,
    parse_inline_citation,
)
from app.core.logging import get_logger
from app.providers.base import Message
from app.providers.registry import get_registry
from app.retrieval.asset_resolver import (
    resolve_attached_asset_context,
    resolve_authoritative_article_id,
)

logger = get_logger(__name__)

CONDENSER_SYSTEM_PROMPT = (
    "You are an expert conversational query reformulator and coreference resolver for a newspaper archive research system.\n"
    "Given the active context, chat history, and latest user query, analyze whether the user is continuing the conversation or shifting topics, "
    "and rewrite the latest query into a single, standalone sentence containing all necessary context (entities, dates, page numbers).\n\n"
    "DECISION PRINCIPLES (Analyze and decide what to keep vs. what to drop):\n"
    "1. CONVERSATION CONTINUATION & PRONOUN RESOLUTION:\n"
    "   - When the user refers to prior results ('those news', 'list all those news', 'show them', 'the articles', 'it', 'its'):\n"
    "     The user is continuing the current topic. Inherit the active newspaper, issue date, and page from the active context.\n"
    "     Example: 'list all those news' (Context: Hindustan Times, 2026-09-03, Page 5)\n"
    "     Rewrite: 'List all news articles on page 5 of Hindustan Times dated 2026-09-03'\n\n"
    "2. CONTINUATION WITH NEW OR MODIFIED CONSTRAINTS:\n"
    "   - When the user asks a follow-up with a new constraint ('what about page 6', 'on page 5', 'in business section', 'calculate its avg word count'):\n"
    "     Inherit the active newspaper and issue date from the conversation, and apply the newly requested constraint.\n"
    "     Example: 'list all those news on page 6' (Context: Hindustan Times, 2026-09-03, previously Page 5)\n"
    "     Rewrite: 'List all news articles on page 6 of Hindustan Times dated 2026-09-03'\n"
    "     Example: 'calculate its avg word count of articles' (Context: Hindustan Times, 2026-09-03, Page 6)\n"
    "     Rewrite: 'Calculate the average word count of articles on page 6 of Hindustan Times dated 2026-09-03'\n\n"
    "3. EXPLICIT CONSTRAINT PRESERVATION:\n"
    "   - NEVER drop or strip explicit filters provided by the user. If the user explicitly mentions a page number (e.g. 'on page 5'), a date, a section, or an author, that constraint MUST appear in the rewritten query.\n\n"
    "4. INDEPENDENT QUERY / TOPIC SHIFT:\n"
    "   - If the user introduces an entirely new subject, entity, date, or publication unrelated to the previous discussion (e.g. 'crime in Panaji', 'who won the election?', 'weather forecast'),\n"
    "     do NOT carry forward previous newspapers, dates, or pages. Keep the query unconstrained.\n"
    "     Example: 'show me weather reports' (Context: The Guardian, 2026-04-12)\n"
    "     Rewrite: 'Show me weather reports'\n\n"
    "5. ATTACHED ASSET REFERENCE / BINDING:\n"
    "   - If a 'CURRENT WORKSPACE ATTACHED ASSET' is provided and referenced by the query ('this photo', 'explain this', 'who is this person'):\n"
    "     Ground the query solely in that attached asset (its newspaper, date, headline, photo ID, caption) and DO NOT inherit or contaminate with prior history.\n"
    "     Example: 'explain this photo' (Attached: Photo #1042 in Financial Times, 2026-05-10)\n"
    "     Rewrite: 'Explain photo #1042 in Financial Times dated 2026-05-10'\n\n"
    "6. COMPARATIVE CONTINUATION:\n"
    "   - If prior turns compared newspapers and the user asks a follow-up ('list all those similar stories', 'show exclusives'):\n"
    "     Preserve both newspaper names, the date, and the comparison intent.\n"
    "     Example: 'list all those similar stories' (Context: The Indian Express and The Hindu, 2026-05-20)\n"
    "     Rewrite: 'List all similar and shared articles between The Indian Express and The Hindu dated 2026-05-20'\n\n"
    "7. ARTICLE SUMMARY / EXPLANATION FOLLOW-UP:\n"
    "   - When the user asks 'summarise it', 'explain it', 'what happened', 'tell me more' after discussing or referencing a specific broadsheet article:\n"
    "     Rewrite the query to explicitly name the article headline, newspaper, and date from the active context or cited articles.\n"
    "     Example: 'summarise it' (Context: Adani leads rush for 37,500 cr coal gasification plan, Hindustan Times, 2026-09-09)\n"
    "     Rewrite: 'Explain article \"Adani leads rush for 37,500 cr coal gasification plan\" in Hindustan Times dated 2026-09-09'\n\n"
    "OUTPUT FORMAT: Output ONLY the plain rewritten query text. Do NOT include quotes, explanations, prefixes, or bullet points."
)



_UNSET = object()


async def condense_conversational_query(
    query: str,
    chat_history: list[dict[str, Any]],
    provider: Any = _UNSET,
    model_override: str | None = None,
    active_issue_id: int | None = None,
    active_newspaper_name: str | None = None,
    active_issue_date: str | None = None,
    attached_asset: dict[str, Any] | None = None,
) -> str:
    """Rewrite a conversational follow-up query into a standalone search query with entities."""
    # Tier 1 Bypass: If query is clean and standalone, return immediately
    if not chat_history and not attached_asset:
        return query
    if not needs_condensation(query, chat_history, attached_asset=attached_asset):
        return query

    # Extract structured workspace attributes
    asset_photo_id = attached_asset.get("photo_id") if attached_asset else None
    asset_art_id = attached_asset.get("article_id") if attached_asset else None
    asset_np = attached_asset.get("newspaper_name") if attached_asset else None
    asset_date = attached_asset.get("issue_date") if attached_asset else None
    asset_iss_id = attached_asset.get("issue_id") if attached_asset else None
    asset_hl = attached_asset.get("headline") if attached_asset else None
    asset_cap = attached_asset.get("caption") if attached_asset else None
    asset_type = attached_asset.get("visual_type") if attached_asset else None
    asset_page = attached_asset.get("page_number") if attached_asset else None

    # Check conflicts between current query and attached asset
    q_params = extract_parameters_from_query(query) if query else {}
    q_cite = parse_inline_citation(query or "")
    explicit_q_date = q_params.get("issue_date") or q_cite.get("issue_date")
    explicit_q_np = q_params.get("newspaper_name") or q_cite.get("newspaper_name")

    has_date_conflict = bool(explicit_q_date and asset_date and explicit_q_date != asset_date)
    has_np_conflict = bool(
        explicit_q_np
        and asset_np
        and explicit_q_np.lower() not in asset_np.lower()
        and asset_np.lower() not in explicit_q_np.lower()
    )

    if attached_asset and (asset_photo_id or asset_art_id) and not has_date_conflict and not has_np_conflict:
        np_name = asset_np or active_newspaper_name
        iss_id = asset_iss_id or active_issue_id
        iss_date = asset_date or active_issue_date
        active_hl = asset_hl
        active_pg = asset_page
        comp_np = None
        is_shared = False
        is_diff = False
    else:
        active_ctx = extract_active_issue_from_history(
            chat_history,
            current_query=query,
            attached_photo_id=asset_photo_id if not has_date_conflict and not has_np_conflict else None,
            attached_article_id=asset_art_id if not has_date_conflict and not has_np_conflict else None,
            attached_issue_date=asset_date if not has_date_conflict and not has_np_conflict else None,
            attached_newspaper_name=asset_np if not has_date_conflict and not has_np_conflict else None,
            attached_headline=asset_hl if not has_date_conflict and not has_np_conflict else None,
        )
        np_name = explicit_q_np or active_newspaper_name or active_ctx.get("newspaper_name")
        iss_id = active_issue_id or active_ctx.get("issue_id")
        iss_date = explicit_q_date or active_issue_date or active_ctx.get("issue_date")
        comp_np = active_ctx.get("comparison_newspaper")
        is_diff = active_ctx.get("is_differential")
        is_shared = active_ctx.get("is_shared")
        active_hl = active_ctx.get("headline")
        active_pg = active_ctx.get("page_number")

    # Deterministic Ordinal Article Coreference Resolution (e.g. "tell me about article 1", "first article")
    ord_m = re.search(
        r"\b(?:this\s+|the\s+)?article\s*(\d{1,2})\b|\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+article\b|\b(first|second|third|fourth|fifth)\s+article\b",
        query,
        re.I,
    )
    if ord_m and chat_history:
        target_idx: int | None = None
        if ord_m.group(1):
            target_idx = int(ord_m.group(1))
        elif ord_m.group(2):
            target_idx = int(ord_m.group(2))
        elif ord_m.group(3):
            _word_to_num = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
            target_idx = _word_to_num.get(ord_m.group(3).lower())

        if target_idx and target_idx >= 1:
            for turn in reversed(chat_history):
                if turn.get("role") == "assistant" and turn.get("citations"):
                    cits = [c for c in turn["citations"] if isinstance(c, dict) and c.get("headline")]
                    if len(cits) >= target_idx:
                        target_cit = cits[target_idx - 1]
                        c_hl = target_cit.get("headline", "").strip()
                        c_np = target_cit.get("newspaper_name", np_name or "").strip()
                        c_dt = target_cit.get("issue_date", iss_date or "").strip()
                        dt_part = f" dated {c_dt}" if c_dt else ""
                        np_part = f" in {c_np}" if c_np else ""
                        return f'Tell me about the article "{c_hl}"{np_part}{dt_part}'

    # Resolve LLM provider
    resolved_provider: Any
    if provider is _UNSET:
        from app.agent.llm_helpers import resolve_provider_candidates

        cands = resolve_provider_candidates(
            task_name="query_condenser",
            model_override=model_override,
            fallback_task_keys=("query_planner",),
            prefer_local=False,
            registry_getter=get_registry,
        )
        resolved_provider = cands[0] if cands else None
    else:
        resolved_provider = provider

    # Assemble Attached Asset Context Block
    attached_asset_block = ""
    if (
        attached_asset
        and not has_date_conflict
        and not has_np_conflict
        and (asset_photo_id or asset_art_id or asset_cap or asset_hl)
    ):
        asset_lines: list[str] = []
        if asset_photo_id:
            asset_lines.append(f"- Asset Type: Photo (ID: #{asset_photo_id})")
        elif asset_art_id:
            asset_lines.append(f"- Asset Type: Article (ID: #{asset_art_id})")
        if asset_type:
            asset_lines.append(f"- Visual Classification: {asset_type}")
        if np_name:
            asset_lines.append(f"- Newspaper: {np_name}")
        if iss_date:
            date_str = f"- Issue Date: {iss_date}"
            if active_pg:
                date_str += f" (Page {active_pg})"
            asset_lines.append(date_str)
        if active_hl:
            asset_lines.append(f"- Associated Article Headline: \"{active_hl}\"")
        if asset_cap:
            asset_lines.append(f"- Caption / Text: \"{asset_cap}\"")

        attached_asset_block = "CURRENT WORKSPACE ATTACHED ASSET:\n" + "\n".join(asset_lines) + "\n\n"

    # Assemble Active Context Block from Conversation (when not bound to attached asset)
    active_context_block = ""
    if not attached_asset or has_date_conflict or has_np_conflict:
        ctx_lines = []
        if np_name:
            ctx_lines.append(f"- Active Newspaper: {np_name}")
        if iss_date:
            ctx_lines.append(f"- Active Issue Date: {iss_date}")
        if active_pg:
            ctx_lines.append(f"- Active Focus Page: {active_pg}")
        if active_hl:
            ctx_lines.append(f"- Active Focus Headline: \"{active_hl}\"")
        if ctx_lines:
            active_context_block = "ACTIVE CONVERSATION CONTEXT:\n" + "\n".join(ctx_lines) + "\n\n"

    # Assemble Conversation History Block
    is_attached_query = bool(attached_asset and (asset_photo_id or asset_art_id or asset_hl))
    if is_attached_query and chat_history and (has_date_conflict or has_np_conflict):
        formatted_history = ""
    else:
        formatted_history = format_chat_history_for_prompt(chat_history) if chat_history else ""

    history_block = f"CONVERSATION HISTORY:\n{formatted_history}\n\n" if formatted_history else ""

    prompt = (
        f"{attached_asset_block}"
        f"{active_context_block}"
        f"{history_block}"
        f"LATEST USER QUERY: {query}\n\n"
        "Rewritten Standalone Query:"
    )

    # Invoke LLM Reformulator
    if resolved_provider is not None:
        try:
            messages = [
                Message(role="system", content=CONDENSER_SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ]
            complete_kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.0,
            }
            if getattr(resolved_provider, "provider_name", "") == "gemini":
                complete_kwargs["thinking_budget"] = 0
            resp = await resolved_provider.complete(**complete_kwargs)
            rewritten = resp.text.strip()

            # Lightweight Normalizer: Clean quotes and conversational prefixes
            rewritten = re.sub(
                r'^(?:Standalone Query:|Rewritten Standalone Query:|Rewritten Query:|Query:|"|\'|`|\*|\-)\s*',
                "",
                rewritten,
                flags=re.IGNORECASE,
            ).strip()
            rewritten = re.sub(r'("|\'|`)$', "", rewritten).strip()

            q_words = query.strip().split()
            rewritten_words = rewritten.strip().split()
            is_truncated = (len(q_words) >= 6 and len(rewritten_words) <= 2) or (
                len(rewritten_words) <= 2
                and rewritten.lower().strip("?.! ") in {"tell", "explain", "summarize", "summarise", "it", "more", "details", "brief"}
            )

            has_unresolved_pronoun = bool(re.search(r"\b(its|it|this\s+paper)\b", rewritten, re.I))
            is_echo = rewritten.lower() == query.lower()
            is_valid = (
                len(rewritten) >= 3
                and not is_truncated
                and not (is_echo and (np_name or iss_id))
                and not (has_unresolved_pronoun and (np_name or iss_id))
            )

            if is_valid and "\n" not in rewritten and not rewritten.startswith("*"):
                logger.info(
                    "Conversational query condensed",
                    extra={"original_query": query, "condensed_query": rewritten},
                )
                return rewritten
        except Exception as e:
            logger.warning(
                "Conversational query condensation LLM call failed, falling back to safe normalizer",
                extra={"error": str(e), "query": query},
            )

    # Safe Fallback Normalizer (Zero Destructive String Suffixing)
    if attached_asset and (asset_photo_id or asset_art_id or asset_hl):
        pub_desc = " ".join([p for p in [np_name, f"dated {iss_date}" if iss_date else ""] if p])
        if asset_photo_id and (
            re.search(r"\b(photo|picture|image|graphic|figure|chart|table|person|who is)\b", query, re.I)
            or asset_type in ("data_chart", "infographic", "table")
        ):
            v_desc = f"{asset_type.replace('_', ' ') if asset_type else 'visual asset'} #{asset_photo_id}"
            if active_hl:
                return f'Explain {v_desc} for article "{active_hl}" in {pub_desc}'.strip()
            return f"Explain {v_desc} in {pub_desc}".strip()
        elif asset_art_id or asset_hl:
            target_hl = active_hl or f"Article #{asset_art_id}"
            return f'Explain article "{target_hl}" in {pub_desc}'.strip()

    # Article explanation / summarization follow-up with active headline from conversation
    if active_hl and (
        GENERIC_FOLLOWUP_SHORT_PATTERN.search(query.strip())
        or AMBIGUOUS_PRONOUNS_PATTERN.search(query.strip())
        or any(w in query.lower() for w in ["summarise", "summarize", "explain", "it", "takeaways"])
    ):
        pub_parts = []
        if np_name:
            pub_parts.append(np_name)
        if iss_date:
            pub_parts.append(f"dated {iss_date}")
        pub_desc = f" in {' '.join(pub_parts)}" if pub_parts else ""
        return f'Explain article "{active_hl}"{pub_desc}'.strip()

    if comp_np and np_name and is_diff:
        dt_suffix = f" dated {iss_date}" if iss_date else ""
        return f"{query} in {np_name} but not in {comp_np}{dt_suffix}"

    if comp_np and np_name:
        dt_suffix = f" dated {iss_date}" if iss_date else ""
        is_shared_mode = is_shared or any(w in query.lower() for w in ["similar", "shared", "common", "same article", "same stories", "both"])
        rel_verb = "between" if is_shared_mode else "comparing"
        return f"{query} {rel_verb} {np_name} and {comp_np}{dt_suffix}"

    # Pronoun resolution with active broadsheet context
    if (np_name or iss_id) and re.search(r"\b(?:all\s+)?its\b", query, re.I):
        pub_parts = [np_name]
        if iss_id:
            pub_parts.append(f"issue {iss_id}")
        if iss_date:
            pub_parts.append(f"dated {iss_date}")
        pub_desc = " ".join([p for p in pub_parts if p])
        return re.sub(r"\b(?:all\s+)?its\s+(.+)$", rf"all \1 from {pub_desc}", query, flags=re.I)

    # Safe return without naive concatenation
    return query


__all__ = [
    "AMBIGUOUS_PRONOUNS_PATTERN",
    "CLEAN_SESSION_CLARIFICATION_MESSAGE",
    "CONDENSER_SYSTEM_PROMPT",
    "GENERIC_FOLLOWUP_SHORT_PATTERN",
    "IN_CONTEXT_META_QUERY_PATTERN",
    "condense_conversational_query",
    "extract_active_issue_from_history",
    "format_chat_history_for_prompt",
    "is_ambiguous_standalone_query",
    "is_in_context_meta_query",
    "needs_condensation",
    "parse_inline_citation",
    "resolve_attached_asset_context",
    "resolve_authoritative_article_id",
]
