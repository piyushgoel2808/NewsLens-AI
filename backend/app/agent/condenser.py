"""Conversational query condensation and coreference resolution engine."""

from __future__ import annotations

import contextlib
import re
from typing import Any

from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    extract_parameters_from_query,
    is_archive_wide_newspaper_query,
)
from app.core.logging import get_logger
from app.providers.base import Message
from app.providers.registry import get_registry

logger = get_logger(__name__)

AMBIGUOUS_PRONOUNS_PATTERN = re.compile(
    r"\b(it|this|that|these|those|they|them|he|him|she|her|its|their|"
    r"the article|the news|the story|the company|the deal|the report|the issue|the paper|the event|the incident|"
    r"the front page|the frontpage|the headline|the headlines|"
    r"this photo|the photo|this image|the image|this picture|the picture|this chart|the chart|in the photo|in this photo|"
    r"who is this|who is in|who is the person|what is the name of the person|"
    r"summarize it|more about this|tell me more|who was involved|what else|why did that happen|"
    r"what happened next|elaborate|explain it|give more details|"
    r"in there|from there|out there|all of them|all there|all their)\b",
    re.IGNORECASE,
)

GENERIC_FOLLOWUP_SHORT_PATTERN = re.compile(
    r"^(can you |please )?(summarize|summarise|explain|elaborate|expand|tell me more|more details|"
    r"what about it|who was involved|why|how|what happened|what else|list all|show all|list|show)( it| this| that| them| there| their)?\??$",
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


def needs_condensation(
    query: str,
    chat_history: list[dict[str, Any]],
    attached_asset: dict[str, Any] | None = None,
) -> bool:
    """Determine whether the query contains coreferences or follow-up ambiguity."""
    if not chat_history and not attached_asset:
        return False

    if is_in_context_meta_query(query, chat_history):
        return False

    q_clean = query.strip()
    words = q_clean.split()

    # If an attached asset is present, check if the query refers to it or contains pronouns
    if attached_asset and (attached_asset.get("photo_id") or attached_asset.get("article_id")):
        if AMBIGUOUS_PRONOUNS_PATTERN.search(q_clean):
            return True
        if re.search(r"\b(photo|picture|image|graphic|figure|chart|article|headline|story|person|who is)\b", q_clean, re.I):
            return True
        if len(words) <= 6 and GENERIC_FOLLOWUP_SHORT_PATTERN.search(q_clean):
            return True

    # Check conversation follow-ups
    if chat_history:
        if len(words) <= 6 and GENERIC_FOLLOWUP_SHORT_PATTERN.search(q_clean):
            return True
        if AMBIGUOUS_PRONOUNS_PATTERN.search(q_clean):
            return True

    return False


def is_ambiguous_standalone_query(
    query: str,
    chat_history: list[dict[str, Any]],
    has_attached_asset: bool = False,
) -> bool:
    """Detect ungrounded ambiguous queries on clean sessions (e.g. 'summarize it' on turn 1)."""
    if has_attached_asset:
        return False
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


def parse_inline_citation(text: str) -> dict[str, Any]:
    """Extract newspaper name, issue date, page number, and headline from inline citations.

    Handles formats like:
    - [4] Hindustan Times, 2026-09-10, Page 4, Headline: "The growing bipolarity..."
    - [{Hindustan Times}, 2026-09-10, Page 4, "The growing bipolarity..."]
    - Hindustan Times, 2026-09-10, Page 4, Headline: "..."
    """
    if not text:
        return {}
    m = re.search(
        r"(?:\[(?:\d+|photo)?\]\s*)?"
        r"(?:\{?([A-Za-z0-9\s\.\'\-]+?)\}?,\s*)"
        r"(?:(\d{4}-\d{2}-\d{2}),\s*)?"
        r"(?:Page\s*(\d+),\s*)?"
        r"(?:Headline:\s*)?[\"“]([^\"”]+)[\"”]",
        text,
        re.IGNORECASE,
    )
    if m:
        np = m.group(1).strip()
        for pat, brand in _KNOWN_BRANDS_PATTERNS:
            if pat.search(np):
                np = brand
                break
        res: dict[str, Any] = {"newspaper_name": np, "target_newspapers": [np]}
        if m.group(2):
            res["issue_date"] = m.group(2).strip()
        if m.group(3):
            with contextlib.suppress(ValueError):
                res["page_number"] = int(m.group(3).strip())
        if m.group(4):
            res["headline"] = m.group(4).strip()
        return res

    # Flexible pattern: article "Headline" [in Newspaper] [dated YYYY-MM-DD]
    hl_m = re.search(r"(?:article|story|headline|titled|report)?\s*[\"“]([^\"”]{8,150})[\"”]", text, re.I)
    if hl_m:
        cand_hl = hl_m.group(1).strip()
        if not any(pat.fullmatch(cand_hl) for pat, _ in _KNOWN_BRANDS_PATTERNS):
            res_flex: dict[str, Any] = {"headline": cand_hl}
            from app.agent.extractor import extract_parameters_from_query
            q_ext = extract_parameters_from_query(text)
            if q_ext.get("newspaper_name"):
                res_flex["newspaper_name"] = q_ext["newspaper_name"]
                res_flex["target_newspapers"] = [q_ext["newspaper_name"]]
            if q_ext.get("issue_date"):
                res_flex["issue_date"] = q_ext["issue_date"]
            return res_flex

    return {}


def extract_active_issue_from_history(
    chat_history: list[dict[str, Any]],
    current_query: str | None = None,
    attached_photo_id: int | None = None,
    attached_article_id: int | None = None,
    attached_issue_date: str | None = None,
    attached_newspaper_name: str | None = None,
    attached_headline: str | None = None,
) -> dict[str, Any]:
    """Scan chat history for previously mentioned newspaper names, issue IDs, and dates.

    Safeguards against context leakage:
    - If current_query targets a cross-newspaper comparison or 'all newspapers', do not inherit a single newspaper.
    - If current_query specifies its own date or date range, do not inherit stale issue_id/newspaper/article from a different date.
    - If current_query specifies its own newspaper brand, do not inherit a different newspaper brand or article.
    - If current_query specifies an inline citation or explicit headline, that citation is authoritative.
    - If an attached photo or article is explicitly supplied on the current turn, its issue, newspaper, and headline are authoritative.
    """
    res: dict[str, Any] = {}
    current_citation = parse_inline_citation(current_query or "")
    if current_citation:
        res.update(current_citation)

    current_params = extract_parameters_from_query(current_query) if current_query else {}
    explicit_query_date = current_params.get("issue_date") or current_citation.get("issue_date")
    explicit_query_np = current_params.get("newspaper_name") or current_citation.get("newspaper_name")
    explicit_query_hl = current_params.get("headline") or current_citation.get("headline")

    has_date_conflict = bool(explicit_query_date and attached_issue_date and explicit_query_date != attached_issue_date)
    has_np_conflict = bool(
        explicit_query_np
        and attached_newspaper_name
        and explicit_query_np.lower() not in attached_newspaper_name.lower()
        and attached_newspaper_name.lower() not in explicit_query_np.lower()
    )

    has_headline_conflict = False
    if attached_headline and current_query:
        if explicit_query_hl and explicit_query_hl.lower() != attached_headline.lower():
            has_headline_conflict = True
        else:
            def _sig_tokens(s: str) -> set[str]:
                stop = {
                    "the", "a", "an", "and", "or", "in", "on", "at", "for", "to", "of", "with", "by", "from",
                    "is", "are", "was", "were", "this", "that", "these", "those", "about", "article", "story",
                    "news", "report", "photo", "image", "picture", "visual", "who", "what", "where", "when",
                    "why", "how", "person", "people",
                }
                return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in stop}
            att_tokens = _sig_tokens(attached_headline)
            q_tokens = _sig_tokens(current_query)
            is_pronoun_ref = bool(
                re.search(
                    r"\b(?:this|that|the|it|ir)\s+(?:article|story|news|report|headline|piece|photo|image|picture|graphic|figure|visual|table|chart|infographic)\b"
                    r"|\b(?:tell me more|explain this|explain it|what is this|who is this|who is the person|who is in this)\b"
                    r"|\b(?:find|read|get|tell|show|explain|summ[ae]ri[sz]e)\s+(?:about|on|for|of)?\s*(?:this|the|that|it|ir)?\s*(?:article|story|news|report|headline|piece)?\b"
                    r"|\bin\s+\d+\s+words?\b",
                    current_query,
                    re.I,
                )
            )
            if (attached_photo_id or attached_article_id) and re.search(r"\b(?:photo|image|picture|visual|this article|this photo|it\b|that\b|the article)\b", current_query, re.I):
                is_pronoun_ref = True
            if not is_pronoun_ref and att_tokens and q_tokens and not (att_tokens & q_tokens):
                has_headline_conflict = True

    if not has_date_conflict and not has_np_conflict and not has_headline_conflict:
        if attached_photo_id:
            with contextlib.suppress(ValueError, TypeError):
                res["photo_id"] = int(attached_photo_id)
        if attached_article_id:
            with contextlib.suppress(ValueError, TypeError):
                res["article_id"] = int(attached_article_id)
        if attached_issue_date:
            res["issue_date"] = attached_issue_date
        if attached_newspaper_name:
            res["newspaper_name"] = attached_newspaper_name
        if attached_headline:
            res["headline"] = attached_headline
    else:
        if explicit_query_date:
            res["issue_date"] = explicit_query_date
        if explicit_query_np:
            res["newspaper_name"] = explicit_query_np
        if explicit_query_hl:
            res["headline"] = explicit_query_hl

    if not chat_history:
        return res

    has_explicit_asset = bool(
        (attached_photo_id or attached_article_id or attached_headline)
        and not has_date_conflict
        and not has_np_conflict
        and not has_headline_conflict
    )

    q_lower = (current_query or "").lower()
    is_cross_newspaper = bool(
        is_archive_wide_newspaper_query(current_query)
        or re.search(r"\b(?:compa[a-z]*|contrast[a-z]*|diff(?:erence[s]?|ering)?|versus|vs\.?)\b", q_lower)
        or any(w in q_lower for w in ["all available", "all newspaper", "both newspaper", "across newspaper", "different newspaper"])
        or re.search(r"\b(?:no|number|count|how many|which|list|total)\s+(?:of\s+)?newspapers?\b", q_lower)
    )
    current_date = explicit_query_date or (attached_issue_date if not has_date_conflict else None)
    current_np = explicit_query_np or (attached_newspaper_name if not has_np_conflict else None)

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
        if params.get("is_shared") and not res.get("is_shared"):
            res["is_shared"] = params["is_shared"]
        if params.get("issue_id") and not res.get("issue_id"):
            res["issue_id"] = params["issue_id"]
        if params.get("issue_date") and not res.get("issue_date"):
            res["issue_date"] = params["issue_date"]

        # 0. Inspect structured citations and attached assets on the turn
        # If current_query provided its own explicit headline/citation or an asset is explicitly attached on current turn,
        # do NOT inherit foreign article/photo IDs or headlines from prior turns
        if not current_citation.get("headline") and not explicit_query_hl and not has_explicit_asset:
            turn_citations = turn.get("citations") or []
            for cit in turn_citations:
                if isinstance(cit, dict):
                    if cit.get("article_id") and not res.get("article_id"):
                        with contextlib.suppress(ValueError):
                            res["article_id"] = int(cit["article_id"])
                    if cit.get("headline") and not res.get("headline"):
                        res["headline"] = str(cit["headline"]).strip()
                    if cit.get("newspaper_name") and not res.get("newspaper_name"):
                        res["newspaper_name"] = str(cit["newspaper_name"]).strip()
                    if cit.get("issue_date") and not res.get("issue_date"):
                        res["issue_date"] = str(cit["issue_date"]).strip()
                    if cit.get("photo_id") and not res.get("photo_id"):
                        with contextlib.suppress(ValueError):
                            res["photo_id"] = int(cit["photo_id"])
                    if cit.get("pages") and not res.get("page_number"):
                        with contextlib.suppress(Exception):
                            res["page_number"] = int(cit["pages"][0])

            attached_asset = turn.get("attachedAsset") or {}
            if isinstance(attached_asset, dict):
                if attached_asset.get("articleId") and not res.get("article_id"):
                    with contextlib.suppress(ValueError):
                        res["article_id"] = int(attached_asset["articleId"])
                if attached_asset.get("photoId") and not res.get("photo_id"):
                    with contextlib.suppress(ValueError):
                        res["photo_id"] = int(attached_asset["photoId"])
                if attached_asset.get("headline") and not res.get("headline"):
                    res["headline"] = str(attached_asset["headline"]).strip()

        # 1. Inspect inline citations in message text
        hist_cite = parse_inline_citation(content)
        if hist_cite:
            if not res.get("newspaper_name") and hist_cite.get("newspaper_name"):
                res["newspaper_name"] = hist_cite["newspaper_name"]
            if not res.get("issue_date") and hist_cite.get("issue_date"):
                res["issue_date"] = hist_cite["issue_date"]
            if not res.get("page_number") and hist_cite.get("page_number"):
                res["page_number"] = hist_cite["page_number"]
            if not res.get("headline") and hist_cite.get("headline") and not current_citation.get("headline") and not explicit_query_hl and not has_explicit_asset:
                res["headline"] = hist_cite["headline"]

        # 2. Inspect executive summary headers: e.g. "The Goan newspaper issue 94 (2026-08-02)"
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

        # 3. Inspect all known brand patterns across turn text
        if not res.get("newspaper_name"):
            for pat, brand in _KNOWN_BRANDS_PATTERNS:
                if pat.search(content):
                    res["newspaper_name"] = brand
                    break

        if res.get("newspaper_name") and res.get("issue_date") and res.get("headline"):
            break

    # Guardrail 1: If current query is cross-newspaper comparison or archive-wide newspaper query, do NOT constrain to a single newspaper
    if is_cross_newspaper:
        if not explicit_query_np:
            res.pop("newspaper_name", None)
            res.pop("comparison_newspaper", None)
            res.pop("target_newspapers", None)
            res.pop("issue_id", None)
            res.pop("article_id", None)
            res.pop("photo_id", None)
        elif not res.get("comparison_newspaper"):
            res.pop("issue_id", None)
            res.pop("article_id", None)
            res.pop("photo_id", None)

    # Guardrail 2: If current query explicitly provides its own date, invalidate stale context if from different date
    if current_date and res.get("issue_date") and res["issue_date"] != current_date:
        res.pop("newspaper_name", None)
        res.pop("issue_id", None)
        res.pop("issue_date", None)
        res.pop("article_id", None)
        res.pop("photo_id", None)
        res.pop("headline", None)
        res.pop("target_newspapers", None)

    # Guardrail 3: If current query explicitly specifies its own newspaper, discard history newspaper and foreign articles
    if current_np and res.get("newspaper_name") and current_np.lower() not in res["newspaper_name"].lower() and res["newspaper_name"].lower() not in current_np.lower():
        res.pop("newspaper_name", None)
        res.pop("issue_id", None)
        res.pop("article_id", None)
        res.pop("photo_id", None)
        res.pop("headline", None)
        res.pop("target_newspapers", None)
        if not current_date:
            res.pop("issue_date", None)

    # Guardrail 4: If current query specifies a date range, do NOT constrain to single-day issue_date
    if current_params.get("date_from") and current_params.get("date_to") and not explicit_query_date:
        res.pop("issue_date", None)
        res.pop("issue_id", None)
        res["date_from"] = current_params["date_from"]
        res["date_to"] = current_params["date_to"]

    # Re-apply explicit current query parameters
    if current_np:
        res["newspaper_name"] = current_np
        res["target_newspapers"] = [current_np]
    if current_date:
        res["issue_date"] = current_date
    if not has_date_conflict and not has_np_conflict:
        if attached_headline:
            res["headline"] = attached_headline
        if attached_photo_id:
            with contextlib.suppress(ValueError, TypeError):
                res["photo_id"] = int(attached_photo_id)
        if attached_article_id:
            with contextlib.suppress(ValueError, TypeError):
                res["article_id"] = int(attached_article_id)
    if current_citation.get("page_number"):
        res["page_number"] = current_citation["page_number"]

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
        citations = turn.get("citations") or []
        if citations and role.lower() == "assistant":
            cit_lines = []
            for c_idx, cit in enumerate(citations, 1):
                if isinstance(cit, dict) and cit.get("headline"):
                    cit_np = cit.get("newspaper_name", "")
                    cit_dt = cit.get("issue_date", "")
                    cit_pg = cit.get("page_number", "")
                    cit_lines.append(f"  [Article {c_idx}]: \"{cit['headline']}\" ({cit_np}, {cit_dt}, Page {cit_pg})")
            if cit_lines:
                content += "\nCited Articles:\n" + "\n".join(cit_lines)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


_UNSET = object()


CONDENSER_SYSTEM_PROMPT = (
    "You are an expert search query reformulator and coreference resolver for a newspaper archive research system.\n"
    "Given the chat history and latest user query, rewrite the latest query into "
    "a single, standalone sentence that contains all necessary context (entities, dates, page numbers).\n\n"
    "DECISION RULES:\n"
    "1. ATTACHED ASSET REFERENCE:\n"
    "   If a 'CURRENT WORKSPACE ATTACHED ASSET' is provided and the user query refers to it "
    "(e.g., 'this photo', 'in the photo', 'this picture', 'the image', 'this article', 'this story', 'the person in the photo', 'who is this'):\n"
    "   - Ground the rewritten query SOLELY in the CURRENT WORKSPACE ATTACHED ASSET (its newspaper, date, headline, photo ID, caption).\n"
    "   - DO NOT inherit or contaminate the query with people, entities, or dates from prior CONVERSATION HISTORY.\n"
    "2. CONVERSATION CONTINUATION:\n"
    "   If the query is a follow-up continuing the discussion from CONVERSATION HISTORY (e.g., 'what else did he say?', 'did other papers cover this?'), "
    "and does NOT refer to an attached asset:\n"
    "   - Resolve pronouns ('it', 'they', 'he', 'this newspaper') using the conversation context.\n"
    "3. TOPIC SWITCH OR INDEPENDENT QUERY:\n"
    "   If the query introduces a NEW or DIFFERENT topic, entity, location, or subject not discussed in CONVERSATION HISTORY "
    "(e.g., 'education policy in Maharashtra', 'ST quota protest in Goa', 'cricket match results'), treat it as a completely independent query:\n"
    "   - STRICTLY DO NOT inject or force previous conversation entities, newspaper brands, dates, or people into the query.\n"
    "   - Keep it as a clean, unconstrained standalone search query.\n"
    "4. CROSS-NEWSPAPER & SHARED COVERAGE FOLLOW-UPS:\n"
    "   If CONVERSATION HISTORY involved comparing two newspapers or finding similar/shared/exclusive articles between them, and the user asks a follow-up "
    "(e.g., 'list all those articles that are similar', 'show the common stories', 'what about the exclusives?'):\n"
    "   - Preserve both newspaper names, the date, and the comparison/shared intent in the rewritten query (e.g., 'list all similar articles between The Goan and The Morning Standard dated 2026-08-01').\n\n"
    "OUTPUT FORMAT: Output ONLY the plain rewritten query text. Do NOT include quotes, explanations, prefixes, or bullet points."
)


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
    """Rewrite a conversational follow-up query into a standalone search query with entities.

    The LLM reformulator is provided both conversation history and current workspace attached asset,
    and decides whether to inherit previous context, bind to the attached asset, or treat as a new topic.
    """
    if not chat_history and not attached_asset:
        return query
    if not needs_condensation(query, chat_history, attached_asset=attached_asset):
        return query

    asset_photo_id = attached_asset.get("photo_id") if attached_asset else None
    asset_art_id = attached_asset.get("article_id") if attached_asset else None
    asset_np = attached_asset.get("newspaper_name") if attached_asset else None
    asset_date = attached_asset.get("issue_date") if attached_asset else None
    asset_iss_id = attached_asset.get("issue_id") if attached_asset else None
    asset_hl = attached_asset.get("headline") if attached_asset else None
    asset_cap = attached_asset.get("caption") if attached_asset else None
    asset_type = attached_asset.get("visual_type") if attached_asset else None
    asset_page = attached_asset.get("page_number") if attached_asset else None

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

    # 0. Deterministic Ordinal Article Coreference Resolution:
    # E.g. "tell me about this article 1", "summarize the 2nd article", "explain the first article"
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
                        q_resolved = f'Tell me about the article "{c_hl}"{np_part}{dt_part}'
                        logger.info(
                            "Conversational query resolved via deterministic ordinal article citation heuristic",
                            extra={"original_query": query, "resolved_query": q_resolved},
                        )
                        return q_resolved

    # Resolve LLM provider (prefer lightweight/fast model like groq_llama or query_planner)
    resolved_provider: Any
    if provider is _UNSET:
        try:
            registry = get_registry()
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

    formatted_history = format_chat_history_for_prompt(chat_history) if chat_history else ""

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

    history_block = f"CONVERSATION HISTORY:\n{formatted_history}\n\n" if formatted_history else ""

    prompt = (
        f"{attached_asset_block}"
        f"{history_block}"
        f"LATEST USER QUERY: {query}\n\n"
        "Rewritten Standalone Query:"
    )

    if resolved_provider is not None:
        try:
            messages = [
                Message(
                    role="system",
                    content=CONDENSER_SYSTEM_PROMPT,
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
    if attached_asset and (asset_photo_id or asset_art_id or asset_hl):
        pub_desc = " ".join([p for p in [np_name, f"dated {iss_date}" if iss_date else ""] if p])
        if asset_photo_id and re.search(r"\b(photo|picture|image|graphic|figure|person|who is)\b", query, re.I):
            base_q = re.sub(r"\b(this|the)\s+(photo|picture|image|graphic|figure)\b", f"photo #{asset_photo_id}", query, flags=re.I)
            if asset_hl:
                q_resolved = f"{base_q} for article '{asset_hl}' in {pub_desc}".strip()
            else:
                q_resolved = f"{base_q} in {pub_desc}".strip()
            logger.info(
                "Conversational query resolved via attached photo fallback",
                extra={"original_query": query, "resolved_query": q_resolved},
            )
            return q_resolved
        elif asset_art_id and re.search(r"\b(article|story|news|report)\b", query, re.I):
            base_q = re.sub(r"\b(this|the)\s+(article|story|news|report)\b", f"article '{asset_hl or asset_art_id}'", query, flags=re.I)
            q_resolved = f"{base_q} in {pub_desc}".strip()
            logger.info(
                "Conversational query resolved via attached article fallback",
                extra={"original_query": query, "resolved_query": q_resolved},
            )
            return q_resolved
        elif asset_hl:
            q_resolved = f"{query} regarding '{asset_hl}' in {pub_desc}".strip()
            logger.info(
                "Conversational query resolved via attached headline fallback",
                extra={"original_query": query, "resolved_query": q_resolved},
            )
            return q_resolved

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
        is_shared_mode = is_shared or any(w in query.lower() for w in ["similar", "shared", "common", "same article", "same stories", "both"])
        if is_shared_mode:
            q_resolved = f"{query} between {np_name} and {comp_np}{dt_suffix}"
        else:
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


async def resolve_attached_asset_context(
    session_factory: Any,
    attached_photo_id: int | None = None,
    attached_article_id: int | None = None,
    attached_issue_date: str | None = None,
    attached_newspaper_name: str | None = None,
) -> dict[str, Any]:
    """Resolve ground truth issue date, newspaper name, article ID, issue ID, headline, and caption for attached assets."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from app.models.article import Article, Photo
    from app.models.newspaper import Issue, Newspaper, Page

    res: dict[str, Any] = {
        "photo_id": int(attached_photo_id) if attached_photo_id else None,
        "article_id": int(attached_article_id) if attached_article_id else None,
        "issue_date": attached_issue_date,
        "newspaper_name": attached_newspaper_name,
        "issue_id": None,
        "headline": None,
        "caption": None,
        "visual_type": None,
        "page_number": None,
    }

    if not session_factory or (not attached_photo_id and not attached_article_id):
        return res

    try:
        async with session_factory() as session:
            if attached_photo_id:
                stmt = (
                    select(Photo)
                    .where(Photo.id == int(attached_photo_id))
                    .options(
                        selectinload(Photo.article)
                        .selectinload(Article.issue)
                        .selectinload(Issue.newspaper)
                    )
                )
                photo = (await session.execute(stmt)).scalar_one_or_none()
                if photo:
                    res["photo_id"] = photo.id
                    if photo.caption:
                        res["caption"] = photo.caption
                    if photo.visual_type:
                        res["visual_type"] = photo.visual_type

                    if photo.page_id:
                        pg_stmt = select(Page.page_number).where(Page.id == photo.page_id)
                        pg_num = (await session.execute(pg_stmt)).scalar_one_or_none()
                        if pg_num is not None:
                            res["page_number"] = pg_num

                    if photo.article:
                        res["article_id"] = photo.article.id
                        if photo.article.headline:
                            res["headline"] = photo.article.headline
                        if photo.article.issue:
                            res["issue_id"] = photo.article.issue.id
                            res["issue_date"] = str(photo.article.issue.issue_date)
                            if photo.article.issue.newspaper:
                                res["newspaper_name"] = photo.article.issue.newspaper.name
            elif attached_article_id:
                stmt = (
                    select(Article)
                    .where(Article.id == int(attached_article_id))
                    .options(
                        selectinload(Article.issue).selectinload(Issue.newspaper)
                    )
                )
                art = (await session.execute(stmt)).scalar_one_or_none()
                if art:
                    res["article_id"] = art.id
                    if art.headline:
                        res["headline"] = art.headline
                    if art.primary_page_id:
                        pg_stmt = select(Page.page_number).where(Page.id == art.primary_page_id)
                        pg_num = (await session.execute(pg_stmt)).scalar_one_or_none()
                        if pg_num is not None:
                            res["page_number"] = pg_num
                    if art.issue:
                        res["issue_id"] = art.issue.id
                        res["issue_date"] = str(art.issue.issue_date)
                        if art.issue.newspaper:
                            res["newspaper_name"] = art.issue.newspaper.name
    except Exception as ex:
        logger.warning("Failed to resolve attached asset metadata", extra={"error": str(ex)})

    return res


async def resolve_authoritative_article_id(
    headline: str,
    newspaper_name: str | None = None,
    issue_date: str | None = None,
) -> int | None:
    """Fast DB lookup to find exact article_id for a known or quoted headline."""
    if not headline or len(headline.strip()) < 8:
        return None
    try:
        from app.models.base import get_session_factory
        from app.models.article import Article
        from app.models.newspaper import Issue, Newspaper
        from sqlalchemy import select

        factory = get_session_factory()
        async with factory() as session:
            stmt = select(Article.id).join(Issue, Article.issue_id == Issue.id)
            if newspaper_name:
                stmt = stmt.join(Newspaper, Issue.newspaper_id == Newspaper.id).where(
                    Newspaper.name.ilike(f"%{newspaper_name}%")
                )
            if issue_date:
                stmt = stmt.where(Issue.issue_date == issue_date)

            clean_hl = headline.strip().strip('"“”\'')
            exact_stmt = stmt.where(Article.headline.ilike(clean_hl))
            res = (await session.execute(exact_stmt)).scalars().first()
            if res:
                return int(res)

            if len(clean_hl) >= 12:
                prefix = clean_hl[:40]
                prefix_stmt = stmt.where(Article.headline.ilike(f"%{prefix}%"))
                res = (await session.execute(prefix_stmt)).scalars().first()
                if res:
                    return int(res)
    except Exception as ex:
        logger.warning("Failed to resolve authoritative article ID by headline", extra={"headline": headline, "error": str(ex)})

    return None


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

