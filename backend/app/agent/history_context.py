"""Deterministic Conversation History, Coreference Gatekeeper, and Inline Citation Parser."""

from __future__ import annotations

import contextlib
import re
from typing import Any

from app.agent.extractor import (
    _KNOWN_BRANDS_PATTERNS,
    extract_parameters_from_query,
    find_matching_brand,
    is_archive_wide_newspaper_query,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tier 1: Deterministic Gatekeeper Patterns & Fast Bypasses
# ---------------------------------------------------------------------------

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

IN_CONTEXT_META_QUERY_PATTERN = re.compile(
    r"\b(which newspaper|what newspaper|what was the date|which date|what date|who wrote|"
    r"what source|which source|which paper|what paper|who is the author|give me the date|"
    r"from which news|from which paper|where was this published|when was this published|"
    r"who reported this|what are the sources|show sources|list the citations|what page was that|"
    r"which edition|who published this)\b",
    re.IGNORECASE,
)

_EXISTENTIAL_PREFIX_PATTERN = re.compile(
    r"^(?:is|are|was|were)\s+there\s+(?:any\s+)?(?:news|articles?|coverage|reports?)\b",
    re.IGNORECASE,
)

CLEAN_SESSION_CLARIFICATION_MESSAGE = (
    "Please specify which article, topic, or newspaper issue you would like me to summarize."
)


def is_in_context_meta_query(query: str, chat_history: list[dict[str, Any]]) -> bool:
    """Detect if the user is asking directly about previous turn's sources, date, or metadata."""
    if not chat_history:
        return False
    return bool(IN_CONTEXT_META_QUERY_PATTERN.search(query.strip()))


def is_ambiguous_standalone_query(
    query: str,
    chat_history: list[dict[str, Any]],
    has_attached_asset: bool = False,
) -> bool:
    """Detect ungrounded ambiguous queries on clean sessions (e.g. 'summarize it' on turn 1)."""
    if has_attached_asset or (chat_history and len(chat_history) > 0):
        return False

    q_clean = query.strip()
    words = q_clean.split()
    if len(words) <= 6:
        if GENERIC_FOLLOWUP_SHORT_PATTERN.search(q_clean):
            return True
        if len(words) < 4 and AMBIGUOUS_PRONOUNS_PATTERN.search(q_clean):
            return True
        if q_clean.lower().strip("?.! ") in {
            "summarize", "summarise", "summarize it", "tell me more",
            "explain it", "what happened", "who is it", "who was involved",
            "details", "give details",
        }:
            return True

    return False


def query_matches_attached_asset(query: str, attached_asset: dict[str, Any] | None) -> bool:
    """Check if user query references or quotes the attached asset headline, caption, or IDs."""
    if not attached_asset:
        return False
    hl = attached_asset.get("headline")
    cap = attached_asset.get("caption")
    q_clean = query.strip()
    q_lower = q_clean.lower()
    if hl:
        hl_lower = hl.lower().strip()
        if hl_lower in q_lower or (len(hl_lower) > 20 and hl_lower[:30] in q_lower):
            return True
        stop = {
            "the", "a", "an", "and", "or", "in", "on", "at", "for", "to", "of", "with", "by", "from",
            "is", "are", "was", "were", "this", "that", "these", "those", "about", "article", "story",
            "news", "report", "photo", "image", "picture", "visual", "who", "what", "where", "when",
            "why", "how", "person", "people", "explain", "summarize", "tell", "details",
        }
        hl_tokens = {w for w in re.findall(r"[a-z0-9]+", hl_lower) if len(w) > 2 and w not in stop}
        q_tokens = {w for w in re.findall(r"[a-z0-9]+", q_lower) if len(w) > 2 and w not in stop}
        if hl_tokens and q_tokens:
            overlap = hl_tokens & q_tokens
            if len(overlap) >= 3 or (len(overlap) / len(hl_tokens) >= 0.3):
                return True
    if cap:
        cap_lower = cap.lower().strip()
        if len(cap_lower) > 15 and cap_lower[:25] in q_lower:
            return True
    p_id = attached_asset.get("photo_id")
    if p_id and re.search(rf"\b(?:photo|image|picture|chart|figure)?\s*#?{p_id}\b", q_clean, re.I):
        return True
    art_id = attached_asset.get("article_id")
    return bool(art_id and re.search(rf"\b(?:article|story)?\s*#?{art_id}\b", q_clean, re.I))


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

    if attached_asset and (attached_asset.get("photo_id") or attached_asset.get("article_id") or attached_asset.get("headline")):
        if query_matches_attached_asset(q_clean, attached_asset):
            return True
        if AMBIGUOUS_PRONOUNS_PATTERN.search(q_clean):
            return True
        if re.search(r"\b(photo|picture|image|graphic|figure|chart|article|headline|story|person|who is)\b", q_clean, re.I):
            return True
        if len(words) <= 6 and GENERIC_FOLLOWUP_SHORT_PATTERN.search(q_clean):
            return True

    if chat_history:
        if AMBIGUOUS_PRONOUNS_PATTERN.search(q_clean):
            return True
        if len(words) <= 6 and GENERIC_FOLLOWUP_SHORT_PATTERN.search(q_clean):
            return True
        if _EXISTENTIAL_PREFIX_PATTERN.search(q_clean):
            return False
        if re.search(r"\b(did any other newspaper|did other newspapers?|similar|shared|exclusives?)\b", q_clean, re.I):
            return True

    return False


def parse_inline_citation(text: str) -> dict[str, Any]:
    """Extract newspaper name, issue date, page number, and headline from inline citations."""
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
        raw_np = m.group(1).strip()
        np = find_matching_brand(raw_np) or raw_np
        res: dict[str, Any] = {"newspaper_name": np, "target_newspapers": [np]}
        if m.group(2):
            res["issue_date"] = m.group(2).strip()
        if m.group(3):
            with contextlib.suppress(ValueError):
                res["page_number"] = int(m.group(3).strip())
        if m.group(4):
            res["headline"] = m.group(4).strip()
        return res

    hl_m = re.search(r"(?:article|story|headline|titled|report)?\s*[\"“]([^\"”]{8,150})[\"”]", text, re.I)
    if hl_m:
        cand_hl = hl_m.group(1).strip()
        if not any(pat.fullmatch(cand_hl) for pat, _ in _KNOWN_BRANDS_PATTERNS):
            res_flex: dict[str, Any] = {"headline": cand_hl}
            q_ext = extract_parameters_from_query(text)
            if q_ext.get("newspaper_name"):
                res_flex["newspaper_name"] = q_ext["newspaper_name"]
                res_flex["target_newspapers"] = [q_ext["newspaper_name"]]
            if q_ext.get("issue_date"):
                res_flex["issue_date"] = q_ext["issue_date"]
            return res_flex

    return {}


# ---------------------------------------------------------------------------
# Tier 2: LLM System Prompt & Context Assembly (4 Rewriting Rules)
# ---------------------------------------------------------------------------


def format_chat_history_for_prompt(chat_history: list[dict[str, Any]], max_turns: int = 5) -> str:
    """Format recent turns of chat history into clean dialog text with citations preserved."""
    recent = chat_history[-max_turns * 2 :]
    lines: list[str] = []
    for turn in recent:
        role = str(turn.get("role", "user")).capitalize()
        content = str(turn.get("content", "")).strip()
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
                    cit_lines.append(f'  [Article {c_idx}]: "{cit["headline"]}" ({cit_np}, {cit_dt}, Page {cit_pg})')
            if cit_lines:
                content += "\nCited Articles:\n" + "\n".join(cit_lines)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured Context Resolver (Backward-compatible adapter for tests)
# ---------------------------------------------------------------------------

def extract_active_issue_from_history(
    chat_history: list[dict[str, Any]],
    current_query: str | None = None,
    attached_photo_id: int | None = None,
    attached_article_id: int | None = None,
    attached_issue_date: str | None = None,
    attached_newspaper_name: str | None = None,
    attached_headline: str | None = None,
) -> dict[str, Any]:
    """Scan chat history for previously mentioned newspaper names, issue IDs, and dates with conflict isolation."""
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

    if not has_date_conflict and not has_np_conflict:
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

    q_lower = (current_query or "").lower()
    is_cross_newspaper = bool(
        is_archive_wide_newspaper_query(current_query)
        or re.search(r"\b(?:compa[a-z]*|contrast[a-z]*|diff(?:erence[s]?|ering)?|versus|vs\.?)\b", q_lower)
        or any(w in q_lower for w in ["all available", "all newspaper", "both newspaper", "across newspaper", "different newspaper"])
    )

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

        summary_m = re.search(
            r"([A-Za-z0-9\s\.\'\-]+?)\s+(?:newspaper\s+)?issue\s+(\d+)(?:\s*\((\d{4}-\d{2}-\d{2})\))?",
            content,
            re.IGNORECASE,
        )
        if summary_m:
            detected_pub = summary_m.group(1).strip().replace("⚡", "").replace("EXECUTIVE SUMMARY", "").strip()
            detected_pub = re.sub(r"^[\s\*\#\-\:]+", "", detected_pub).strip()
            if detected_pub and len(detected_pub) >= 3 and not res.get("newspaper_name"):
                matched_b = find_matching_brand(detected_pub)
                if matched_b:
                    res["newspaper_name"] = matched_b
            if summary_m.group(2) and not res.get("issue_id"):
                with contextlib.suppress(ValueError):
                    res["issue_id"] = int(summary_m.group(2))
            if summary_m.group(3) and not res.get("issue_date"):
                res["issue_date"] = summary_m.group(3)

        # Check citations in turn for active headline, newspaper, date
        citations = turn.get("citations") or []
        for cit in citations:
            if isinstance(cit, dict):
                if cit.get("headline") and not res.get("headline"):
                    res["headline"] = cit["headline"]
                if cit.get("newspaper_name") and not res.get("newspaper_name"):
                    res["newspaper_name"] = cit["newspaper_name"]
                if cit.get("issue_date") and not res.get("issue_date"):
                    res["issue_date"] = cit["issue_date"]
                if cit.get("page_number") and not res.get("page_number"):
                    with contextlib.suppress(ValueError):
                        res["page_number"] = int(cit["page_number"])

        # Also check for quoted headlines or "Explain article ..." in content
        if not res.get("headline"):
            q_hl = re.search(r"(?:article|story|headline|titled|report)?\s*[\"“]([^\"”]{8,150})[\"”]", content, re.I)
            if q_hl:
                res["headline"] = q_hl.group(1).strip()

        if not res.get("newspaper_name"):
            for pat, brand in _KNOWN_BRANDS_PATTERNS:
                if pat.search(content):
                    res["newspaper_name"] = brand
                    break

        if res.get("newspaper_name") and res.get("issue_date") and res.get("headline"):
            break

    if is_cross_newspaper and not explicit_query_np:
        res.pop("newspaper_name", None)
        res.pop("comparison_newspaper", None)
        res.pop("target_newspapers", None)
        res.pop("issue_id", None)
        res.pop("article_id", None)
        res.pop("photo_id", None)

    if explicit_query_np and res.get("newspaper_name") and explicit_query_np.lower() not in res["newspaper_name"].lower() and res["newspaper_name"].lower() not in explicit_query_np.lower():
        res.pop("newspaper_name", None)
        res.pop("issue_id", None)
        res.pop("article_id", None)
        res.pop("photo_id", None)
        res.pop("headline", None)
        res.pop("target_newspapers", None)
        if not explicit_query_date:
            res.pop("issue_date", None)

    if explicit_query_date and res.get("issue_date") and explicit_query_date != res["issue_date"]:
        if not explicit_query_np:
            res.pop("newspaper_name", None)
            res.pop("target_newspapers", None)
            res.pop("comparison_newspaper", None)
        res.pop("issue_id", None)
        res.pop("article_id", None)
        res.pop("photo_id", None)
        res.pop("headline", None)

    if explicit_query_np:
        res["newspaper_name"] = explicit_query_np
        res["target_newspapers"] = [explicit_query_np]
    if explicit_query_date:
        res["issue_date"] = explicit_query_date

    return res


# ---------------------------------------------------------------------------
# Tier 3: Async LLM Reformulation & Lightweight Normalizer
# ---------------------------------------------------------------------------


__all__ = [
    "AMBIGUOUS_PRONOUNS_PATTERN",
    "CLEAN_SESSION_CLARIFICATION_MESSAGE",
    "GENERIC_FOLLOWUP_SHORT_PATTERN",
    "IN_CONTEXT_META_QUERY_PATTERN",
    "_EXISTENTIAL_PREFIX_PATTERN",
    "extract_active_issue_from_history",
    "format_chat_history_for_prompt",
    "is_ambiguous_standalone_query",
    "is_in_context_meta_query",
    "needs_condensation",
    "parse_inline_citation",
    "query_matches_attached_asset",
]
