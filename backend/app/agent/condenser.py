"""Conversational query condensation and coreference resolution engine.

3-Tier Architecture:
1. Deterministic Bypass / Gatekeeper (clean sessions, in-context meta-queries)
2. Structured Context Assembly & LLM Few-Shot Rewriter (4 intent rules)
3. Lightweight Normalizer & Safe Fallback (zero destructive string mutations)
"""

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
from app.retrieval.asset_resolver import (
    resolve_attached_asset_context,
    resolve_authoritative_article_id,
)

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

CONDENSER_SYSTEM_PROMPT = (
    "You are an expert search query reformulator and coreference resolver for a newspaper archive research system.\n"
    "Given the chat history and latest user query, rewrite the latest query into "
    "a single, standalone sentence that contains all necessary context (entities, dates, page numbers).\n\n"
    "DECISION RULES:\n"
    "1. ATTACHED ASSET REFERENCE / BINDING:\n"
    "   If a 'CURRENT WORKSPACE ATTACHED ASSET' is provided and the user query refers to it "
    "(\"this photo\", \"the article\", \"explain this\", \"who is this person\", or quotes/mentions the attached headline):\n"
    "   - Ground the rewritten query SOLELY in the CURRENT WORKSPACE ATTACHED ASSET (its newspaper, date, headline, photo ID, caption).\n"
    "   - Keep dates in ISO format (YYYY-MM-DD).\n"
    "   - DO NOT inherit or contaminate the query with people, entities, dates, or newspapers from prior CONVERSATION HISTORY.\n"
    "   Example: \"explain this photo\" (Attached: Photo #1042 in Financial Times, 2026-05-10)\n"
    "   Rewrite: \"Explain photo #1042 in Financial Times dated 2026-05-10\"\n\n"
    "2. CONVERSATION CONTINUATION:\n"
    "   If the query continues the discussion from CONVERSATION HISTORY and does NOT refer to an attached asset:\n"
    "   - Resolve pronouns (\"it\", \"they\", \"he\", \"this newspaper\", \"its\") using the conversation context.\n"
    "   Example: \"what about on page 4?\" (History discussing: The Indian Express, 2026-06-15)\n"
    "   Rewrite: \"What articles appeared on page 4 of The Indian Express on 2026-06-15?\"\n\n"
    "3. INDEPENDENT QUERY / TOPIC SHIFT:\n"
    "   If the user query introduces an entirely new topic, entity, date, or publication (e.g. \"crime in Panaji\", \"who won the election?\"), "
    "do NOT inject prior publications, people, or dates. Leave it unconstrained.\n"
    "   Example: \"show me weather reports\" (History discussing: The Guardian, 2026-04-12)\n"
    "   Rewrite: \"Show me weather reports\"\n\n"
    "4. COMPARATIVE CONTINUATION:\n"
    "   If prior turns involved a cross-newspaper comparison and the user asks a follow-up (e.g. \"list all those articles that are similar\", \"show the common stories\", \"what about the exclusives?\"), "
    "preserve both newspaper names, the date, and the comparison/shared intent in the rewritten query.\n"
    "   Example: \"list all those similar stories\" (History comparing: The Indian Express and The Hindu, 2026-05-20)\n"
    "   Rewrite: \"List all similar and shared articles between The Indian Express and The Hindu dated 2026-05-20\"\n\n"
    "OUTPUT FORMAT: Output ONLY the plain rewritten query text. Do NOT include quotes, explanations, prefixes, or bullet points."
)


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
                for pat, brand in _KNOWN_BRANDS_PATTERNS:
                    if pat.search(detected_pub):
                        res["newspaper_name"] = brand
                        break
            if summary_m.group(2) and not res.get("issue_id"):
                with contextlib.suppress(ValueError):
                    res["issue_id"] = int(summary_m.group(2))
            if summary_m.group(3) and not res.get("issue_date"):
                res["issue_date"] = summary_m.group(3)

        if not res.get("newspaper_name"):
            for pat, brand in _KNOWN_BRANDS_PATTERNS:
                if pat.search(content):
                    res["newspaper_name"] = brand
                    break

        if res.get("newspaper_name") and res.get("issue_date"):
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
        try:
            registry = get_registry()
            try:
                resolved_provider = registry.get_chat_provider(model_override or "query_condenser")
            except Exception:
                resolved_provider = registry.get_chat_provider(model_override or "query_planner")
        except Exception as reg_err:
            logger.warning("Could not obtain provider for query condensation", extra={"error": str(reg_err)})
            resolved_provider = None
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

    # Assemble Conversation History Block
    is_attached_query = bool(attached_asset and (asset_photo_id or asset_art_id or asset_hl))
    if is_attached_query and chat_history and (has_date_conflict or has_np_conflict):
        formatted_history = ""
    else:
        formatted_history = format_chat_history_for_prompt(chat_history) if chat_history else ""

    history_block = f"CONVERSATION HISTORY:\n{formatted_history}\n\n" if formatted_history else ""

    prompt = (
        f"{attached_asset_block}"
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
            resp = await resolved_provider.complete(
                messages=messages,
                max_tokens=128,
                temperature=0.0,
            )
            rewritten = resp.text.strip()

            # Lightweight Normalizer: Clean quotes and conversational prefixes
            rewritten = re.sub(
                r'^(?:Standalone Query:|Rewritten Standalone Query:|Rewritten Query:|Query:|"|\'|`|\*|\-)\s*',
                "",
                rewritten,
                flags=re.IGNORECASE,
            ).strip()
            rewritten = re.sub(r'("|\'|`)$', "", rewritten).strip()

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
