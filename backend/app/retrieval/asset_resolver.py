"""Database Asset Context Resolver for NewsLens-AI.

Resolves ground-truth database metadata for attached workspace assets (photos, articles)
and authoritative article ID lookup for quoted headlines.
"""

from __future__ import annotations

import contextlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.article import Article, Photo
from app.models.base import get_session_factory
from app.models.newspaper import Issue, Newspaper, Page

logger = get_logger(__name__)


async def resolve_attached_asset_context(
    session_factory: Any,
    attached_photo_id: int | None = None,
    attached_article_id: int | None = None,
    attached_issue_date: str | None = None,
    attached_newspaper_name: str | None = None,
) -> dict[str, Any]:
    """Resolve ground truth issue date, newspaper name, article ID, issue ID, headline, and caption for attached assets."""
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
    session_factory: Any = None,
) -> int | None:
    """Fast DB lookup to find exact article_id for a known or quoted headline."""
    if not headline or len(headline.strip()) < 8:
        return None
    try:
        factory = session_factory or get_session_factory()
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
        logger.warning(
            "Failed to resolve authoritative article ID by headline",
            extra={"headline": headline, "error": str(ex)},
        )

    return None


from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ConversationWorkingContext:
    """Consolidated Ground Truth context and conflict resolution for a conversation turn."""

    active_ctx: dict[str, Any]
    attached_ctx: dict[str, Any]
    eff_attached_article_id: int | None
    eff_attached_photo_id: int | None
    has_attached_asset: bool
    is_followup: bool
    has_date_conflict: bool
    has_np_conflict: bool
    has_headline_conflict: bool


async def resolve_conversation_working_context(
    query: str,
    chat_history: list[dict[str, Any]] | None = None,
    attached_photo_id: int | None = None,
    attached_article_id: int | None = None,
    attached_issue_date: str | None = None,
    attached_newspaper_name: str | None = None,
    session_factory: Any = None,
) -> ConversationWorkingContext:
    """Authoritative reconciliation of attached assets, query parameters, headline binding, and active issue context."""
    from app.agent.condenser import (
        extract_active_issue_from_history,
        needs_condensation,
        parse_inline_citation,
    )
    from app.agent.extractor import extract_parameters_from_query

    history = chat_history or []
    attached_ctx = await resolve_attached_asset_context(
        session_factory=session_factory,
        attached_photo_id=attached_photo_id,
        attached_article_id=attached_article_id,
        attached_issue_date=attached_issue_date,
        attached_newspaper_name=attached_newspaper_name,
    )

    res_date = attached_ctx.get("issue_date")
    res_np = attached_ctx.get("newspaper_name")
    res_art_id = attached_ctx.get("article_id")
    res_iss_id = attached_ctx.get("issue_id")

    q_params = extract_parameters_from_query(query) if query else {}
    q_cite = parse_inline_citation(query or "")
    explicit_q_date = q_params.get("issue_date") or q_cite.get("issue_date")
    explicit_q_date_from = q_params.get("date_from")
    explicit_q_date_to = q_params.get("date_to")
    explicit_q_np = q_params.get("newspaper_name") or q_cite.get("newspaper_name")
    explicit_q_hl = q_params.get("headline") or q_cite.get("headline")

    has_date_conflict = bool(
        (explicit_q_date and res_date and explicit_q_date != res_date)
        or (explicit_q_date_from and explicit_q_date_to and res_date and not (explicit_q_date_from <= res_date <= explicit_q_date_to))
    )
    has_np_conflict = bool(
        explicit_q_np
        and res_np
        and explicit_q_np.lower() not in res_np.lower()
        and res_np.lower() not in explicit_q_np.lower()
    )

    attached_hl = attached_ctx.get("headline")
    has_headline_conflict = False
    if attached_hl and query:
        if explicit_q_hl and explicit_q_hl.lower() != attached_hl.lower():
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

            att_tokens = _sig_tokens(attached_hl)
            q_tokens = _sig_tokens(query)
            is_pronoun_ref = bool(
                re.search(
                    r"\b(?:this|that|the|it|ir)\s+(?:article|story|news|report|headline|piece|photo|image|picture|graphic|figure|visual|table|chart|infographic)\b"
                    r"|\b(?:tell me more|explain this|explain it|what is this|who is this|who is the person|who is in this)\b"
                    r"|\b(?:find|read|get|tell|show|explain|summ[ae]ri[sz]e)\s+(?:about|on|for|of)?\s*(?:this|the|that|it|ir)?\s*(?:article|story|news|report|headline|piece)?\b"
                    r"|\bin\s+\d+\s+words?\b",
                    query,
                    re.I,
                )
            )
            if (attached_photo_id or attached_article_id or res_art_id) and re.search(
                r"\b(?:photo|image|picture|visual|this article|this photo|it\b|that\b|the article)\b",
                query,
                re.I,
            ):
                is_pronoun_ref = True
            if not is_pronoun_ref and att_tokens and q_tokens and not (att_tokens & q_tokens):
                has_headline_conflict = True

    eff_attached_article_id = (
        (res_art_id or attached_article_id)
        if not has_date_conflict and not has_np_conflict and not has_headline_conflict
        else None
    )
    eff_attached_photo_id = (
        attached_photo_id
        if not has_date_conflict and not has_np_conflict and not has_headline_conflict
        else None
    )

    # Authoritative headline DB binding: if query has an explicit or quoted headline,
    # resolve its true article_id and override attached IDs
    cand_hl = explicit_q_hl
    if not cand_hl and query:
        hl_m = re.search(r"[\"“]([^\"”]{8,150})[\"”]", query, re.I)
        if hl_m:
            cand_hl = hl_m.group(1).strip()
    if cand_hl:
        bound_art_id = await resolve_authoritative_article_id(
            headline=cand_hl,
            newspaper_name=explicit_q_np or res_np,
            issue_date=explicit_q_date or res_date,
            session_factory=session_factory,
        )
        if bound_art_id:
            eff_attached_article_id = bound_art_id
            if res_art_id and res_art_id != bound_art_id:
                eff_attached_photo_id = None
            if not attached_ctx.get("headline") or has_headline_conflict:
                attached_ctx["headline"] = cand_hl

    active_ctx = extract_active_issue_from_history(
        history,
        current_query=query,
        attached_photo_id=eff_attached_photo_id,
        attached_article_id=eff_attached_article_id,
        attached_issue_date=res_date if not has_date_conflict else None,
        attached_newspaper_name=res_np if not has_np_conflict else None,
        attached_headline=attached_ctx.get("headline") if not has_date_conflict and not has_np_conflict and not has_headline_conflict else None,
    )

    if not has_date_conflict and not has_np_conflict and not has_headline_conflict:
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
        if eff_attached_article_id:
            active_ctx["article_id"] = eff_attached_article_id
        if cand_hl:
            active_ctx["headline"] = cand_hl

    if explicit_q_date_from and explicit_q_date_to and not explicit_q_date:
        active_ctx["issue_date"] = None

    has_attached_asset = bool(
        (eff_attached_photo_id or eff_attached_article_id)
        and not has_date_conflict
        and not has_np_conflict
        and not has_headline_conflict
    )
    is_followup = bool(
        (history or has_attached_asset)
        and needs_condensation(query, history, attached_asset=attached_ctx)
    )

    return ConversationWorkingContext(
        active_ctx=active_ctx,
        attached_ctx=attached_ctx,
        eff_attached_article_id=eff_attached_article_id,
        eff_attached_photo_id=eff_attached_photo_id,
        has_attached_asset=has_attached_asset,
        is_followup=is_followup,
        has_date_conflict=has_date_conflict,
        has_np_conflict=has_np_conflict,
        has_headline_conflict=has_headline_conflict,
    )


__all__ = [
    "ConversationWorkingContext",
    "resolve_attached_asset_context",
    "resolve_authoritative_article_id",
    "resolve_conversation_working_context",
]
