"""Citation Extraction and Authoritative Database Resolution for NewsLens-AI.

Extracts inline broadsheet, visual asset, and web citations from synthesized text
and resolves authoritative page numbers and publication metadata from MySQL.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.state import AgentCitation
from app.core.logging import get_logger
from app.retrieval.sql_analytics import sanitize_headline

logger = get_logger(__name__)


def make_citation(item: dict[str, Any], headline: str | None = None) -> AgentCitation:
    """Create an AgentCitation object from an evidence item."""
    is_web = bool(item.get("is_web") or item.get("source_tool") == "web_search")
    hl = headline or item.get("headline", "Untitled")
    if is_web:
        return AgentCitation(
            newspaper_name=item.get("newspaper_name", "Live Web"),
            issue_date=item.get("issue_date", "Live Web"),
            page_number=1,
            headline=hl,
            article_id=0,
            snippet=(item.get("snippet") or "")[:300],
            issue_id=0,
            bboxes=[],
            url=item.get("url") or "",
            source_type="web",
            is_web=True,
        )
    p_raw = item.get("page_number")
    pages = item.get("pages", [])
    page_val = int(p_raw) if p_raw is not None else (int(pages[0]) if pages and pages[0] else 1)
    return AgentCitation(
        newspaper_name=item.get("newspaper_name", "Daily News"),
        issue_date=item.get("issue_date", ""),
        page_number=page_val,
        headline=hl,
        article_id=item.get("article_id", 0),
        snippet=(item.get("snippet") or "")[:300],
        issue_id=item.get("issue_id", 0),
        bboxes=item.get("bboxes", []),
        url=None,
        source_type="visual_asset" if item.get("is_visual_asset") or item.get("source_tool") == "inspect_visual_asset" else "newspaper",
        is_web=False,
        photo_id=item.get("photo_id"),
        image_url=item.get("image_url"),
        visual_type=item.get("visual_type"),
    )


async def resolve_authoritative_citations(
    citations: list[AgentCitation],
    session_factory: Any | None = None,
    target_page: int | None = None,
) -> list[AgentCitation]:
    """Resolve and verify authoritative page numbers and metadata directly from MySQL."""
    if not citations:
        return []

    art_ids = [
        int(c["article_id"])
        for c in citations
        if c.get("article_id") and int(c.get("article_id", 0)) > 0 and not c.get("is_web")
    ]
    if art_ids and session_factory:
        try:
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            from app.models.article import Article
            from app.models.newspaper import Issue

            async with session_factory() as db:
                stmt = (
                    select(Article)
                    .where(Article.id.in_(art_ids))
                    .options(
                        selectinload(Article.issue).selectinload(Issue.newspaper),
                        selectinload(Article.article_pages),
                    )
                )
                res = await db.execute(stmt)
                db_articles = {a.id: a for a in res.scalars().all()}

                for c in citations:
                    c_art_id = c.get("article_id")
                    if c_art_id and c_art_id in db_articles:
                        art = db_articles[c_art_id]
                        true_pages = sorted({ap.page_number for ap in art.article_pages}) if art.article_pages else []
                        if true_pages:
                            c["page_number"] = true_pages[0]
                        if art.issue:
                            c["issue_id"] = art.issue.id
                            if art.issue.issue_date:
                                c["issue_date"] = str(art.issue.issue_date)
                            if art.issue.newspaper and art.issue.newspaper.name:
                                c["newspaper_name"] = art.issue.newspaper.name
        except Exception as e:
            logger.warning("Failed to resolve authoritative citation metadata from DB", extra={"error": str(e)})

    if target_page is not None and target_page > 0:
        scoped = [c for c in citations if c.get("is_web") or c.get("page_number") == target_page]
        if scoped:
            return scoped

    return citations


def extract_citations_from_text(
    text: str,
    evidence_items: list[dict[str, Any]],
    target_page: int | None = None,
    archetype: str | None = None,
) -> list[AgentCitation]:
    """Extract and structure verified citations mentioned in the text or used from evidence."""
    citations: list[AgentCitation] = []
    seen_keys: set[str] = set()
    text_lower = (text or "").lower()

    candidate_items = [
        item for item in evidence_items
        if (
            bool(item.get("is_web") or item.get("source_tool") == "web_search")
            or int(item.get("article_id") or 0) > 0
            or bool(item.get("is_visual_asset") or item.get("source_tool") == "inspect_visual_asset")
        )
    ]
    if not candidate_items and evidence_items:
        candidate_items = [
            item for item in evidence_items
            if not str(item.get("headline", "")).startswith(
                ("Issue Manifest:", "Coverage Audit:", "Article Count Analysis:", "Archive Topic", "Frontpage Prominence")
            )
        ]

    if target_page is not None and target_page > 0:
        page_filtered = [
            item for item in candidate_items
            if item.get("is_web")
            or item.get("page_number") == target_page
            or (item.get("pages") and target_page in item.get("pages"))
        ]
        if page_filtered:
            candidate_items = page_filtered

    if archetype == "article_catalog":
        for item in candidate_items:
            if int(item.get("article_id") or 0) > 0:
                raw_hl = item.get("headline", "")
                sub_hl = item.get("subheadline")
                byline = item.get("byline_author")
                snip = item.get("snippet") or item.get("summary") or ""
                hl, _ = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
                photo_id = item.get("photo_id")
                dedup_key = f"visual_{photo_id}" if photo_id else f"{item.get('newspaper_name')}_{item.get('issue_date')}_{hl}"
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    citations.append(make_citation(item, headline=hl))

    for item in candidate_items:
        is_web = bool(item.get("is_web") or item.get("source_tool") == "web_search")
        raw_hl = item.get("headline", "")
        sub_hl = item.get("subheadline")
        byline = item.get("byline_author")
        snip = item.get("snippet") or item.get("summary") or ""
        hl, _ = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
        hl_clean = hl.strip().lower()
        url = item.get("url") or ""

        is_referenced = False
        is_visual = bool(item.get("is_visual_asset") or item.get("source_tool") == "inspect_visual_asset")

        if is_web:
            if (url and url.lower() in text_lower) or (hl_clean and len(hl_clean) > 4 and hl_clean in text_lower):
                is_referenced = True
        elif is_visual:
            photo_id_str = str(item.get("photo_id") or "")
            v_type = str(item.get("visual_type") or "chart").replace("_", " ").lower()
            if (
                (photo_id_str and photo_id_str in text_lower)
                or (hl_clean and len(hl_clean) > 5 and hl_clean in text_lower)
                or any(w in text_lower for w in (v_type, "chart", "infographic", "table", "graph", "trend", "metric", "figure"))
            ):
                is_referenced = True
            elif hl_clean:
                tokens = [w for w in re.findall(r"\w+", hl_clean) if len(w) > 3]
                if len(tokens) >= 3 and sum(1 for tok in tokens if tok in text_lower) / len(tokens) >= 0.5:
                    is_referenced = True
        else:
            if hl_clean and len(hl_clean) > 5 and hl_clean in text_lower:
                is_referenced = True
            elif hl_clean:
                tokens = [w for w in re.findall(r"\w+", hl_clean) if len(w) > 3]
                if len(tokens) >= 3 and sum(1 for tok in tokens if tok in text_lower) / len(tokens) >= 0.65:
                    is_referenced = True
            if not is_referenced and hl_clean:
                hl_simple = re.sub(r"[^\w\s]", "", hl_clean).strip()
                if len(hl_simple) > 8 and hl_simple in re.sub(r"[^\w\s]", "", text_lower):
                    is_referenced = True

        if is_referenced:
            photo_id = item.get("photo_id")
            dedup_key = f"visual_{photo_id}" if photo_id else f"{item.get('newspaper_name')}_{item.get('issue_date')}_{hl}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                citations.append(make_citation(item, headline=hl))

    if not citations and candidate_items:
        for item in candidate_items[:15]:
            photo_id = item.get("photo_id")
            dedup_key = f"visual_{photo_id}" if photo_id else f"{item.get('newspaper_name')}_{item.get('issue_date')}_{item.get('headline')}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                raw_hl = item.get("headline", "")
                sub_hl = item.get("subheadline")
                byline = item.get("byline_author")
                snip = item.get("snippet") or item.get("summary") or ""
                hl, _ = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
                citations.append(make_citation(item, headline=hl))
    elif len(citations) < len(candidate_items) and any(it.get("source_tool", "").startswith("sql_analytics") for it in candidate_items):
        for item in candidate_items[:15]:
            raw_hl = item.get("headline", "")
            sub_hl = item.get("subheadline")
            byline = item.get("byline_author")
            snip = item.get("snippet") or item.get("summary") or ""
            hl, _ = sanitize_headline(raw_hl, subheadline=sub_hl, byline_author=byline, snippet=snip)
            dedup_key = f"{item.get('newspaper_name')}_{item.get('issue_date')}_{hl}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                citations.append(make_citation(item, headline=hl))

    return citations


__all__ = [
    "extract_citations_from_text",
    "make_citation",
    "resolve_authoritative_citations",
]
