"""Visual Asset Inspection and Multimodal VLM Extraction Engine.

Provides deep broadsheet visual crop inspection, on-demand VLM chart/table
extraction, companion visual asset resolution, and standardized evidence formatting.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.article import Article, ArticlePage, Photo
from app.models.newspaper import Issue, Newspaper

logger = get_logger(__name__)


class VisualInspectionEngine:
    """Engine for resolving, transcribing, and formatting visual assets from broadsheet archives."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def inspect_visual_asset(
        self,
        args: dict[str, Any],
        state_context: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Deep multimodal visual inspection, on-demand VLM extraction, and table transcription."""
        state = state_context or {}
        params = self.resolve_visual_target_params(args, state)
        async with self._session_factory() as session:
            photos, early_items = await self.resolve_photos_for_inspection(session, params, args, state)
            if early_items is not None:
                return early_items, len(early_items)

            if not photos:
                logger.warning("No photo found for inspect_visual_asset", extra={"tool_args": args})
                return [], 0

            await self.enrich_photo_vlm_descriptions(session, photos)
            items = [self.format_visual_asset_item(p, params, state) for p in photos]
            return items, len(items)

    def resolve_visual_target_params(
        self,
        args: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve target photo_id, article_id, newspaper, date, and headline filters."""
        photo_id = args.get("photo_id") or state.get("attached_photo_id")
        article_id = args.get("article_id") or state.get("attached_article_id")
        query_text = (args.get("query") or state.get("query") or "").strip()
        newspaper_name = (args.get("newspaper_name") or state.get("active_newspaper_name") or "").strip()
        issue_date = (args.get("issue_date") or state.get("active_issue_date") or "").strip()
        page_filter = args.get("page_filter") or ""
        target_headline: str | None = None
        from app.agent.condenser import parse_inline_citation

        # 0. Check for authoritative inline citation or extracted parameters in query_text or current state
        q_citation = (
            parse_inline_citation(query_text)
            or parse_inline_citation(state.get("query") or "")
            or parse_inline_citation(state.get("original_query") or "")
        )
        if q_citation.get("newspaper_name") and not newspaper_name:
            newspaper_name = q_citation["newspaper_name"]
        if q_citation.get("issue_date") and not issue_date:
            issue_date = q_citation["issue_date"]
        if q_citation.get("page_number") and not page_filter:
            page_filter = str(q_citation["page_number"])
        if q_citation.get("headline"):
            target_headline = q_citation["headline"]

        from app.agent.extractor import extract_parameters_from_query

        q_ext = (
            extract_parameters_from_query(query_text)
            or extract_parameters_from_query(state.get("query") or "")
            or extract_parameters_from_query(state.get("original_query") or "")
        )
        if not newspaper_name and q_ext.get("newspaper_name"):
            newspaper_name = q_ext["newspaper_name"]
        if not issue_date and q_ext.get("issue_date"):
            issue_date = q_ext["issue_date"]
        if not page_filter and q_ext.get("page_filter"):
            page_filter = str(q_ext["page_filter"])

        # 1. If photo_id or article_id not provided and no current query headline, try resolving from conversation history
        if not photo_id and not article_id and not target_headline:
            chat_history = state.get("chat_history") or []
            for turn in reversed(chat_history):
                # Check turn citations (only inherit if matching known newspaper/date filters)
                turn_citations = turn.get("citations") or []
                for cit in turn_citations:
                    if isinstance(cit, dict):
                        c_np = cit.get("newspaper_name") or ""
                        c_dt = cit.get("issue_date") or ""
                        np_ok = not newspaper_name or not c_np or (newspaper_name.lower() in c_np.lower() or c_np.lower() in newspaper_name.lower())
                        dt_ok = not issue_date or not c_dt or c_dt == issue_date

                        if np_ok and dt_ok:
                            if cit.get("photo_id") and not photo_id:
                                with contextlib.suppress(ValueError):
                                    photo_id = int(cit["photo_id"])
                            if cit.get("article_id") and not article_id:
                                with contextlib.suppress(ValueError):
                                    article_id = int(cit["article_id"])
                            if cit.get("headline") and not target_headline:
                                target_headline = str(cit["headline"]).strip()

                # Check turn attached asset (only inherit if matching known newspaper/date filters)
                att = turn.get("attachedAsset") or {}
                if isinstance(att, dict):
                    att_np = att.get("newspaperName") or att.get("newspaper_name") or ""
                    att_dt = att.get("issueDate") or att.get("issue_date") or ""
                    att_np_ok = not newspaper_name or not att_np or (newspaper_name.lower() in att_np.lower() or att_np.lower() in newspaper_name.lower())
                    att_dt_ok = not issue_date or not att_dt or att_dt == issue_date
                    if att_np_ok and att_dt_ok:
                        if att.get("photoId") and not photo_id:
                            with contextlib.suppress(ValueError):
                                photo_id = int(att["photoId"])
                        if att.get("articleId") and not article_id:
                            with contextlib.suppress(ValueError):
                                article_id = int(att["articleId"])
                        if att.get("headline") and not target_headline:
                            target_headline = str(att["headline"]).strip()

                # Check turn message text for inline citation
                c_text = str(turn.get("content", ""))
                cite_m = parse_inline_citation(c_text)
                if cite_m:
                    c_np = cite_m.get("newspaper_name") or ""
                    c_dt = cite_m.get("issue_date") or ""
                    np_ok = not newspaper_name or not c_np or (newspaper_name.lower() in c_np.lower() or c_np.lower() in newspaper_name.lower())
                    dt_ok = not issue_date or not c_dt or c_dt == issue_date

                    if np_ok and dt_ok:
                        if not newspaper_name and c_np:
                            newspaper_name = c_np
                        if not issue_date and c_dt:
                            issue_date = c_dt
                        if not page_filter and cite_m.get("page_number"):
                            page_filter = str(cite_m["page_number"])
                        if not target_headline and cite_m.get("headline"):
                            target_headline = cite_m["headline"]

                if photo_id or article_id or target_headline:
                    break

        return {
            "photo_id": photo_id,
            "article_id": article_id,
            "query_text": query_text,
            "newspaper_name": newspaper_name,
            "issue_date": issue_date,
            "page_filter": page_filter,
            "target_headline": target_headline,
        }

    async def resolve_photos_for_inspection(
        self,
        session: AsyncSession,
        params: dict[str, Any],
        args: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[list[Photo], list[dict[str, Any]] | None]:
        """Find and filter matching photos via five cascade discovery strategies."""
        photo_id = params["photo_id"]
        article_id = params["article_id"]
        target_headline = params["target_headline"]
        newspaper_name = params["newspaper_name"]
        issue_date = params["issue_date"]
        page_filter = params["page_filter"]
        query_text = params["query_text"]

        photos_to_process: list[Photo] = []

        # Strategy A: Explicit Photo ID provided or resolved
        if photo_id:
            stmt = (
                select(Photo)
                .where(Photo.id == int(photo_id))
                .options(
                    selectinload(Photo.article)
                    .selectinload(Article.issue)
                    .selectinload(Issue.newspaper),
                    selectinload(Photo.article).selectinload(Article.article_pages),
                )
            )
            res = await session.execute(stmt)
            single_photo = res.scalar_one_or_none()
            if single_photo:
                is_compat = True
                p_art = single_photo.article
                if p_art and p_art.issue:
                    if newspaper_name and p_art.issue.newspaper:
                        np_m = (newspaper_name.lower() in p_art.issue.newspaper.name.lower() or p_art.issue.newspaper.name.lower() in newspaper_name.lower())
                        if not np_m:
                            is_compat = False
                    if issue_date and str(p_art.issue.issue_date) != issue_date:
                        is_compat = False

                is_attached_photo = bool(
                    (state.get("attached_photo_id") and int(state["attached_photo_id"]) == single_photo.id)
                    or (args.get("photo_id") and int(args["photo_id"]) == single_photo.id)
                )

                if is_compat or is_attached_photo:
                    photos_to_process.append(single_photo)
                    if is_attached_photo and p_art and p_art.issue:
                        params["issue_date"] = str(p_art.issue.issue_date)
                        issue_date = params["issue_date"]
                        if p_art.issue.newspaper:
                            params["newspaper_name"] = p_art.issue.newspaper.name
                            newspaper_name = params["newspaper_name"]
                else:
                    logger.warning("Photo ID %s rejected for visual inspection due to newspaper/date mismatch", photo_id)
                    params["photo_id"] = None

                if photos_to_process and single_photo in photos_to_process and single_photo.article_id:
                    comp_stmt = (
                        select(Photo)
                        .where(
                            Photo.article_id == single_photo.article_id,
                            Photo.id != single_photo.id,
                        )
                        .order_by(Photo.id.asc())
                        .options(
                            selectinload(Photo.article)
                            .selectinload(Article.issue)
                            .selectinload(Issue.newspaper),
                            selectinload(Photo.article).selectinload(Article.article_pages),
                        )
                    )
                    comp_res = await session.execute(comp_stmt)
                    companion_photos = comp_res.scalars().all()
                    for cp in companion_photos:
                        if cp.visual_type in ("data_chart", "infographic", "table") and len(photos_to_process) < 6:
                            photos_to_process.append(cp)

        # Strategy B: Target headline resolved from query citation or context, find article
        if target_headline and not photos_to_process:
            art_stmt = (
                select(Article)
                .join(Issue, Article.issue_id == Issue.id)
                .join(Newspaper, Issue.newspaper_id == Newspaper.id)
                .where(
                    (Article.headline == target_headline)
                    | (Article.headline.ilike(f"%{target_headline[:50]}%"))
                )
            )
            if newspaper_name:
                art_stmt = art_stmt.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
            if issue_date:
                art_stmt = art_stmt.where(Issue.issue_date == issue_date)

            matched_art = (await session.execute(art_stmt.order_by(Article.id.desc()).limit(1))).scalars().first()
            if matched_art:
                params["article_id"] = matched_art.id
                article_id = matched_art.id

        # Strategy C: Explicit Article ID provided or resolved
        if article_id and not photos_to_process:
            chk_stmt = (
                select(Article)
                .where(Article.id == int(article_id))
                .options(
                    selectinload(Article.issue).selectinload(Issue.newspaper),
                )
            )
            chk_art = (await session.execute(chk_stmt)).scalar_one_or_none()
            is_explicit_article = bool(
                (state.get("attached_article_id") and int(state["attached_article_id"]) == int(article_id))
                or (args.get("article_id") and int(args["article_id"]) == int(article_id))
            )
            is_compat = True
            if chk_art and chk_art.issue:
                if newspaper_name and chk_art.issue.newspaper:
                    np_m = (newspaper_name.lower() in chk_art.issue.newspaper.name.lower() or chk_art.issue.newspaper.name.lower() in newspaper_name.lower())
                    if not np_m:
                        is_compat = False
                if issue_date and str(chk_art.issue.issue_date) != issue_date:
                    is_compat = False

            if is_compat:
                pass
            elif is_explicit_article:
                from app.agent.extractor import extract_parameters_from_query

                q_text = state.get("query") or args.get("query", "")
                q_ext = extract_parameters_from_query(q_text)
                explicit_query_date = q_ext.get("issue_date")
                explicit_query_np = q_ext.get("newspaper_name")
                has_date_conflict = bool(
                    explicit_query_date
                    and chk_art
                    and chk_art.issue
                    and str(chk_art.issue.issue_date) != explicit_query_date
                )
                has_np_conflict = bool(
                    explicit_query_np
                    and chk_art
                    and chk_art.issue
                    and chk_art.issue.newspaper
                    and explicit_query_np.lower() not in chk_art.issue.newspaper.name.lower()
                    and chk_art.issue.newspaper.name.lower() not in explicit_query_np.lower()
                )
                if not has_date_conflict and not has_np_conflict:
                    if chk_art and chk_art.issue:
                        params["issue_date"] = str(chk_art.issue.issue_date)
                        issue_date = params["issue_date"]
                        if chk_art.issue.newspaper:
                            params["newspaper_name"] = chk_art.issue.newspaper.name
                            newspaper_name = params["newspaper_name"]
                else:
                    params["article_id"] = None
                    article_id = None
            else:
                if article_id:
                    logger.warning(
                        "Article ID %s rejected for visual inspection due to newspaper/date mismatch: requested %s (%s)",
                        article_id,
                        newspaper_name,
                        issue_date,
                    )
                params["article_id"] = None
                article_id = None

        if article_id and not photos_to_process:
            stmt = (
                select(Photo)
                .where(Photo.article_id == int(article_id))
                .order_by(Photo.id.asc())
                .options(
                    selectinload(Photo.article)
                    .selectinload(Article.issue)
                    .selectinload(Issue.newspaper),
                    selectinload(Photo.article).selectinload(Article.article_pages),
                )
            )
            res = await session.execute(stmt)
            all_photos = res.scalars().all()
            charts = [p for p in all_photos if p.visual_type in ("data_chart", "infographic", "table")]
            other = [p for p in all_photos if p not in charts]
            photos_to_process = (charts + other)[:6]

            # If article genuinely has 0 visual assets, return clear verification notice
            if not photos_to_process:
                art_stmt = (
                    select(Article)
                    .where(Article.id == int(article_id))
                    .options(
                        selectinload(Article.issue).selectinload(Issue.newspaper),
                        selectinload(Article.article_pages),
                    )
                )
                art_obj = (await session.execute(art_stmt)).scalar_one_or_none()
                if art_obj:
                    np_name = art_obj.issue.newspaper.name if art_obj.issue and art_obj.issue.newspaper else "Archive Publication"
                    iss_d = str(art_obj.issue.issue_date) if art_obj.issue else "Current"
                    pg = art_obj.article_pages[0].page_number if art_obj.article_pages else 1
                    early_items = [{
                        "article_id": art_obj.id,
                        "issue_id": art_obj.issue_id,
                        "headline": art_obj.headline or "Article",
                        "newspaper_name": np_name,
                        "issue_date": iss_d,
                        "pages": [pg],
                        "snippet": (
                            f"=== VISUAL ASSET INSPECTION RESULT ===\n"
                            f"Article: {art_obj.headline}\n"
                            f"Publication: {np_name} ({iss_d}), Page {pg}\n\n"
                            f"Visual Asset Audit: Verified that no infographics, data charts, graphs, or visual tables are attached to this article in the broadsheet layout."
                        ),
                        "prominence_score": 0.9,
                        "source_tool": "inspect_visual_asset",
                        "is_visual_asset": False,
                    }]
                    return [], early_items

        # Strategy D: Multi-criteria database search by metadata (newspaper, date, page, query)
        has_explicit_attached_asset = bool(
            state.get("attached_photo_id")
            or state.get("attached_article_id")
            or args.get("photo_id")
            or args.get("article_id")
        )
        if not photos_to_process and not has_explicit_attached_asset:
            cand_stmt = (
                select(Article)
                .join(Issue, Article.issue_id == Issue.id)
                .join(Newspaper, Issue.newspaper_id == Newspaper.id)
            )
            if newspaper_name:
                cand_stmt = cand_stmt.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
            if issue_date:
                cand_stmt = cand_stmt.where(Issue.issue_date == issue_date)
            if page_filter and str(page_filter).strip().isdigit():
                cand_stmt = cand_stmt.join(ArticlePage, ArticlePage.article_id == Article.id).where(
                    ArticlePage.page_number == int(str(page_filter).strip())
                )
            if target_headline:
                cand_stmt = cand_stmt.where(
                    (Article.headline == target_headline)
                    | (Article.headline.ilike(f"%{target_headline[:50]}%"))
                )

            cand_res = await session.execute(cand_stmt.limit(30))
            candidates = cand_res.scalars().all()

            q_tokens = [w.lower() for w in re.findall(r"\w+", query_text) if len(w) > 2]
            scored_cands: list[tuple[int, Article]] = []
            for cand in candidates:
                hl_lower = (cand.headline or "").lower()
                score = sum(1 for tok in q_tokens if tok in hl_lower)
                if score > 0 or target_headline:
                    scored_cands.append((score, cand))
            scored_cands.sort(key=lambda x: x[0], reverse=True)

            if scored_cands:
                best_art = scored_cands[0][1]
                p_stmt = (
                    select(Photo)
                    .where(Photo.article_id == best_art.id)
                    .order_by(Photo.id.asc())
                    .options(
                        selectinload(Photo.article)
                        .selectinload(Article.issue)
                        .selectinload(Issue.newspaper),
                        selectinload(Photo.article).selectinload(Article.article_pages),
                    )
                )
                p_res = await session.execute(p_stmt)
                all_art_photos = p_res.scalars().all()
                charts = [p for p in all_art_photos if p.visual_type in ("data_chart", "infographic", "table")]
                other = [p for p in all_art_photos if p not in charts]
                photos_to_process = (charts + other)[:6]

        # Strategy E: Fallback search on Photo captions and VLM descriptions
        if not photos_to_process and query_text:
            _photo_stopwords = {
                "the", "and", "for", "any", "have", "with", "from", "that", "this",
                "what", "does", "dated", "photo", "photos", "picture", "pictures",
                "image", "images", "there",
            }
            q_tokens = [w.lower() for w in re.findall(r"\w+", query_text) if len(w) >= 3 and w.lower() not in _photo_stopwords]
            if q_tokens:
                p_search_stmt = (
                    select(Photo)
                    .join(Article, Photo.article_id == Article.id)
                    .where(Photo.visual_type.in_(["data_chart", "infographic", "table", "photo"]))
                    .options(
                        selectinload(Photo.article)
                        .selectinload(Article.issue)
                        .selectinload(Issue.newspaper),
                        selectinload(Photo.article).selectinload(Article.article_pages),
                    )
                    .order_by(Photo.id.desc())
                    .limit(10)
                )
                if newspaper_name or issue_date:
                    p_search_stmt = (
                        p_search_stmt.join(Issue, Article.issue_id == Issue.id)
                        .join(Newspaper, Issue.newspaper_id == Newspaper.id)
                    )
                    if newspaper_name:
                        p_search_stmt = p_search_stmt.where(Newspaper.name.ilike(f"%{newspaper_name}%"))
                    if issue_date:
                        p_search_stmt = p_search_stmt.where(Issue.issue_date == issue_date)

                p_search_res = await session.execute(p_search_stmt)
                cand_photos = p_search_res.scalars().all()
                scored_photos: list[tuple[int, Photo]] = []
                for cp in cand_photos:
                    blob = f"{cp.caption or ''} {cp.vlm_description or ''} {(cp.article.headline if cp.article else '')}".lower()
                    sc = sum(1 for tok in q_tokens if tok in blob)
                    if sc > 0:
                        scored_photos.append((sc, cp))
                scored_photos.sort(key=lambda x: x[0], reverse=True)
                if scored_photos:
                    photos_to_process = [sp[1] for sp in scored_photos[:4]]

        return photos_to_process, None

    async def enrich_photo_vlm_descriptions(
        self,
        session: AsyncSession,
        photos: list[Photo],
    ) -> None:
        """On-demand MinIO image crop fetch and Gemini VLM extraction."""
        for target_photo in photos:
            desc = (target_photo.vlm_description or "").strip()
            is_placeholder = (
                not desc
                or desc.startswith("Visual asset:")
                or desc in (
                    "Editorial news photograph.",
                    "Visual asset: infographic from broadsheet.",
                    "Visual asset: table from broadsheet.",
                    "Visual asset: data_chart from broadsheet.",
                )
            )

            if is_placeholder and target_photo.object_key:
                try:
                    from app.core.config import get_settings
                    from app.ingestion.visual_extractor import VisualDataExtractor
                    from app.storage import get_object_store

                    cfg = get_settings()
                    store = get_object_store(cfg)
                    crop_bytes = await store.get(
                        bucket=cfg.minio.bucket_pages,
                        key=target_photo.object_key,
                    )
                    if crop_bytes:
                        extractor = VisualDataExtractor()
                        classification, extraction = await extractor.process_image_crop(
                            image_bytes=crop_bytes,
                            ocr_text=target_photo.caption or "",
                        )
                        if extraction:
                            parts = [extraction.summary]
                            if extraction.key_metrics:
                                parts.append("\nKey Data Points & Metrics:\n• " + "\n• ".join(extraction.key_metrics))
                            if extraction.markdown_table:
                                parts.append("\n" + extraction.markdown_table)
                            new_desc = "\n".join(parts)

                            target_photo.visual_type = classification.visual_type or target_photo.visual_type
                            target_photo.vlm_description = new_desc
                            session.add(target_photo)
                            await session.commit()
                            await session.refresh(target_photo)
                except Exception as vlm_err:
                    logger.warning(
                        "On-demand visual extraction failed in inspect_visual_asset",
                        extra={"photo_id": target_photo.id, "error": str(vlm_err)},
                    )

    def format_visual_asset_item(
        self,
        target_photo: Photo,
        params: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Format a resolved Photo into a standardized evidence item."""
        article = target_photo.article
        hl = article.headline if article else "Visual Intelligence Asset"
        issue = article.issue if article else None
        np_name = issue.newspaper.name if issue and issue.newspaper else (params.get("newspaper_name") or state.get("active_newspaper_name") or "Archive Publication")
        iss_d = str(issue.issue_date) if issue else (params.get("issue_date") or state.get("active_issue_date") or "Current")
        page_filter = params.get("page_filter") or ""
        page_val = article.article_pages[0].page_number if (article and article.article_pages) else (int(page_filter) if str(page_filter).isdigit() else 1)

        v_type_clean = (target_photo.visual_type or "Infographic").replace("_", " ").title()
        vlm_content = target_photo.vlm_description or target_photo.caption or "Visual graphic from broadsheet."

        snippet = (
            f"=== VISUAL DATA ASSET: {v_type_clean} (Asset #{target_photo.id}) ===\n"
            f"Headline: {hl}\n"
            f"Publication: {np_name} ({iss_d}), Page {page_val}\n"
            f"Visual Type: {v_type_clean}\n"
            f"Printed Caption: {target_photo.caption or None}\n\n"
            f"Extracted Infographic Data & Visual Table Content:\n{vlm_content}"
        )

        return {
            "article_id": target_photo.article_id or (article.id if article else 0),
            "issue_id": article.issue_id if article else 0,
            "headline": hl,
            "newspaper_name": np_name,
            "issue_date": iss_d,
            "pages": [page_val],
            "bboxes": target_photo.bbox_json.get("bbox", []) if target_photo.bbox_json else [],
            "snippet": snippet,
            "prominence_score": 1.0,
            "source_tool": "inspect_visual_asset",
            "is_visual_asset": True,
            "photo_id": target_photo.id,
            "visual_type": target_photo.visual_type,
            "image_url": f"/api/photos/{target_photo.id}/image",
            "vlm_description": target_photo.vlm_description,
            "caption": target_photo.caption,
        }


__all__ = ["VisualInspectionEngine"]
