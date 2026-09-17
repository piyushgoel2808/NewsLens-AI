"""Core ingestion pipeline orchestrator for NewsLens-AI.

Unified Two-Phase Architecture (Gemini Vision / Gemma 4):
1. PDF Rasterization & Digital Text Detection (PyMuPDF)
2. Phase 1: High-Speed Parallel Page Layout & Article Boundary Extraction
3. Issue-wide Multi-Page Brand & Publication Date Consensus
4. Phase 2: Selective Deep Enrichment (Body Text, NER, Topics, Tables) for Major Articles
5. Cross-Page Story Continuation Linking (CrossPageAssembler)
6. Atomic Relational Database Persistence (Articles, Entities, Topics, Tables, Photos)
7. Contextual Newspaper Chunking & Qdrant Vector Indexing
8. Diagnostic JSON Manifest Generation (DebugArtifactsExporter)
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Sequence
from datetime import date
from typing import Any

import pymupdf
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.celery_app import celery_app
from app.ingestion.chunker import NewspaperChunker
from app.ingestion.classifier import ArticleClassifier
from app.ingestion.detector import PDFPageDetector, check_is_advertisement_text
from app.ingestion.embedder import ArticleEmbedder
from app.ingestion.layout import (
    ArticleSegmenter,
    AssembledArticle,
    CrossPageAssembler,
    LayoutAnalyzer,
    SegmentedArticle,
)
from app.ingestion.media_extractor import MediaExtractor
from app.ingestion.metadata import ConsensusExtractor
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.parsers import (
    DoclingLayoutParser,
    ExtractedPhotoData,
    PageLayoutExtraction,
    UnifiedExtractor,
)
from app.ingestion.rasterizer import PDFRasterizer, RasterizedPage
from app.ingestion.storage import DebugArtifactsExporter
from app.models.article import Article, ArticleCategory, ArticleChunk, ArticlePage, Photo
from app.models.entity import ArticleEntity, ArticleTopic, Topic
from app.models.newspaper import Issue, Newspaper, Page
from app.providers.openrouter_provider import RateLimitExhaustedError
from app.storage import get_object_store
from app.storage.base import ObjectStore

logger = get_logger(__name__)



def detect_masthead_and_date(blocks: Sequence[Any], height_px: float) -> tuple[str | None, date | None]:
    """Helper to detect masthead and date from top header blocks."""
    from app.ingestion.metadata import (
        _DATE_PATTERNS,
        _KNOWN_MASTHEADS,
        _parse_extracted_date,
    )
    detected_brand: str | None = None
    detected_date: date | None = None

    header_zone = height_px * 0.15
    for b in blocks:
        raw_bbox = getattr(b, "bbox", (0, 0, 0, 0))
        y0 = raw_bbox[1] if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 2 else 0.0
        if y0 <= header_zone:
            txt = getattr(b, "text", "")
            txt_upper = txt.upper()
            if not detected_brand:
                for token, canonical in _KNOWN_MASTHEADS:
                    if token in txt_upper:
                        detected_brand = canonical
                        break
            if not detected_date:
                for pattern in _DATE_PATTERNS:
                    match = pattern.search(txt)
                    if match:
                        parsed = _parse_extracted_date(match.groups())
                        if parsed:
                            detected_date = parsed
                            break
    return detected_brand, detected_date


async def run_ingestion_pipeline(
    issue_id: int,
    pdf_bytes: bytes,
    dpi: int = 150,
    parser_engine: str = "auto",
    minio: ObjectStore | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end high-speed single-pass ingestion pipeline for an issue."""
    settings = get_settings()
    store = minio or get_object_store(settings)

    if session_factory is None:
        engine = create_async_engine(settings.database.async_url, echo=False)
        maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    else:
        maker = session_factory

    async with maker() as db:
        # Step 1: Rasterize PDF pages to PNG (MinIO upload)
        rasterizer = PDFRasterizer(db=db, minio=store)
        rendered_pages = await rasterizer.rasterize_pdf_bytes(
            pdf_bytes=pdf_bytes,
            issue_id=issue_id,
            dpi=dpi,
        )

        # Step 2: Digital text extraction (PyMuPDF)
        detector = PDFPageDetector()
        analysis_results = detector.analyze_document_bytes(pdf_bytes, dpi=dpi)

        # Fetch Issue record
        issue_stmt = (
            select(Issue).where(Issue.id == issue_id).options(selectinload(Issue.newspaper))
        )
        issue_res = await db.execute(issue_stmt)
        issue = issue_res.scalar_one_or_none()
        issue_lang = issue.language if issue else "en"

        # Step 3: Run Multi-Page Consensus for Brand & Date
        extractor_engine = UnifiedExtractor(engine_name=parser_engine)
        consensus_extractor = ConsensusExtractor(max_pages=15)
        con_brand, con_date, con_telem = consensus_extractor.extract_consensus(
            pdf_bytes=pdf_bytes,
            filename=issue.edition if issue else None,
        )

        if issue:
            issue.total_pages = len(rendered_pages)
            issue.ingestion_status = "processing"

        if con_brand and issue:
            np_stmt = select(Newspaper).where(Newspaper.name == con_brand)
            np_res = await db.execute(np_stmt)
            np_obj = np_res.scalar_one_or_none()
            if not np_obj:
                np_obj = Newspaper(name=con_brand, default_language=issue_lang)
                db.add(np_obj)
                await db.flush()
            issue.newspaper_id = np_obj.id
            issue.newspaper = np_obj
            logger.info("Consensus identified newspaper brand", extra={"newspaper": con_brand, "issue_id": issue_id})

        if con_date and issue:
            issue.issue_date = con_date
            logger.info("Consensus identified publication date", extra={"issue_date": str(con_date), "issue_id": issue_id})
        await db.flush()

        is_docling_engine = not parser_engine or "docling" in parser_engine.lower() or parser_engine.lower() == "auto"
        phase1_layouts: dict[int, PageLayoutExtraction] = {}

        if not is_docling_engine:
            # Step 4: Phase 1 Bounded Parallel Extraction (Page Skeletons & Layout)
            # 4 concurrent requests for cloud (Gemini / OpenRouter), 1 for local Ollama
            is_local = "ollama" in (parser_engine or "").lower()
            concurrency_limit = 1 if is_local else 4
            semaphore = asyncio.Semaphore(concurrency_limit)

            async def _extract_page_phase1(idx: int, rendered: RasterizedPage) -> tuple[int, PageLayoutExtraction]:
                from app.providers.openrouter_provider import RateLimitExhaustedError

                async with semaphore:
                    digital_hint = analysis_results[idx].full_text if idx < len(analysis_results) else None
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            layout = await extractor_engine.extract_page_layout(
                                page_number=rendered.page_number,
                                image_bytes=rendered.image_bytes,
                                digital_text_hint=digital_hint,
                            )
                            return (rendered.page_number, layout)
                        except RateLimitExhaustedError as rle:
                            if attempt == max_attempts - 1:
                                logger.error(
                                    "Rate limits exhausted on page %d after %d attempts: %s",
                                    rendered.page_number, max_attempts, rle
                                )
                                raise
                            wait_s = getattr(rle, "retry_after_seconds", 15.0) * (attempt + 1)
                            logger.warning(
                                "OpenRouter rate limits exhausted on page %d. Backing off for %.1fs (attempt %d/%d)...",
                                rendered.page_number, wait_s, attempt + 1, max_attempts
                            )
                            await asyncio.sleep(wait_s)
                    return (rendered.page_number, PageLayoutExtraction(page_number=rendered.page_number))

            phase1_tasks = [_extract_page_phase1(i, r) for i, r in enumerate(rendered_pages)]
            phase1_results_list = await asyncio.gather(*phase1_tasks)
            phase1_layouts = dict(phase1_results_list)

        # Step 5: Convert Skeletons & Run Selective Phase 2 Enrichment
        media_extractor = MediaExtractor(minio=store, db=db)
        all_pages_articles: dict[int, list[SegmentedArticle]] = {}
        page_media_items: dict[int, list[ExtractedPhotoData]] = {}
        parsed_doc_items_by_page: dict[int, list[Any]] = {}
        page_extractions: list[dict[str, Any]] = []

        for i, rendered in enumerate(rendered_pages):
            page_num = rendered.page_number
            analysis = analysis_results[i]
            layout_data = phase1_layouts.get(page_num, PageLayoutExtraction(page_number=page_num))

            # Database Page update
            stmt = select(Page).where(Page.issue_id == issue_id, Page.page_number == page_num)
            res = await db.execute(stmt)
            page_record = res.scalar_one_or_none()

            if page_record:
                page_record.is_advertisement_page = layout_data.is_advertisement_page or analysis.is_advertisement
                page_record.ingestion_status = "layout_done"

            # Identify if using pure Google Cloud Vision layout (already has complete OCR blocks for the page)
            is_gcv_engine = "google" in (parser_engine or "").lower() or "vision" in (parser_engine or "").lower()

            # Process articles on this page
            page_segmented_articles: list[SegmentedArticle] = []

            # If docling engine is selected, OR Phase 1 layout extraction returned 0 articles (or empty layout)
            if is_docling_engine or not layout_data.articles:
                logger.info(
                    "Extracting articles using DoclingLayoutParser",
                    extra={"page_number": page_num, "is_docling_engine": is_docling_engine},
                )
                try:
                    docling_parser = DoclingLayoutParser()
                    src_pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    single_doc = pymupdf.open()
                    single_doc.insert_pdf(src_pdf, from_page=i, to_page=i)
                    page_pdf_bytes = single_doc.tobytes()
                    single_doc.close()
                    src_pdf.close()

                    parsed_doc_items = await asyncio.get_running_loop().run_in_executor(
                        None,
                        docling_parser.parse_docling_document,
                        page_pdf_bytes,
                        page_num,
                        int(rendered.width_px),
                        int(rendered.height_px),
                    )
                    docling_articles = docling_parser.assemble_articles(
                        page_number=page_num,
                        items=parsed_doc_items,
                        width_px=int(rendered.width_px),
                        height_px=int(rendered.height_px),
                        is_advertisement_page=page_record.is_advertisement_page if page_record else False,
                    )
                    page_media_items[page_num] = docling_parser.extract_page_media_items(parsed_doc_items)
                    parsed_doc_items_by_page[page_num] = parsed_doc_items
                    if docling_articles:
                        page_segmented_articles.extend(docling_articles)
                except Exception as docling_err:
                    logger.warning(
                        "DoclingLayoutParser failed on page, attempting legacy fallback",
                        extra={"page_number": page_num, "error": str(docling_err)},
                    )
                    if len(analysis.blocks) > 0:
                        try:
                            layout_analyzer = LayoutAnalyzer()
                            page_layout_res = layout_analyzer.analyze_from_text_blocks(
                                page_number=page_num,
                                width_px=int(rendered.width_px),
                                height_px=int(rendered.height_px),
                                digital_blocks=analysis.blocks,
                            )
                            segmenter = ArticleSegmenter()
                            seg_fallback = segmenter.segment_page(
                                page_number=page_num,
                                ordered_blocks=page_layout_res.reading_order,
                                is_advertisement_page=page_record.is_advertisement_page if page_record else False,
                            )
                            if seg_fallback:
                                page_segmented_articles.extend(seg_fallback)
                        except Exception as fallback_err:
                            logger.warning(
                                "LayoutAnalyzer fallback failed on page",
                                extra={"page_number": page_num, "error": str(fallback_err)},
                            )
                    else:
                        # Corrupted font stream or scanned page: run pure OCR on rendered image
                        try:
                            from app.providers.google_vision_provider import GoogleCloudVisionOCR
                            gcv_key = settings.google_api_key or settings.gemini_api_key
                            gcv_ocr = GoogleCloudVisionOCR(api_key=gcv_key)
                            ocr_res = await gcv_ocr.ocr(image_bytes=rendered.image_bytes, lang_hint=issue_lang)
                            if ocr_res.blocks:
                                layout_analyzer = LayoutAnalyzer()
                                page_layout_res = layout_analyzer.analyze_from_text_blocks(
                                    page_number=page_num,
                                    width_px=int(rendered.width_px),
                                    height_px=int(rendered.height_px),
                                    ocr_blocks=ocr_res.blocks,
                                )
                                segmenter = ArticleSegmenter()
                                seg_fallback = segmenter.segment_page(
                                    page_number=page_num,
                                    ordered_blocks=page_layout_res.reading_order,
                                    is_advertisement_page=page_record.is_advertisement_page if page_record else False,
                                )
                                if seg_fallback:
                                    page_segmented_articles.extend(seg_fallback)
                        except Exception as ocr_fb_err:
                            logger.warning(
                                "GCV OCR fallback failed on page",
                                extra={"page_number": page_num, "error": str(ocr_fb_err)},
                            )

            # Process LLM/VLM articles on this page if not already populated by Docling
            if not page_segmented_articles:
                for art_idx, skel in enumerate(layout_data.articles):
                    is_minor = (
                        skel.prominence in ("minor", "filler")
                        or skel.article_type in ("advertisement", "photo_caption", "table_data", "index", "teaser")
                    )

                    # Convert bounding box to [x0, y0, x1, y1] normalized/pixel
                    bbox = skel.bbox if len(skel.bbox) == 4 else [0.0, 0.0, 1000.0, 1000.0]

                    # Phase 2 enrichment:
                    # When using Google Cloud Vision, full-page OCR blocks are already extracted with 98%+ accuracy.
                    # Avoid wasteful per-article crop API calls and extract directly from skel.body_text / page text.
                    if is_gcv_engine or is_minor:
                        full_body = skel.body_text or (
                            f"{skel.headline}\n\n{skel.subheadline}" if skel.subheadline else (skel.headline or "News item.")
                        )
                    else:
                        # For LLM-based extraction (Gemini / Gemma / OpenRouter), enrich major articles with VLM crop
                        from app.providers.openrouter_provider import RateLimitExhaustedError

                        crop_bytes = extractor_engine.crop_article_image(rendered.image_bytes, bbox)
                        full_body = skel.body_text or skel.headline or "News item."
                        for art_attempt in range(3):
                            try:
                                enrichment = await extractor_engine.enrich_article(
                                    headline=skel.headline,
                                    article_crop_bytes=crop_bytes,
                                    digital_slice=analysis.full_text,
                                )
                                full_body = enrichment.body_text
                                break
                            except RateLimitExhaustedError as rle:
                                if art_attempt == 2:
                                    logger.warning("All keys exhausted for article %s, using skeleton text", skel.headline)
                                    break
                                wait_s = getattr(rle, "retry_after_seconds", 15.0) * (art_attempt + 1)
                                logger.warning("Rate limits exhausted enriching article, sleeping %.1fs...", wait_s)
                                await asyncio.sleep(wait_s)

                    # Convert normalized [ymin, xmin, ymax, xmax] (0..1000) to standard [x0, y0, x1, y1] pixels
                    if max(bbox) <= 1000.0 and (rendered.width_px > 1000 or rendered.height_px > 1000):
                        # Normalized 0..1000 coordinates [ymin, xmin, ymax, xmax]
                        y0 = float(bbox[0]) / 1000.0 * float(rendered.height_px)
                        x0 = float(bbox[1]) / 1000.0 * float(rendered.width_px)
                        y1 = float(bbox[2]) / 1000.0 * float(rendered.height_px)
                        x1 = float(bbox[3]) / 1000.0 * float(rendered.width_px)
                        box_tuple = (x0, y0, x1, y1)
                    else:
                        # Already absolute pixel coordinates [x0, y0, x1, y1] or [y0, x0, y1, x1]
                        box_tuple = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

                    seg = SegmentedArticle(
                        article_temp_id=f"page_{page_num}_art_{art_idx}",
                        headline=skel.headline,
                        subheadline=skel.subheadline,
                        byline_author=skel.byline,
                        body_text=full_body,
                        word_count=len(full_body.split()),
                        bbox_list=[box_tuple],
                        jump_to_page=skel.continues_to_page,
                        jump_from_page=skel.continued_from_page,
                        is_teaser=(skel.article_type == "teaser"),
                    )
                    page_segmented_articles.append(seg)

            # Fallback if 0 articles detected on a non-empty page
            if not page_segmented_articles:
                full_page_txt = ""
                if hasattr(analysis, "full_text") and analysis.full_text.strip():
                    full_page_txt = analysis.full_text.strip()
                elif "parsed_doc_items" in locals() and parsed_doc_items:
                    full_page_txt = "\n\n".join(it.text for it in parsed_doc_items if it.text.strip()).strip()

                is_ad_page = (page_record.is_advertisement_page if page_record else False) or (
                    bool(full_page_txt and check_is_advertisement_text(full_page_txt))
                )

                if full_page_txt and len(full_page_txt.split()) >= 6:
                    fallback_hl = (
                        f"[Advertisement] Page {page_num} Feature"
                        if is_ad_page
                        else f"Page {page_num} Feature"
                    )
                    lines = [line_str.strip() for line_str in full_page_txt.split("\n") if line_str.strip()]
                    if lines and 3 <= len(lines[0].split()) <= 12:
                        fallback_hl = f"[Advertisement] {lines[0]}" if is_ad_page else lines[0]

                    page_segmented_articles.append(
                        SegmentedArticle(
                            article_temp_id=f"page_{page_num}_art_fallback",
                            headline=fallback_hl,
                            body_text=full_page_txt,
                            word_count=len(full_page_txt.split()),
                            bbox_list=[(0.0, 0.0, float(rendered.width_px), float(rendered.height_px))],
                        )
                    )

            all_pages_articles[page_num] = page_segmented_articles

            page_extractions.append({
                "page_number": page_num,
                "is_advertisement_page": page_record.is_advertisement_page if page_record else False,
                "articles_count": len(page_segmented_articles),
            })

        # Step 6: Cross-Page Continuation Assembly
        assembler = CrossPageAssembler()
        assembled_articles = assembler.assemble_issue_articles(all_pages_articles)

        # Step 7: Classification, Enrichment, Embedding & DB Persistence
        classifier = ArticleClassifier()
        meta_extractor = MetadataExtractor(db=db)
        chunker = NewspaperChunker()
        embedder = ArticleEmbedder(db=db)

        # Purge previous articles for idempotent re-ingestion
        with contextlib.suppress(Exception):
            await embedder.delete_issue_vectors(issue_id)

        existing_arts_res = await db.execute(select(Article.id).where(Article.issue_id == issue_id))
        existing_art_ids = existing_arts_res.scalars().all()
        if existing_art_ids:
            await db.execute(delete(ArticleEntity).where(ArticleEntity.article_id.in_(existing_art_ids)))
            await db.execute(delete(ArticleTopic).where(ArticleTopic.article_id.in_(existing_art_ids)))
            await db.execute(delete(ArticleChunk).where(ArticleChunk.article_id.in_(existing_art_ids)))
            await db.execute(delete(ArticlePage).where(ArticlePage.article_id.in_(existing_art_ids)))
            await db.execute(delete(Article).where(Article.issue_id == issue_id))
            await db.flush()

        newspaper_name = issue.newspaper.name if issue and issue.newspaper else "Daily"
        issue_date_str = str(issue.issue_date) if issue else ""

        page_id_map: dict[int, int] = {}
        pages_fetch = await db.execute(select(Page).where(Page.issue_id == issue_id))
        for p in pages_fetch.scalars().all():
            page_id_map[p.page_number] = p.id

        articles_manifest: list[dict[str, Any]] = []
        total_chunks_created = 0

        # Step 7a: First Pass - Persist Articles to obtain article IDs
        persisted_articles: list[tuple[Article, AssembledArticle, Any]] = []
        article_envelopes_by_page: dict[int, list[Any]] = {}

        for assembled in assembled_articles:
            class_res = classifier.classify_and_score(
                article=assembled,
                total_issue_pages=len(rendered_pages),
                printed_section=assembled.printed_section,
            )

            is_explicit_ad = assembled.headline.startswith(("[Advertisement]", "[Public Notice]"))
            full_txt = assembled.full_text or ""
            text_is_ad = check_is_advertisement_text(full_txt)
            has_editorial_protection = (
                bool(assembled.byline_author and assembled.byline_author.strip())
                or (len(full_txt.split()) >= 300 and class_res.article_type != "advertisement")
            )

            if is_explicit_ad:
                is_ad = True
            elif has_editorial_protection:
                is_ad = False
            else:
                is_ad = (class_res.article_type == "advertisement" or text_is_ad)

            primary_page_id = page_id_map.get(assembled.primary_page_number)
            clean_hl = (assembled.headline or "")[:1024]
            if is_ad and not clean_hl.startswith(("[Advertisement]", "[Public Notice]")):
                clean_hl = f"[Advertisement] {clean_hl}"[:1024]

            # Resolve Canonical Category ID if available
            cat_id: int | None = None
            if class_res.category:
                cat_fetch = await db.execute(
                    select(ArticleCategory.id).where(ArticleCategory.name == class_res.category)
                )
                cat_id = cat_fetch.scalar_one_or_none()
                if not cat_id:
                    new_cat = ArticleCategory(name=class_res.category)
                    db.add(new_cat)
                    await db.flush()
                    cat_id = new_cat.id

            article_record = Article(
                issue_id=issue_id,
                primary_page_id=primary_page_id,
                category_id=cat_id,
                category_confidence=class_res.category_confidence,
                headline=clean_hl,
                subheadline=assembled.subheadline[:1024] if assembled.subheadline else None,
                byline_author=assembled.byline_author[:512] if assembled.byline_author else None,
                section=class_res.section[:255] if class_res.section else "National",
                printed_section=class_res.printed_section[:128] if class_res.printed_section else None,
                article_type="advertisement" if is_ad else class_res.article_type,
                language=issue_lang,
                prominence_score=class_res.prominence_score,
                word_count=assembled.word_count,
                full_text=assembled.full_text,
            )
            db.add(article_record)
            await db.flush()

            # Save Multi-Topic Secondary Categories if present
            if class_res.secondary_categories:
                for sec_cat_name, sec_conf in class_res.secondary_categories:
                    t_fetch = await db.execute(
                        select(Topic.id).where(Topic.name == sec_cat_name)
                    )
                    t_id = t_fetch.scalar_one_or_none()
                    if not t_id:
                        new_t = Topic(name=sec_cat_name, taxonomy_path=f"Newsroom > {sec_cat_name}")
                        db.add(new_t)
                        await db.flush()
                        t_id = new_t.id
                    art_topic = ArticleTopic(
                        article_id=article_record.id,
                        topic_id=t_id,
                        confidence=sec_conf,
                    )
                    db.add(art_topic)

            persisted_articles.append((article_record, assembled, class_res))

            # Save ArticlePage mappings and record spatial envelopes for photo binding
            for p_map in assembled.pages_mapping:
                mapped_pid = page_id_map.get(p_map.page_number)
                if mapped_pid:
                    art_page = ArticlePage(
                        article_id=article_record.id,
                        page_id=mapped_pid,
                        page_number=p_map.page_number,
                        bbox_json={"bboxes": [list(b) for b in p_map.bbox_list]},
                        block_order=p_map.block_order,
                    )
                    db.add(art_page)

                    if p_map.bbox_list:
                        env_x0 = min(b[0] for b in p_map.bbox_list)
                        env_y0 = min(b[1] for b in p_map.bbox_list)
                        env_x1 = max(b[2] for b in p_map.bbox_list)
                        env_y1 = max(b[3] for b in p_map.bbox_list)
                        envelope = (env_x0, env_y0, env_x1, env_y1)
                    else:
                        envelope = (0.0, 0.0, 1000.0, 1000.0)

                    if p_map.page_number not in article_envelopes_by_page:
                        article_envelopes_by_page[p_map.page_number] = []
                    article_envelopes_by_page[p_map.page_number].append(
                        (
                            article_record.id,
                            envelope,
                            article_record.headline or "",
                            p_map.bbox_list or [],
                        )
                    )
            await db.flush()

        # Step 7b: Extract & Store Visual Regions for each page (Single-Pass 1 Call Per Page)
        article_photos_map: dict[int, list[Photo]] = {}

        # Concurrency semaphore for single-pass visual extraction (capped at 2 to protect TPM limits)
        visual_semaphore = asyncio.Semaphore(2)

        async def _process_page_visuals(rendered: RasterizedPage, analysis: Any) -> list[tuple[int | None, Photo]]:
            p_num = rendered.page_number
            p_id = page_id_map.get(p_num)
            if not p_id:
                return []

            page_envs = article_envelopes_by_page.get(p_num, [])

            # Harvest media items from Docling
            media_list = list(page_media_items.get(p_num, []))

            # Also incorporate any PyMuPDF image_boxes not covered by Docling
            for ibox in analysis.image_boxes:
                if not any(
                    abs(ibox[0] - m.bbox[0]) < 30 and abs(ibox[1] - m.bbox[1]) < 30
                    for m in media_list
                    if m.bbox
                ):
                    media_list.append(ExtractedPhotoData(bbox=ibox, caption=""))

            if not media_list:
                return []

            async with visual_semaphore:
                return await media_extractor.extract_and_store_all_photos_single_pass(
                    page_image_bytes=rendered.image_bytes,
                    page_id=p_id,
                    article_envelopes=page_envs,
                    media_items=media_list,
                    all_docling_items=parsed_doc_items_by_page.get(p_num, []),
                    page_width_px=rendered.width_px,
                    page_height_px=rendered.height_px,
                    start_photo_index=1,
                )

        # Execute single-pass visual extraction across pages with Semaphore(2)
        visual_tasks = [
            _process_page_visuals(rendered, analysis)
            for rendered, analysis in zip(rendered_pages, analysis_results, strict=False)
        ]
        page_visual_results = await asyncio.gather(*visual_tasks)

        for pairs in page_visual_results:
            for bound_art_id, photo_rec in pairs:
                if bound_art_id:
                    if bound_art_id not in article_photos_map:
                        article_photos_map[bound_art_id] = []
                    article_photos_map[bound_art_id].append(photo_rec)

        # Step 7c: Chunking, Visual Tag Markup, and Vector Indexing
        for article_record, assembled, _class_res in persisted_articles:
            is_ad = article_record.article_type == "advertisement"
            art_photos = article_photos_map.get(article_record.id, [])
            has_photo = bool(art_photos)

            # Check if article contains tabular or graphic statistics
            has_table = (
                "table_data" in (article_record.article_type or "")
                or "table" in (assembled.headline or "").lower()
                or bool(re.search(r"\|\s*[-:]+\s*\|", assembled.full_text))
            )

            # Inject visual markup into article full text if visuals exist
            annotated_text = assembled.full_text
            if has_photo:
                annotated_text = f"{annotated_text}\n\n[🖼️ Attached Image/Photo: Visual coverage included on Page {assembled.primary_page_number}]"
            if has_table:
                annotated_text = f"{annotated_text}\n\n[📊 Attached Data / Infographic: Structured statistical representation included]"

            # Update article full text with visual tags
            article_record.full_text = annotated_text

            # Extract & Persist Metadata (NER & Topics)
            meta_res = await meta_extractor.process_and_persist_metadata(
                article_id=article_record.id,
                headline=assembled.headline,
                full_text=annotated_text,
            )

            # Chunk & Embed (skip vector indexing for advertisements)
            if not is_ad:
                # 1. Generate text paragraph chunks
                chunks = chunker.chunk_article(
                    full_text=annotated_text,
                    newspaper_name=newspaper_name,
                    issue_date=issue_date_str,
                    headline=assembled.headline,
                    section=article_record.section or "National",
                    pages=[pm.page_number for pm in assembled.pages_mapping],
                )

                # 2. Generate dedicated visual chunks for infographics, data charts, and tables
                visual_chunks = []
                for p_idx, photo in enumerate(art_photos, start=len(chunks)):
                    if photo.vlm_description and photo.visual_type in {"data_chart", "table", "infographic"}:
                        v_chunk = chunker.create_visual_chunk(
                            visual_markdown=photo.vlm_description,
                            visual_type=photo.visual_type,
                            summary=photo.caption or "",
                            newspaper_name=newspaper_name,
                            issue_date=issue_date_str,
                            headline=assembled.headline,
                            section=article_record.section or "National",
                            pages=[pm.page_number for pm in assembled.pages_mapping],
                            chunk_index=p_idx,
                        )
                        photo_bbox = photo.bbox_json.get("bbox", []) if photo.bbox_json else []
                        visual_chunks.append((v_chunk, photo.visual_type, photo_bbox, photo.vlm_description))

                # Index text chunks
                if chunks:
                    await embedder.embed_and_index_chunks(
                        article_id=article_record.id,
                        issue_id=issue_id,
                        newspaper_name=newspaper_name,
                        issue_date=issue_date_str,
                        headline=article_record.headline or "Untitled",
                        section=article_record.section,
                        article_type=article_record.article_type,
                        prominence_score=article_record.prominence_score,
                        page_numbers=[pm.page_number for pm in assembled.pages_mapping],
                        entities=[e.name for e in meta_res.entities],
                        topics=[t.name for t in meta_res.topics],
                        chunks=chunks,
                        has_photo=has_photo,
                        has_table=has_table,
                        chunk_type="text",
                    )
                    total_chunks_created += len(chunks)

                # Index dedicated visual chunks with bbox and photo_description payload
                for v_chunk, v_type, p_bbox, p_desc in visual_chunks:
                    await embedder.embed_and_index_chunks(
                        article_id=article_record.id,
                        issue_id=issue_id,
                        newspaper_name=newspaper_name,
                        issue_date=issue_date_str,
                        headline=article_record.headline or "Untitled",
                        section=article_record.section,
                        article_type=article_record.article_type,
                        prominence_score=article_record.prominence_score,
                        page_numbers=[pm.page_number for pm in assembled.pages_mapping],
                        entities=[e.name for e in meta_res.entities],
                        topics=[t.name for t in meta_res.topics],
                        chunks=[v_chunk],
                        has_photo=True,
                        has_table=True,
                        chunk_type="visual",
                        has_visual_data=True,
                        visual_type=v_type,
                        bbox=p_bbox,
                        photo_description=p_desc,
                    )
                    total_chunks_created += 1

            articles_manifest.append({
                "article_id": article_record.id,
                "headline": article_record.headline,
                "section": article_record.section,
                "article_type": article_record.article_type,
                "prominence_score": article_record.prominence_score,
                "word_count": article_record.word_count,
                "has_photo": has_photo,
                "has_table": has_table,
            })

        if issue:
            issue.ingestion_status = "completed"
        await db.commit()

        # Step 8: Diagnostic telemetry export
        exporter = DebugArtifactsExporter()
        debug_dir = exporter.export_issue_artifacts(
            issue_id=issue_id,
            newspaper_name=newspaper_name,
            issue_date=issue_date_str,
            edition=issue.edition if issue and issue.edition else "morning",
            page_extractions=page_extractions,
            articles=articles_manifest,
            rag_chunks=[],
            advertisements=[],
        )

        return {
            "issue_id": issue_id,
            "status": "completed",
            "total_articles": len(articles_manifest),
            "total_chunks": total_chunks_created,
            "debug_dir": debug_dir,
        }


@celery_app.task(
    bind=True,
    name="app.ingestion.tasks.process_issue_ingestion_task",
    max_retries=3,
    autoretry_for=(RateLimitExhaustedError,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def process_issue_ingestion_task(self, issue_id: int, pdf_bytes: bytes, dpi: int = 150) -> dict[str, Any]:
    """Synchronous entry point for Celery worker with exponential backoff on total rate limit exhaustion."""
    try:
        return asyncio.run(run_ingestion_pipeline(issue_id=issue_id, pdf_bytes=pdf_bytes, dpi=dpi))
    except RateLimitExhaustedError as exc:
        retries = getattr(self.request, "retries", 0) if hasattr(self, "request") else 0
        logger.warning(
            "Celery task hit RateLimitExhaustedError for issue %d, scheduling retry %d/3...",
            issue_id,
            retries + 1,
        )
        raise self.retry(exc=exc, countdown=min(60 * (2 ** retries), 300)) from exc


__all__ = [
    "process_issue_ingestion_task",
    "run_ingestion_pipeline",
]

