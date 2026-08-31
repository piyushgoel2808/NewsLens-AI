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
from app.ingestion.chunker import NewspaperChunker
from app.ingestion.classifier import ArticleClassifier
from app.ingestion.consensus_extractor import ConsensusExtractor
from app.ingestion.cross_page_assembler import AssembledArticle, CrossPageAssembler
from app.ingestion.debug_exporter import DebugArtifactsExporter
from app.ingestion.detector import PDFPageDetector
from app.ingestion.docling_parser import DoclingLayoutParser, ExtractedPhotoData
from app.ingestion.embedder import ArticleEmbedder
from app.ingestion.extraction_schemas import (
    PageLayoutExtraction,
)
from app.ingestion.folio_detector import FolioDetector
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.media_extractor import MediaExtractor
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.rasterizer import PDFRasterizer, RasterizedPage
from app.ingestion.segmenter import ArticleSegmenter, SegmentedArticle
from app.ingestion.unified_extractor import UnifiedExtractor
from app.models.article import Article, ArticleCategory, ArticleChunk, ArticlePage, Photo
from app.models.entity import ArticleEntity, ArticleTopic, Topic
from app.models.newspaper import Issue, Newspaper, Page
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)


def detect_masthead_and_date(blocks: Sequence[Any], height_px: float) -> tuple[str | None, date | None]:
    """Helper to detect masthead and date from top header blocks."""
    from app.ingestion.consensus_extractor import (
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


def check_is_advertisement_text(text: str) -> bool:
    """Fast check for ad keywords in text."""
    lower = text.lower()
    ad_kws = (
        "advertisement", "advertorial", "special feature", "public notice",
        "tender notice", "ipo", "red herring", "terms and conditions apply",
        "t&c apply", "statutory notice"
    )
    return any(kw in lower for kw in ad_kws)


async def run_ingestion_pipeline(
    issue_id: int,
    pdf_bytes: bytes,
    dpi: int = 300,
    parser_engine: str = "auto",
    minio: MinioStore | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end high-speed single-pass ingestion pipeline for an issue."""
    settings = get_settings()
    store = minio or MinioStore(settings.minio)

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
        analysis_results = detector.analyze_document_bytes(pdf_bytes)

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
            # 4 concurrent requests for cloud Gemini, 1 for local Ollama
            concurrency_limit = 1 if "gemma" in (parser_engine or "").lower() else 4
            semaphore = asyncio.Semaphore(concurrency_limit)

            async def _extract_page_phase1(idx: int, rendered: RasterizedPage) -> tuple[int, PageLayoutExtraction]:
                async with semaphore:
                    digital_hint = analysis_results[idx].full_text if idx < len(analysis_results) else None
                    layout = await extractor_engine.extract_page_layout(
                        page_number=rendered.page_number,
                        image_bytes=rendered.image_bytes,
                        digital_text_hint=digital_hint,
                    )
                    return (rendered.page_number, layout)

            phase1_tasks = [_extract_page_phase1(i, r) for i, r in enumerate(rendered_pages)]
            phase1_results_list = await asyncio.gather(*phase1_tasks)
            phase1_layouts = dict(phase1_results_list)

        # Step 5: Convert Skeletons & Run Selective Phase 2 Enrichment
        folio_detector = FolioDetector()
        media_extractor = MediaExtractor(minio=store, db=db)
        all_pages_articles: dict[int, list[SegmentedArticle]] = {}
        page_media_items: dict[int, list[ExtractedPhotoData]] = {}
        page_extractions: list[dict[str, Any]] = []

        last_known_folio: int | None = None
        last_known_pdf_page: int | None = None

        for i, rendered in enumerate(rendered_pages):
            page_num = rendered.page_number
            analysis = analysis_results[i]
            layout_data = phase1_layouts.get(page_num, PageLayoutExtraction(page_number=page_num))

            # Database Page update
            stmt = select(Page).where(Page.issue_id == issue_id, Page.page_number == page_num)
            res = await db.execute(stmt)
            page_record = res.scalar_one_or_none()

            printed_folio = layout_data.printed_page_number or folio_detector.extract_printed_page_number(
                page_number=page_num,
                height_px=float(rendered.height_px),
                width_px=float(rendered.width_px),
                blocks=analysis.blocks,
                is_advertisement_page=layout_data.is_advertisement_page or analysis.is_advertisement,
                last_known_folio_num=last_known_folio,
                last_known_pdf_page=last_known_pdf_page,
                total_issue_pages=len(rendered_pages),
            )

            if page_record:
                page_record.printed_page_number = printed_folio
                page_record.is_advertisement_page = layout_data.is_advertisement_page or analysis.is_advertisement
                page_record.ingestion_status = "layout_done"

            if printed_folio and printed_folio.isdigit():
                last_known_folio = int(printed_folio)
                last_known_pdf_page = page_num

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
                        # For LLM-based extraction (Gemini / Gemma), enrich major articles with VLM crop
                        crop_bytes = extractor_engine.crop_article_image(rendered.image_bytes, bbox)
                        enrichment = await extractor_engine.enrich_article(
                            headline=skel.headline,
                            article_crop_bytes=crop_bytes,
                            digital_slice=analysis.full_text,
                        )
                        full_body = enrichment.body_text

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
                "printed_page_number": printed_folio,
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
        page_folio_map: dict[int, str] = {}
        pages_fetch = await db.execute(select(Page).where(Page.issue_id == issue_id))
        for p in pages_fetch.scalars().all():
            page_id_map[p.page_number] = p.id
            page_folio_map[p.page_number] = p.printed_page_number or str(p.page_number)

        articles_manifest: list[dict[str, Any]] = []
        total_chunks_created = 0

        # Step 7a: First Pass - Persist Articles to obtain article IDs
        persisted_articles: list[tuple[Article, AssembledArticle, Any]] = []
        article_envelopes_by_page: dict[int, list[tuple[int, tuple[float, float, float, float], str]]] = {}

        for assembled in assembled_articles:
            class_res = classifier.classify_and_score(
                article=assembled,
                total_issue_pages=len(rendered_pages),
                printed_section=assembled.printed_section,
            )

            is_ad = (
                class_res.article_type == "advertisement"
                or assembled.headline.startswith(("[Advertisement]", "[Public Notice]"))
                or check_is_advertisement_text(assembled.full_text)
            )

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
                        printed_page_number=page_folio_map.get(p_map.page_number, str(p_map.page_number)),
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
                        (article_record.id, envelope, article_record.headline or "")
                    )
            await db.flush()

        # Step 7b: Extract & Store 100% Photos / Graphic Regions for each page
        article_photos_map: dict[int, list[Photo]] = {}
        for rendered, analysis in zip(rendered_pages, analysis_results, strict=False):
            p_num = rendered.page_number
            p_id = page_id_map.get(p_num)
            if not p_id:
                continue

            page_envs = article_envelopes_by_page.get(p_num, [])

            # Harvest photos from Docling media items and PDF image boxes
            media_list = page_media_items.get(p_num, [])
            all_media_boxes: list[tuple[tuple[float, float, float, float], str]] = [
                (m.bbox, m.caption or "") for m in media_list if m.bbox
            ]

            # Also incorporate any PyMuPDF xrefs not already covered
            for ibox in analysis.image_boxes:
                # Check if already covered by Docling
                if not any(
                    abs(ibox[0] - mb[0][0]) < 30 and abs(ibox[1] - mb[0][1]) < 30
                    for mb in all_media_boxes
                ):
                    all_media_boxes.append((ibox, ""))

            has_large_canvas = False
            img_idx = 1
            for m_box, m_cap in all_media_boxes:
                b_area = max(0.0, m_box[2] - m_box[0]) * max(0.0, m_box[3] - m_box[1])
                if (b_area / max(1.0, float(rendered.width_px * rendered.height_px))) >= 0.75:
                    has_large_canvas = True

                # Resolve spatial binding to article
                bound_art_id = media_extractor.resolve_photo_article_binding(
                    photo_bbox=m_box,
                    article_envelopes=page_envs,
                    caption=m_cap,
                )
                try:
                    photo_rec = await media_extractor.extract_and_store_photo(
                        page_image_bytes=rendered.image_bytes,
                        page_id=p_id,
                        article_id=bound_art_id,
                        bbox=m_box,
                        caption=m_cap,
                        photo_index=img_idx,
                    )
                    if photo_rec:
                        if bound_art_id:
                            if bound_art_id not in article_photos_map:
                                article_photos_map[bound_art_id] = []
                            article_photos_map[bound_art_id].append(photo_rec)
                        img_idx += 1
                except Exception as ex:
                    logger.warning("Photo extraction failed for box", extra={"page_number": p_num, "bbox": m_box, "error": str(ex)})

            # Fallback: if page yielded 0 discrete photos AND has a large canvas (>= 75%), run VLM Grounding Sweep
            page_art_photos = [
                p for art_id, p_list in article_photos_map.items()
                if art_id in [a.id for a, _, _ in persisted_articles if a.primary_page_id == p_id]
                for p in p_list
            ]
            if not page_art_photos and has_large_canvas:
                logger.info(
                    "Triggering VLM Grounding Sweep fallback on large visual canvas",
                    extra={"page_number": p_num, "page_id": p_id},
                )
                grounded_subphotos = await media_extractor.extract_subphotos_vlm_fallback(
                    page_image_bytes=rendered.image_bytes,
                    page_id=p_id,
                    article_envelopes=page_envs,
                    width_px=rendered.width_px,
                    height_px=rendered.height_px,
                    start_photo_index=img_idx,
                )
                for bound_art_id, photo_rec in grounded_subphotos:
                    if bound_art_id:
                        if bound_art_id not in article_photos_map:
                            article_photos_map[bound_art_id] = []
                        article_photos_map[bound_art_id].append(photo_rec)
                    img_idx += 1

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
                    printed_pages=[page_folio_map.get(pm.page_number, str(pm.page_number)) for pm in assembled.pages_mapping],
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
                            printed_pages=[page_folio_map.get(pm.page_number, str(pm.page_number)) for pm in assembled.pages_mapping],
                            chunk_index=p_idx,
                        )
                        visual_chunks.append((v_chunk, photo.visual_type))

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
                        printed_pages=[page_folio_map.get(pm.page_number, str(pm.page_number)) for pm in assembled.pages_mapping],
                        entities=[e.name for e in meta_res.entities],
                        topics=[t.name for t in meta_res.topics],
                        chunks=chunks,
                        has_photo=has_photo,
                        has_table=has_table,
                        chunk_type="text",
                    )
                    total_chunks_created += len(chunks)

                # Index dedicated visual chunks
                for v_chunk, v_type in visual_chunks:
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
                        printed_pages=[page_folio_map.get(pm.page_number, str(pm.page_number)) for pm in assembled.pages_mapping],
                        entities=[e.name for e in meta_res.entities],
                        topics=[t.name for t in meta_res.topics],
                        chunks=[v_chunk],
                        has_photo=True,
                        has_table=True,
                        chunk_type="visual",
                        has_visual_data=True,
                        visual_type=v_type,
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


def process_issue_ingestion_task(issue_id: int, pdf_bytes: bytes, dpi: int = 300) -> dict[str, Any]:
    """Synchronous entry point for Celery worker."""
    return asyncio.run(run_ingestion_pipeline(issue_id=issue_id, pdf_bytes=pdf_bytes, dpi=dpi))
