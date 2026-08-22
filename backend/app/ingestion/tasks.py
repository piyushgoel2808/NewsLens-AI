from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.celery_app import celery_app
from app.ingestion.chunker import NewspaperChunker
from app.ingestion.classifier import ArticleClassifier
from app.ingestion.cross_page_assembler import CrossPageAssembler
from app.ingestion.detector import PDFPageDetector
from app.ingestion.embedder import ArticleEmbedder
from app.ingestion.folio_detector import FolioDetector
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.ocr_service import OCRService
from app.ingestion.rasterizer import PDFRasterizer
from app.ingestion.reading_order import BlockType, OrderedReadingBlock
from app.ingestion.segmenter import ArticleSegmenter, SegmentedArticle
from app.models.article import Article, ArticleChunk, ArticlePage, ArticleTable, Photo
from app.models.entity import ArticleEntity, ArticleTopic
from app.models.ingestion import IngestionJob
from app.models.newspaper import Issue, Newspaper, Page
from app.providers.base import OCRBlock
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)

_KNOWN_MASTHEADS = [
    ("MINT", "Mint"),
    ("LIVEMINT", "Mint"),
    ("BUSINESS STANDARD", "Business Standard"),
    ("THE HINDU", "The Hindu"),
    ("ECONOMIC TIMES", "The Economic Times"),
    ("TIMES OF INDIA", "The Times of India"),
    ("FINANCIAL EXPRESS", "Financial Express"),
    ("INDIAN EXPRESS", "The Indian Express"),
    ("THE TRIBUNE", "The Tribune"),
    ("DECCAN HERALD", "Deccan Herald"),
    ("THE TELEGRAPH", "The Telegraph"),
]

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_DATE_EXTRACTION_PATTERNS = [
    re.compile(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?[\s\.\,\-\/]+(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[\s\.\,\-\/]+(\d{4})\b"
    ),
    re.compile(
        r"(?i)\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[\s\.\,\-\/]+(\d{1,2})(?:st|nd|rd|th)?[\s\.\,\-\/]+(\d{4})\b"
    ),
]


def detect_masthead_and_date(
    blocks: Sequence[Any], height_px: float
) -> tuple[str | None, date | None]:
    """Detect authentic newspaper masthead brand and publication date from Page 1 top header."""
    detected_name: str | None = None
    detected_date: date | None = None

    for block in blocks:
        text = ""
        bbox = None
        if hasattr(block, "text") and hasattr(block, "bbox"):
            text = str(block.text)
            bbox = block.bbox
        elif isinstance(block, dict):
            text = str(block.get("text", ""))
            bbox = block.get("bbox")

        if not text:
            continue

        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            y1 = float(bbox[3])
            if y1 > height_px * 0.25:
                continue

        text_upper = text.upper()

        if not detected_name:
            for keyword, brand in _KNOWN_MASTHEADS:
                if keyword in text_upper:
                    detected_name = brand
                    break

        if not detected_date:
            for pat in _DATE_EXTRACTION_PATTERNS:
                m = pat.search(text)
                if m:
                    g1, g2, g3 = m.groups()
                    try:
                        if g1.isdigit() and g3.isdigit():
                            day = int(g1)
                            mon = _MONTH_MAP.get(g2.lower(), 1)
                            yr = int(g3)
                        else:
                            mon = _MONTH_MAP.get(g1.lower(), 1)
                            day = int(g2)
                            yr = int(g3)
                        if 1 <= day <= 31 and 1990 <= yr <= 2050:
                            detected_date = date(yr, mon, day)
                            break
                    except Exception:
                        pass

    return detected_name, detected_date


async def run_ingestion_pipeline(
    issue_id: int,
    pdf_bytes: bytes,
    dpi: int = 300,
    minio: MinioStore | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end Phase 1 & 2 ingestion pipeline for an issue.

    Steps:
    1. Rasterize all PDF pages to PNG (at specified DPI) and upload to MinIO.
    2. Analyze page text layers (digital vs scanned detection).
    3. Run OCR on scanned/image-only pages.
    4. Run layout analysis & reading order resolution on all pages.
    5. Update MySQL Page records and Issue state.
    """
    settings = get_settings()
    store = minio or MinioStore(settings.minio)

    if session_factory is None:
        engine = create_async_engine(settings.database.async_url, echo=False)
        maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    else:
        maker = session_factory

    async with maker() as db:
        # Step 1: Rasterize PDF pages
        rasterizer = PDFRasterizer(db=db, minio=store)
        rendered_pages = await rasterizer.rasterize_pdf_bytes(
            pdf_bytes=pdf_bytes,
            issue_id=issue_id,
            dpi=dpi,
        )

        # Step 2: Detect text layer
        detector = PDFPageDetector()
        analysis_results = detector.analyze_document_bytes(pdf_bytes)

        ocr_service = OCRService(db=db, minio=store)
        layout_analyzer = LayoutAnalyzer()

        pages_summary: list[dict[str, Any]] = []
        all_pages_articles: dict[int, list[SegmentedArticle]] = {}

        # Fetch issue upfront
        issue_stmt = (
            select(Issue).where(Issue.id == issue_id).options(selectinload(Issue.newspaper))
        )
        issue_res = await db.execute(issue_stmt)
        issue = issue_res.scalar_one_or_none()
        issue_lang = issue.language if issue else "en"

        # Step 3 & 4: OCR and Layout Analysis per page
        folio_detector = FolioDetector()
        last_known_folio: int | None = None
        last_known_pdf_page: int | None = None

        for i, rendered in enumerate(rendered_pages):
            page_num = rendered.page_number
            analysis = analysis_results[i]

            stmt = select(Page).where(
                Page.issue_id == issue_id,
                Page.page_number == page_num,
            )
            res = await db.execute(stmt)
            page = res.scalar_one_or_none()
            if not page:
                continue

            extracted_ocr_blocks: list[OCRBlock] = []

            # Run OCR if page is scanned
            if analysis.requires_ocr:
                try:
                    ocr_res = await ocr_service.process_page_ocr(
                        page_id=page.id,
                        image_bytes=rendered.image_bytes,
                        lang_hint=issue_lang,
                    )
                    extracted_ocr_blocks = ocr_res.blocks
                    page.ocr_confidence = ocr_res.mean_confidence
                except Exception as e:
                    logger.warning(
                        "OCR fallback on page",
                        extra={"page_id": page.id, "error": str(e)},
                    )

            # Extract printed page number using FolioDetector with strict DPI synchronization
            target_blocks: Sequence[Any]
            if analysis.requires_ocr:
                folio_height = float(rendered.height_px)
                folio_width = float(rendered.width_px)
                target_blocks = extracted_ocr_blocks
            else:
                folio_height = float(analysis.page_height or rendered.height_px)
                folio_width = float(analysis.page_width or rendered.width_px)
                target_blocks = analysis.blocks

            # Dynamic Page 1 Masthead & Date Detection
            if page_num == 1 and issue:
                det_brand, det_date = detect_masthead_and_date(target_blocks, folio_height)
                if det_brand:
                    np_stmt = select(Newspaper).where(Newspaper.name == det_brand)
                    np_res = await db.execute(np_stmt)
                    np_obj = np_res.scalar_one_or_none()
                    if not np_obj:
                        np_obj = Newspaper(name=det_brand, default_language=issue_lang)
                        db.add(np_obj)
                        await db.flush()
                    issue.newspaper_id = np_obj.id
                    issue.newspaper = np_obj
                    logger.info(
                        "Dynamically identified newspaper masthead from Page 1",
                        extra={"newspaper": det_brand, "issue_id": issue_id},
                    )
                if det_date:
                    issue.issue_date = det_date
                    logger.info(
                        "Dynamically identified publication date from Page 1",
                        extra={"issue_date": str(det_date), "issue_id": issue_id},
                    )
                await db.flush()

            printed_folio = folio_detector.extract_printed_page_number(
                page_number=page_num,
                height_px=folio_height,
                width_px=folio_width,
                blocks=target_blocks,
                is_advertisement_page=analysis.is_advertisement,
                last_known_folio_num=last_known_folio,
                last_known_pdf_page=last_known_pdf_page,
                total_issue_pages=len(rendered_pages),
            )
            page.printed_page_number = printed_folio
            page.is_advertisement_page = analysis.is_advertisement

            if printed_folio and printed_folio.isdigit():
                last_known_folio = int(printed_folio)
                last_known_pdf_page = page_num

            # Run layout analysis
            layout_res = await layout_analyzer.analyze_page(
                page_number=page_num,
                width_px=rendered.width_px,
                height_px=rendered.height_px,
                image_bytes=rendered.image_bytes,
                digital_blocks=analysis.blocks if not analysis.requires_ocr else None,
                ocr_blocks=extracted_ocr_blocks if analysis.requires_ocr else None,
            )

            # Store page layout reading blocks for segmentation
            segmenter = ArticleSegmenter()
            page_articles = segmenter.segment_page(
                page_number=page_num,
                ordered_blocks=layout_res.reading_order,
                is_advertisement_page=analysis.is_advertisement,
            )

            # Robust fallback: if 0 articles generated but OCR or layout text exists
            if not page_articles:
                text_blocks = [b for b in layout_res.reading_order if b.text and b.text.strip()]
                if not text_blocks and extracted_ocr_blocks:
                    text_blocks = [
                        OrderedReadingBlock(
                            reading_order_index=idx,
                            element_id=idx + 1,
                            block_type=BlockType.BODY_TEXT,
                            text=b.text,
                            bbox=b.bbox,
                        )
                        for idx, b in enumerate(extracted_ocr_blocks)
                        if b.text and b.text.strip()
                    ]
                if text_blocks:
                    combined_text = "\n\n".join(b.text.strip() for b in text_blocks)
                    first_line = combined_text.split("\n")[0][:200].strip()
                    fallback_hl = first_line if first_line else f"Page {page_num} Report"
                    page_articles = [
                        SegmentedArticle(
                            article_temp_id=f"p{page_num}_art_fallback_1",
                            headline=fallback_hl,
                            body_text=combined_text,
                            word_count=len(combined_text.split()),
                            bbox_list=[b.bbox for b in text_blocks],
                            raw_blocks=text_blocks,
                        )
                    ]

            # Persist extracted tables
            for tbl in layout_res.tables:
                db_table = ArticleTable(
                    page_id=page.id,
                    bbox_json={"bbox": list(tbl.bbox)},
                    extracted_json={
                        "headers": tbl.headers,
                        "rows": tbl.rows,
                        "raw_markdown": tbl.raw_markdown,
                        "raw_html": tbl.raw_html,
                    },
                )
                db.add(db_table)

            # Persist extracted photos
            for pht in layout_res.photos:
                db_photo = Photo(
                    page_id=page.id,
                    bbox_json={"bbox": list(pht.bbox)},
                    caption=pht.caption,
                )
                db.add(db_photo)

            all_pages_articles[page_num] = page_articles

            page.ingestion_status = "segmented"

            pages_summary.append(
                {
                    "page_number": page_num,
                    "width_px": rendered.width_px,
                    "height_px": rendered.height_px,
                    "object_key": rendered.object_key,
                    "type": analysis.page_type.value,
                    "requires_ocr": analysis.requires_ocr,
                    "ocr_confidence": page.ocr_confidence,
                    "char_count": analysis.character_count,
                    "layout_elements": len(layout_res.elements),
                    "reading_blocks": len(layout_res.reading_order),
                    "tables_count": len(layout_res.tables),
                    "photos_count": len(layout_res.photos),
                    "layout_source": layout_res.source,
                    "articles_count": len(page_articles),
                }
            )

        # Step 5: Cross-Page Assembly
        assembler = CrossPageAssembler()
        assembled_articles = assembler.assemble_issue_articles(all_pages_articles)

        # Step 6: Classification, Metadata Extraction, Chunking, Embedding & Persistence
        classifier = ArticleClassifier()
        meta_extractor = MetadataExtractor(db=db)
        chunker = NewspaperChunker()
        embedder = ArticleEmbedder(db=db)

        # Idempotent Atomic Cleanup: Purge any previously inserted articles/chunks for this issue_id
        with contextlib.suppress(Exception):
            await embedder.delete_issue_vectors(issue_id)

        existing_arts_res = await db.execute(select(Article.id).where(Article.issue_id == issue_id))
        existing_art_ids = existing_arts_res.scalars().all()
        if existing_art_ids:
            await db.execute(
                delete(ArticleEntity).where(ArticleEntity.article_id.in_(existing_art_ids))
            )
            await db.execute(
                delete(ArticleTopic).where(ArticleTopic.article_id.in_(existing_art_ids))
            )
            await db.execute(
                delete(ArticleChunk).where(ArticleChunk.article_id.in_(existing_art_ids))
            )
            await db.execute(
                delete(ArticlePage).where(ArticlePage.article_id.in_(existing_art_ids))
            )
            await db.execute(delete(Article).where(Article.issue_id == issue_id))
            await db.flush()
            logger.info(
                "Purged previous articles and chunks for idempotent re-ingestion",
                extra={"issue_id": issue_id, "deleted_articles": len(existing_art_ids)},
            )

        # Get updated newspaper metadata
        newspaper_name = issue.newspaper.name if issue and issue.newspaper else "Daily"
        issue_date_str = str(issue.issue_date) if issue else ""

        page_id_map: dict[int, int] = {}
        page_folio_map: dict[int, str] = {}
        pages_fetch = await db.execute(select(Page).where(Page.issue_id == issue_id))
        all_db_pages = pages_fetch.scalars().all()
        for p in all_db_pages:
            page_id_map[p.page_number] = p.id
            page_folio_map[p.page_number] = p.printed_page_number or str(p.page_number)

        created_articles: list[dict[str, Any]] = []
        total_chunks_created = 0

        for assembled in assembled_articles:
            class_res = classifier.classify_and_score(
                article=assembled,
                total_issue_pages=len(rendered_pages),
            )

            primary_page_id = page_id_map.get(assembled.primary_page_number)

            clean_hl = (assembled.headline or "")[:1024]
            article_record = Article(
                issue_id=issue_id,
                primary_page_id=primary_page_id,
                headline=clean_hl if clean_hl else "Untitled Article",
                subheadline=assembled.subheadline[:1024] if assembled.subheadline else None,
                byline_author=assembled.byline_author[:512] if assembled.byline_author else None,
                section=class_res.section[:255] if class_res.section else None,
                article_type=class_res.article_type,
                language=issue_lang,
                prominence_score=class_res.prominence_score,
                word_count=assembled.word_count,
                full_text=assembled.full_text,
            )
            db.add(article_record)
            await db.flush()

            # Insert junction table article_pages
            for p_map in assembled.pages_mapping:
                target_pid = page_id_map.get(p_map.page_number)
                if target_pid:
                    art_page = ArticlePage(
                        article_id=article_record.id,
                        page_id=target_pid,
                        page_number=p_map.page_number,
                        printed_page_number=page_folio_map.get(p_map.page_number),
                        bbox_json={"bboxes": [list(b) for b in p_map.bbox_list]},
                        block_order=p_map.block_order,
                    )
                    db.add(art_page)

            # Metadata extraction (NER, topics, summary)
            art_hl = article_record.headline or "Untitled Article"
            art_text = article_record.full_text or art_hl
            if not art_text:
                continue

            meta_res = await meta_extractor.process_and_persist_metadata(
                article_id=article_record.id,
                headline=art_hl,
                full_text=art_text,
            )

            # Hierarchical Chunking
            pages_list = [pm.page_number for pm in assembled.pages_mapping] or [1]
            printed_pages_list = [page_folio_map.get(p, str(p)) for p in pages_list]
            chunks = chunker.chunk_article(
                full_text=art_text,
                newspaper_name=newspaper_name,
                issue_date=issue_date_str,
                headline=art_hl,
                section=article_record.section,
                pages=pages_list,
                printed_pages=printed_pages_list,
            )

            # Vector Embedding: Skip ads and trivial fragments from Qdrant vector index
            is_trivial_fragment = (
                article_record.word_count < 10 or len(art_text.strip().split()) < 8
            )
            if article_record.article_type == "advertisement" or is_trivial_fragment:
                logger.info(
                    "Skipping Qdrant vector indexing for advertisement or trivial fragment",
                    extra={
                        "article_id": article_record.id,
                        "headline": art_hl[:60],
                        "type": article_record.article_type,
                        "word_count": article_record.word_count,
                    },
                )
            else:
                # Dense Vector Embedding & Qdrant Upsert
                await embedder.embed_and_index_chunks(
                    article_id=article_record.id,
                    issue_id=issue_id,
                    newspaper_name=newspaper_name,
                    issue_date=issue_date_str,
                    headline=art_hl,
                    section=article_record.section,
                    article_type=article_record.article_type,
                    prominence_score=article_record.prominence_score,
                    page_numbers=pages_list,
                    printed_pages=printed_pages_list,
                    entities=[e.name for e in meta_res.entities],
                    topics=[t.name for t in meta_res.topics],
                    chunks=chunks,
                )
                total_chunks_created += len(chunks)

            created_articles.append(
                {
                    "id": article_record.id,
                    "headline": article_record.headline,
                    "type": article_record.article_type,
                    "prominence_score": article_record.prominence_score,
                    "word_count": article_record.word_count,
                    "chunks_count": len(chunks),
                    "entities_count": len(meta_res.entities),
                    "topics_count": len(meta_res.topics),
                    "pages_spanned": pages_list,
                }
            )

        # Update Pages & Issue status
        for p in all_db_pages:
            p.ingestion_status = "indexed"

        if issue:
            issue.ingestion_status = "completed"
            if issue.source_zip_id:
                job_stmt = select(IngestionJob).where(IngestionJob.id == issue.source_zip_id)
                job_res = await db.execute(job_stmt)
                job = job_res.scalar_one_or_none()
                if job:
                    job.status = "completed"
                    job.completed_at = func.now()

        await db.commit()

        logger.info(
            "Phase 4 ingestion pipeline completed successfully",
            extra={
                "issue_id": issue_id,
                "rendered_pages": len(rendered_pages),
                "articles_created": len(created_articles),
                "chunks_created": total_chunks_created,
            },
        )

        return {
            "issue_id": issue_id,
            "total_pages": len(rendered_pages),
            "pages": pages_summary,
            "articles": created_articles,
            "total_chunks": total_chunks_created,
        }


@celery_app.task(name="app.ingestion.tasks.process_issue_ingestion_task")  # type: ignore[untyped-decorator]
def process_issue_ingestion_task(
    issue_id: int,
    pdf_bytes: bytes,
    dpi: int = 300,
) -> dict[str, Any]:
    """Celery worker task to process an issue asynchronously."""
    return asyncio.run(run_ingestion_pipeline(issue_id, pdf_bytes, dpi=dpi))
