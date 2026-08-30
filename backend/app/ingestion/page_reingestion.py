"""Interactive Single-Page Re-Ingestion Service for NewsLens-AI.

Orchestrates end-to-end page re-processing:
1. Slices the target page from the original PDF in MinIO.
2. Atomically purges previous page-exclusive articles, entities, topics, chunks, photos, and Qdrant vectors.
3. Re-runs Docling OCR, picture-nested text recovery, and broadsheet masthead noise filtering.
4. Re-extracts photos, filters full-page ad canvases, and runs Qwen-VL visual scene analysis.
5. Re-segments and classifies articles on the page.
6. Re-extracts metadata & Named Entities (NER) and taxonomy topics.
7. Generates dense vector embeddings for all new chunks and indexes them into Qdrant.
8. Persists all updated records to MySQL (articles, article_pages, article_chunks, photos, tables, article_entities, article_topics, pages).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import re
from typing import Any

import pymupdf
from PIL import Image
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.chunker import NewspaperChunker
from app.ingestion.classifier import ArticleClassifier
from app.ingestion.cross_page_assembler import CrossPageAssembler
from app.ingestion.detector import PDFPageDetector
from app.ingestion.docling_parser import DoclingLayoutParser, ExtractedPhotoData
from app.ingestion.embedder import ArticleEmbedder
from app.ingestion.folio_detector import FolioDetector
from app.ingestion.media_extractor import MediaExtractor
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.rasterizer import PDFRasterizer
from app.ingestion.segmenter import SegmentedArticle
from app.models.article import (
    Article,
    ArticleCategory,
    ArticleChunk,
    ArticlePage,
    ArticleTable,
    Photo,
)
from app.models.entity import ArticleEntity, ArticleTopic, Entity, Topic
from app.models.newspaper import Issue, Page
from app.storage.minio_store import MinioStore
from app.storage.qdrant_store import QdrantStore

logger = get_logger(__name__)


def check_is_advertisement_text(text: str) -> bool:
    """Detect if text has strong advertising or commercial markers."""
    t_lower = text.lower()
    ad_keywords = [
        "advertisement",
        "ad vt",
        "public notice",
        "classified",
        "tender notice",
        "e-tender",
        "terms & conditions apply",
        "t&c apply",
        "all rights reserved",
        "visit our website",
        "www.",
        "http://",
        "https://",
        "call now",
        "toll free",
        "limited period offer",
        "smarter steels",
        "net zero steel",
        "anniversary",
    ]
    return any(kw in t_lower for kw in ad_keywords)


class PageReingestionService:
    """Orchestrates selective, atomic single-page re-ingestion."""

    def __init__(
        self,
        db: AsyncSession,
        minio: MinioStore | None = None,
        qdrant: QdrantStore | None = None,
    ) -> None:
        self._db = db
        self._settings = get_settings()
        self._minio = minio or MinioStore(self._settings.minio)
        self._qdrant = qdrant or QdrantStore(self._settings.qdrant)

    async def _fetch_original_pdf(self, issue_id: int, job_id: int | None) -> bytes | None:
        """Locate and download the original PDF for an issue from MinIO."""
        client = self._minio._client
        bucket = self._settings.minio.bucket_originals

        # 1. Search by job_id prefix
        if job_id:
            try:
                objects = list(client.list_objects(bucket, prefix=f"originals/{job_id}/"))
                if objects:
                    resp = client.get_object(bucket, objects[0].object_name)
                    return resp.read()
            except Exception as ex:
                logger.debug("Original lookup by job_id failed", extra={"job_id": job_id, "error": str(ex)})

        # 2. Search by issue_id prefix
        for pfx in [f"originals/{issue_id}/", f"issues/{issue_id}/"]:
            try:
                objects = list(client.list_objects(bucket, prefix=pfx))
                if objects:
                    resp = client.get_object(bucket, objects[0].object_name)
                    return resp.read()
            except Exception as ex:
                logger.debug("Original lookup by issue_id failed", extra={"issue_id": issue_id, "error": str(ex)})

        # 3. Fallback: Search all objects in bucket_originals for matching job_id or issue_id
        try:
            for obj in client.list_objects(bucket, recursive=True):
                name = obj.object_name or ""
                if (job_id and f"/{job_id}/" in name) or f"/{issue_id}/" in name:
                    resp = client.get_object(bucket, name)
                    return resp.read()
        except Exception as ex:
            logger.warning("Bucket-wide search for original PDF failed", extra={"error": str(ex)})

        return None

    async def reingest_page(
        self,
        issue_id: int,
        page_number: int,
        parser_engine: str = "auto",
    ) -> dict[str, Any]:
        """Execute complete re-ingestion for a single broadsheet page."""
        logger.info(
            "Starting single-page re-ingestion",
            extra={"issue_id": issue_id, "page_number": page_number, "parser_engine": parser_engine},
        )

        # 1. Retrieve Issue and Page
        issue_stmt = (
            select(Issue)
            .where(Issue.id == issue_id)
            .options(selectinload(Issue.newspaper))
        )
        issue_res = await self._db.execute(issue_stmt)
        issue = issue_res.scalar_one_or_none()
        if not issue:
            raise ValueError(f"Issue {issue_id} not found")

        page_stmt = select(Page).where(Page.issue_id == issue_id, Page.page_number == page_number)
        page_res = await self._db.execute(page_stmt)
        page = page_res.scalar_one_or_none()
        if not page:
            raise ValueError(f"Page {page_number} not found for Issue {issue_id}")

        newspaper_name = issue.newspaper.name if issue.newspaper else "Newspaper"
        issue_date_str = str(issue.issue_date)
        issue_lang = issue.language or "en"
        page.ingestion_status = "processing"
        with contextlib.suppress(Exception):
            await self._db.execute(text("SET SESSION innodb_lock_wait_timeout = 180;"))
        await self._db.flush()

        # 2. Download original PDF and extract single page PDF bytes
        pdf_bytes = await self._fetch_original_pdf(issue_id=issue_id, job_id=issue.source_zip_id)
        if not pdf_bytes:
            raise ValueError(f"Original PDF file for Issue {issue_id} could not be retrieved from MinIO storage")

        src_pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        if page_number < 1 or page_number > len(src_pdf):
            src_pdf.close()
            raise ValueError(f"Page {page_number} out of bounds (document has {len(src_pdf)} pages)")

        single_doc = pymupdf.open()
        single_doc.insert_pdf(src_pdf, from_page=page_number - 1, to_page=page_number - 1)
        single_pdf_bytes = single_doc.tobytes()
        single_doc.close()
        src_pdf.close()

        # 3. Retrieve or render high-resolution raster image
        page_image_bytes: bytes | None = None
        if page.raster_object_key:
            try:
                page_image_bytes = await self._minio.get(
                    bucket=self._settings.minio.bucket_pages,
                    key=page.raster_object_key,
                )
            except Exception as img_err:
                logger.warning(
                    "Failed to fetch cached raster image, will re-render",
                    extra={"error": str(img_err), "key": page.raster_object_key},
                )

        if not page_image_bytes:
            rasterizer = PDFRasterizer(db=self._db, minio=self._minio)
            rendered = await rasterizer.rasterize_single_page(
                pdf_bytes=single_pdf_bytes,
                issue_id=issue_id,
                page_number=page_number,
                dpi=300,
            )
            page_image_bytes = rendered.image_bytes
            page.raster_object_key = rendered.object_key
            page.width_px = rendered.width_px
            page.height_px = rendered.height_px

        pil_img = Image.open(io.BytesIO(page_image_bytes))
        width_px, height_px = pil_img.size
        page.width_px = width_px
        page.height_px = height_px

        # 4. Atomic multi-tier purge of previous page data
        ap_stmt = select(ArticlePage).where(ArticlePage.page_id == page.id)
        ap_res = await self._db.execute(ap_stmt)
        old_art_pages = ap_res.scalars().all()
        old_art_ids = list({ap.article_id for ap in old_art_pages})

        purged_articles_count = 0
        purged_vectors_count = 0

        for a_id in old_art_ids:
            # Check if this article exists on any other page
            other_pages_cnt = (
                await self._db.execute(
                    select(func.count(ArticlePage.id)).where(
                        ArticlePage.article_id == a_id,
                        ArticlePage.page_id != page.id,
                    )
                )
            ).scalar() or 0

            if other_pages_cnt == 0:
                # Article belongs ONLY to this page -> full multi-tier purge
                chunk_res = await self._db.execute(
                    select(ArticleChunk).where(ArticleChunk.article_id == a_id)
                )
                old_chunks = chunk_res.scalars().all()
                old_vec_ids = [c.embedding_vector_id for c in old_chunks if c.embedding_vector_id]

                # A. Purge Qdrant vectors
                if old_vec_ids:
                    with contextlib.suppress(Exception):
                        await self._qdrant.delete(old_vec_ids)
                        purged_vectors_count += len(old_vec_ids)

                with contextlib.suppress(Exception):
                    await self._qdrant.delete_by_filter({"article_id": a_id})

                # B. Purge MySQL relations
                await self._db.execute(delete(ArticleChunk).where(ArticleChunk.article_id == a_id))
                await self._db.execute(delete(ArticleEntity).where(ArticleEntity.article_id == a_id))
                await self._db.execute(delete(ArticleTopic).where(ArticleTopic.article_id == a_id))
                await self._db.execute(delete(Photo).where(Photo.article_id == a_id))
                await self._db.execute(delete(ArticleTable).where(ArticleTable.article_id == a_id))
                await self._db.execute(delete(ArticlePage).where(ArticlePage.article_id == a_id))
                await self._db.execute(delete(Article).where(Article.id == a_id))
                purged_articles_count += 1
            else:
                # Article spans multiple pages -> remove only this page's mapping
                await self._db.execute(
                    delete(ArticlePage).where(
                        ArticlePage.article_id == a_id,
                        ArticlePage.page_id == page.id,
                    )
                )

        # Purge standalone photos & tables linked directly to page_id
        standalone_photos = (
            await self._db.execute(select(Photo).where(Photo.page_id == page.id))
        ).scalars().all()
        for sp in standalone_photos:
            if sp.object_key:
                with contextlib.suppress(Exception):
                    await self._minio.delete(self._settings.minio.bucket_pages, sp.object_key)
            await self._db.delete(sp)

        await self._db.execute(delete(ArticleTable).where(ArticleTable.page_id == page.id))
        await self._db.flush()

        logger.info(
            "Purged previous page assets",
            extra={
                "page_id": page.id,
                "purged_articles": purged_articles_count,
                "purged_vectors": purged_vectors_count,
            },
        )

        # 5. Extract Layout & Articles using DoclingLayoutParser
        detector = PDFPageDetector()
        digital_analysis = detector.analyze_document_bytes(single_pdf_bytes)
        single_analysis = digital_analysis[0] if digital_analysis else None

        docling_parser = DoclingLayoutParser()
        parsed_doc_items = await asyncio.get_running_loop().run_in_executor(
            None,
            docling_parser.parse_docling_document,
            single_pdf_bytes,
            page_number,
            width_px,
            height_px,
        )

        docling_articles = docling_parser.assemble_articles(
            page_number=page_number,
            items=parsed_doc_items,
            width_px=width_px,
            height_px=height_px,
            is_advertisement_page=page.is_advertisement_page,
        )

        page_segmented_articles: list[SegmentedArticle] = list(docling_articles)
        page_media_items: list[ExtractedPhotoData] = docling_parser.extract_page_media_items(parsed_doc_items)

        # Fallback if 0 articles detected
        if not page_segmented_articles:
            full_page_txt = ""
            if single_analysis and single_analysis.full_text.strip():
                full_page_txt = single_analysis.full_text.strip()
            elif parsed_doc_items:
                full_page_txt = "\n\n".join(it.text for it in parsed_doc_items if it.text.strip()).strip()

            is_ad_page = (page.is_advertisement_page) or bool(
                full_page_txt and check_is_advertisement_text(full_page_txt)
            )

            if full_page_txt and len(full_page_txt.split()) >= 6:
                fallback_hl = (
                    f"[Advertisement] Page {page_number} Feature"
                    if is_ad_page
                    else f"Page {page_number} Feature"
                )
                lines = [line_str.strip() for line_str in full_page_txt.split("\n") if line_str.strip()]
                if lines and 3 <= len(lines[0].split()) <= 12:
                    fallback_hl = f"[Advertisement] {lines[0]}" if is_ad_page else lines[0]

                page_segmented_articles.append(
                    SegmentedArticle(
                        article_temp_id=f"page_{page_number}_art_fallback",
                        headline=fallback_hl,
                        body_text=full_page_txt,
                        word_count=len(full_page_txt.split()),
                        bbox_list=[(0.0, 0.0, float(width_px), float(height_px))],
                    )
                )

        # Update folio and ad status
        folio_detector = FolioDetector()
        blocks = single_analysis.blocks if single_analysis else []
        printed_folio = folio_detector.extract_printed_page_number(
            page_number=page_number,
            height_px=float(height_px),
            width_px=float(width_px),
            blocks=blocks,
            is_advertisement_page=page.is_advertisement_page,
            total_issue_pages=issue.total_pages or 1,
        )
        if printed_folio:
            page.printed_page_number = printed_folio
        final_printed_folio = str(page.printed_page_number or printed_folio or page_number)

        # 6. Assemble and Classify Articles
        assembler = CrossPageAssembler()
        assembled_articles = assembler.assemble_issue_articles({page_number: page_segmented_articles})
        classifier = ArticleClassifier()
        metadata_extractor = MetadataExtractor(db=self._db)
        chunker = NewspaperChunker()
        embedder = ArticleEmbedder(db=self._db, qdrant=self._qdrant)

        persisted_articles: list[tuple[Article, Any]] = []
        article_envelopes: list[tuple[int, tuple[float, float, float, float], str]] = []

        for assembled in assembled_articles:
            class_res = classifier.classify_and_score(
                article=assembled,
                total_issue_pages=issue.total_pages or 1,
                printed_section=assembled.printed_section,
            )

            is_ad = (
                class_res.article_type == "advertisement"
                or assembled.headline.startswith(("[Advertisement]", "[Public Notice]"))
                or check_is_advertisement_text(assembled.full_text)
            )

            clean_hl = (assembled.headline or "")[:1024]
            if is_ad and not clean_hl.startswith(("[Advertisement]", "[Public Notice]")):
                clean_hl = f"[Advertisement] {clean_hl}"[:1024]

            # Canonical Category
            cat_id: int | None = None
            if class_res.category:
                cat_fetch = await self._db.execute(
                    select(ArticleCategory.id).where(ArticleCategory.name == class_res.category)
                )
                cat_id = cat_fetch.scalar_one_or_none()
                if not cat_id:
                    new_cat = ArticleCategory(name=class_res.category)
                    self._db.add(new_cat)
                    await self._db.flush()
                    cat_id = new_cat.id

            article_record = Article(
                issue_id=issue_id,
                primary_page_id=page.id,
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
            self._db.add(article_record)
            await self._db.flush()

            # Save ArticlePage mapping
            art_page = ArticlePage(
                article_id=article_record.id,
                page_id=page.id,
                page_number=page_number,
                printed_page_number=page.printed_page_number or str(page_number),
                bbox_json={"bboxes": [list(b) for b in assembled.pages_mapping[0].bbox_list]},
                block_order=1,
            )
            self._db.add(art_page)

            # Metadata extraction (NER Entities, Topics, Summary)
            await metadata_extractor.process_and_persist_metadata(
                article_id=article_record.id,
                headline=article_record.headline or "",
                full_text=assembled.full_text,
            )

            # Record spatial envelope for photo binding (filter out full-page canvas background items)
            page_area = float(width_px * height_px)
            real_bboxes = [
                b for b in assembled.pages_mapping[0].bbox_list
                if (max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])) < (0.50 * page_area)
            ] if assembled.pages_mapping[0].bbox_list else []

            if real_bboxes:
                env = (
                    min(b[0] for b in real_bboxes),
                    min(b[1] for b in real_bboxes),
                    max(b[2] for b in real_bboxes),
                    max(b[3] for b in real_bboxes),
                )
            elif assembled.pages_mapping[0].bbox_list:
                bboxes = assembled.pages_mapping[0].bbox_list
                env = (
                    min(b[0] for b in bboxes),
                    min(b[1] for b in bboxes),
                    max(b[2] for b in bboxes),
                    max(b[3] for b in bboxes),
                )
            else:
                env = (0.0, 0.0, float(width_px), float(height_px))

            article_envelopes.append((article_record.id, env, article_record.headline or ""))
            persisted_articles.append((article_record, assembled))

        await self._db.flush()

        # 7. Extract Photos & Visual Intelligence
        media_extractor = MediaExtractor(minio=self._minio, db=self._db)
        all_media_boxes: list[tuple[tuple[float, float, float, float], str]] = [
            (m.bbox, m.caption or "") for m in page_media_items if m.bbox
        ]

        if single_analysis:
            for ibox in single_analysis.image_boxes:
                if not any(
                    abs(ibox[0] - mb[0][0]) < 30 and abs(ibox[1] - mb[0][1]) < 30
                    for mb in all_media_boxes
                ):
                    all_media_boxes.append((ibox, ""))

        article_photos_map: dict[int, list[Photo]] = {}
        has_large_canvas = False
        img_idx = 1
        for m_box, m_cap in all_media_boxes:
            b_area = max(0.0, m_box[2] - m_box[0]) * max(0.0, m_box[3] - m_box[1])
            if (b_area / max(1.0, float(width_px * height_px))) >= 0.75:
                has_large_canvas = True

            bound_art_id = media_extractor.resolve_photo_article_binding(
                photo_bbox=m_box,
                article_envelopes=article_envelopes,
                caption=m_cap,
            )
            try:
                photo_rec = await media_extractor.extract_and_store_photo(
                    page_image_bytes=page_image_bytes,
                    page_id=page.id,
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
            except Exception as p_err:
                logger.warning(
                    "Photo extraction failed on page",
                    extra={"page_number": page_number, "error": str(p_err)},
                )

        # Fallback: if page yielded 0 discrete photos AND has a large canvas (>= 75%), run VLM Grounding Sweep
        if not article_photos_map and has_large_canvas:
            logger.info(
                "Triggering VLM Grounding Sweep fallback on large visual canvas",
                extra={"page_number": page_number, "page_id": page.id},
            )
            grounded_subphotos = await media_extractor.extract_subphotos_vlm_fallback(
                page_image_bytes=page_image_bytes,
                page_id=page.id,
                article_envelopes=article_envelopes,
                width_px=width_px,
                height_px=height_px,
                start_photo_index=img_idx,
            )
            for bound_art_id, photo_rec in grounded_subphotos:
                if bound_art_id:
                    if bound_art_id not in article_photos_map:
                        article_photos_map[bound_art_id] = []
                    article_photos_map[bound_art_id].append(photo_rec)
                img_idx += 1

        # 8. Chunking, Dense Vector Embeddings, and Qdrant Indexing
        total_chunks_created = 0
        total_vectors_created = 0

        for article_rec, assembled_art in persisted_articles:
            art_photos = article_photos_map.get(article_rec.id, [])
            has_photo = bool(art_photos)
            has_table = (
                "table" in (article_rec.article_type or "")
                or "table" in (article_rec.headline or "").lower()
                or bool(re.search(r"\|\s*[-:]+\s*\|", assembled_art.full_text))
            )

            # Annotate full text with visual tags
            annotated_text = assembled_art.full_text
            if has_photo:
                annotated_text = f"[VISUAL: PHOTO INCLUDED]\n{annotated_text}"
            if has_table:
                annotated_text = f"[VISUAL: DATA TABLE INCLUDED]\n{annotated_text}"

            # 1. Generate text chunks
            chunks = chunker.chunk_article(
                full_text=annotated_text,
                newspaper_name=newspaper_name,
                issue_date=issue_date_str,
                headline=article_rec.headline or "",
                section=article_rec.section or "National",
                pages=[page_number],
                printed_pages=[final_printed_folio],
            )

            # 2. Dedicated visual chunks for charts and infographics
            visual_chunks = []
            for p_num_idx, photo in enumerate(art_photos, start=len(chunks)):
                if photo.vlm_description and photo.visual_type in {"data_chart", "table", "infographic"}:
                    v_chunk = chunker.create_visual_chunk(
                        visual_markdown=photo.vlm_description,
                        visual_type=photo.visual_type,
                        summary=photo.caption or "",
                        newspaper_name=newspaper_name,
                        issue_date=issue_date_str,
                        headline=article_rec.headline or "",
                        section=article_rec.section or "National",
                        pages=[page_number],
                        printed_pages=[final_printed_folio],
                        chunk_index=p_num_idx,
                    )
                    visual_chunks.append(v_chunk)

            # Fetch entity and topic names for payload filtering
            ent_res = await self._db.execute(
                select(Entity.name)
                .join(ArticleEntity, ArticleEntity.entity_id == Entity.id)
                .where(ArticleEntity.article_id == article_rec.id)
            )
            entity_names = [name for (name,) in ent_res.all()]

            top_res = await self._db.execute(
                select(Topic.name)
                .join(ArticleTopic, ArticleTopic.topic_id == Topic.id)
                .where(ArticleTopic.article_id == article_rec.id)
            )
            topic_names = [name for (name,) in top_res.all()]

            # 3. Embed and upsert text chunks to Qdrant + MySQL
            if chunks:
                vec_ids = await embedder.embed_and_index_chunks(
                    article_id=article_rec.id,
                    issue_id=issue_id,
                    newspaper_name=newspaper_name,
                    issue_date=issue_date_str,
                    headline=article_rec.headline or "",
                    section=article_rec.section,
                    article_type=article_rec.article_type or "news",
                    prominence_score=article_rec.prominence_score,
                    page_numbers=[page_number],
                    entities=entity_names,
                    topics=topic_names,
                    chunks=chunks,
                    printed_pages=[final_printed_folio],
                    has_photo=has_photo,
                    has_table=has_table,
                    chunk_type="text",
                    has_visual_data=False,
                )
                total_chunks_created += len(chunks)
                total_vectors_created += len(vec_ids)

            # 4. Embed and upsert visual chunks to Qdrant + MySQL
            for v_chunk, photo in zip(visual_chunks, art_photos, strict=False):
                v_vec_ids = await embedder.embed_and_index_chunks(
                    article_id=article_rec.id,
                    issue_id=issue_id,
                    newspaper_name=newspaper_name,
                    issue_date=issue_date_str,
                    headline=article_rec.headline or "",
                    section=article_rec.section,
                    article_type=article_rec.article_type or "news",
                    prominence_score=article_rec.prominence_score,
                    page_numbers=[page_number],
                    entities=entity_names,
                    topics=topic_names,
                    chunks=[v_chunk],
                    printed_pages=[final_printed_folio],
                    has_photo=True,
                    has_table=photo.visual_type == "table",
                    chunk_type="visual",
                    has_visual_data=True,
                    visual_type=photo.visual_type,
                )
                total_chunks_created += 1
                total_vectors_created += len(v_vec_ids)

        # 9. Finalize page status and commit transaction
        page.ingestion_status = "indexed"
        await self._db.commit()

        logger.info(
            "Page re-ingestion successfully completed",
            extra={
                "issue_id": issue_id,
                "page_number": page_number,
                "articles_count": len(persisted_articles),
                "chunks_count": total_chunks_created,
                "vectors_count": total_vectors_created,
            },
        )

        return {
            "status": "success",
            "issue_id": issue_id,
            "page_number": page_number,
            "printed_page_number": final_printed_folio,
            "is_advertisement_page": page.is_advertisement_page,
            "articles_count": len(persisted_articles),
            "photos_count": sum(len(p) for p in article_photos_map.values()),
            "chunks_count": total_chunks_created,
            "vectors_count": total_vectors_created,
            "articles": [
                {
                    "id": art.id,
                    "headline": art.headline,
                    "article_type": art.article_type,
                    "section": art.section,
                    "word_count": art.word_count,
                    "prominence_score": art.prominence_score,
                }
                for art, _ in persisted_articles
            ],
        }
