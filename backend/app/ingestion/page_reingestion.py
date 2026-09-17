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
from app.ingestion.detector import PDFPageDetector, check_is_advertisement_text
from app.ingestion.embedder import ArticleEmbedder
from app.ingestion.layout import (
    ArticleSegmenter,
    CrossPageAssembler,
    LayoutAnalyzer,
    SegmentedArticle,
)
from app.ingestion.media_extractor import MediaExtractor
from app.ingestion.metadata_extractor import MetadataExtractor
from app.ingestion.parsers import (
    CorruptedPdfTextLayerError,
    DoclingLayoutParser,
    ExtractedPhotoData,
)
from app.ingestion.rasterizer import DEFAULT_DPI, PDFRasterizer
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
                dpi=DEFAULT_DPI,
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

        # 5. Extract Layout & Articles
        detector = PDFPageDetector()
        digital_analysis = detector.analyze_document_bytes(single_pdf_bytes)
        single_analysis = digital_analysis[0] if digital_analysis else None

        # 5. Extract Layout & Articles using DoclingLayoutParser
        page_segmented_articles: list[SegmentedArticle] = []
        page_media_items: list[ExtractedPhotoData] = []
        parsed_doc_items: list[Any] = []
        try:
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

            page_segmented_articles = list(docling_articles)
            page_media_items = docling_parser.extract_page_media_items(parsed_doc_items)
        except CorruptedPdfTextLayerError as font_err:
            logger.warning(
                "Docling failed due to corrupted PDF font layer, escalating to image OCR fallback",
                extra={"page_number": page_number, "error": str(font_err)},
            )
        except Exception as docling_err:
            logger.warning(
                "DoclingLayoutParser encountered error, checking image OCR fallback",
                extra={"page_number": page_number, "error": str(docling_err)},
            )

        # Fallback to Pure Image OCR (Google Cloud Vision / LayoutAnalyzer) if 0 articles extracted
        if not page_segmented_articles:
            logger.info(
                "Running Pure Image OCR fallback for page",
                extra={"page_number": page_number, "width_px": width_px, "height_px": height_px},
            )
            try:
                from app.providers.google_vision_provider import GoogleCloudVisionOCR
                gcv_api_key = self._settings.google_api_key or self._settings.gemini_api_key
                gcv_ocr = GoogleCloudVisionOCR(api_key=gcv_api_key)
                ocr_result = await gcv_ocr.ocr(image_bytes=page_image_bytes, lang_hint=issue_lang)

                if ocr_result.blocks:
                    layout_analyzer = LayoutAnalyzer()
                    page_layout = layout_analyzer.analyze_from_text_blocks(
                        page_number=page_number,
                        width_px=width_px,
                        height_px=height_px,
                        ocr_blocks=ocr_result.blocks,
                    )
                    segmenter = ArticleSegmenter()
                    ocr_articles = segmenter.segment_page(
                        page_number=page_number,
                        ordered_blocks=page_layout.reading_order,
                        is_advertisement_page=page.is_advertisement_page,
                    )
                    if ocr_articles:
                        page_segmented_articles = ocr_articles
                        logger.info(
                            "Image OCR successfully recovered segmented articles",
                            extra={"page_number": page_number, "articles_count": len(ocr_articles)},
                        )
            except Exception as ocr_err:
                logger.warning(
                    "Image OCR fallback encountered error",
                    extra={"page_number": page_number, "error": str(ocr_err)},
                )

        # Ultimate fallback if still 0 articles detected
        if not page_segmented_articles:
            full_page_txt = ""
            if single_analysis and single_analysis.full_text.strip():
                full_page_txt = single_analysis.full_text.strip()

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

        # 6. Assemble and Classify Articles
        assembler = CrossPageAssembler()
        assembled_articles = assembler.assemble_issue_articles({page_number: page_segmented_articles})
        classifier = ArticleClassifier()
        metadata_extractor = MetadataExtractor(db=self._db)
        chunker = NewspaperChunker()
        embedder = ArticleEmbedder(db=self._db, qdrant=self._qdrant)

        persisted_articles: list[tuple[Article, Any]] = []
        article_envelopes: list[Any] = []

        for assembled in assembled_articles:
            class_res = classifier.classify_and_score(
                article=assembled,
                total_issue_pages=issue.total_pages or 1,
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

            # Record spatial envelope for photo binding
            art_bboxes = assembled.pages_mapping[0].bbox_list if assembled.pages_mapping else []
            if art_bboxes:
                env_x0 = min(b[0] for b in art_bboxes)
                env_y0 = min(b[1] for b in art_bboxes)
                env_x1 = max(b[2] for b in art_bboxes)
                env_y1 = max(b[3] for b in art_bboxes)
                env = (env_x0, env_y0, env_x1, env_y1)
            else:
                env = (0.0, 0.0, float(width_px), float(height_px))

            article_envelopes.append((article_record.id, env, article_record.headline or "", art_bboxes))
            persisted_articles.append((article_record, assembled))

        await self._db.flush()

        # 7. Extract Photos & Visual Intelligence
        media_extractor = MediaExtractor(minio=self._minio, db=self._db)
        media_list = list(page_media_items)
        if single_analysis:
            for ibox in single_analysis.image_boxes:
                if not any(
                    abs(ibox[0] - m.bbox[0]) < 30 and abs(ibox[1] - m.bbox[1]) < 30
                    for m in media_list
                    if m.bbox
                ):
                    media_list.append(ExtractedPhotoData(bbox=ibox, caption=""))

        article_photos_map: dict[int, list[Photo]] = {}
        has_large_canvas = False
        for m in media_list:
            if m.bbox:
                b_area = max(0.0, m.bbox[2] - m.bbox[0]) * max(0.0, m.bbox[3] - m.bbox[1])
                if (b_area / max(1.0, float(width_px * height_px))) >= 0.75:
                    has_large_canvas = True

        raw_items = parsed_doc_items if "parsed_doc_items" in locals() else []
        stored_pairs = await media_extractor.extract_and_store_all_photos_single_pass(
            page_image_bytes=page_image_bytes,
            page_id=page.id,
            article_envelopes=article_envelopes,
            media_items=media_list,
            all_docling_items=raw_items,
            page_width_px=width_px,
            page_height_px=height_px,
            start_photo_index=1,
        )

        for bound_art_id, photo_rec in stored_pairs:
            if bound_art_id:
                if bound_art_id not in article_photos_map:
                    article_photos_map[bound_art_id] = []
                article_photos_map[bound_art_id].append(photo_rec)

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
                start_photo_index=len(stored_pairs) + 1,
            )
            for bound_art_id, photo_rec in grounded_subphotos:
                if bound_art_id:
                    if bound_art_id not in article_photos_map:
                        article_photos_map[bound_art_id] = []
                    article_photos_map[bound_art_id].append(photo_rec)

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


__all__ = [
    "PageReingestionService",
    "check_is_advertisement_text",
]

