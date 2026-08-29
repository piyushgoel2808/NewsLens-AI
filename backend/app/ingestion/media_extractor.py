"""Photo region cropping and table metadata extractor for newspaper pages."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.visual_extractor import VisualDataExtractor, VisualExtractionResult
from app.models.article import ArticleTable, Photo
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)


@dataclass
class ExtractedPhoto:
    """Cropped photo asset metadata."""

    object_key: str
    caption: str
    bbox: tuple[float, float, float, float]
    visual_type: str | None = None
    vlm_description: str | None = None
    extracted_data: VisualExtractionResult | None = None


class MediaExtractor:
    """Extracts photo image crops and structured table assets from newspaper pages."""

    def __init__(
        self,
        db: AsyncSession,
        minio: MinioStore | None = None,
        visual_extractor: VisualDataExtractor | None = None,
    ) -> None:
        self._db = db
        self._settings = get_settings()
        self._minio = minio or MinioStore(self._settings.minio)
        self._visual_extractor = visual_extractor or VisualDataExtractor()

    def resolve_photo_article_binding(
        self,
        photo_bbox: tuple[float, float, float, float],
        article_envelopes: list[tuple[int, tuple[float, float, float, float], str]],
        caption: str = "",
    ) -> int | None:
        """Resolve which article owns a photo using convex envelope overlap & ambiguity tie-breaking.

        article_envelopes: list of (article_id, (ax0, ay0, ax1, ay1), headline)
        """
        if not article_envelopes:
            return None

        px0, py0, px1, py1 = photo_bbox
        pw = max(px1 - px0, 1.0)

        candidate_scores: list[tuple[int, float, str]] = []

        for art_id, (ax0, ay0, ax1, ay1), headline in article_envelopes:
            px_center = (px0 + px1) / 2.0
            py_center = (py0 + py1) / 2.0
            in_envelope = (ax0 <= px_center <= ax1 and ay0 <= py_center <= ay1)

            # 1. Check Horizontal Column Overlap
            h_overlap = max(0.0, min(px1, ax1) - max(px0, ax0))
            h_overlap_ratio = h_overlap / pw

            # 2. Check Vertical Edge Proximity
            # Photo inside envelope
            if py0 >= ay0 and py1 <= ay1:
                v_dist = 0.0
            elif py1 < ay0:
                v_dist = ay0 - py1  # photo above article
            else:
                v_dist = py0 - ay1  # photo below article

            norm_v_dist = max(0.0, v_dist / max(ay1 - ay0, 100.0))

            # Proximity Score (higher is better)
            containment_bonus = 0.5 if in_envelope else 0.0
            score = (h_overlap_ratio * 0.5) + (max(0.0, 1.0 - norm_v_dist) * 0.3) + containment_bonus
            candidate_scores.append((art_id, score, headline))

        if not candidate_scores:
            # Fallback to closest article envelope on the page
            def _center_dist(env: tuple[float, float, float, float]) -> float:
                cx, cy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
                ecx, ecy = (env[0] + env[2]) / 2.0, (env[1] + env[3]) / 2.0
                return ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5

            closest_art = min(article_envelopes, key=lambda it: _center_dist(it[1]))
            return closest_art[0]

        candidate_scores.sort(key=lambda item: item[1], reverse=True)
        top_art_id, top_score, top_hl = candidate_scores[0]

        if top_score < 0.20:
            # Fallback to nearest candidate
            return top_art_id

        # Check Ambiguity: if second best is within 10% score margin
        if len(candidate_scores) > 1:
            second_art_id, second_score, second_hl = candidate_scores[1]
            if (top_score - second_score) / max(top_score, 0.01) < 0.10:
                # Tie-breaker 1: Caption keyword / entity match
                if caption:
                    cap_lower = caption.lower()
                    top_match = any(w.lower() in cap_lower for w in top_hl.split() if len(w) > 4)
                    sec_match = any(w.lower() in cap_lower for w in second_hl.split() if len(w) > 4)
                    if top_match and not sec_match:
                        return top_art_id
                    elif sec_match and not top_match:
                        return second_art_id

        return top_art_id

    async def extract_and_store_photo(
        self,
        page_image_bytes: bytes,
        page_id: int,
        article_id: int | None,
        bbox: tuple[float, float, float, float],
        caption: str = "",
        photo_index: int = 1,
    ) -> Photo:
        """Crop photo region from high-resolution page image and persist to MinIO and MySQL."""
        image = Image.open(io.BytesIO(page_image_bytes))
        width, height = image.size

        # Clamp bounding box coordinates
        x0 = max(0, min(int(bbox[0]), width - 1))
        y0 = max(0, min(int(bbox[1]), height - 1))
        x1 = max(x0 + 1, min(int(bbox[2]), width))
        y1 = max(y0 + 1, min(int(bbox[3]), height))

        cropped = image.crop((x0, y0, x1, y1))
        crop_bytes_io = io.BytesIO()
        cropped.save(crop_bytes_io, format="PNG")
        crop_bytes = crop_bytes_io.getvalue()

        object_key = f"photos/{page_id}/photo_{photo_index}.png"

        # Upload crop to MinIO
        await self._minio.put(
            bucket=self._settings.minio.bucket_pages,
            key=object_key,
            data=crop_bytes,
            content_type="image/png",
        )

        # Execute 3-Stage Visual Intelligence Pipeline
        visual_type: str | None = "photo"
        vlm_desc: str | None = None
        extracted_res: VisualExtractionResult | None = None

        try:
            classification, extraction = await self._visual_extractor.process_image_crop(
                crop_bytes, ocr_text=caption
            )
            visual_type = classification.visual_type
            if extraction:
                extracted_res = extraction
                desc_parts = [extraction.summary]
                if extraction.key_metrics:
                    desc_parts.append("\nKey Metrics:\n• " + "\n• ".join(extraction.key_metrics))
                if extraction.markdown_table:
                    desc_parts.append("\n" + extraction.markdown_table)
                vlm_desc = "\n".join(desc_parts)

                # If classified as table, also create ArticleTable entry
                if visual_type == "table":
                    await self.store_table_metadata(
                        page_id=page_id,
                        article_id=article_id,
                        bbox=bbox,
                        table_data={
                            "summary": extraction.summary,
                            "markdown": extraction.markdown_table,
                            "metrics": extraction.key_metrics,
                            "confidence": extraction.confidence,
                        },
                        table_index=photo_index,
                    )
        except Exception as vlm_err:
            logger.warning(
                "Visual intelligence pass skipped or failed on crop",
                extra={"error": str(vlm_err), "page_id": page_id},
            )

        photo_record = Photo(
            article_id=article_id,
            page_id=page_id,
            bbox_json={"bbox": [x0, y0, x1, y1]},
            caption=caption,
            vlm_description=vlm_desc,
            visual_type=visual_type,
            object_key=object_key,
        )
        self._db.add(photo_record)
        await self._db.flush()

        logger.info(
            "Extracted and stored photo asset with visual intelligence",
            extra={
                "page_id": page_id,
                "object_key": object_key,
                "visual_type": visual_type,
                "has_vlm_desc": bool(vlm_desc),
            },
        )
        return photo_record

    async def store_table_metadata(
        self,
        page_id: int,
        article_id: int | None,
        bbox: tuple[float, float, float, float],
        table_data: dict[str, Any] | None = None,
        table_index: int = 1,
    ) -> ArticleTable:
        """Persist structured table metadata to MySQL."""
        table_record = ArticleTable(
            article_id=article_id,
            page_id=page_id,
            bbox_json={"bbox": list(bbox)},
            extracted_json=table_data or {},
            object_key=f"tables/{page_id}/table_{table_index}.json",
        )
        self._db.add(table_record)
        await self._db.flush()
        return table_record
