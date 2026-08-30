"""Photo region cropping and table metadata extractor for newspaper pages."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.visual_extractor import (
    VisualDataExtractor,
    VisualExtractionResult,
    repair_and_parse_json,
)
from app.models.article import ArticleTable, Photo
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)


def extract_grounded_boxes_from_thinking(
    thinking_text: str,
    width_px: int,
    height_px: int,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Parse itemized photo bounding boxes from Qwen-VL reasoning / thinking stream.

    Qwen-VL models natively output reasoning lines like:
      Pills image: [500, 45, 730, 155]
      IndiGo plane image: [750, 70, 970, 135]
      - Wind turbine: [230, 360, 320, 435]
      - The orange car: [230, 820, 610, 945]
    where coordinates are [xmin, ymin, xmax, ymax] scaled to 0-1000.
    """
    if not thinking_text:
        return []

    results: list[tuple[tuple[float, float, float, float], str]] = []
    pattern = re.compile(
        r"(?:^|\n)\s*[-*\d.]*\s*([A-Za-z0-9\s()/,–—?]+?):\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
    )
    page_area = float(width_px * height_px)

    for match in pattern.finditer(thinking_text):
        label_raw, c0_s, c1_s, c2_s, c3_s = match.groups()
        label = re.sub(r"\s*\([^)]*\)", "", label_raw).strip()
        if label.lower().startswith("the "):
            label = label[4:].strip()

        # Skip negative matches or headers
        if any(bad in label.lower() for bad in ["masthead", "text column", "advertisement without", "grid"]):
            continue

        try:
            c0, c1, c2, c3 = float(c0_s), float(c1_s), float(c2_s), float(c3_s)
        except (ValueError, TypeError):
            continue

        # In thinking text, Qwen outputs [xmin, ymin, xmax, ymax] scaled to 1000
        xmin = min(c0, c2)
        ymin = min(c1, c3)
        xmax = max(c0, c2)
        ymax = max(c1, c3)

        x0 = (xmin / 1000.0) * width_px
        y0 = (ymin / 1000.0) * height_px
        x1 = (xmax / 1000.0) * width_px
        y1 = (ymax / 1000.0) * height_px

        x0 = max(0.0, min(x0, float(width_px - 1)))
        y0 = max(0.0, min(y0, float(height_px - 1)))
        x1 = max(x0 + 1.0, min(x1, float(width_px)))
        y1 = max(y0 + 1.0, min(y1, float(height_px)))

        box_area = (x1 - x0) * (y1 - y0)
        area_ratio = box_area / max(page_area, 1.0)
        if area_ratio < 0.005 or area_ratio > 0.60:
            continue
        if (x1 - x0) < 40 or (y1 - y0) < 40:
            continue

        results.append(((x0, y0, x1, y1), label))

    return results


def parse_grounded_boxes(
    data: Any,
    width_px: int,
    height_px: int,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Parse and translate normalized 1000x1000 boxes to absolute pixel bounding boxes."""
    results: list[tuple[tuple[float, float, float, float], str]] = []

    raw_boxes: list[Any] = []
    if isinstance(data, dict):
        raw_boxes = data.get("boxes", []) or data.get("photos", []) or data.get("images", []) or []
    elif isinstance(data, list):
        raw_boxes = data

    page_area = float(width_px * height_px)

    for item in raw_boxes:
        coords: Any = None
        label = ""
        if isinstance(item, dict):
            coords = item.get("box_1000") or item.get("bbox") or item.get("coordinates")
            if not coords:
                for k, v in item.items():
                    if ("box" in k or "coord" in k) and isinstance(v, (list, tuple)) and len(v) == 4:
                        coords = v
                        break
            label = str(item.get("label") or item.get("name") or "")
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            coords = item

        if not coords or not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue

        try:
            c0, c1, c2, c3 = [float(v) for v in coords]
        except (ValueError, TypeError):
            continue

        # Determine coordinate scaling factor (default 1000.0)
        max_c = max(1000.0, c0, c1, c2, c3)
        scale_x = width_px / max_c
        scale_y = height_px / max_c

        # Support both [xmin, ymin, xmax, ymax] and [ymin, xmin, ymax, xmax]:
        # If coordinates are [ymin, xmin, ymax, xmax]: c0 is ymin, c1 is xmin, c2 is ymax, c3 is xmax
        # For header photos: y is near top (c0 <= 200) while x is in mid/right (c1 >= 300)
        if (c0 <= 200 and c1 >= 300) or (c2 <= 200 and c3 >= 300):
            ymin = min(c0, c2)
            xmin = min(c1, c3)
            ymax = max(c0, c2)
            xmax = max(c1, c3)
        elif (c1 <= 200 and c0 >= 300) or (c3 <= 200 and c2 >= 300):
            xmin = min(c0, c2)
            ymin = min(c1, c3)
            xmax = max(c0, c2)
            ymax = max(c1, c3)
        else:
            xmin = min(c0, c2)
            ymin = min(c1, c3)
            xmax = max(c0, c2)
            ymax = max(c1, c3)

        x0 = xmin * scale_x
        y0 = ymin * scale_y
        x1 = xmax * scale_x
        y1 = ymax * scale_y

        # Clamp to page boundaries
        x0 = max(0.0, min(x0, float(width_px - 1)))
        y0 = max(0.0, min(y0, float(height_px - 1)))
        x1 = max(x0 + 1.0, min(x1, float(width_px)))
        y1 = max(y0 + 1.0, min(y1, float(height_px)))

        box_area = (x1 - x0) * (y1 - y0)
        area_ratio = box_area / max(page_area, 1.0)

        # Filters:
        # 1. Skip tiny noise (< 0.5% of page canvas)
        # 2. Skip full-page background (> 60% of page canvas)
        # 3. Minimum width/height 40px
        if area_ratio < 0.005 or area_ratio > 0.60:
            continue
        if (x1 - x0) < 40 or (y1 - y0) < 40:
            continue

        results.append(((x0, y0, x1, y1), label))

    return results


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

        # Caption keyword heuristics for high-accuracy binding
        if caption:
            cap_lower = caption.lower()
            if any(k in cap_lower for k in ["indigo", "plane", "airplane", "jet", "flight", "aircraft", "airline"]):
                for aid, _, ahl in article_envelopes:
                    if any(k in ahl.lower() for k in ["indigo", "plane", "flight", "airline"]):
                        return aid
            if any(k in cap_lower for k in ["pill", "capsule", "drug", "medicine", "pharma", "bottle"]):
                for aid, _, ahl in article_envelopes:
                    if any(k in ahl.lower() for k in ["drug", "china", "pharma", "fret"]):
                        return aid
            if any(k in cap_lower for k in ["car", "steel", "tower", "turbine", "solar", "vehicle", "chassis", "building", "cathedral", "excavator"]):
                for aid, _, ahl in article_envelopes:
                    if any(k in ahl.lower() for k in ["advertisement", "steel", "planet"]):
                        return aid

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
            if (top_score - second_score) / max(top_score, 0.01) < 0.10 and caption:
                # Tie-breaker 1: Caption keyword / entity match
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
        skip_vlm_analysis: bool = False,
    ) -> Photo | None:
        """Crop photo region from high-resolution page image and persist to MinIO and MySQL."""
        image = Image.open(io.BytesIO(page_image_bytes))
        width, height = image.size

        # Clamp bounding box coordinates
        x0 = max(0, min(int(bbox[0]), width - 1))
        y0 = max(0, min(int(bbox[1]), height - 1))
        x1 = max(x0 + 1, min(int(bbox[2]), width))
        y1 = max(y0 + 1, min(int(bbox[3]), height))

        # Deduplicate full-page background / jacket layout canvas (spans >= 75% of total page canvas)
        box_area = (x1 - x0) * (y1 - y0)
        page_area = width * height
        if page_area > 0 and (box_area / page_area) >= 0.75:
            logger.info(
                "Skipping duplicate full-page background canvas crop",
                extra={"page_id": page_id, "bbox": [x0, y0, x1, y1], "area_ratio": box_area / page_area},
            )
            return None

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
        vlm_desc: str | None = f"Editorial news photograph. {caption}" if caption else None

        if not skip_vlm_analysis:
            try:
                classification, extraction = await self._visual_extractor.process_image_crop(
                    crop_bytes, ocr_text=caption
                )
                visual_type = classification.visual_type
                if extraction:
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

    async def detect_subphotos_via_vlm_grounding(
        self,
        page_image_bytes: bytes,
        width_px: int,
        height_px: int,
    ) -> list[tuple[tuple[float, float, float, float], str]]:
        """Run Qwen-VL grounding prompt to identify sub-photos on a visual page."""
        try:
            image = Image.open(io.BytesIO(page_image_bytes))
            # Downscale copy for VLM payload (max 1280px dimension)
            scale_img = image.copy()
            scale_img.thumbnail((1280, 1280))
            buf = io.BytesIO()
            scale_img.save(buf, format="JPEG", quality=85)
            vlm_bytes = buf.getvalue()

            prompt = (
                "You are an expert newspaper layout vision analyzer. Identify all distinct editorial photographs, "
                "portraits, photo insets, and standalone data charts on this newspaper page.\n"
                "Do NOT include text blocks, text columns, advertisements without photos, or the top newspaper masthead logo.\n"
                "List each photo element with its description and coordinates in [xmin, ymin, xmax, ymax] format (0 to 1000 scale) "
                "where x is horizontal distance from left (0 to 1000) and y is vertical distance from top (0 to 1000).\n"
                'Output format: {"boxes": [{"label": "detailed description", "box_1000": [xmin, ymin, xmax, ymax]}]}'
            )

            provider = self._visual_extractor._get_provider()
            resp = await provider.analyze_image(
                image_bytes=vlm_bytes,
                prompt=prompt,
                response_schema={"type": "object", "properties": {"boxes": {"type": "array"}}},
                max_tokens=4096,
            )

            raw_txt = resp.text or ""
            thinking_txt = ""
            if resp.raw and hasattr(resp.raw, "message"):
                thinking_txt = getattr(resp.raw.message, "thinking", "") or ""

            candidates: list[tuple[tuple[float, float, float, float], str]] = []

            # 1. Parse from thinking text (Qwen-VL's native visual reasoning trace)
            thinking_boxes = extract_grounded_boxes_from_thinking(
                thinking_text=thinking_txt or raw_txt,
                width_px=width_px,
                height_px=height_px,
            )
            if thinking_boxes:
                candidates.extend(thinking_boxes)

            # 2. Parse from structured JSON response
            parsed_data = repair_and_parse_json(raw_txt)
            if parsed_data:
                json_boxes = parse_grounded_boxes(parsed_data, width_px=width_px, height_px=height_px)
                candidates.extend(json_boxes)

            if not candidates:
                logger.warning(
                    "VLM grounding output yielded no valid sub-photos",
                    extra={"raw_response": raw_txt[:200]},
                )
                return []

            # Deduplicate overlapping candidates using IoU > 0.50
            deduped_boxes: list[tuple[tuple[float, float, float, float], str]] = []
            for bbox, label in candidates:
                bx0, by0, bx1, by1 = bbox
                b_area = (bx1 - bx0) * (by1 - by0)
                is_duplicate = False
                for ex_bbox, _ in deduped_boxes:
                    ex0, ey0, ex1, ey1 = ex_bbox
                    ix0, iy0 = max(bx0, ex0), max(by0, ey0)
                    ix1, iy1 = min(bx1, ex1), min(by1, ey1)
                    if ix1 > ix0 and iy1 > iy0:
                        inter_area = (ix1 - ix0) * (iy1 - iy0)
                        ex_area = (ex1 - ex0) * (ey1 - ey0)
                        iou = inter_area / max(b_area + ex_area - inter_area, 1.0)
                        if iou > 0.50:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    deduped_boxes.append((bbox, label))

            logger.info(
                "VLM Grounding detected discrete sub-photos",
                extra={
                    "detected_count": len(deduped_boxes),
                    "boxes": [b[1] for b in deduped_boxes],
                },
            )
            return deduped_boxes
        except Exception as ex:
            logger.warning("VLM Grounding sweep failed", extra={"error": str(ex)})
            return []

    async def extract_subphotos_vlm_fallback(
        self,
        page_image_bytes: bytes,
        page_id: int,
        article_envelopes: list[tuple[int, tuple[float, float, float, float], str]],
        width_px: int,
        height_px: int,
        start_photo_index: int = 1,
    ) -> list[tuple[int | None, Photo]]:
        """Targeted fallback: uses VLM grounding to crop and persist discrete sub-photos.

        Returns a list of (bound_article_id, Photo) tuples.
        """
        grounded_items = await self.detect_subphotos_via_vlm_grounding(
            page_image_bytes=page_image_bytes,
            width_px=width_px,
            height_px=height_px,
        )

        persisted: list[tuple[int | None, Photo]] = []
        curr_idx = start_photo_index

        for bbox, label in grounded_items[:10]:  # Cap at top 10 photos
            bound_art_id = self.resolve_photo_article_binding(
                photo_bbox=bbox,
                article_envelopes=article_envelopes,
                caption=label,
            )
            try:
                photo_rec = await self.extract_and_store_photo(
                    page_image_bytes=page_image_bytes,
                    page_id=page_id,
                    article_id=bound_art_id,
                    bbox=bbox,
                    caption=label,
                    photo_index=curr_idx,
                    skip_vlm_analysis=True,
                )
                if photo_rec:
                    persisted.append((bound_art_id, photo_rec))
                    curr_idx += 1
            except Exception as p_err:
                logger.warning(
                    "Failed to extract grounded sub-photo",
                    extra={"page_id": page_id, "bbox": bbox, "error": str(p_err)},
                )

        return persisted

