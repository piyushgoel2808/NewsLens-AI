"""Photo region cropping and table metadata extractor for newspaper pages."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.article import ArticleTable, Photo
from app.storage.minio_store import MinioStore

logger = get_logger(__name__)


@dataclass
class ExtractedPhoto:
    """Cropped photo asset metadata."""

    object_key: str
    caption: str
    bbox: tuple[float, float, float, float]


class MediaExtractor:
    """Extracts photo image crops and structured table assets from newspaper pages."""

    def __init__(self, db: AsyncSession, minio: MinioStore | None = None) -> None:
        self._db = db
        self._settings = get_settings()
        self._minio = minio or MinioStore(self._settings.minio)

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

        photo_record = Photo(
            article_id=article_id,
            page_id=page_id,
            bbox_json={"bbox": [x0, y0, x1, y1]},
            caption=caption,
            object_key=object_key,
        )
        self._db.add(photo_record)
        await self._db.flush()

        logger.info(
            "Extracted and stored photo asset",
            extra={"page_id": page_id, "object_key": object_key},
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
