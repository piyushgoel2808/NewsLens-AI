#!/usr/bin/env python3
"""
Backfill script to run visual table & data extraction on existing infographic / data chart photos
that currently have fallback placeholder descriptions in the database.
"""
import argparse
import asyncio
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select, or_
from app.models.base import get_session_factory, close_db
from app.models.article import Photo
from app.storage.minio_store import MinioStore
from app.ingestion.visual_extractor import VisualDataExtractor
from app.core.config import get_settings


async def backfill_infographics(limit: int = 10, dry_run: bool = False):
    cfg = get_settings()
    minio = MinioStore(cfg.minio)
    extractor = VisualDataExtractor()
    factory = get_session_factory()

    print(f"[*] Scanning for infographic / chart photos needing extraction (limit={limit})...")
    async with factory() as db:
        stmt = (
            select(Photo)
            .where(
                or_(
                    Photo.visual_type.in_(["infographic", "data_chart", "table"]),
                    Photo.vlm_description.is_(None),
                    Photo.vlm_description.like("%Visual asset: infographic%"),
                    Photo.vlm_description.like("%Visual asset%"),
                )
            )
            .order_by(Photo.id.asc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        photos = result.scalars().all()
        print(f"[*] Found {len(photos)} visual asset candidate(s) to process.")

        processed = 0
        for photo in photos:
            print(f"--- Processing Photo #{photo.id} (type={photo.visual_type}) ---")
            if not photo.object_key:
                print(f"  [!] Photo #{photo.id} has no object_key, skipping.")
                continue

            try:
                img_bytes = await minio.get(
                    bucket=cfg.minio.bucket_pages,
                    key=photo.object_key,
                )
                if not img_bytes:
                    print(f"  [!] Could not fetch image bytes for key '{photo.object_key}'.")
                    continue

                if dry_run:
                    print(f"  [Dry Run] Would extract data for Photo #{photo.id} ({len(img_bytes)} bytes)")
                    processed += 1
                    continue

                classification, extraction = await extractor.process_image_crop(
                    image_bytes=img_bytes,
                    ocr_text=photo.caption or "",
                )

                if extraction:
                    parts = [extraction.summary]
                    if extraction.key_metrics:
                        parts.append("\nKey Data Points & Metrics:\n• " + "\n• ".join(extraction.key_metrics))
                    if extraction.markdown_table:
                        parts.append("\n" + extraction.markdown_table)
                    photo.vlm_description = "\n".join(parts)
                    photo.visual_type = classification.visual_type or photo.visual_type
                    db.add(photo)
                    await db.commit()
                    print(f"  [✓] Updated Photo #{photo.id} as {photo.visual_type} with {len(photo.vlm_description)} chars of extracted data.")
                    processed += 1
                else:
                    print(f"  [-] No data extracted for Photo #{photo.id}.")
            except Exception as e:
                print(f"  [!] Error extracting Photo #{photo.id}: {e}")
                await db.rollback()

        print(f"[*] Backfill finished. Successfully updated {processed} visual asset(s).")
    await close_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill infographic visual extraction")
    parser.add_argument("--limit", type=int, default=5, help="Max number of photos to process")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without modifying database")
    args = parser.parse_args()

    asyncio.run(backfill_infographics(limit=args.limit, dry_run=args.dry_run))
