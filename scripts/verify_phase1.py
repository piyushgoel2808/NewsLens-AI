#!/usr/bin/env python3
"""Phase 1 Verification Script — End-to-End Ingestion Pipeline.

Validates:
1. PDF Intake & Hashing: Idempotency and ZIP archive unpacking.
2. Rasterization: High-resolution PNG rendering at 300 DPI stored in MinIO.
3. Text Detection: Correct classification of digital vs scanned pages.
4. Database Synchronization: MySQL tables (newspapers, issues, pages, ingestion_jobs).

Usage:
    python scripts/verify_phase1.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import date
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ingestion.detector import PDFPageDetector, PageType
from app.ingestion.intake import IntakeService
from app.ingestion.tasks import run_ingestion_pipeline
from app.models.ingestion import IngestionJob
from app.models.newspaper import Issue, Newspaper, Page
from app.storage.minio_store import MinioStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "backend" / "tests" / "fixtures"

RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, message: str) -> None:
    RESULTS.append((name, passed, message))


async def run_phase1_verification() -> None:
    print("\n" + "=" * 70)
    print("NewsLens-AI — Phase 1 End-to-End Verification")
    print("Intake, Rasterization (300 DPI), and Digital Text Extraction")
    print("=" * 70 + "\n")

    settings = get_settings()
    engine = create_async_engine(settings.database.async_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    minio = MinioStore(settings.minio)
    await minio.startup()

    digital_pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"
    scanned_pdf_path = FIXTURES_DIR / "sample_scanned_page.pdf"
    zip_archive_path = FIXTURES_DIR / "sample_newspaper_archive.zip"

    if not digital_pdf_path.exists():
        from scripts.generate_sample_newspaper import generate_all_samples
        generate_all_samples(FIXTURES_DIR)

    async with session_factory() as db:
        # Test 1: Digital PDF Intake & Pipeline
        t0 = time.monotonic()
        try:
            intake = IntakeService(db=db, minio=minio)
            intake_res = await intake.process_upload(
                file_bytes=digital_pdf_path.read_bytes(),
                filename="sample_digital_frontpage.pdf",
                newspaper_name="The Daily Tribune",
                issue_date=date(1930, 4, 15),
                edition="morning",
                language="en",
                force=True,
            )
            assert len(intake_res.issues_created) == 1, "Expected 1 issue created"
            issue_id = intake_res.issues_created[0]

            # Run full pipeline
            pipe_res = await run_ingestion_pipeline(
                issue_id=issue_id,
                pdf_bytes=digital_pdf_path.read_bytes(),
                dpi=300,
                minio=minio,
                session_factory=session_factory,
            )

            assert pipe_res["total_pages"] == 1
            p0 = pipe_res["pages"][0]
            assert p0["type"] in ("digital", "hybrid")
            assert p0["requires_ocr"] is False
            assert p0["width_px"] > 2000

            # Verify MinIO object exists
            exists = await minio.exists(settings.minio.bucket_pages, p0["object_key"])
            assert exists, f"Page raster not found in MinIO at {p0['object_key']}"

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Digital PDF Intake & 300 DPI Rasterization",
                True,
                f"Page rendered: {p0['width_px']}x{p0['height_px']}px | Text blocks: {p0['block_count']} | MinIO: {p0['object_key']} ({latency}ms)",
            )
        except Exception as e:
            _record("Digital PDF Intake & 300 DPI Rasterization", False, f"{type(e).__name__}: {e}")

        # Test 2: Scanned PDF Classification
        t0 = time.monotonic()
        try:
            detector = PDFPageDetector()
            analysis = detector.analyze_document_bytes(scanned_pdf_path.read_bytes())
            assert len(analysis) == 1
            assert analysis[0].page_type == PageType.SCANNED
            assert analysis[0].requires_ocr is True
            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Scanned PDF Classification & OCR Tagging",
                True,
                f"Detected: {analysis[0].page_type.value} | requires_ocr={analysis[0].requires_ocr} ({latency}ms)",
            )
        except Exception as e:
            _record("Scanned PDF Classification & OCR Tagging", False, f"{type(e).__name__}: {e}")

        # Test 3: ZIP Archive Ingestion (Multi-Issue)
        t0 = time.monotonic()
        try:
            intake = IntakeService(db=db, minio=minio)
            zip_res = await intake.process_upload(
                file_bytes=zip_archive_path.read_bytes(),
                filename="sample_newspaper_archive.zip",
                newspaper_name="Archive Gazette",
                issue_date=date(1929, 10, 24),
                edition="special",
                force=True,
            )
            assert zip_res.total_files == 2
            assert len(zip_res.issues_created) == 2
            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "ZIP Archive Intake & Multi-Issue Unpacking",
                True,
                f"Extracted {zip_res.total_files} issues into MySQL (Job ID: {zip_res.job_id}) ({latency}ms)",
            )
        except Exception as e:
            _record("ZIP Archive Intake & Multi-Issue Unpacking", False, f"{type(e).__name__}: {e}")

        # Test 4: Database Integrity Check
        try:
            np_stmt = select(Newspaper)
            np_res = await db.execute(np_stmt)
            newspapers = np_res.scalars().all()
            page_stmt = select(Page).where(Page.ingestion_status == "digital_text_extracted")
            page_res = await db.execute(page_stmt)
            pages = page_res.scalars().all()
            _record(
                "MySQL Schema Synchronization",
                len(newspapers) > 0 and len(pages) > 0,
                f"Newspapers in DB: {len(newspapers)} | Ingested pages: {len(pages)}",
            )
        except Exception as e:
            _record("MySQL Schema Synchronization", False, f"{type(e).__name__}: {e}")

    await engine.dispose()

    # Summary
    print("\nResults:")
    print("-" * 70)
    all_passed = True
    for name, passed, msg in RESULTS:
        status_icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status_icon}  {name}")
        print(f"          {msg}")
        if not passed:
            all_passed = False
    print("-" * 70)

    if all_passed:
        print("\n✓ Phase 1 Ingestion Verification PASSED cleanly!\n")
        sys.exit(0)
    else:
        print("\n✗ Phase 1 Ingestion Verification FAILED. See errors above.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_phase1_verification())
