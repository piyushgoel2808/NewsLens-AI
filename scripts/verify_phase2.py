#!/usr/bin/env python3
"""Phase 2 Verification Script — OCR & Layout Analysis Pipeline.

Validates:
1. Tesseract OCR Engine: Real OCR execution on scanned bitmap pages with confidence scoring.
2. Layout Analysis: Headline identification, column bounding boxes, and reading order resolution.
3. Database State Transitions: Page ingestion_status to 'layout_done' and ocr_confidence persistence.

Usage:
    python scripts/verify_phase2.py
"""
from __future__ import annotations

import asyncio
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

from app.core.config import get_settings
from app.ingestion.detector import PDFPageDetector
from app.ingestion.intake import IntakeService
from app.ingestion.layout_analyzer import LayoutAnalyzer
from app.ingestion.ocr_service import OCRService
from app.ingestion.rasterizer import PDFRasterizer
from app.ingestion.reading_order import BlockType
from app.ingestion.tasks import run_ingestion_pipeline
from app.models.newspaper import Page
from app.storage.minio_store import MinioStore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "backend" / "tests" / "fixtures"

RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, message: str) -> None:
    RESULTS.append((name, passed, message))


async def run_phase2_verification() -> None:
    print("\n" + "=" * 70)
    print("NewsLens-AI — Phase 2 End-to-End Verification")
    print("Scanned-PDF OCR, Layout Extraction & Reading Order Resolution")
    print("=" * 70 + "\n")

    settings = get_settings()
    engine = create_async_engine(settings.database.async_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    minio = MinioStore(settings.minio)
    await minio.startup()

    scanned_pdf_path = FIXTURES_DIR / "sample_scanned_page.pdf"
    digital_pdf_path = FIXTURES_DIR / "sample_digital_frontpage.pdf"

    if not scanned_pdf_path.exists():
        from scripts.generate_sample_newspaper import generate_all_samples
        generate_all_samples(FIXTURES_DIR)

    async with session_factory() as db:
        # Test 1: Real Tesseract OCR on Scanned Page
        t0 = time.monotonic()
        try:
            # 1. Intake scanned file
            intake = IntakeService(db=db, minio=minio)
            intake_res = await intake.process_upload(
                file_bytes=scanned_pdf_path.read_bytes(),
                filename="sample_scanned_page.pdf",
                newspaper_name="The Historic Record",
                issue_date=date(1898, 4, 25),
                edition="morning",
                language="en",
                force=True,
            )
            issue_id = intake_res.issues_created[0]

            # 2. Rasterize
            rasterizer = PDFRasterizer(db=db, minio=minio)
            rendered = await rasterizer.rasterize_pdf_bytes(
                pdf_bytes=scanned_pdf_path.read_bytes(),
                issue_id=issue_id,
                dpi=300,
            )
            assert len(rendered) == 1

            # 3. OCR Page
            page_stmt = select(Page).where(Page.issue_id == issue_id, Page.page_number == 1)
            page_res = await db.execute(page_stmt)
            page = page_res.scalar_one()

            ocr_service = OCRService(db=db, minio=minio)
            ocr_out = await ocr_service.process_page_ocr(
                page_id=page.id,
                image_bytes=rendered[0].image_bytes,
                lang_hint="en",
            )

            assert len(ocr_out.blocks) > 0, "No OCR text blocks extracted"
            assert ocr_out.mean_confidence > 0.40, f"Low OCR confidence: {ocr_out.mean_confidence}"
            assert "WAR" in ocr_out.full_text or "CARIBBEAN" in ocr_out.full_text or "HISTORIC" in ocr_out.full_text

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Tesseract OCR on Scanned Bitmap Page",
                True,
                f"Extracted {len(ocr_out.blocks)} blocks | Mean confidence: {ocr_out.mean_confidence:.2%} | Text sample: {ocr_out.full_text[:50]!r}... ({latency}ms)",
            )
        except Exception as e:
            _record("Tesseract OCR on Scanned Bitmap Page", False, f"{type(e).__name__}: {e}")

        # Test 2: Layout Analysis & Reading Order Resolution on Digital Multi-Column Page
        t0 = time.monotonic()
        try:
            detector = PDFPageDetector()
            analysis_list = detector.analyze_document_bytes(digital_pdf_path.read_bytes())
            analysis = analysis_list[0]

            layout_analyzer = LayoutAnalyzer()
            layout_res = layout_analyzer.analyze_from_text_blocks(
                page_number=1,
                width_px=2480,
                height_px=3508,
                digital_blocks=analysis.blocks,
            )

            assert len(layout_res.elements) >= 3
            assert len(layout_res.reading_order) >= 3

            # Check that wide headlines lead the reading order
            first_block = layout_res.reading_order[0]
            assert first_block.block_type in (BlockType.BANNER_HEADLINE, BlockType.HEADLINE)

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Layout Analysis & Reading Order Linearization",
                True,
                f"Elements: {len(layout_res.elements)} | Reading stream: {len(layout_res.reading_order)} blocks | First: {first_block.text[:40]!r}... ({latency}ms)",
            )
        except Exception as e:
            _record("Layout Analysis & Reading Order Linearization", False, f"{type(e).__name__}: {e}")

        # Test 3: End-to-End Task Pipeline with Layout State Transition
        t0 = time.monotonic()
        try:
            pipe_res = await run_ingestion_pipeline(
                issue_id=issue_id,
                pdf_bytes=scanned_pdf_path.read_bytes(),
                dpi=300,
                minio=minio,
                session_factory=session_factory,
            )
            p0 = pipe_res["pages"][0]
            assert p0["requires_ocr"] is True
            assert p0["ocr_confidence"] is not None
            assert p0["layout_elements"] > 0

            # Verify page record in DB using fresh session
            async with session_factory() as check_db:
                page_stmt = select(Page).where(Page.issue_id == issue_id, Page.page_number == 1)
                page_res = await check_db.execute(page_stmt)
                updated_page = page_res.scalar_one()
                assert updated_page.ingestion_status == "layout_done"

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "End-to-End Pipeline & DB State Transitions",
                True,
                f"Page status: '{updated_page.ingestion_status}' | OCR confidence: {updated_page.ocr_confidence:.2%} ({latency}ms)",
            )
        except Exception as e:
            import traceback
            _record(
                "End-to-End Pipeline & DB State Transitions",
                False,
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

    await engine.dispose()

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
        print("\n✓ Phase 2 OCR & Layout Analysis Verification PASSED cleanly!\n")
        sys.exit(0)
    else:
        print("\n✗ Phase 2 Verification FAILED. See details above.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_phase2_verification())
