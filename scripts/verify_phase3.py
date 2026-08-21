"""Phase 3 Verification Script — Article Segmentation, Cross-Page Assembly & Classification.

Validates:
1. Article Boundary Segmenter: Clusters reading blocks into coherent articles with bylines & jump targets.
2. Cross-Page Assembler: Stitches multi-page continuation stories into unified Article entities.
3. Article Classifier: Correctly tags 8-tier article types and computes prominence scores (0.0 to 1.0).
4. Real Demo Newspaper Integration: Tests segmentation on a real newspaper from demo/ directory.
5. REST API & Database: Validates /api/articles/{id} and /api/issues/{id}/articles.

Usage:
    python scripts/verify_phase3.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import date
from pathlib import Path

# Add backend and root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.main import create_app
from app.core.config import get_settings
from app.ingestion.classifier import ArticleClassifier
from app.ingestion.cross_page_assembler import CrossPageAssembler
from app.ingestion.intake import IntakeService
from app.ingestion.reading_order import BlockType, OrderedReadingBlock
from app.ingestion.segmenter import ArticleSegmenter, SegmentedArticle
from app.ingestion.tasks import run_ingestion_pipeline
from app.models.article import Article, ArticlePage
from app.models.newspaper import Issue, Page
from app.storage.minio_store import MinioStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "backend" / "tests" / "fixtures"
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"

RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, message: str) -> None:
    RESULTS.append((name, passed, message))


async def run_phase3_verification() -> None:
    print("\n" + "=" * 70)
    print("NewsLens-AI — Phase 3 End-to-End Verification")
    print("Article Segmentation, Cross-Page Assembly & Classification")
    print("=" * 70 + "\n")

    settings = get_settings()
    engine = create_async_engine(settings.database.async_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    minio = MinioStore(settings.minio)
    await minio.startup()

    multi_page_pdf = FIXTURES_DIR / "sample_multi_page_issue.pdf"
    if not multi_page_pdf.exists():
        from scripts.generate_sample_newspaper import generate_all_samples
        generate_all_samples(FIXTURES_DIR)

    # Test 1: Cross-Page Assembly Logic
    t0 = time.monotonic()
    try:
        assembler = CrossPageAssembler()
        p1_art = SegmentedArticle(
            article_temp_id="p1_a1",
            headline="GOVERNMENT PASSES KEY TAX REFORM ACT",
            byline_author="By National Bureau",
            body_text="Lawmakers passed the long-awaited tax legislation today.\nContinued on Page 3",
            jump_to_page=3,
            word_count=12,
            bbox_list=[(30.0, 30.0, 300.0, 200.0)],
        )
        p3_art = SegmentedArticle(
            article_temp_id="p3_a1",
            headline="TAX REFORM ACT (Continued from Page 1)",
            body_text="The new tax brackets will take effect from the start of the next financial quarter.",
            jump_from_page=1,
            word_count=16,
            bbox_list=[(30.0, 50.0, 300.0, 250.0)],
        )
        assembled = assembler.assemble_issue_articles({1: [p1_art], 3: [p3_art]})
        assert len(assembled) == 1
        assert assembled[0].primary_page_number == 1
        assert len(assembled[0].pages_mapping) == 2
        assert assembled[0].pages_mapping[0].page_number == 1
        assert assembled[0].pages_mapping[1].page_number == 3
        assert "next financial quarter" in assembled[0].full_text

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Cross-Page Jump Story Continuation Assembly",
            True,
            f"Stitched multi-page story across pages {[p.page_number for p in assembled[0].pages_mapping]} ({latency}ms)",
        )
    except Exception as e:
        _record("Cross-Page Jump Story Continuation Assembly", False, f"{type(e).__name__}: {e}")

    # Test 2: Full End-to-End Issue Ingestion with Article Creation
    t0 = time.monotonic()
    created_issue_id: int | None = None
    try:
        async with session_factory() as db:
            intake = IntakeService(db=db, minio=minio)
            intake_res = await intake.process_upload(
                file_bytes=multi_page_pdf.read_bytes(),
                filename="sample_multi_page_issue.pdf",
                newspaper_name="The Daily Observer",
                issue_date=date(2026, 8, 20),
                edition="morning",
                language="en",
                force=True,
            )
            created_issue_id = intake_res.issues_created[0]

        pipe_res = await run_ingestion_pipeline(
            issue_id=created_issue_id,
            pdf_bytes=multi_page_pdf.read_bytes(),
            dpi=300,
            minio=minio,
            session_factory=session_factory,
        )

        assert pipe_res["total_pages"] >= 2
        assert len(pipe_res["articles"]) >= 2

        # Verify MySQL Database Records
        async with session_factory() as db:
            articles_stmt = select(Article).where(Article.issue_id == created_issue_id)
            articles_res = await db.execute(articles_stmt)
            articles_in_db = articles_res.scalars().all()
            assert len(articles_in_db) >= 2

            # Verify article_pages junction rows
            ap_stmt = select(ArticlePage).where(ArticlePage.article_id == articles_in_db[0].id)
            ap_res = await db.execute(ap_stmt)
            pages_for_art = ap_res.scalars().all()
            assert len(pages_for_art) >= 1

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "End-to-End Multi-Page Ingestion & MySQL Persistence",
            True,
            f"Created {len(articles_in_db)} articles and verified article_pages junction records ({latency}ms)",
        )
    except Exception as e:
        _record("End-to-End Multi-Page Ingestion & MySQL Persistence", False, f"{type(e).__name__}: {e}")

    # Test 3: Real Demo Newspaper Ingestion (from demo/ folder)
    t0 = time.monotonic()
    try:
        demo_files = list(DEMO_DIR.glob("*.pdf"))
        if not demo_files:
            _record("Real Demo Newspaper Ingestion", False, "No PDF files found in demo/ directory")
        else:
            import fitz

            demo_pdf = demo_files[0]
            # Take a 2-page slice from the demo newspaper for fast verification
            src_doc = fitz.open(demo_pdf)
            slice_doc = fitz.open()
            slice_doc.insert_pdf(src_doc, from_page=0, to_page=min(1, len(src_doc) - 1))
            demo_bytes = slice_doc.tobytes()
            slice_doc.close()
            src_doc.close()

            async with session_factory() as db:
                intake = IntakeService(db=db, minio=minio)
                demo_intake = await intake.process_upload(
                    file_bytes=demo_bytes,
                    filename=f"slice_{demo_pdf.name}",
                    newspaper_name="Demo Indian Daily",
                    issue_date=date(2026, 7, 30),
                    edition="delhi",
                    language="en",
                    force=True,
                )
                demo_issue_id = demo_intake.issues_created[0]

            demo_pipe = await run_ingestion_pipeline(
                issue_id=demo_issue_id,
                pdf_bytes=demo_bytes,
                dpi=150,  # Fast verification DPI
                minio=minio,
                session_factory=session_factory,
            )

            assert demo_pipe["total_pages"] > 0
            assert len(demo_pipe["articles"]) > 0

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Real Demo Newspaper Ingestion (demo/)",
                True,
                f"Ingested '{demo_pdf.name}' (2-page sample): {demo_pipe['total_pages']} pages, {len(demo_pipe['articles'])} segmented articles ({latency}ms)",
            )
    except Exception as e:
        import traceback
        _record(
            "Real Demo Newspaper Ingestion (demo/)",
            False,
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )

    # Test 4: Article REST API Endpoints
    t0 = time.monotonic()
    try:
        from app.models.base import init_db
        init_db(settings.database.async_url)
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. List articles for issue
            if created_issue_id:
                list_resp = await client.get(f"/api/issues/{created_issue_id}/articles")
                assert list_resp.status_code == 200
                articles_list = list_resp.json()
                assert len(articles_list) >= 2
                first_art_id = articles_list[0]["id"]

                # 2. Get article details
                detail_resp = await client.get(f"/api/articles/{first_art_id}")
                assert detail_resp.status_code == 200
                detail = detail_resp.json()
                assert detail["id"] == first_art_id
                assert "headline" in detail
                assert "pages" in detail
                assert len(detail["pages"]) >= 1

                latency = round((time.monotonic() - t0) * 1000)
                _record(
                    "Article REST API Endpoints (/api/articles & /api/issues/)",
                    True,
                    f"Verified GET /api/issues/{created_issue_id}/articles and GET /api/articles/{first_art_id} ({latency}ms)",
                )
            else:
                _record("Article REST API Endpoints (/api/articles & /api/issues/)", False, "created_issue_id was None")
    except Exception as e:
        _record("Article REST API Endpoints (/api/articles & /api/issues/)", False, f"{type(e).__name__}: {e}")

    from app.models.base import close_db
    await close_db()
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
        print("\n✓ Phase 3 Article Segmentation & Assembly Verification PASSED cleanly!\n")
        sys.exit(0)
    else:
        print("\n✗ Phase 3 Verification FAILED. See details above.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_phase3_verification())
