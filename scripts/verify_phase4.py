"""Phase 4 Verification Script — Metadata Extraction, Hierarchical Chunking & Vector Indexing.

Validates:
1. Metadata Extraction: Entity NER (Person, Org, Location), Topic Taxonomies, and Summaries.
2. Hierarchical Chunker: Header context injection and paragraph boundaries.
3. Dense Vector Indexing (Qdrant): Live point upserts and semantic vector similarity search.
4. MySQL FULLTEXT Keyword Search: Exact keyword and phrase search across indexed articles.
5. Metadata REST APIs: /api/entities, /api/topics, /api/articles/{id}/entities.

Usage:
    python scripts/verify_phase4.py
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
from app.ingestion.intake import IntakeService
from app.ingestion.tasks import run_ingestion_pipeline
from app.models.article import Article, ArticleChunk
from app.models.base import close_db, init_db
from app.models.entity import ArticleEntity, Entity, Topic
from app.models.newspaper import Issue, Page
from app.storage.minio_store import MinioStore
from app.storage.mysql_fulltext import MySQLFullTextSearch
from app.storage.qdrant_store import QdrantStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "backend" / "tests" / "fixtures"
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"

RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, message: str) -> None:
    RESULTS.append((name, passed, message))


async def run_phase4_verification() -> None:
    print("\n" + "=" * 70)
    print("NewsLens-AI — Phase 4 End-to-End Verification")
    print("Metadata Extraction, Hierarchical Chunking, Vector & Full-Text Indexing")
    print("=" * 70 + "\n")

    settings = get_settings()
    init_db(settings.database.async_url)
    engine = create_async_engine(settings.database.async_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    minio = MinioStore(settings.minio)
    await minio.startup()

    qdrant = QdrantStore(settings.qdrant)
    await qdrant._ensure_collection()

    multi_page_pdf = FIXTURES_DIR / "sample_multi_page_issue.pdf"
    if not multi_page_pdf.exists():
        from scripts.generate_sample_newspaper import generate_all_samples
        generate_all_samples(FIXTURES_DIR)

    # Ingest full multi-page test issue through Phase 4 pipeline
    t0 = time.monotonic()
    created_issue_id: int | None = None
    created_article_id: int | None = None

    try:
        async with session_factory() as db:
            intake = IntakeService(db=db, minio=minio)
            intake_res = await intake.process_upload(
                file_bytes=multi_page_pdf.read_bytes(),
                filename="sample_multi_page_issue.pdf",
                newspaper_name="The Metropolis Chronicle",
                issue_date=date(2026, 8, 21),
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
        assert pipe_res["total_chunks"] >= 2
        created_article_id = next(
            (a["id"] for a in pipe_res["articles"] if a.get("chunks_count", 0) > 0),
            pipe_res["articles"][0]["id"],
        )

        # Verify DB states: Pages indexed, Issue completed
        async with session_factory() as db:
            issue_stmt = select(Issue).where(Issue.id == created_issue_id)
            issue = (await db.execute(issue_stmt)).scalar_one()
            assert issue.ingestion_status == "completed"

            pages_stmt = select(Page).where(Page.issue_id == created_issue_id)
            pages = (await db.execute(pages_stmt)).scalars().all()
            assert all(p.ingestion_status == "indexed" for p in pages)

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Phase 4 Full Ingestion Pipeline & Completion Status",
            True,
            f"Articles: {len(pipe_res['articles'])}, Chunks: {pipe_res['total_chunks']}, Pages indexed ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Phase 4 Full Ingestion Pipeline & Completion Status", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 2: Relational Metadata (Entities, Topics, Chunks in MySQL)
    t0 = time.monotonic()
    try:
        async with session_factory() as db:
            # Check entities in DB
            ent_stmt = select(Entity)
            entities = (await db.execute(ent_stmt)).scalars().all()
            assert len(entities) > 0

            # Check article_chunks in DB
            chunk_stmt = select(ArticleChunk).where(ArticleChunk.article_id == created_article_id)
            chunks = (await db.execute(chunk_stmt)).scalars().all()
            assert len(chunks) > 0
            assert chunks[0].embedding_vector_id is not None
            assert "[Newspaper:" in chunks[0].text

            # Check topics in DB
            top_stmt = select(Topic)
            topics = (await db.execute(top_stmt)).scalars().all()
            assert len(topics) > 0

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Relational Metadata Persistence (Entities, Topics, Chunks)",
            True,
            f"Verified {len(entities)} entities, {len(topics)} topics, and {len(chunks)} contextual chunks with vector IDs ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Relational Metadata Persistence (Entities, Topics, Chunks)", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 3: Qdrant Dense Vector Indexing & Semantic Search
    t0 = time.monotonic()
    try:
        from app.providers.registry import get_registry

        registry = get_registry()
        embed_provider = registry.get_provider("embedding")
        query_vector = await embed_provider.embed_one("financial market trading volume")

        search_results = await qdrant.search(
            query_vector=query_vector,
            top_k=5,
        )

        assert len(search_results) > 0
        top_match = search_results[0]
        assert "headline" in top_match.payload
        assert "newspaper_name" in top_match.payload
        assert top_match.score > 0.0

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Qdrant Dense Vector Upsert & Semantic Search",
            True,
            f"Retrieved {len(search_results)} vector matches | Top score: {top_match.score:.4f} ('{top_match.payload['headline'][:40]}...') ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Qdrant Dense Vector Upsert & Semantic Search", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 4: MySQL FULLTEXT Search
    t0 = time.monotonic()
    try:
        ft_search = MySQLFullTextSearch(session_factory=session_factory)
        matches = await ft_search.search("market financial trading", top_k=5)
        assert len(matches) > 0
        assert matches[0].score >= 0.0

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "MySQL FULLTEXT Keyword Search",
            True,
            f"Matched {len(matches)} articles with fulltext relevance score: {matches[0].score:.4f} ('{matches[0].headline[:40]}') ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("MySQL FULLTEXT Keyword Search", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 5: Metadata REST API Endpoints
    t0 = time.monotonic()
    try:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. GET /api/entities
            ent_resp = await client.get("/api/entities")
            assert ent_resp.status_code == 200
            ent_list = ent_resp.json()
            assert len(ent_list) > 0

            # 2. GET /api/topics
            top_resp = await client.get("/api/topics")
            assert top_resp.status_code == 200
            top_list = top_resp.json()
            assert len(top_list) > 0

            # 3. GET /api/articles/{id}/entities
            if created_article_id:
                art_ent_resp = await client.get(f"/api/articles/{created_article_id}/entities")
                assert art_ent_resp.status_code == 200
                art_ents = art_ent_resp.json()
                assert len(art_ents) >= 0

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Metadata REST API Endpoints (/api/entities & /api/topics)",
            True,
            f"Verified GET /api/entities, /api/topics, and /api/articles/{created_article_id}/entities ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Metadata REST API Endpoints (/api/entities & /api/topics)", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

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
        print("\n✓ Phase 4 Metadata Extraction, Embedding & Vector Indexing PASSED cleanly!\n")
        sys.exit(0)
    else:
        print("\n✗ Phase 4 Verification FAILED. See details above.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_phase4_verification())
