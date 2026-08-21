"""Phase 5 Verification Script — Agentic Retrieval Engine & LangGraph Workflow.

Validates:
1. Toolbelt: Hybrid Reciprocal Rank Fusion (RRF k=60) dense/sparse search.
2. Toolbelt: Entity-grounded search with salience thresholds.
3. Toolbelt: Chronological Timeline Builder.
4. Toolbelt: SQL Analytics Engine (mention trends, topic volume, frontpage ratio).
5. Agentic Workflow: LangGraph StateGraph execution across all 5 Query Archetypes with citation generation:
   - Factual / Point-in-Time Lookup
   - Thematic / Multi-Issue Timeline
   - Cross-Newspaper Comparison
   - Entity Deep Dive
   - Quantitative & Frequency Trends
6. FastAPI Query REST API: POST /api/query, POST /api/query/plan, GET /api/query/history.

Usage:
    python scripts/verify_phase5.py
"""
from __future__ import annotations

import asyncio
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

from app.agent.graph import AgentWorkflow
from app.api.main import create_app
from app.core.config import get_settings
from app.ingestion.intake import IntakeService
from app.ingestion.tasks import run_ingestion_pipeline
from app.models.base import close_db, init_db
from app.retrieval.entity_filter import EntitySearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.sql_analytics import SQLAnalyticsEngine
from app.retrieval.timeline_builder import TimelineBuilder
from app.storage.minio_store import MinioStore
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "backend" / "tests" / "fixtures"

RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, message: str) -> None:
    RESULTS.append((name, passed, message))


async def run_phase5_verification() -> None:
    print("\n" + "=" * 70)
    print("NewsLens-AI — Phase 5 End-to-End Verification")
    print("Agentic Retrieval Engine, Toolbelt & LangGraph State Machine")
    print("=" * 70 + "\n")

    settings = get_settings()
    init_db(settings.database.async_url)
    engine = create_async_engine(settings.database.async_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    minio = MinioStore(settings.minio)
    await minio.startup()

    # Ingest fixture issue if not already present
    multi_page_pdf = FIXTURES_DIR / "sample_multi_page_issue.pdf"
    if not multi_page_pdf.exists():
        from scripts.generate_sample_newspaper import generate_all_samples
        generate_all_samples(FIXTURES_DIR)

    async with session_factory() as db:
        intake = IntakeService(db=db, minio=minio)
        intake_res = await intake.process_upload(
            file_bytes=multi_page_pdf.read_bytes(),
            filename="sample_multi_page_issue.pdf",
            newspaper_name="The Daily Record",
            issue_date=date(2026, 8, 21),
            edition="morning",
            language="en",
            force=True,
        )
        issue_id = intake_res.issues_created[0]

    await run_ingestion_pipeline(
        issue_id=issue_id,
        pdf_bytes=multi_page_pdf.read_bytes(),
        dpi=300,
        minio=minio,
        session_factory=session_factory,
    )

    # Test 1: Hybrid Reciprocal Rank Fusion Search
    t0 = time.monotonic()
    try:
        hybrid = HybridSearchEngine(session_factory=session_factory)
        results = await hybrid.search("market financial district trading", top_k=5)
        assert len(results) > 0
        top = results[0]
        assert top.rrf_score > 0.0
        assert top.headline is not None
        assert len(top.pages) > 0

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Toolbelt: Hybrid Reciprocal Rank Fusion Search (RRF)",
            True,
            f"Fused dense + sparse search: {len(results)} hits | Top: '{top.headline}' (RRF: {top.rrf_score:.6f}) ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Toolbelt: Hybrid Reciprocal Rank Fusion Search (RRF)", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 2: Entity Search & Salience Filter
    t0 = time.monotonic()
    try:
        entity_engine = EntitySearchEngine(session_factory=session_factory)
        ent_results = await entity_engine.search_by_entity(min_salience=0.1, top_k=5)
        assert len(ent_results) > 0
        top_ent = ent_results[0]
        assert top_ent.entity_name != ""
        assert top_ent.salience_score >= 0.1

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Toolbelt: Entity Search & Salience Filter",
            True,
            f"Retrieved {len(ent_results)} entity matches | Top: '{top_ent.entity_name}' ({top_ent.entity_type}, Salience: {top_ent.salience_score}) ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Toolbelt: Entity Search & Salience Filter", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 3: Chronological Timeline Builder
    t0 = time.monotonic()
    try:
        timeline_engine = TimelineBuilder(session_factory=session_factory)
        tl = await timeline_engine.build_timeline(query="market", limit=10)
        assert tl.total_articles > 0
        assert tl.total_dates > 0

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Toolbelt: Chronological Timeline Builder",
            True,
            f"Built timeline across {tl.total_dates} dates with {tl.total_articles} articles ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Toolbelt: Chronological Timeline Builder", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 4: SQL Analytics Engine (Trends, Topics, Frontpage Ratios)
    t0 = time.monotonic()
    try:
        analytics = SQLAnalyticsEngine(session_factory=session_factory)
        dist = await analytics.get_topic_distribution()
        assert len(dist) > 0
        frontpage_ratio = await analytics.get_frontpage_prominence_ratio()
        assert frontpage_ratio["total_articles"] > 0

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Toolbelt: SQL Analytics Engine",
            True,
            f"Computed topic distributions ({len(dist)} sections) and frontpage ratio ({frontpage_ratio['frontpage_ratio'] * 100:.1f}%) ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Toolbelt: SQL Analytics Engine", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 5: LangGraph Agent Workflow Across 5 Archetypes
    archetype_queries = [
        ("factual_lookup", "What happened in the financial district market?"),
        ("thematic_timeline", "Provide a chronological timeline of the market tumble"),
        ("quantitative_trend", "How many articles covered the market expansion?"),
        ("cross_newspaper_comparison", "Compare the coverage across newspapers for the transit plans"),
        ("entity_deep_dive", "Show me everything about the Mayor"),
    ]

    t0 = time.monotonic()
    try:
        workflow = AgentWorkflow(session_factory=session_factory)
        all_archetypes_passed = True

        for expected_arch, query_text in archetype_queries:
            t_sub = time.monotonic()
            state = await workflow.run(query=query_text, user_id="verifier")
            sub_latency = round((time.monotonic() - t_sub) * 1000)

            assert state["query"] == query_text
            assert state["archetype"] == expected_arch
            assert len(state["plan"]) > 0
            assert len(state["evidence_items"]) > 0
            assert len(state["citations"]) > 0
            assert len(state["synthesized_answer"]) > 20

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Agentic Workflow: LangGraph State Machine (5 Archetypes)",
            True,
            f"Successfully executed and synthesized answers with citations across all 5 archetypes ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Agentic Workflow: LangGraph State Machine (5 Archetypes)", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 6: FastAPI Query REST API Endpoints
    t0 = time.monotonic()
    try:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. POST /api/query/plan
            plan_resp = await client.post("/api/query/plan", json={"query": "Provide a timeline of events"})
            assert plan_resp.status_code == 200
            plan_data = plan_resp.json()
            assert plan_data["archetype"] == "thematic_timeline"
            assert len(plan_data["planned_tools"]) > 0

            # 2. POST /api/query
            query_resp = await client.post("/api/query", json={"query": "What happened to the markets?"})
            assert query_resp.status_code == 200
            ans_data = query_resp.json()
            assert len(ans_data["answer"]) > 10
            assert len(ans_data["citations"]) > 0

            # 3. GET /api/query/history
            hist_resp = await client.get("/api/query/history")
            assert hist_resp.status_code == 200
            hist_data = hist_resp.json()
            assert len(hist_data) > 0

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Query REST API Endpoints (POST /query, POST /plan, GET /history)",
            True,
            f"Verified POST /api/query, POST /api/query/plan, and GET /api/query/history ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Query REST API Endpoints (POST /query, POST /plan, GET /history)", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

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
        print("\n✓ Phase 5 Agentic Retrieval Engine & LangGraph State Machine PASSED cleanly!\n")
        sys.exit(0)
    else:
        print("\n✗ Phase 5 Verification FAILED. See details above.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_phase5_verification())
