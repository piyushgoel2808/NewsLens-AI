"""Phase 6.1 Verification Script — API Hardening & Functional Data Flow.

Validates:
1. Server-Sent Events (SSE) Streaming: POST /api/query/stream stage transitions and token streaming.
2. Corpus APIs: GET /api/newspapers and GET /api/issues with aggregation metrics.
3. Issue Details & Page Image API: GET /api/issues/{id} and GET /api/pages/{id}/image.
4. Settings APIs: GET & PUT /api/settings/model-bindings with runtime invalidation.
5. Frontend React SPA: Vite build output verification.

Usage:
    python scripts/verify_phase6_1.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.main import create_app
from app.core.config import get_settings
from app.ingestion.intake import IntakeService
from app.ingestion.tasks import run_ingestion_pipeline
from app.models.base import close_db, init_db
from app.storage.minio_store import MinioStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "backend" / "tests" / "fixtures"

RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, message: str) -> None:
    RESULTS.append((name, passed, message))


async def run_phase6_1_verification() -> None:
    print("\n" + "=" * 70)
    print("NewsLens-AI — Phase 6.1 End-to-End Verification")
    print("API Hardening, SSE Streaming & Functional Data Flow")
    print("=" * 70 + "\n")

    settings = get_settings()
    init_db(settings.database.async_url)
    engine = create_async_engine(settings.database.async_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    minio = MinioStore(settings.minio)
    await minio.startup()

    # Ingest fixture issue if not present
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

    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test 1: SSE Streaming Query Endpoint
        t0 = time.monotonic()
        try:
            resp = await client.post(
                "/api/query/stream",
                json={"query": "What happened to the financial markets?"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            lines = resp.text.split("\n")
            events = [line for line in lines if line.startswith("event:")]
            tokens = [line for line in lines if line.startswith("data:") and "delta" in line]

            assert len(events) >= 5
            assert any("planning" in l for l in lines)
            assert any("synthesizing" in l for l in lines)
            assert len(tokens) > 0

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Streaming API: POST /api/query/stream (SSE)",
                True,
                f"Streamed {len(tokens)} token events and {len(events)} stage events ({latency}ms)",
            )
        except Exception as e:
            import traceback
            _record("Streaming API: POST /api/query/stream (SSE)", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

        # Test 2: Corpus API (Newspapers & Issues)
        t0 = time.monotonic()
        try:
            np_resp = await client.get("/api/newspapers")
            assert np_resp.status_code == 200
            newspapers = np_resp.json()
            assert len(newspapers) > 0
            total_articles = sum(n.get("article_count", 0) for n in newspapers)
            assert total_articles > 0

            iss_resp = await client.get("/api/issues")
            assert iss_resp.status_code == 200
            issues = iss_resp.json()
            assert len(issues) > 0

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Corpus API: GET /api/newspapers & GET /api/issues",
                True,
                f"Retrieved {len(newspapers)} newspapers ({total_articles} total articles) and {len(issues)} issues ({latency}ms)",
            )
        except Exception as e:
            import traceback
            _record("Corpus API: GET /api/newspapers & GET /api/issues", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

        # Test 3: Issue Details & Page Image Proxy API
        t0 = time.monotonic()
        try:
            detail_resp = await client.get(f"/api/issues/{issue_id}")
            assert detail_resp.status_code == 200
            detail = detail_resp.json()
            assert len(detail["pages"]) > 0
            assert len(detail["articles"]) > 0

            page_1_id = detail["pages"][0]["id"]
            img_resp = await client.get(f"/api/pages/{page_1_id}/image")
            assert img_resp.status_code == 200
            assert img_resp.headers.get("content-type") == "image/png"
            assert len(img_resp.content) > 1000

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Issue Details & Page Image API: GET /api/issues/{id} & /api/pages/{id}/image",
                True,
                f"Fetched issue details ({len(detail['pages'])} pages, {len(detail['articles'])} articles) and streamed 300 DPI page scan ({len(img_resp.content)} bytes) ({latency}ms)",
            )
        except Exception as e:
            import traceback
            _record("Issue Details & Page Image API: GET /api/issues/{id} & /api/pages/{id}/image", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

        # Test 4: Settings API (Model Bindings Runtime Swapping)
        t0 = time.monotonic()
        try:
            get_bind_resp = await client.get("/api/settings/model-bindings")
            assert get_bind_resp.status_code == 200
            bind_data = get_bind_resp.json()
            assert "task_bindings" in bind_data

            put_bind_resp = await client.put(
                "/api/settings/model-bindings",
                json={
                    "task_bindings": {
                        "query_planner": "ollama_chat",
                        "answerer": "ollama_chat",
                    }
                },
            )
            assert put_bind_resp.status_code == 200
            updated_data = put_bind_resp.json()
            assert updated_data["task_bindings"]["query_planner"] == "ollama_chat"

            latency = round((time.monotonic() - t0) * 1000)
            _record(
                "Settings API: GET & PUT /api/settings/model-bindings",
                True,
                f"Verified active model configuration and dynamic task-binding update at runtime ({latency}ms)",
            )
        except Exception as e:
            import traceback
            _record("Settings API: GET & PUT /api/settings/model-bindings", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test 5: Frontend Vite React Build
    t0 = time.monotonic()
    try:
        frontend_dir = ROOT_DIR / "frontend"
        build_proc = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(frontend_dir),
            capture_output=True,
            text=True,
        )
        assert build_proc.returncode == 0, f"Vite build failed: {build_proc.stderr}"
        dist_html = frontend_dir / "dist" / "index.html"
        assert dist_html.exists()

        latency = round((time.monotonic() - t0) * 1000)
        _record(
            "Frontend Scaffolding: Vite React SPA Build",
            True,
            f"Vite production build succeeded cleanly: dist/index.html generated ({latency}ms)",
        )
    except Exception as e:
        import traceback
        _record("Frontend Scaffolding: Vite React SPA Build", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

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
        print("\n✓ Phase 6.1 API Hardening & Functional Data Flow PASSED cleanly!\n")
        sys.exit(0)
    else:
        print("\n✗ Phase 6.1 Verification FAILED. See details above.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_phase6_1_verification())
