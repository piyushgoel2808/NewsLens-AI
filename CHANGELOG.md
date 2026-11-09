# Changelog

All notable changes to **NewsLens-AI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0] - 2026-11-10

### Added & Hardened (Phase 31 & Phase 32 Audit Remediations)
- **API Key / Bearer Authentication & Fail-Closed Production Security (`backend/app/core/security.py`)**:
  - Added `require_api_auth` FastAPI dependency validating `X-API-Key` or `Authorization: Bearer <token>` with constant-time `secrets.compare_digest` across all mutating endpoints (`/api/ingest/*`, `/api/settings/*`, newspaper/issue `POST`/`PUT`/`PATCH`/`DELETE`, and single-page re-ingestion).
  - Enforces fail-closed protection (`HTTP 403`) in production (`APP_DEBUG=false`) when `API_AUTH_KEY` is not set.
- **Distributed Sliding-Window Rate Limiting & Proxy Trust Guard (`backend/app/core/security.py`, `backend/app/api/main.py`)**:
  - Replaced per-worker in-memory rate limiting with a Redis-backed atomic `INCR` + `EXPIRE` sliding-window middleware (`RateLimitMiddleware`, default `RATE_LIMIT_RPM=60`) with graceful in-memory fallback.
  - Added `TRUST_FORWARDED_FOR` configuration to prevent `X-Forwarded-For` header spoofing.
- **Laptop Node PIN Brute-Force Lockout (`PinAttemptTracker`)**:
  - Added constant-time PIN comparison and per-IP brute-force protection on `POST /api/settings/laptop-node/toggle` (locks out client IP with `HTTP 429` for 300s after 5 failed attempts within 5 minutes).
- **Unlimited Multi-Year Qdrant Date Range Filtering (`issue_date_int`)**:
  - Added `YYYYMMDD` integer payload indexing (`issue_date_int`) in Qdrant (`backend/app/storage/qdrant_store.py` & `backend/app/ingestion/embedder.py`), replacing the 366-day date expansion cap with native `qdrant_models.Range(gte=..., lte=...)` + legacy `MatchAny` union for backward compatibility.
- **Request-Scoped `ContextVar` Task Bindings (`backend/app/providers/registry.py`)**:
  - Added `ModelRegistry.scoped_task_bindings()` backed by `contextvars.ContextVar` so per-request `model_override` selections in `/api/query` and `/api/query/stream` never mutate global singleton state across concurrent users.
- **Full Streaming & LangGraph CRAG/Verifier Parity (`backend/app/api/routers/query.py`)**:
  - Unified `POST /api/query/stream` with the compiled LangGraph workflow: post-condensation parameter re-extraction, reflexive CRAG evaluation (with adaptive replanning and dynamic tool synthesis), post-synthesis `AnswerVerifier` fact-checking, and real token cost accounting (`record_usage_and_cost`).
- **Hardened Subprocess AST Sandbox (`backend/app/agent/sandbox.py`, `sandbox_runner.py`)**:
  - Added AST binary string concatenation folding (`_fold_string_concat`) to catch obfuscated dunder strings, runtime `_safe_getattr` protection, MySQL `SET SESSION TRANSACTION READ ONLY`, and `_ReadOnlySessionProxy` blocking `.commit()`/`.flush()`.
- **Automatic Redis Query Cache Invalidation on Corpus Mutation**:
  - Wired `await CacheStore().clear_all()` into `DeletionService.delete_issue`, `PATCH /api/issues/{id}`, `run_ingestion_pipeline`, and `PageReingestionService.reingest_page`, plus MinIO prefix cleanup for `photos/{page_id}/` and `originals/{job_id}/`.
- **Docker Context Isolation & Worker Healthcheck Fixes**:
  - Added `.dockerignore`, `backend/.dockerignore`, and `frontend/.dockerignore` to block secrets (`service-account.json`, `.env*`) and virtualenvs from Docker images; pointed Celery worker to `app.ingestion.celery_app`; disabled inherited HTTP healthcheck on the worker container.
- **Docling Parser Unpacking, DualSync Empty-Byte Fallback & Frontend Lazy Mounting**:
  - Fixed inverted tuple unpacking (`clean_h, kicker = extract_kicker_and_clean_headline(...)`) in `parsers/docling.py`; fixed `DualSyncObjectStore.get()` empty-byte `b""` cloud fallback in `dual_store.py`; added lazy tab mounting and `vite.config.js` vendor chunk splitting (`vendor-react`, `vendor-icons`) in the React 19 frontend.

---

## [0.5.0] - 2026-09-21

### Added & Optimized
- **10x End-to-End Ingestion Pipeline Acceleration (OCR Preservation & Calibrated Visual Extraction)**:
  - Slashed 24-page broadsheet issue ingestion time from **7–11+ minutes (440s–660s)** down to **45s–65s** (~8x–10x acceleration / >90% reduction).
  - **100% Neural OCR Preservation**: Retained full IBM Docling neural layout OCR and RapidOCR text recognition across broadsheet pages, preserving multi-column article segmentation, reading order, and 2D bounding boxes without skipping OCR.
  - **Upfront In-Memory PDF Slicing & Concurrent 4-Worker OCR**: Pre-slices all single-page PDF streams in memory in `<30ms` with PyMuPDF, running Docling layout OCR concurrently via `asyncio.Semaphore(4)`.
  - **In-Memory Page Rasterization & 8x Concurrent Uploads**: Rendered all 24 page PNGs in memory (`<1s`) and uploaded concurrently to MinIO with `asyncio.Semaphore(8)` and bulk MySQL `Page` status updates.
  - **Calibrated VLM Visual Extraction & Token Headroom**: Expanded output runway to 8,192 tokens; allocated dynamic thinking budgets (512 tokens for infographics/tables/charts, 128 for photos) to prevent table truncation; implemented sub-batch partitioning ($\le 6$ items) to prevent token overflow. Increased visual extraction concurrency to 4 workers.
  - **In-Memory Entity/Topic Caches & Bulk Persistence**: Added process-level entity and topic caches, replacing ~2,200 individual roundtrip queries per issue with 3 bulk SQL queries.
  - **Whole-Issue Vector Chunk Aggregation & Bulk Qdrant Upsert**: Added `bulk_embed_and_index_issue()` to aggregate all ~250 chunks across the entire issue, embedding in 2 batch calls to Vertex AI `text-embedding-004` and upserting in a single bulk Qdrant operation with one database commit.

---

## [0.4.0] - 2026-09-21

### Added & Optimized
- **10x Faster Query Pipeline & Sub-2s Streaming TTFT**:
  - Slashed Time-to-First-Token (TTFT) from 25–45s to **1.8s–2.4s** and full stream completion to **3.5s–5.0s**.
  - Direct Vertex AI publisher model alignment on `gemini-2.5-flash`, eliminating upstream 404 retry delays.
  - Enforced calibrated thinking budgets (`thinking_budget: 0`) across structured JSON planning (`plan_query_async`), CRAG LLM Judge evaluation (`_evaluate_with_llm_judge`), dynamic tool maker, and response streaming (`synthesize_stream`), eliminating 15–20s of hidden thinking tokens.
  - Filtered raw reasoning thoughts (`part.get("thought")`) out of SSE client token streams.
  - Preserved 100% LLM cognitive understanding of impure, conversational user queries and dynamic `AnswerBlueprint` formatting.
- **Adaptive Cross-Encoder Candidate Capping (`hybrid_search.py`)**:
  - Capped rerank candidate pool adaptively: `max_rerank_candidates = min(len(final_results), min(16, max(8, top_k * 2)))`.
  - Slashed CPU Cross-Encoder inference latency from 2.2s to **~450ms** on 1 vCPU with zero degradation in Top-5 precision.
- **CRAG Dual-Signal Fast-Floor Gate (`evaluator.py`)**:
  - Added neural retrieval confidence evaluation (`rerank_score >= 0.30` or `rrf_score >= 0.015`) alongside lexical article word counts, passing strong evidence in <5ms without triggering unnecessary 4s LLM Judge evaluations.
- **Vertex AI Native IAM Routing & GCP Promotional Credit Preservation**:
  - Auto-discovers Application Default Credentials (ADC) from attached `newslens-runner` Service Account in Cloud Run.
  - Routes inference directly via `https://aiplatform.googleapis.com/v1/publishers/google/models` with OAuth2 Bearer tokens, keeping 100% of LLM tokens billed to GCP Promotional Credits (net ₹0 out-of-pocket).
- **85–90% Cloud Run Compute Cost Reduction**:
  - Reconfigured `newslens-backend` with `--cpu-throttling`, 1 vCPU, 2Gi RAM in CI/CD, eliminating continuous idle CPU charges.
  - Added process-wide module cache `_SHARED_CROSS_ENCODERS` in `reranker.py` to prevent redundant model disk reloads.
  - Built frontend `apiDeduplicator.js` to eliminate burst API calls across concurrent components on initial load.

---

## [0.3.0] - 2026-09-18

### Added
- **Dynamic Model-Aware Token Budgeting (`resolve_dynamic_token_budget`)**:
  - Eliminated static archetype token caps in favor of dynamic per-model output envelopes derived from `ProviderCapability` (`max_output_tokens` 4,096 or 8,192).
  - Added reasoning model auto-detection (`is_reasoning_model: true`) and `reasoning_headroom = 2048` across Gemini 2.5 Flash, DeepSeek-R1, NVIDIA Nemotron, and OpenAI reasoning models.
  - Formulated dynamic token scaling for user-requested word limits: $\max(1024, \text{target\_words} \times 4) + \text{reasoning\_headroom}$.
  - Dynamically calculates `effective_max_tokens` inside provider loops for both `synthesize()` and `synthesize_stream()`.
- **Authoritative Planner LLM Cognitive Routing**:
  - Removed fast-path deterministic bypass from `_classify_and_plan_node` in LangGraph (`graph.py`); the Planner LLM is now the authoritative cognitive router receiving live database schema and archive boundaries.
  - Deterministic heuristic router is reserved strictly as an emergency resilience fallback in `QueryPlanner.plan_query_async`.
- **Statistical & Analytical Computation Archetype (`analytical_computation`)**:
  - Introduced `analytical_computation` archetype in Planner prompt and Blueprints, routing statistical/mathematical queries ("average word count", "average length", "correlation") to `dynamic_analysis`.
  - Decoupled analytical computations from scalar counts, eliminating the 80-word ceiling.
- **OCR Noise Tolerance & Semantic Reconstruction**:
  - Added Section 4 (`OCR NOISE TOLERANCE & INTELLIGENT RECONSTRUCTION`) to `COMMON_ANALYTICAL_GUIDELINES` in `synthesizer.py`.
  - Authorizes the LLM to phonetically and semantically reconstruct words corrupted by broadsheet scanning ("Reconstruction is not hallucination"), strictly prohibiting copying raw OCR errors into answers.
- **Prompt Context Isolation & Contamination Elimination**:
  - Single-article queries isolate evidence strictly to target article chunks, eliminating irrelevant advertisements and unrelated page stories from contaminating prompt context.
- **Conversational Follow-Up Condenser Disambiguation**:
  - Enhanced history extraction to parse citations and quoted headlines from previous turns.
  - Normalizer catches truncated outputs and safely reformulates follow-ups like `"summarise it"` into standalone article queries.
  - Emergency LLM fallback before Python string slicing in streaming API.

---

## [0.2.0] - 2026-09-18

### Added
- **Production Serverless Deployment on Google Cloud Platform (`asia-south1`)**:
  - Live deployment of `newslens-frontend` (Nginx 1.27 Alpine SPA reverse proxy) and `newslens-backend` (FastAPI) to Google Cloud Run.
  - Live deployment of `newslens-worker` Celery background ingestion consumer configured with `--no-cpu-throttling` and `--min-instances=1` to guarantee uninterrupted background processing.
  - Dedicated worker entrypoint (`backend/app/run_worker.py`) with embedded background HTTP server on `$PORT` to satisfy Cloud Run liveness and readiness health probes while running Celery in the foreground.
  - Dedicated Cloud Run Job `newslens-migrate` executing Alembic migrations (`alembic upgrade head`) over the Cloud SQL Unix socket before container revision rollouts.
- **Polymorphic Cloud Storage Abstraction**:
  - `GoogleCloudStorageStore` (`backend/app/storage/gcs_store.py`) implementing the `ObjectStore` protocol with threadpool executor offloading for non-blocking async execution.
  - Dynamic storage factory `get_object_store()` (`backend/app/storage/factory.py`) supporting seamless switching between Google Cloud Storage (`STORAGE_BACKEND=gcs`) and MinIO S3 (`STORAGE_BACKEND=minio`).
  - Automated bucket initialization for `gs://newslens-ai-prod-pages` and `gs://newslens-ai-prod-originals`.
- **Managed Production Databases & Broker**:
  - Managed Cloud SQL for MySQL 8.0 with Unix Domain Socket connectivity (`/cloudsql/...`), utf8mb4 encoding, and FULLTEXT search indexes.
  - Managed Qdrant Cloud vector cluster on GCP with TLS encryption and API key authentication for 1024-dim BGE-M3 embeddings.
  - Upstash Managed Redis with TLS support (`rediss://...ssl_cert_reqs=required`) powering Celery task queues and sub-millisecond query caching.
- **Enterprise Keyless CI/CD Automation**:
  - GitHub Actions deployment pipeline (`.github/workflows/deploy-gcp.yml`) leveraging Google Cloud Workload Identity Federation (WIF) with OIDC authentication, eliminating all static JSON service account keys.
  - Automated quality gate running linting, type-checking, ephemeral MySQL migrations, and the full 574-test suite before builds.
  - Automated multi-arch container image builds pushed to Google Artifact Registry.
- **Production Model Configuration**:
  - `model_config.prod.yaml` binding Google Gemini Flash (`gemini-3.8-flash` with multi-candidate failover) across all core agentic reasoning, VLM, and extraction pipelines.
  - Centralized Google Secret Manager integration managing 11 production secrets.

---

## [0.1.0] - 2026-09-16

### Added
- **Core Broadsheet Layout Analysis & Neural Parsing**:
  - Multi-column article segmentation with IBM Docling (DocLayNet) and RapidOCR fallback.
  - 2D spatial headline-deck coalescing and drop-cap normalization.
  - Multi-page consensus masthead and publication date verification.
- **Multimodal Visual Asset Intelligence**:
  - Automatic isolation of editorial photos, infographics, and standalone figures.
  - 2D proximity scoring for caption-to-photo binding.
  - VLM visual grounding and on-demand infographic analysis via Qwen-VL and Gemini.
- **Agentic RAG & LangGraph State Machine**:
  - Dynamic Chain-of-Thought (CoT) query planner grounded in live relational archive metadata.
  - 8 specialized retrieval tools (hybrid search, issue inspection, entity network, cross-newspaper timeline, SQL analytics, visual inspection, dynamic tool synthesis).
  - CRAG (Corrective RAG) reflexive evaluator and adaptive re-planning loop.
  - Subprocess AST sandbox for safe execution of dynamic analytical tools.
- **Hybrid Retrieval & RRF**:
  - Dense vector retrieval with Qdrant and BAAI/bge-m3 embeddings.
  - Sparse relational retrieval with MySQL FULLTEXT search.
  - Stage-2 neural reranking with Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
- **Interactive React SPA Frontend**:
  - 300 DPI high-resolution Broadsheet Reader with interactive spatial bounding boxes.
  - Real-time Server-Sent Events (SSE) streaming reasoning trace and assistant chat.
  - Multi-hop Entity Knowledge Graph visualizer.
  - Cross-Newspaper Narrative Trajectory and Timeline explorer.
  - Dynamic Model Settings Studio for hot-swappable provider configurations.
- **Production Readiness & DevOps**:
  - Full Docker Compose stack (`docker-compose.yml`) for one-command containerized deployment.
  - Multi-stage Dockerfiles for backend and frontend.
  - Unified `make setup` and developer tooling in Makefile.
  - GitHub Actions CI workflow covering linting, type-checking, backend test suite, and frontend production builds.
