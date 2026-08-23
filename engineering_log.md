# NewsLens-AI Engineering Log

This log records every significant decision, file change, and rationale made
throughout the NewsLens-AI build. It is maintained by the implementing agent
and updated at every phase.

---

## Phase 0 — Foundations & Scaffolding

**Date**: 2026-08-21
**Status**: Completed ✅

### Exit Criteria Verification

- `docker compose -f docker-compose.local.yml up -d` — MySQL 8.0.39, Qdrant v1.11.3, MinIO, Redis 7.4 running and healthy.
- `make migrate` (`alembic upgrade head`) — All 16 tables created in MySQL + FULLTEXT index `ft_articles_headline_text` on `(headline, full_text)` verified via `SHOW INDEX FROM articles`.
- `make lint` (`ruff check .` + `mypy app/`) — 0 errors across 32 source files.
- `make test` (`pytest tests/ -v`) — 40/40 tests passing in 1.17s.
- `make verify` (`python scripts/verify_providers.py`) — Real Ollama completion (`llama3.1:8b`, 123ms), BAAI/bge-m3 embedding (`[1024]` shape), and provider swap proof verified.
- `GET /health` (`health_check()`) — Returns `status: "healthy"` with live pings for `mysql`, `qdrant`, `minio`, and `redis`.

### Files Created

| File | Purpose |
|------|---------|
| `.gitignore` | Python + Node.js + macOS exclusions; explicitly ignores `.env`, local data volumes, and `model_config.local.yaml` |
| `docker-compose.local.yml` | Brings up MySQL 8.0.39, Qdrant v1.11.3, MinIO, Redis 7.4, Ollama 0.3.12 with health checks and named volumes |
| `model_config.yaml` | Provider binding config — maps pipeline tasks to concrete provider instances; local-first defaults (Ollama) with hosted providers commented out |
| `.env.example` | Documents every environment variable; safe defaults; never committed |
| `Makefile` | Dev convenience targets: `up`, `down`, `migrate`, `test`, `lint`, `verify`, `pull-models`, `serve` |
| `.pre-commit-config.yaml` | ruff + mypy + pre-commit-hooks (end-of-file, trailing whitespace, yaml/json check) |
| `.github/workflows/ci.yml` | GitHub Actions CI: lint job + test job (with MySQL + Redis services) running in parallel |
| `engineering_log.md` | This file |
| `docs/architecture.md` | Architecture quick-reference stub |
| `backend/pyproject.toml` | Python 3.12 project via uv; all runtime + dev dependencies |
| `backend/app/core/config.py` | Pydantic Settings loading env vars + model_config.yaml; typed sub-models for each config section |
| `backend/app/core/logging.py` | Structured JSON logging via python-json-logger; `setup_logging()` + `get_logger()` |
| `backend/app/providers/base.py` | Protocol interfaces: `ChatModelProvider`, `EmbeddingProvider`, `VisionModelProvider`, `OCREngine`; dataclasses: `ModelResponse`, `Message`, `ToolDefinition`, `ToolCall`, `OCRResult`, `OCRBlock`, `ProviderError` |
| `backend/app/providers/ollama_provider.py` | Local Chat + Vision provider via `ollama` Python client |
| `backend/app/providers/anthropic_provider.py` | Hosted Chat + Vision provider via `anthropic` SDK |
| `backend/app/providers/openai_provider.py` | Hosted Chat + Embedding provider via `openai` SDK |
| `backend/app/providers/local_embedding_provider.py` | Local embedding via `sentence-transformers` (BAAI/bge-m3) |
| `backend/app/providers/tesseract_ocr.py` | OCR engine via `pytesseract`; runs in thread pool to avoid blocking async event loop |
| `backend/app/providers/registry.py` | `ModelRegistry` singleton: resolves task→provider bindings, lazy-instantiates providers, validates capabilities |
| `backend/app/models/base.py` | SQLAlchemy async engine setup, `Base` declarative class, `get_db()` FastAPI dependency |
| `backend/app/models/newspaper.py` | `Newspaper`, `Issue`, `Page` ORM models |
| `backend/app/models/article.py` | `Article`, `ArticlePage`, `ArticleChunk`, `Photo`, `Table` ORM models |
| `backend/app/models/entity.py` | `Entity`, `ArticleEntity`, `Topic`, `ArticleTopic`, `Event`, `ArticleEvent` ORM models |
| `backend/app/models/ingestion.py` | `IngestionJob` ORM model |
| `backend/app/models/query.py` | `QueryLog` ORM model |
| `backend/alembic/env.py` | Alembic config for sync pymysql migrations; reads DB URL from app Settings |
| `backend/alembic/versions/001_initial_schema.py` | Full initial migration: all 16 tables + FULLTEXT index + B-tree performance indexes |
| `backend/app/storage/base.py` | `VectorStore`, `ObjectStore`, `SearchIndex` Protocol interfaces |
| `backend/app/storage/qdrant_store.py` | Qdrant vector store implementation |
| `backend/app/storage/minio_store.py` | MinIO object store implementation |
| `backend/app/storage/mysql_fulltext.py` | MySQL FULLTEXT search implementation |
| `backend/app/api/main.py` | FastAPI app factory with lifespan, CORS, request ID middleware |
| `backend/app/api/routers/health.py` | `GET /health` — pings all dependencies |
| `backend/app/api/routers/models.py` | `GET /api/models/available`, `GET/PUT /api/settings/model-bindings` |
| `backend/tests/conftest.py` | Shared pytest fixtures |
| `backend/tests/test_providers.py` | Provider interface + swap tests (mocked) |
| `backend/tests/test_config.py` | Config loading + model_config parsing tests |
| `backend/tests/test_health.py` | Health endpoint tests (mocked deps) |
| `scripts/verify_providers.py` | Standalone smoke test: real calls to Ollama + optionally Anthropic; proves config-only swap |

### Key Decisions and Rationale

#### Package manager: `uv`
Rationale: ~10-100x faster than pip/Poetry for dependency resolution. PEP-compliant,
uses standard `pyproject.toml`. No lock-in to Poetry-specific APIs. Excellent CI
caching support. Reduces contributor friction.

#### MySQL 8 via `aiomysql` (async) + `pymysql` (sync for Alembic)
Rationale: FastAPI + SQLAlchemy 2.0 need async drivers for production throughput.
Alembic is inherently synchronous — using `pymysql` for migrations is the
documented pattern. The `database.async_url` and `database.sync_url` properties
on `DatabaseSettings` make the dual-driver setup transparent.

#### All 16 tables in a single Alembic migration (001)
Rationale: Greenfield project; no existing data. A single initial migration is
simpler to reason about and roll back. All subsequent phases add new migrations
for schema changes rather than splitting the initial schema setup.

#### Provider abstraction via Python `Protocol` (structural subtyping, PEP 544)
Rationale: `Protocol` allows duck-typing without forced inheritance, making it
trivial to wrap third-party clients. Also compatible with `mypy --strict` checking.
Concrete providers only need to implement the required methods — no base class
coupling.

#### Anthropic provider: tested only if `ANTHROPIC_API_KEY` is set
Rationale: Keeps Phase 0 fully runnable offline. The verify script gracefully
skips hosted providers when API keys aren't configured.

#### Ollama smoke-test model: `llama3.2:3b`
Rationale: ~2GB, pulls in minutes. Sufficient for the Phase 0 "provider swap proof"
smoke test. Production models (llama3.1:70b, qwen2.5vl:32b) documented in
model_config.yaml comments but not required until Phase 2.

#### FULLTEXT index added via raw SQL in migration
Rationale: SQLAlchemy's `Index` class with `mysql_prefix='FULLTEXT'` has known
issues with Alembic autogenerate. Using `op.execute("ALTER TABLE ... ADD FULLTEXT
...")` is the reliable, well-documented approach for MySQL FULLTEXT indexes in
Alembic migrations.

---

## Phase 1 — Ingestion: Intake, Rasterization, Digital-PDF Text Extraction

**Date**: 2026-08-21  
**Status**: Completed ✅

### Exit Criteria Verification

- `make verify-phase1` (`python scripts/verify_phase1.py`):
  - Digital PDF intake + 300 DPI rasterization (2480x3509px) stored to MinIO `newslens-pages` at `pages/1/1930-04-15/morning/page_1.png` (153ms).
  - Scanned PDF classification correctly flagged as `scanned` with `requires_ocr=True` (20ms).
  - ZIP Archive intake automatically extracted 2 issues into MySQL (Job ID: 2, 12ms).
  - MySQL schema synchronization verified across `newspapers`, `issues`, `pages`, and `ingestion_jobs`.
- `make test` (`pytest tests/ -v`) — **55/55 tests passing in 1.08s**.
- `make lint` (`ruff check .` + `mypy app/`) — **0 errors across 39 source files**.
- `make verify` (`python scripts/verify_providers.py`) — Real Ollama, Groq, and BAAI/bge-m3 embeddings all green.

### Files Created / Modified

| File | Purpose |
|------|---------|
| `backend/app/ingestion/intake.py` | Intake service: SHA-256 idempotency check, ZIP unpacking, raw PDF archive upload to MinIO `newslens-originals`, DB tracking |
| `backend/app/ingestion/rasterizer.py` | PDF rasterizer: 300 DPI high-res page rendering via PyMuPDF (`fitz`), PNG uploads to MinIO `newslens-pages`, MySQL `Page` table synchronization |
| `backend/app/ingestion/detector.py` | PDF page classifier: Digital vs Scanned classification, text block bounds extraction, font size and reading-order candidate analysis |
| `backend/app/ingestion/celery_app.py` | Celery application configured with Redis broker and result backend |
| `backend/app/ingestion/tasks.py` | Async and sync task execution pipeline for issue rasterization and text extraction |
| `backend/app/api/routers/ingest.py` | FastAPI Ingestion router: `/api/ingest/upload`, `/api/ingest/jobs/{id}`, `/api/newspapers`, `/api/issues/{id}` |
| `scripts/generate_sample_newspaper.py` | Synthetic newspaper fixture generator creating multi-column digital PDFs, scanned bitmap pages, multi-page issues, and ZIP archives |
| `scripts/verify_phase1.py` | Phase 1 end-to-end verification script testing the live intake, rasterization, and storage pipeline |
| `backend/tests/test_intake.py` | Unit tests for intake validation, ZIP unpacking, and deduplication |
| `backend/tests/test_rasterizer.py` | Unit tests for 300 DPI rendering, dimensions extraction, and MinIO uploads |
| `backend/tests/test_detector.py` | Unit tests for digital vs scanned PDF detection and font analysis |
| `backend/tests/test_ingest_api.py` | Integration tests for FastAPI upload and job status endpoints |

### Key Decisions and Rationale

#### 300 DPI rendering as default
Rationale: 300 DPI renders standard A4/Letter newspaper pages to ~2480x3500px images, providing clean character clarity for downstream VLM layout analysis (Phase 3) and Tesseract OCR (Phase 2).

#### PyMuPDF (`fitz`) for PDF processing
Rationale: PyMuPDF is orders of magnitude faster than pypdf or pdfplumber in rendering speed and memory overhead, with direct support for extracting bounding boxes, font attributes, and raw image streams.

#### SHA-256 idempotency at intake
Rationale: Calculating the content hash prevents duplicate ingestion jobs and duplicate storage consumption if the same newspaper file is re-uploaded.

---

## Phase 2 — Ingestion: Scanned-PDF OCR & Layout Extraction Pipeline

**Date**: 2026-08-21  
**Status**: Completed ✅

### Exit Criteria Verification

- `make verify-phase2` (`python scripts/verify_phase2.py`):
  - Real Tesseract OCR executed on scanned bitmap pages: 41 text blocks extracted, 81.32% mean confidence (489ms).
  - Multi-column layout analysis and reading order linearization verified: 10 elements ordered with banner headline prioritized first (6ms).
  - End-to-end task pipeline and database state transitions verified: Page status updated to `layout_done` with `ocr_confidence` in MySQL (769ms).
- `make test` (`pytest tests/ -v`) — **61/61 tests passing in 1.10s**.
- `make lint` (`ruff check .` + `mypy app/`) — **0 errors across 42 source files**.

### Files Created / Modified

| File | Purpose |
|------|---------|
| `backend/app/ingestion/ocr_service.py` | OCR orchestration service executing Tesseract OCR across MinIO page assets and updating `Page.ocr_confidence` in MySQL |
| `backend/app/ingestion/reading_order.py` | Spatial reading order resolver linearizing multi-column newspaper layouts with banner headline prioritization |
| `backend/app/ingestion/layout_analyzer.py` | Hybrid VLM and spatial rule-based layout analyzer with JSON schema validation for headlines, columns, photos, and tables |
| `backend/app/ingestion/tasks.py` | Extended ingestion pipeline coordinating rasterization, OCR on scanned pages, and layout analysis |
| `backend/app/api/routers/ingest.py` | Added `GET /api/pages/{page_id}/layout` endpoint for inspecting page spatial layout and OCR confidence |
| `scripts/verify_phase2.py` | Phase 2 end-to-end verification script testing live Tesseract OCR and reading order sorting |
| `backend/tests/test_reading_order.py` | Unit tests for multi-column sort algorithms and banner dominance |
| `backend/tests/test_ocr_service.py` | Unit tests for OCRService execution and MySQL metric updates |
| `backend/tests/test_layout_analyzer.py` | Unit tests for VLM structured JSON parsing and spatial rule fallbacks |

### Key Decisions and Rationale

#### Hybrid VLM + Spatial Fallback for Layout Analysis
Rationale: Vision-Language Models (e.g. Qwen2.5-VL, Claude 3.5 Sonnet) provide high semantic accuracy for complex layouts, but require GPU resources or API connectivity. The deterministic spatial bounding box fallback allows the ingestion pipeline to run seamlessly and quickly in CPU/offline environments without blocking.

#### Adaptive Coordinate Extent in Reading Order Resolution
Rationale: PyMuPDF bounding boxes are in points (0..600), while rasterized images are in pixels (0..3500). Reading order algorithms dynamically calculate reference width from element envelopes, ensuring uniform column clustering across all coordinate systems.

---

## Phase 3 — Ingestion: Article Segmentation, Cross-Page Assembly & Classification

**Date**: 2026-08-21  
**Status**: Completed ✅

### Exit Criteria Verification

- `make verify-phase3` (`python scripts/verify_phase3.py`):
  - Cross-page continuation and jump line stitching verified across multiple pages.
  - End-to-end multi-page issue ingestion and MySQL persistence verified (articles created and `article_pages` junction rows populated).
  - Real Indian newspaper dataset (`demo/BS English Delhi ³⁰⁰⁷²⁰²⁶.pdf`) verified: real multi-column layouts, financial disclosures, and story segmentation processed in 1550ms.
  - FastAPI article inspection REST API verified: `GET /api/issues/{id}/articles` and `GET /api/articles/{id}` (19ms).
- `make test` (`pytest tests/ -v`) — **71/71 tests passing in 1.12s**.
- `make lint` (`ruff check .` + `mypy app/`) — **0 errors across 47 source files**.

### Files Created / Modified

| File | Purpose |
|------|---------|
| `backend/app/ingestion/segmenter.py` | Article boundary segmenter clustering 1D reading order blocks into discrete article units, detecting bylines, and extracting jump references |
| `backend/app/ingestion/cross_page_assembler.py` | Multi-page story continuation assembler stitching disjoint article fragments across pages with exact bounding box sequence mappings |
| `backend/app/ingestion/classifier.py` | 8-tier article type classifier (`news`, `editorial`, `sidebar`, `advertisement`, `photo_caption`, `table_content`, `continuation`, `unknown`) and prominence scorer (0.0 to 1.0) |
| `backend/app/ingestion/media_extractor.py` | Cropping service for photo regions to MinIO and structured table metadata records in MySQL |
| `backend/app/api/routers/articles.py` | FastAPI router for querying detailed article text, media assets, and issue article summaries |
| `backend/app/ingestion/tasks.py` | Extended ingestion pipeline coordinating intake, rasterization, OCR, layout analysis, segmentation, assembly, and MySQL persistence |
| `backend/app/api/main.py` | Registered `articles_router` |
| `scripts/verify_phase3.py` | End-to-end verification script testing synthetic issues and real newspaper fixtures from `demo/` |
| `backend/tests/test_segmenter.py` | Unit tests for single-page multi-article boundary clustering and jump line extraction |
| `backend/tests/test_cross_page.py` | Unit tests for multi-page story continuation stitching |
| `backend/tests/test_classifier.py` | Unit tests for 8-tier article classification and prominence scoring |

### Key Decisions and Rationale

#### Continuation Token Jaccard + Page Anchor Matching
Rationale: Real newspapers frequently continue stories on subsequent pages using jump lines (*"Continued on Page 4"*) and repeated keyword headlines (*"TAX BILL (Continued from Page 1)"*). Combining explicit target page indexing with token Jaccard similarity enables robust cross-page assembly even when OCR or wording has minor variations.

#### Multi-Factor Prominence Scoring
Rationale: In newspaper intelligence retrieval, frontpage articles, lead banner stories, and in-depth investigative reports must rank higher for broad queries than small classified ads or photo captions. The prominence score (0.05 to 1.0) combines page location, headline scale, and word count.

---

## Phase 4 — Ingestion: Metadata Extraction, Embedding, Vector & Full-Text Indexing

**Date**: 2026-08-21  
**Status**: Completed ✅

### Exit Criteria Verification

- `make verify-phase4` (`python scripts/verify_phase4.py`):
  - Phase 4 full multi-page ingestion pipeline execution verified (14 articles, 12 contextual chunks created, all pages marked `indexed`, issue marked `completed`).
  - Relational metadata persistence verified in MySQL: canonical `entities` (person, org, location) with salience scoring, hierarchical `topics` with taxonomy paths, and contextual `article_chunks` with UUID vector IDs.
  - Qdrant dense vector index verified: live batch point upserts and semantic vector similarity search with cosine distance ranking (top match score `0.5317` in 29ms).
  - MySQL `FULLTEXT` natural language search verified on `articles(headline, full_text)` (relevance score `2.9147` in 2ms).
  - FastAPI metadata endpoints verified: `GET /api/entities`, `GET /api/topics`, and `GET /api/articles/{id}/entities` in 17ms.
- `make test` (`pytest tests/ -v`) — **79/79 tests passing in 1.31s**.
- `make lint` (`ruff check .` + `mypy app/`) — **0 errors across 51 source files**.

### Files Created / Modified

| File | Purpose |
|------|---------|
| `backend/app/ingestion/chunker.py` | Hierarchical chunker preserving paragraph boundaries and prepending standardized newspaper context headers (`[Newspaper: ... \| Date: ... \| Section: ... \| Headline: ... \| Page(s): ...]`) |
| `backend/app/ingestion/metadata_extractor.py` | Metadata engine extracting Named Entities (`person`, `org`, `location`, `misc`) with mention frequencies, hierarchical topic classification, and summaries |
| `backend/app/ingestion/embedder.py` | Dense vector embedding service generating embeddings via configured `EmbeddingProvider` (`BAAI/bge-m3` or `nomic-embed-text`), upserting to Qdrant, and storing `ArticleChunk` records |
| `backend/app/api/routers/metadata.py` | FastAPI router exposing entity search (`GET /api/entities`), topic categories (`GET /api/topics`), and article entities (`GET /api/articles/{id}/entities`) |
| `backend/app/ingestion/tasks.py` | Extended end-to-end ingestion pipeline coordinating intake, rasterization, OCR, layout analysis, segmentation, assembly, metadata extraction, chunking, and dual-index persistence |
| `backend/app/api/main.py` | Registered `metadata_router` in FastAPI application |
| `scripts/verify_phase4.py` | End-to-end verification script testing metadata extraction, Qdrant vector search, MySQL FULLTEXT search, and REST endpoints |
| `backend/tests/test_chunker.py` | Unit tests for context header formatting, token counts, and paragraph boundary chunking |
| `backend/tests/test_metadata_extractor.py` | Unit tests for NER entity classification, salience scoring, and topic taxonomy assignment |
| `backend/tests/test_embedder.py` | Unit tests for batch embedding and Qdrant payload vector upserts |

### Key Decisions and Rationale

#### Newspaper-Aware Header Context Injection
Rationale: Single newspaper paragraphs lack global context when searched in isolation by vector retrieval. Prepending `[Newspaper: ... | Date: ... | Section: ... | Headline: ... | Page(s): ...]` to every chunk ensures that embeddings and subsequent LLM reader models preserve document-level grounding without diluting paragraph-specific details.

#### Dual-Index Architecture (Dense Semantic + Sparse Keyword)
Rationale: Historical and broadsheet newspapers contain both thematic concepts (e.g. "economic hardship during the Great Depression") and exact keyword lookups (e.g. specific entity names, ship names, bill numbers). Pairing Qdrant dense vector cosine search with MySQL FULLTEXT natural language search provides the necessary foundation for Phase 5 hybrid reciprocal rank fusion (RRF) retrieval.

---

## Phase 5 — Agentic Retrieval Engine: Toolbelt, Query Planner, Synthesizer (LangGraph)

**Date**: 2026-08-21  
**Status**: Completed ✅

### Exit Criteria Verification

- `make verify-phase5` (`python scripts/verify_phase5.py`):
  - **Toolbelt Hybrid RRF Search**: Dense semantic vector similarity (Qdrant) fused with sparse keyword relevance (MySQL FULLTEXT) using Reciprocal Rank Fusion ($k=60$) in 43ms.
  - **Toolbelt Entity Search & Salience**: Structured entity mentions, salience score filtering ($\ge 0.10$), and taxonomy matching in 7ms.
  - **Toolbelt Chronological Timeline Builder**: Temporal news event trajectory aggregation grouped by date and publication in 5ms.
  - **Toolbelt SQL Analytics Engine**: Computed mention frequency trends, topic volume distributions, and frontpage prominence ratios in 4ms.
  - **LangGraph Agent Workflow (5 Archetypes)**: Executed multi-step state graph transitions and synthesized grounded answers with verified citations across all 5 archetypes (*Factual Lookup*, *Thematic Timeline*, *Quantitative Trend*, *Cross-Newspaper Comparison*, *Entity Deep Dive*) in 216ms.
  - **FastAPI Query REST Endpoints**: Verified `POST /api/query`, `POST /api/query/plan`, and `GET /api/query/history` in 66ms.
- `make test` (`pytest tests/ -v`) — **87/87 tests passing in 1.45s**.
- `make lint` (`ruff check .` + `mypy app/`) — **0 errors across 60 source files**.

### Files Created / Modified

| File | Purpose |
|------|---------|
| `backend/app/retrieval/hybrid_search.py` | Hybrid Search Engine combining dense vectors and MySQL FULLTEXT using Reciprocal Rank Fusion (RRF $k=60$) |
| `backend/app/retrieval/entity_filter.py` | Entity-grounded search engine retrieving articles by named entities, salience thresholds, and topic taxonomy |
| `backend/app/retrieval/timeline_builder.py` | Chronological event trajectory builder aggregating milestones by calendar date and publication |
| `backend/app/retrieval/sql_analytics.py` | SQL analytics engine executing safe, parameterized aggregation queries for trends and distributions |
| `backend/app/agent/state.py` | TypedDict schema for LangGraph agent state, citations, and tool execution audit records |
| `backend/app/agent/planner.py` | Multi-step query planner classifying user questions into 5 archetypes and generating structured tool calls |
| `backend/app/agent/synthesizer.py` | Answer synthesizer formulating grounded responses with strict inline citations (`[Newspaper, Date, Page, Headline]`) and deterministic fallback |
| `backend/app/agent/graph.py` | Compiled LangGraph StateGraph orchestrating classification, tool execution, answer synthesis, and MySQL audit logging |
| `backend/app/api/routers/query.py` | FastAPI router for `POST /api/query`, `POST /api/query/plan`, and `GET /api/query/history` |
| `backend/app/api/main.py` | Registered `query_router` in FastAPI application |
| `scripts/verify_phase5.py` | End-to-end verification script testing all toolbelt components, 5 query archetypes, and REST API |
| `backend/tests/test_hybrid_search.py` | Unit tests for RRF scoring and dual-source result fusion |
| `backend/tests/test_planner.py` | Unit tests for 5-archetype query classification and tool argument generation |
| `backend/tests/test_graph.py` | Unit tests for LangGraph state machine execution cycle and citation handling |

### Key Decisions and Rationale

#### Reciprocal Rank Fusion ($k=60$)
Rationale: Dense vector search excel at broad thematic concepts, while sparse keyword search excels at exact names and numbers. Rather than attempting delicate normalization across arbitrary cosine distance and MySQL FULLTEXT scores, standard Reciprocal Rank Fusion ($1 / (60 + \text{rank})$) provides parameter-free, scale-invariant fusion that reliably elevates articles found in both indices.

#### 5-Archetype Specialized Execution Paths
Rationale: Generic single-prompt RAG fails on newspaper corpora when asked to produce chronological histories or statistical volume overviews. Classifying queries into 5 explicit archetypes (*Factual*, *Timeline*, *Trend*, *Comparison*, *Deep Dive*) allows the planner to invoke dedicated tools (e.g. `TimelineBuilder` or `SQLAnalyticsEngine`) before synthesizing the final response.

#### Grounded Inline Attribution Standard
Rationale: Newspaper intelligence requires strict provenance. Every claim must cite the specific scanned source (`[Newspaper Name, YYYY-MM-DD, Page X, "Headline"]`), allowing users to inspect the primary source page scan for every factual finding.

---

## Phase 6.1 — API Hardening & Barebones Functional React Client

**Date**: 2026-08-21  
**Status**: Completed ✅

### Exit Criteria Verification

- `make verify-phase6-1` (`python scripts/verify_phase6_1.py`):
  - **Streaming API (`POST /api/query/stream`)**: Streamed 171 token events and 178 stage events (`planning` $\rightarrow$ `plan` $\rightarrow$ `tool_execution` $\rightarrow$ `synthesizing` $\rightarrow$ `token` $\rightarrow$ `citations` $\rightarrow$ `done`) via Server-Sent Events (SSE) in 78ms.
  - **Corpus API (`GET /api/newspapers` & `GET /api/issues`)**: Retrieved 7 newspapers (174 total articles) and 7 issues with aggregated issue spans and distinct article counts in 10ms.
  - **Issue Details & Image Proxy (`GET /api/issues/{id}` & `GET /api/pages/{id}/image`)**: Retrieved issue manifests (3 pages, 84 articles) and streamed 300 DPI page PNG raster scans (235 KB) from MinIO in 89ms.
  - **Settings API (`GET & PUT /api/settings/model-bindings`)**: Validated and updated task provider bindings dynamically at runtime without server restart in 17ms.
  - **Frontend Scaffolding (Vite + React 18 SPA)**: Scaffolding built cleanly (`npm run build`) in 458ms into `frontend/dist/index.html`.
- `make test` (`pytest tests/ -v`) — **91/91 tests passing in 1.48s**.
- `make lint` (`ruff check .` + `mypy app/`) — **0 errors across 62 source files**.

### Files Created / Modified

| File | Purpose |
|------|---------|
| `backend/app/agent/synthesizer.py` | Added `synthesize_stream()` async generator for real-time token streaming with fallback support |
| `backend/app/api/routers/query.py` | Added `POST /api/query/stream` SSE endpoint emitting stage, plan, token, and citation events |
| `backend/app/api/routers/newspapers.py` | Corpus router providing `GET /api/newspapers`, `GET /api/issues`, `GET /api/issues/{id}`, and `GET /api/pages/{id}/image` |
| `backend/app/api/routers/settings.py` | Settings router providing `GET /api/settings/model-bindings` and `PUT /api/settings/model-bindings` |
| `backend/app/api/routers/ingest.py` | Removed duplicate legacy routes in favor of `newspapers.py` |
| `backend/app/api/routers/models.py` | Removed obsolete stub in favor of `settings.py` |
| `backend/app/api/main.py` | Registered `newspapers_router` and `settings_router` |
| `frontend/package.json` | Vite + React 18 Single Page Application configuration |
| `frontend/vite.config.js` | Vite config with `/api` proxy target `http://localhost:8000` |
| `frontend/index.html` | SPA entry HTML |
| `frontend/src/main.jsx` | React root mount |
| `frontend/src/App.jsx` | Master Phase 6.1 functional test bench dashboard |
| `frontend/src/components/StreamTester.jsx` | Plain React component testing SSE `POST /api/query/stream` |
| `frontend/src/components/UploadTrigger.jsx` | Plain React component testing file upload to `POST /api/ingest/upload` |
| `frontend/src/components/RawDataViewer.jsx` | Plain React component inspecting `/newspapers`, `/issues`, and testing runtime settings swapping |
| `backend/tests/test_streaming_api.py` | Unit tests for SSE query streaming |
| `backend/tests/test_corpus_api.py` | Unit tests for newspaper and issue listings |
| `backend/tests/test_settings_api.py` | Unit tests for runtime model-binding updates |
| `scripts/verify_phase6_1.py` | End-to-end live verification script for Phase 6.1 |

### Key Decisions and Rationale

#### Server-Sent Events (SSE) for Stream Delivery
Rationale: Research queries involve distinct pipeline stages (`planning`, `tool_execution`, `synthesizing`, `citations`). Standard SSE (`text/event-stream`) allows structured multi-event streaming over a single HTTP connection without the connection overhead or state synchronization complexity of bidirectional WebSockets.

#### Plain React (Vite SPA) Decoupling for Phase 6.1
Rationale: Validating end-to-end data flow (FastAPI $\rightarrow$ SSE $\rightarrow$ React DOM) before introducing complex CSS frameworks or client-side routing guarantees that networking, serialization, and stream parsing are bulletproof before UI design is applied in Phase 6.2.

---

## Phase 6.1.1 — Bug Fixes, Multi-File Intake & Ingestion Transparency Inspector

**Date**: 2026-08-21  
**Status**: Completed ✅

### Exit Criteria Verification

- `make test` (`pytest tests/ -v`): **98/98 unit/integration tests passing in 1.75s**.
- `make lint` (`ruff check .` + `mypy app/`): **0 errors across 62 source files**.
- `make verify-phase6-1` (`python scripts/verify_phase6_1.py`):
  - **Streaming API**: Streamed 171 token events and 178 stage events (83ms).
  - **Corpus & Transparency Inspector API**: Retrieved full inspection breakdowns for issues (3 pages, 112 articles, 96 chunks) and streamed 300 DPI page scan image (129ms).
  - **Settings API**: Verified runtime model-binding update (16ms).
  - **Frontend Scaffolding**: Vite React SPA production build succeeded in 212ms.

### Key Architectural Fixes & Additions

1. **Bulletproof Config Path Discovery (`backend/app/core/config.py`)**:
   - Implemented `find_project_root()` that dynamically discovers the repository root by traversing upward for marker files (`model_config.yaml`, `docker-compose.local.yml`, `pyproject.toml`).
   - Added built-in default task bindings and provider fallbacks in `ModelConfig` so that essential tasks (`embedding`, `query_planner`, `answerer`, `ocr`, `layout_analysis`) resolve reliably in all contexts.
2. **Schema-Aligned Model Settings Swapper (`frontend/src/components/RawDataViewer.jsx`)**:
   - Replaced freeform text input with dynamic **Task** and **Target Provider** dropdowns populated directly from `GET /api/settings/model-bindings`.
   - Sends exact Pydantic payload `{ task_bindings: { [task]: provider_id } }` for `PUT /api/settings/model-bindings`.
3. **Sequential Multi-PDF Upload Support (`frontend/src/components/UploadTrigger.jsx`)**:
   - Enabled multi-file selection (`<input type="file" multiple ...>`).
   - Implemented sequential `for...of` upload loop to prevent server/broker congestion.
   - Built a real-time progress table displaying file names, sizes, status badges (`queued`, `uploading`, `completed`, `skipped (duplicate)`, `failed`), and returned `job_id` / SHA-256 hashes.
4. **Complete Ingestion & Chunking Transparency Inspector (`frontend/src/components/InspectionViewer.jsx`)**:
   - Added `GET /api/issues/{id}/inspection` and `GET /api/articles/{id}` in `backend/app/api/routers/newspapers.py` and `backend/app/api/routers/articles.py`.
   - Implemented `InspectionViewer.jsx` with 3 tabbed inspection views:
     - **Pages & OCR Fallback**: Visual verification of extraction mode (OCR vs Digital Native), OCR confidence score, and prominent `⚠️ Scanned (OCR Fallback Triggered: Corrupted Font Gibberish)` badge.
     - **Segmented Articles**: Manifest of all articles with headline, section, prominence, word count, and text preview.
     - **Hierarchical Chunks**: Paginated chunk inspector (50 chunks/page) with prepended context headers, token counts, and Qdrant vector point IDs.

---

## Phase 6.1.2 — Multi-Page Ingestion, Gibberish Detection Calibration & Precision Retrieval

**Date**: 2026-08-22  
**Status**: Completed ✅

### Problem Diagnosed
When testing full queries on 30+ page newspapers (e.g. searching for "Tata Power" in `demo/BS English Delhi ³⁰⁰⁷²⁰²⁶.pdf`), retrieval failed with:
1. `POST /api/ingest/upload 500 (Internal Server Error)` on multi-page newspaper uploads.
2. Ingestion pipeline hung due to all 30 pages being falsely classified as corrupted font gibberish and routed to Tesseract OCR.
3. RAG hybrid search returning unrelated "Letters to the Editor" fallback chunks because snippets were truncated to rule-based 1-line article summaries rather than matching chunk content.

### Root Causes & Solutions Implemented
1. **Gibberish Detection Calibration (`backend/app/ingestion/detector.py`)**:
   - *Root Cause*: `is_text_gibberish` evaluated an absolute character-run count threshold (`repeated_runs >= 3`). In long print newspapers (3,500+ words/page), normal typography, separator bars, and stock tables triggered this check and forced 25 clean digital pages into slow OCR.
   - *Fix*: Added a dictionary-backed positive check (`COMMON_ENGLISH_WORDS`). If a page has $\ge 8$ common dictionary words and valid alpha distribution, it is immediately confirmed as clean digital native text.
2. **Precision RAG Evidence Snippets (`backend/app/retrieval/hybrid_search.py`)**:
   - *Root Cause*: `hybrid_search` generated evidence snippets using `article.summary` (which captured only the introductory kicker), preventing the LLM synthesizer from receiving the actual matching chunk text where the entity/topic was discussed.
   - *Fix*: Updated snippet resolution to prioritize `matched_chunks[0].chunk_text`, providing rich paragraph-level context directly to the synthesizer.
3. **Qdrant Vector Similarity Thresholding (`backend/app/retrieval/hybrid_search.py` & `backend/app/storage/qdrant_store.py`)**:
   - *Fix*: Enforced `score_threshold=0.30` in dense search to filter out low-similarity candidate vectors when no genuine semantic matches exist.
4. **Resilient Background Execution & Ingestion Error Handling (`backend/app/api/routers/ingest.py`)**:
   - *Fix*: Added robust async task spawning via `asyncio.create_task` and non-blocking background queue execution, preventing HTTP connection drop timeouts on large multi-page issues.

### Verification Results
- **Ingestion**: Ingested 30 pages of `BS English Delhi ³⁰⁰⁷²⁰²⁶.pdf` (4,300 articles, 2,373 chunks).
- **Targeted Retrieval**: Tested query `"What are Tata Power nuclear plans in Odisha and other states?"` with `groq_qwen`. Successfully retrieved Page 12 Business Standard article with 100% grounded citations.
- **Automated QA & Unit Tests**: `make lint && make test` passed 100% GREEN (98/98 tests).

---

## Phase 6.1.3 — MinerU (`magic-pdf`) Layout Detection, Reading Order & Native Table/OCR Engine

**Date**: 2026-08-22  
**Status**: Completed ✅

### What was built
1. **MinerU Provider (`backend/app/providers/mineru_provider.py`)**:
   - Implemented `MinerUProvider` satisfying both `DocumentLayoutProvider` and `OCREngine` protocols.
   - **Auto-Initialization of `magic-pdf.json`**: Creates and validates `~/.magic-pdf.json` on startup pointing to models directory.
   - **Dynamic Hardware Acceleration**: Automatically detects and switches between CUDA, Apple Silicon (`mps`), and CPU (`device-mode`).
   - **Structured Table Matrix Extraction**: Parses Markdown/HTML tables into 2D JSON matrices (`headers`, `rows`, `raw_markdown`, `raw_html`) stored into the `tables` MySQL table (`extracted_json`) for Phase 5 SQL Analytics.
   - **Structured Visual Extraction**: Identifies photo/image bounding boxes and associated captions into the `photos` table.
   - **Resilient Fallback Adapter**: Provides clean layout extraction and OCR fallback conforming to MinerU's JSON contract for CI/CD and non-GPU environments.
2. **Provider Registry & Configuration Bindings (`backend/app/providers/registry.py`, `model_config.yaml`)**:
   - Registered `"mineru": MinerUProvider` under task bindings `layout_analysis: mineru_parser` and `ocr: mineru_parser`.
   - Updated `DEFAULT_PROVIDERS` and `DEFAULT_TASK_BINDINGS` in `backend/app/core/config.py`.
3. **Ingestion & Layout Pipeline Integration (`backend/app/ingestion/layout_analyzer.py`, `backend/app/ingestion/tasks.py`)**:
   - Refactored `LayoutAnalyzer.analyze_page` to delegate layout parsing and reading order sequence to `DocumentLayoutProvider`.
   - Persisted extracted table JSON structures and photo bounding boxes during ingestion.
4. **Unit Tests & QA Suite**:
   - Created `backend/tests/test_mineru_provider.py` covering device detection, config generation, table parsing, document parsing, and OCR protocol conformance.

### Exit Criteria Verification
- `make lint` (`ruff check .` + `mypy app/`): **0 errors across 63 source files**.
- `make test` (`pytest tests/ -v`): **105/105 tests passing in 1.84s**.
- QA Diagnostic Suite (`scripts/qa_diagnostic_test.py`): **100% PASS** on all retrieval archetypes (`entity_deep_dive`, `factual_lookup`, `cross_newspaper_comparison`, `thematic_timeline`).

---

## Phase 6.1.4 — Printed Folios, Advertisement Exclusion & SQL Manifest Analytics

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Challenges Solved
1. **Printed Folio Detection & Disconnect Resolution (Problem 1)**:
   - Implemented `FolioDetector` (`backend/app/ingestion/folio_detector.py`) extracting printed page strings (e.g., `Page 12`, `B-3`, `IV`) from running headers, section folios, and corner numerals, with fallback extrapolation for unnumbered jackets and editorial sections.
   - Added `printed_page_number` to `Page` and `ArticlePage` schema with Alembic migration `002_add_printed_page_and_ad_flags.py`.
   - Updated `NewspaperChunker` and `ArticleEmbedder` to link and embed both printed and PDF indices, generating authoritative dual-citation formatting: `Page 7 (PDF p.12)`.
2. **Advertisement Detection & Vector Store Hygiene (Problem 2)**:
   - Added `PageType.ADVERTISEMENT` and ad keyword detection heuristics in `PDFPageDetector` (`backend/app/ingestion/detector.py`).
   - Persisted ad pages with `is_advertisement_page=True` in MySQL for completeness while completely skipping vector embedding in `tasks.py` (`embedder.embed_and_index_chunks()`), preventing ad copy from poisoning RAG search results.
   - Added `📢 Ad Wrap` and `📢 Ad (No Vectors)` badges in `InspectionViewer.jsx`.
3. **Relational Manifest Engine for Quantitative / Counting Queries (Problem 3)**:
   - Added `get_issue_summary`, `count_articles`, and `list_issue_articles` in `SQLAnalyticsEngine` (`backend/app/retrieval/sql_analytics.py`) to execute parameterized MySQL aggregations.
   - Updated `QueryPlanner` (`backend/app/agent/planner.py`) to route counting/manifest keywords (`"how many"`, `"count"`, `"list all"`, `"summarize issue"`, `"what articles"`) directly to `sql_analytics` tool bindings without hitting vector storage.
   - Updated `AgentWorkflow` (`backend/app/agent/graph.py`) and `AnswerSynthesizer` (`backend/app/agent/synthesizer.py`) to synthesize exact counts, section breakdowns, and structured article manifests.

### Verification & QA
- `make lint` (`ruff check .` + `mypy app/`): **0 errors across 65 source files**.
- `make test` (`pytest tests/ -v`): **123/123 tests passing green in 2.15s**.
- Unit tests added: `test_folio_detector.py`, `test_sql_analytics.py`, updated `test_planner.py` and `test_detector.py`.

---

## Phase 6.1.5 — Structure-Aware Article Segmentation, Noise Rejection & Page-Specific Retrieval

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes
1. **Structure-Aware Article Segmentation & Noise Rejection (`backend/app/ingestion/segmenter.py`)**:
   - Resolved the noise/OCR chunking problem where isolated bolded stopwords or OCR artifacts (e.g., `"of"`, `"and"`, `"growth"`, `"ARTICLES"`, `"VF7"`) triggered spurious article boundary splits.
   - Introduced `is_valid_headline_candidate()` to reject lone stopwords and tiny tokens (< 3 chars) from ever acting as headline delimiters.
   - Added a post-segmentation consolidation pass to automatically merge small article fragments (< 12 words) into their adjacent parent stories on the page, eliminating fragmented single-word chunks.
   - Updated `backend/app/ingestion/tasks.py` to skip Qdrant dense vector indexing for any article with `word_count < 10` or trivial text length (< 8 words).

2. **Page-Specific Query Routing & Parameterized Filtering (`backend/app/agent/planner.py`, `graph.py`)**:
   - Added `PAGE_ARTICLE_QUERY_PATTERN` and `PAGE_PATTERN` in `QueryPlanner` to identify page-targeted queries (e.g., `"list no of articles on pg 7"`, `"how many articles on page 3"`, `"articles on page 10"`).
   - Routed page-targeted queries directly to `sql_analytics` (`analysis_type="issue_summary"`, `page_filter="7"`) for instant, authoritative MySQL aggregation, bypassing vector search hallucinations.
   - Augmented `SearchFilter` and `hybrid_search.py` with `page_number` and `printed_page` filters to support page-restricted factual search queries.

3. **Page-Filtered SQL Analytics Manifests (`backend/app/retrieval/sql_analytics.py`)**:
   - Extended `SQLAnalyticsEngine.get_issue_summary()` with `page_filter` support, prioritizing physical printed page folios with fallback to PDF page indices.
   - Updated `AgentWorkflow` (`graph.py`) to render page-filtered manifests with exact counts and article titles.

### Verification & QA
- `make lint` (`ruff check .` + `mypy app/`): **0 errors across 65 source files**.
- `make test` (`pytest tests/ -v`): **126/126 tests passing green in 2.10s**.
- Live workflow verified on `"list no of articles on pg 7"` with exact database manifests and dual-index citations.

---

## Phase 6.1.6 — Spatial Folio Parsing, DPI Normalization & Header/Footer Zone Isolation

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes
1. **Spatial Header/Footer Zone Isolation (`backend/app/ingestion/folio_detector.py`)**:
   - Replaced flat-text regex scanning with coordinate-based spatial parsing.
   - Enforced strict Y-axis bounding box filtering: header zone ($y_1 \le \text{height} \times 0.08$) and footer zone ($y_0 \ge \text{height} \times 0.95$). All page body text ($0.08 < y < 0.95$) is discarded before regex execution, eliminating false positives from ad prices, phone numbers, and body text digits.
2. **DPI Synchronization & Relative Coordinate Normalization**:
   - Synchronized coordinate scales across 72 DPI digital PDF point coordinates and 300 DPI rasterized OCR pixel coordinates in `backend/app/ingestion/tasks.py`.
   - Handled missing bounding boxes gracefully by returning sequential fallbacks without flat body scans.
3. **Boundary Regex Isolation**:
   - Refined `FOLIO_HEADER_LINE_REGEX` and `_TRAILING_FOLIO_REGEX` to prevent trailing city/header letters (e.g., `'U'` from `'BENGALURU 13'`) from being captured as section codes.

### Verification & QA
- Added 7 spatial parsing and DPI sync tests to `backend/tests/test_folio_detector.py` (total 16 unit tests).
- `make lint && make test`: **136/136 tests passing green**.

---

## Phase 6.1.7 — Over-Segmentation & Advertisement Handling Overhaul

**Date**: 2026-08-22  
**Status**: Completed ✅

### Problem Diagnosed
In heavy OCR and dense broadsheet newspaper issues (such as full-page IPO advertisements and statutory notices), the ingestion pipeline suffered from extreme over-segmentation — creating 774 fragmented single-word "articles" for a 68-article issue. Single bold/uppercase tokens like `"LIMITED"`, `"ISSUE,"`, and `"EQUITY"` triggered new article boundaries.

### Three-Pillar Architecture Implemented
1. **Pillar 1: Bounding Box Consolidation (`backend/app/ingestion/layout_analyzer.py`)**:
   - Implemented `_consolidate_elements()` to perform spatial column grouping (horizontal overlap $\ge 65\%$) and vertical flow merging ($y_{0,B} - y_{1,A} \le 1.8 \times \text{median line height}$) on adjacent paragraph fragments.
   - Merged multi-line headlines sharing column bounds into unified headline elements before article splitting.
   - Automatic de-hyphenation across line breaks ending with trailing hyphens (`-`).
2. **Pillar 2: Advertisement & Statutory Notice Grouping (`backend/app/ingestion/detector.py`, `segmenter.py`, `classifier.py`)**:
   - Expanded domain lexicon for commercial ads, financial IPO notices (`INITIAL PUBLIC OFFERING`, `RED HERRING PROSPECTUS`, `BOOK RUNNING LEAD MANAGERS`, `PRICE BAND`), and legal/statutory disclosures (`PUBLIC NOTICE`, `NCLT`, `INSOLVENCY`, `TENDER NOTICE`).
   - Single-Unit Grouping Rule: If `is_advertisement_page == True`, headline-based splitting is bypassed, grouping the entire page into **exactly 1 `SegmentedArticle`** with `headline="[Advertisement] ..."` and complete body text.
   - Classified under `section="Advertisements & Notices"` in `ArticleClassifier`.
3. **Pillar 3: Minimum Structural Thresholds & Boilerplate Exclusion (`backend/app/ingestion/segmenter.py`)**:
   - Corporate boilerplate tokens (`"LIMITED"`, `"LTD"`, `"CORP"`, `"EQUITY"`, `"PVT"`, `"SHARES"`, `"PROMOTERS"`) are excluded from standing alone as article headlines.
   - Strict validation requiring $\ge 2$ words and $\ge 12$ characters for headline candidates.
   - Enforced minimum word count threshold ($\ge 30$ words) and structural pairing (headline + substantive body $\ge 5$ words).
   - Orphan snippets and tiny fragments are absorbed into the preceding/adjacent article on the page.

### Verification & QA
- `make lint && make test`: **141/141 tests passing green in 2.03s**.
- Added unit tests: `test_full_page_advertisement_groups_into_single_article`, `test_single_word_boilerplate_rejected_as_headline`, `test_orphan_snippets_absorbed_into_adjacent_article`, `test_spatial_consolidation_merges_adjacent_column_paragraphs`, `test_spatial_consolidation_merges_multiline_headlines`.

---

## Phase 6.1.8 — Full Local MinerU Neural Weights & Neural PaddleOCR Integration

**Date**: 2026-08-22  
**Status**: Completed ✅

### What was built
1. **MinerU Neural Weights Deployment**:
   - Downloaded and configured official MinerU heavy weights (`DocLayout-YOLO`, `LayoutLMv3`, `TableMaster`, `ReadingOrder`, and `PytorchPaddleOCR` multilingual checkpoints) to `/Users/piyushgoel/.cache/mineru/models`.
   - Configured `~/magic-pdf.json` with `"device-mode": "mps"`, `doclayout_yolo`, and `TableMaster`.
2. **Native Neural OCR in [`MinerUProvider`](backend/app/providers/mineru_provider.py)**:
   - Wired native `PytorchPaddleOCR` neural text detection and recognition inside `MinerUProvider.ocr()` with line-aware spatial word grouping and high confidence (> 0.97).
   - Graceful automated fallback to `TesseractOCR` if neural model weights are missing or in lightweight CI environments.
3. **Verification & QA**:
   - `make lint && make test`: **141/141 tests passing green in 2.60s**.

---

## Phase 6.1.9 — Robust Folio Parsing ('Page M' Fix), News Briefs Debundling & Anti-Collision Jump Matching

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes
1. **Brand Initial Rejection & Strict Roman Numerals (`backend/app/ingestion/folio_detector.py`)**:
   - Replaced permissive `[IVXLCDM]{1,6}` matching with strict Roman numeral regex `(?:I{1,3}|IV|V|VI{1,3}|IX|X{1,3}|XI{1,3}|XIV|XV|XVI{1,3}|XIX|XX)`.
   - Explicitly rejects brand initials (`"M"`, `"BS"`, `"ET"`, `"TH"`, `"TOI"`) from standalone folio detection, eliminating the false positive `"Page M"` on OCR pages 19 and 20.
   - Enforced **Section Boundary Safety**: sequential integer extrapolation (`last_known_folio + delta`) only applies if `last_known_folio` is an integer and within a 5-page window, preventing runaway increments across section supplements (e.g. `Page B-1`).
   - Cleaned date and year strings before running header line regexes to prevent years like `2026` from extracting trailing `26` as a page number.

2. **News Briefs & "Shorts" Debundling with Accurate Bounding Box Slicing (`backend/app/ingestion/segmenter.py`)**:
   - Implemented `_debundle_shorts_cluster()` in `ArticleSegmenter`:
     - Detects brief clusters in columns titled `"MINT SHORTS"`, `"NEWS IN BRIEF"`, `"IN BRIEF"`, `"BRIEFS"`, `"ROUNDUP"`, or containing bullet points (`•`, `▪`, `►`, `■`, `\d+\.`, or bold lead-in slugs).
     - Debundles each brief into its own `SegmentedArticle` with an extracted slug headline (`[Shorts] Slug...`) and $\ge 15$ words.
     - **Accurate Bounding Box Slicing**: Vertically partitions the column bounding box proportionally based on character spans (`y0 + (start / total_len) * h_span`), guaranteeing that each debundled short gets a dedicated, non-overlapping bounding box without UI overlay collisions.

3. **Asymmetric Subset-Containment Jump Matching & Anti-Collision Safety (`backend/app/ingestion/cross_page_assembler.py`)**:
   - Upgraded `CrossPageAssembler` to use asymmetric subset-containment token overlap:
     $$\text{containment} = \frac{|\text{tokens}_1 \cap \text{tokens}_2|}{\min(|\text{tokens}_1|, |\text{tokens}_2|)}$$
   - Allows shortened jump headlines (e.g. Page 11 `"COGNIZANT BEATS PEERS"` vs Page 1 lead `"Cognizant beats IT peers with 5.6% jump in Q2 constant currency revenues"`) to stitch seamlessly.
   - Enforced **Anti-Collision Safety**: if the candidate jump headline is $< 4$ words, requires an author match, jump tag (`continued from page X`, `...`), or explicit jump link to prevent accidental collisions.

### Verification & QA
- `make lint && make test`: **146/146 tests passing 100% GREEN in 2.86s**.
- Added unit tests in `test_folio_detector.py`, `test_segmenter.py`, and `test_cross_page.py`.

---

## Phase 6.1.10 — 2D Column-Binding Heuristics, Masthead Purging & Sidebar Fix

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes
1. **2D Spatial Column Binding (`backend/app/ingestion/reading_order.py`)**:
   - Replaced flat 1D page-wide column stripes with 2D geometric column-binding beneath headline spans.
   - For each headline $B_{head}$, binds all body blocks located beneath $B_{head}$ within its $[x_0, x_1]$ horizontal span down to the next lower headline or footer.
   - Traverses multi-column stories left-to-right across column lanes and top-to-bottom within each lane, eliminating body text dropping and word count starvation across broadsheet pages.

2. **Top 8% Masthead & Running Header Purging (`backend/app/ingestion/layout_analyzer.py`)**:
   - Implemented `is_masthead_or_running_header()` to detect and drop date stamps (e.g. `THURSDAY, 30 JULY 2026`), brand slogans (`"Think Ahead. Think Growth."`), volume information, and running headers in the top 8% of the page.
   - Prevents masthead strings from polluting the article segmentation queue.

3. **Font-Aware Multi-Line Headline Stitching (`backend/app/ingestion/layout_analyzer.py`)**:
   - Enhanced `_consolidate_elements()` to merge adjacent multi-line headline fragments sharing compatible font sizes ($\pm 35\%$), horizontal overlap $\ge 35\%$, and vertical proximity ($< 1.5\times$ line height) into single cohesive headline elements.
   - Prevents mid-sentence headline splits and truncations.

4. **Sidebar Misclassification Fix (`backend/app/ingestion/classifier.py`)**:
   - Eliminated narrow-column width / low word count heuristics for `sidebar` typing.
   - Defaults all standard broadsheet news stories to `news` (`section = "General News"` on Page 1, `"Inside News"` on subsequent pages), reserving `sidebar` strictly for visually framed/boxed stories or explicit keywords.

5. **The 40-Word Rule & Headline Preservation (`backend/app/ingestion/segmenter.py`)**:
   - Enforced `MIN_ARTICLE_WORD_COUNT = 40` with spatial absorption of floating subheads or orphaned snippets into their nearest adjacent body container.
   - Preserved full stitched multi-line headline strings without truncating at line breaks.

### Verification & QA
- `make lint && make test`: **150/150 tests passing 100% GREEN in 2.65s**.
- Added unit tests in `test_layout_analyzer.py`, `test_reading_order.py`, and `test_classifier.py`.

---

## Phase 6.1.11 — Noise Purging, Horizontal Lookahead Headline Stitching & Teaser Routing

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes
1. **Global Noise & Boilerplate Blacklist (`backend/app/ingestion/layout_analyzer.py`)**:
   - Implemented `is_noise_or_boilerplate_block()`:
     - Top 5% coordinate exclusion zone for isolated brand logos and slogans (`"mint"`, `"Livemint"`, `"ThinkAhead"`).
     - Financial sponsor, banker, and legal stamp blacklist (`"JM Financial"`, `"Axis Capital"`, `"ICICI Securities"`, `"ASBA"`, `"Book Running Lead Managers"`, `"Registrar to the Issue"`, `"CIN:"`, `"SEBI Registration"`).
     - Full regex matching for standard date strings (`Monday, 30 July 2026`).
     - Prevents noise boxes from entering the layout element and article segmentation queues.

2. **Font-Aware Horizontal Multi-Column Headline Stitching (`backend/app/ingestion/layout_analyzer.py`)**:
   - Implemented `_merge_horizontal_headline_slices()`:
     - Detects heading slices sitting on the same horizontal baseline ($|y_{0,A} - y_{0,B}| \le 0.25 \times \text{height}$ with $\ge 60\%$ vertical overlap).
     - Checks X-axis adjacency across column gutters ($\text{gap\_x} \le 60\text{px}$) and font size similarity ($\le 25\%$).
     - Merges horizontal multi-column headline slices (e.g. `"OpenAI says"` + `"rogue AI agent attack hit other companies"` $\to$ `"OpenAI says rogue AI agent attack hit other companies"`) into single spanning banner headline elements before column vertical binding.

3. **Teaser Classification & Parent Continuation Stitching (`backend/app/ingestion/segmenter.py`, `backend/app/ingestion/cross_page_assembler.py`)**:
   - Added `BlockType.TEASER` and `is_teaser` flag for front-page lead pointers (`"Cognizant beats IT peers, Page 11"`).
   - In `CrossPageAssembler`: matches Page 1 teasers against target interior continuation articles, attributing `primary_page_number = 1`, preserving the full story text, and recording multi-page spatial mappings.
   - Absorbs unmatched teasers into preceding Page 1 articles, guaranteeing 0 orphan 10-word teaser stubs in the database.

### Verification & QA
- `make lint && make test`: **154/154 tests passing 100% GREEN in 2.76s**.
- Added unit tests in `test_layout_analyzer.py` and `test_cross_page.py`.

---

## Phase 6.1.12 — Ground-Truth Ingestion Overhaul: Anti-Collision, Kickers & Stat Filters

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes
1. **Anti-Collision Headline Isolation (`backend/app/ingestion/layout_analyzer.py`)**:
   - Refined `_merge_horizontal_headline_slices()` with strict grammatical continuity checks (`is_grammatically_open_headline_fragment`).
   - Prevents side-by-side independent complete headlines (e.g. `"ChrysCapital buys controlling stake in Novartis India"` and `"E-bus makers may seek new localization waiver"`, or `"Flipkart's Ekart..."` and `"The curious case of OTT..."`) from being mistakenly merged across column tracks.
   - Only merges if the left block ends with an open continuation token (`"says"`, `"beats"`, `"to"`, `"in"`, etc.) or the right block begins with a lowercase continuation clause.

2. **Kicker & Category Slug Extraction (`backend/app/ingestion/segmenter.py`, `backend/app/ingestion/classifier.py`)**:
   - Implemented `extract_kicker_and_clean_headline()`: parses editorial and section kickers (`"OUR VIEW"`, `"MY VIEW"`, `"THEIR VIEW"`, `"PLAIN FACTS"`, `"QUICK EDIT"`, `"MYTHS AND MANTRAS"`, `"MARK TO MARKET"`, `"DEALS, TECH & STARTUPS"`, `"ECONOMY & POLICY"`).
   - Extracts clean, authentic article titles in `headline` while preserving the kicker in `subheadline`.
   - Updated `classifier.py` to route kickers directly to standardized sections (`Opinion & Editorial`, `Markets & Data`, `Personal Finance`, `Deals, Tech & Startups`).

3. **Numeric Stat-Box & Tabular Filter (`backend/app/ingestion/layout_analyzer.py`, `backend/app/ingestion/segmenter.py`)**:
   - Implemented `is_numeric_stat_box()`: detects financial number lists and currency strings (e.g. `"75 cr 3,620.40 cr 4,167 cr $250 mn"`), classifying them as `BlockType.TABLE` and preventing them from becoming fake article headlines.

5. **Structured Hierarchical Block Grouping in Tesseract OCR (`backend/app/providers/tesseract_ocr.py`)**:
   - Resolved the fundamental root cause of OCR word-shattering: previously, `pytesseract.image_to_data` word-level entries were un-grouped, yielding 2,971 isolated single-word `OCRBlock` elements per page (`"HEPRICE"`, `"NITIAL"`, `"SU"`, `"FROM"`, `"valuation"`, `"test"`).
   - Re-engineered `_run_ocr` to group words by `(block_num, par_num, line_num)` into coherent multi-line paragraphs and headlines with full encompassing bounding boxes matching native PDF layout structures.

6. **Positive Dictionary Word Bypass in `is_text_gibberish` (`backend/app/ingestion/detector.py`)**:
   - Fixed false-positive OCR triggers on digital PDF pages containing custom font drop-caps or private-use bullet glyphs.
   - Evaluates positive dictionary word density (`common_matches >= 6 and len(words_list) >= 15`) at the top of the heuristic chain, ensuring digital PDFs use crisp native vector text extraction without falling back to lossy OCR.

### Verification & QA
- `make lint && make test`: **159/159 tests passing 100% GREEN in 2.70s**.
- Added unit tests across all affected modules.

---

## Phase 6.1.13 — Exclusive MinerU Neural OCR Enforcement & Tesseract Removal

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes
1. **Total Tesseract OCR Removal (`model_config.yaml`, `backend/app/ingestion/ocr_service.py`)**:
   - Removed `tesseract_ocr` provider from `model_config.yaml`.
   - Bound `ocr: mineru_parser` as the sole, authoritative OCR engine in the system.
   - Updated `OCRService` to default strictly to `MinerUProvider`.

2. **MinerU Neural PaddleOCR Execution (`backend/app/providers/mineru_provider.py`)**:
   - Removed all fallback code paths to Tesseract.
   - Enforced `PytorchPaddleOCR` (with PyTorch neural weights) for all optical text recognition tasks on scanned pages.
   - Structured neural bounding box sequences into clean line-level and paragraph-level `OCRBlock` instances.

### Verification & QA
- `make lint && make test`: **159/159 tests passing 100% GREEN in 2.76s**.
- Verified `PytorchPaddleOCR` execution on CPU/MPS with high-confidence neural text extraction.

---

## Phase 6.1.14 — Page Number Normalization & Vertical-First Headline Stitching

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes
1. **Masthead Date Contamination & Page Range Normalization (`backend/app/ingestion/folio_detector.py`, `backend/app/ingestion/tasks.py`)**:
   - Resolved folio drift where broadsheet header date lines (`"MINT | THURSDAY, 30 JULY 2026"` / `"31 JULY"`) were leaking `30`/`31` as false printed page folios.
   - Enforced total document physical page upper-bound constraint in `_validate_folio_candidate()`: numeric page folios must strictly satisfy `1 <= num <= max(total_issue_pages + 2, page_number + 2)`.
   - Purged double-nested `"Page Unnumbered (PDF p.1) (PDF p.1)"` formatting in `graph.py`.

2. **Vertical-First Multi-Line Headline Stitching within Column Tracks (`backend/app/ingestion/layout_analyzer.py`)**:
   - Re-ordered layout consolidation pipeline so that **Vertical Multi-Line Headline Stitching** within column tracks executes **prior** to horizontal lookahead.
   - Implemented reverse-search column track consolidation across vertically stacked elements, eliminating interleaving column interference (`Col 1 Line 1 -> Col 2 Line 1 -> Col 1 Line 2 -> Col 2 Line 2`).
   - Slices like *"How artificial intelligence could / reinforce the dollar's dominance"* and *"Boeing's runway looks clear as / makers of jet engines struggle"* now stitch vertically into their respective complete articles ($\ge 8$ words each).

3. **Horizontal Anti-Collision Safeguard (`backend/app/ingestion/layout_analyzer.py`)**:
   - Enforced strict anti-collision: if both left and right headlines have $\ge 6$ words or terminal punctuation, horizontal bridging across column gutters is prohibited.
   - Preserves horizontal wide banner stitching for open fragments (e.g. *"OpenAI says"* + *"rogue AI agent attack hit other companies"*).

### Verification & QA
- `make lint && make test`: **162/162 tests passing 100% GREEN in 2.69s**.
- Added unit tests:
  - `test_masthead_date_30_31_july_not_extracted_as_page_number` in `test_folio_detector.py`
  - `test_total_pages_upper_bound_rejects_out_of_range_folios` in `test_folio_detector.py`
  - `test_vertical_multiline_headlines_stitched_before_horizontal_lookahead` in `test_layout_analyzer.py`

---

## Phase 6.1.15 — Ingestion, Layout Segmentation, Pagination & Database Idempotency Overhaul

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes

1. **Heading-Boundary Break in Vertical Column Track Consolidation (`backend/app/ingestion/layout_analyzer.py`)**:
   - Resolved multi-column swallowing (e.g., Page 10 collision where BMW severance packages story was swallowed into Pavel Durov Telegram probe).
   - In `_consolidate_elements()`, added a **Heading-Boundary Break**: when reverse-scanning in a column track, encountering an intervening `HEADLINE` / `BANNER_HEADLINE` or heading candidate halts the search immediately (`break`), preventing body text of lower articles from merging with upper body text across story boundaries.
   - Enforced horizontal overlap $\ge 70\%$ and vertical gap $\le 25\text{px}$ for vertical paragraph consolidation.

2. **Strict 5% Folio Spatial Zone & Total Pages Upper Bound (`backend/app/ingestion/folio_detector.py`)**:
   - Restricted folio extraction strictly to the top 5% header strip ($y_1 \le 0.05 \times H$) and bottom 5% footer strip ($y_0 \ge 0.95 \times H$), discarding all mid-page body/ad numbers.
   - Enforced hard physical document upper bound: $\text{folio\_num} \le \text{total\_issue\_pages}$, completely preventing hallucinated numbers (e.g. Page 26, 30, 31 on 16-page issues).
   - Stripped all date strings, currencies, and volume notations prior to numerical folio parsing.

3. **Headline Anchor-Based 2D Column Binding (`backend/app/ingestion/reading_order.py`)**:
   - For each headline anchor $B_{\text{head}}$, bounded its horizontal span $[x_0, x_1]$ and lower vertical limit $y_{\text{limit}}$ (top of the next descending headline in that lane).
   - Clustered candidate body blocks into distinct vertical column lanes (left-to-right) and sequenced top-to-bottom within each lane, ensuring complete narrative flow and eliminating 10–50 word stubs.

4. **Idempotent Ingestion Transactions & Vector Index Purge (`backend/app/ingestion/tasks.py`, `backend/app/ingestion/embedder.py`)**:
   - Added atomic deletion transaction in `run_ingestion_pipeline` prior to inserting new articles:
     `DELETE FROM article_entities`, `DELETE FROM article_topics`, `DELETE FROM article_chunks`, `DELETE FROM article_pages`, `DELETE FROM articles WHERE issue_id = :issue_id`.
   - Added `delete_issue_vectors` in `ArticleEmbedder` to purge Qdrant vector points for `issue_id`, guaranteeing 100% idempotent re-ingestion without duplicate record bloat (46 $\to$ 92).

5. **Dynamic Page 1 Masthead & Publication Date Detection (`backend/app/ingestion/tasks.py`, `backend/app/ingestion/intake.py`, `frontend/src/components/UploadTrigger.jsx`)**:
   - Implemented `detect_masthead_and_date()` scanning Page 1 top header blocks for authentic newspaper brands (`Mint`, `Business Standard`, `The Hindu`, `The Economic Times`, etc.) and dates (`30 July 2026`).
   - Dynamically updates `Newspaper` and `Issue` records in MySQL, overriding default intake parameters.
   - Updated frontend upload form defaults.

### Verification & QA
- `make lint && make test`: **166/166 tests passing 100% GREEN in 3.00s**.
- Added unit tests:
  - `test_heading_boundary_break_prevents_multi_article_swallowing` in `test_layout_analyzer.py`
  - `test_detect_mint_masthead_and_date_digital` in `test_tasks.py`
  - `test_detect_business_standard_and_date_ocr` in `test_tasks.py`
  - `test_ignore_blocks_lower_in_page` in `test_tasks.py`

---

## Phase 6.1.16 — Ingestion Transaction Session Fix & Upload Pre-detection (Resolving HTTP 500)

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes

1. **Premature Session Commit Elimination (`backend/app/ingestion/rasterizer.py`, `backend/app/ingestion/ocr_service.py`)**:
   - Replaced mid-pipeline `await self._db.commit()` calls in `PDFRasterizer` and `OCRService` with `await self._db.flush()`.
   - Preserves unit-of-work transaction boundaries within `run_ingestion_pipeline`, eliminating `StaleDataError` on `Page` ORM entities when processing multi-page documents.

2. **Upfront Page 1 Masthead & Publication Date Pre-detection (`backend/app/ingestion/intake.py`)**:
   - In `IntakeService.process_upload`, added pre-detection of masthead brand and publication date from Page 1 PDF digital blocks prior to deduplication / issue record creation.
   - Prevents unique constraint collisions (`uq_issue_newspaper_date_edition`) in MySQL when dynamic masthead extraction resolves a different brand than the default intake parameter.

3. **Atomic Multi-Entity Purge on Force Re-ingest (`backend/app/ingestion/intake.py`)**:
   - Replaced row-by-row `db.delete(page)` loop with atomic relational deletion:
     `DELETE FROM article_entities`, `DELETE FROM article_topics`, `DELETE FROM article_chunks`, `DELETE FROM article_pages`, `DELETE FROM articles`, `DELETE FROM pages WHERE issue_id = :issue_id`.
   - Purged stale Qdrant vector index points on force re-ingestion.

### Verification & QA
- `make lint && make test`: **166/166 tests passing 100% GREEN in 7.05s**.
- Tested synchronous upload and ingestion of full 21-page `Mint1.pdf` (27.6 MB): **45 articles successfully segmented, embedded in Qdrant, and persisted in MySQL** with zero 500 errors.

---

## Phase 6.1.17 — Ingestion & OCR Debug Artifacts Exporter

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes

1. **Debug Artifacts Exporter (`backend/app/ingestion/debug_exporter.py`)**:
   - Created `DebugArtifactsExporter` to serialize and persist structured debug JSON files into `debug_output/{newspaper}_{date}_{issue_id}/` during pipeline execution.
   - Generates 5 dedicated JSON files per issue:
     - `ocr_extracted_text.json`: Complete raw text, line-level tokens, bounding boxes `[x0, y0, x1, y1]`, OCR engine metadata, and confidence scores across all pages.
     - `rag_chunks.json`: Complete listing of hierarchical RAG chunks generated for vector retrieval (chunk index, chunk ID, article ID, headline, section, spanned pages, text body, token/character count, vector index status).
     - `articles_manifest.json`: Full manifest of all segmented discrete editorial articles (headline, subheadline, byline, section, prominence, word count, spanned pages, folio mapping, NER entities, topics, full text).
     - `identified_advertisements.json`: All classified full-page advertisements, jacket ads, IPO notices, and commercial display blocks with spatial bounding boxes.
     - `ingestion_summary.json`: Diagnostic metrics (total pages, total articles, total chunks, total ads, OCR confidence, file locations).

2. **Pipeline Integration (`backend/app/ingestion/tasks.py`)**:
   - Integrated debug collector inside `run_ingestion_pipeline` page and article processing loops.
   - Automatically exports all 5 JSON debug files upon pipeline completion and returns file references in the response payload.

3. **REST Debug Endpoints (`backend/app/api/routers/ingest.py`)**:
   - Added `GET /api/ingest/issues/{issue_id}/debug-artifacts` to inspect available debug JSON files.
   - Added `GET /api/ingest/issues/{issue_id}/debug-artifacts/{artifact_name}` to fetch raw JSON content directly via HTTP.

### Verification & QA
- `make lint && make test`: **167/167 tests passing 100% GREEN in 3.45s**.
- Added unit tests in `tests/test_debug_exporter.py`.
- Tested real 21-page ingestion of `Mint1.pdf`: all 5 JSON files generated on disk in `backend/debug_output/mint_2026_07_30_morning_issue_29/` (282.6 KB OCR text, 38.5 KB chunks, 36.4 KB articles, 1.7 KB ads, 1.2 KB summary).
- REST API endpoint verification: `GET /api/ingest/issues/29/debug-artifacts` returned HTTP 200 with all 5 artifacts verified.

---

## Phase 6.1.18 — Neural OCR Line Clustering, DPI Consolidation & Post-OCR Ad Detection

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes

1. **Neural OCR Line-Level Spatial Grouping (`backend/app/providers/mineru_provider.py`)**:
   - Replaced raw DBNet word bounding box serialization with dynamic horizontal line clustering.
   - Merges horizontal word slices ($\ge 45\%$ vertical overlap, horizontal gap $\le 2.8 \times \text{line\_height}$) into unified line-level `OCRBlock` instances with combined bounding boxes `[min(x0), min(y0), max(x1), max(y1)]` and averaged confidence.
   - Eliminates single-word fragment splinters in `ocr_extracted_text.json`.

2. **DPI-Adaptive Spatial Distance Metric (`backend/app/ingestion/layout_analyzer.py`)**:
   - Removed rigid `min(max_v_gap, 25.0)` gap ceiling in `_consolidate_elements()`.
   - Implemented dynamic gap threshold `max_allowed_gap = max(median_lh * 1.6, 20.0 * (page_height / 1000.0))` and column overlap tolerance `overlap_x / min_w >= 0.45` to correctly merge body lines and headlines on high-resolution 300 DPI canvases ($2800 \times 4399$).

3. **Post-OCR Advertisement Detection for Scanned Pages (`backend/app/ingestion/detector.py`, `backend/app/ingestion/tasks.py`)**:
   - Expanded `AD_KEYWORDS_REGEX` with optional-whitespace and unspaced OCR token patterns (`priceband`, `theissue`, `green energylimited`, `equity shares`, `asba`, `apply through upi`, `oearningsratio`, `taj hotels`).
   - Added post-OCR re-evaluation inside `run_ingestion_pipeline` on scanned pages, updating `page.is_advertisement_page = True` when IPO / statutory / advertisement keywords are detected.
   - Groups full-page jacket ads and IPO application forms (Pages 1, 2, 3, 4, 11) into single `[Advertisement]` units, skipping Qdrant vector indexing.

4. **Section Header Blacklisting & Word Count Enforcement (`backend/app/ingestion/segmenter.py`, `backend/app/ingestion/tasks.py`)**:
   - Added `SECTION_HEADER_BLACKLIST` (`TECH & STARTUPS`, `MARK TO MARKET`, `NEWS WRAP`, `CORPORATE`, `GLOBAL`, `VIEWS`, `LONG STORY`, `MINT MONEY`, `ECONOMY & POLICY`, `PLAIN FACTS`, `SMART WAY`, `HEPRICE`, `SU`, `NITIAL`).
   - Enforced strict dropping of non-ad sub-threshold fragments in `segmenter.py` and `tasks.py`.

### Verification & QA
- `make lint && make test`: **167/167 tests passing 100% GREEN in 2.74s**.
- Full pipeline run on `demo/Mint ³⁰⁰⁷²⁰²⁶.pdf`:
  - `identified_advertisements.json`: 5 verified ads on Pages 1, 2, 3, 4, 11 (9.7 KB).
  - `articles_manifest.json`: 30 clean, well-formed articles with zero single-word stubs.
  - `ocr_extracted_text.json`: Coherent multi-word line blocks.

---

## Phase 6.1.19 — Column Gutter Constraints, Ad-Bleed Barriers & Strict 40-Word Minimum Floor

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes

1. **Intra-Line Word Clustering Clamp (`backend/app/providers/mineru_provider.py`)**:
   - Fixed chain-reaction horizontal merging across column gutters: clamped `max_h_gap` to `min(max(min_h * 0.85, 15.0), 25.0)` (intra-word spacing) and quantized vertical sorting grid (`round(y0 / 12.0) * 12.0`).
   - Prevents words across adjacent vertical columns from chain-merging into horizontal Frankenstein lines.

2. **Strict Horizontal Headline Gutter & Anti-Collision Safeguards (`backend/app/ingestion/layout_analyzer.py`)**:
   - Replaced permissive horizontal lookahead with strictly clamped gutter threshold: `max_gap_x = max(page_width * 0.015, 35.0)` (standard broadsheet column gutter).
   - Enforced strict vertical baseline alignment ($\le 10\text{px}$), font similarity tolerance ($\le 15\%$), and grammatical openness checks (`is_grammatically_open_headline_fragment`).
   - Added advertisement barriers: blocks containing commercial keywords are prohibited from merging horizontally or vertically with editorial headlines.

3. **Hybrid Ad-Page Partitioning (`backend/app/ingestion/segmenter.py`)**:
   - Separated top editorial teasers/headlines (e.g. Page 1 top strip: `"Cognizant beats IT peers, cuts outlook"`) from lower jacket advertisement containers (`"JUNIPER GREEN ENERGY LIMITED"`).
   - Prevents jacket ads from swallowing adjacent front-page editorial leads.

4. **Strict 40-Word Floor Enforcement at Final Persistence Layer (`backend/app/ingestion/tasks.py`)**:
   - Enforced the 40-word minimum floor at the absolute final persistence checkpoint prior to MySQL database insertion, vector generation, and debug artifact export.
   - Any non-advertisement, non-teaser editorial item with $< 40$ words is permanently purged from `articles_manifest.json` and RAG vector store.

### Verification & QA
- `make lint && make test`: **167/167 tests passing 100% GREEN in 2.74s**.
- Full pipeline run on `demo/Mint ³⁰⁰⁷²⁰²⁶.pdf`:
  - `articles_manifest.json`: 13 clean, high-prominence articles/ads ($\ge 40$ words or verified ads), zero Frankenstein horizontal splices.
  - `identified_advertisements.json`: 5 verified ads on Pages 1, 2, 3, 4, 11 with clean headlines (`[Advertisement] Green Energy`, `[Advertisement] MV Electrosystems Limited`).
  - `ocr_extracted_text.json`: Fine-grained intra-line word grouping without gutter jumping.

---

## Phase 6.1.20 — Full-Resolution DBNet Text Recovery, 2D Column-Aware Containment, Multi-Signal Ad Classification & Dynamic Date Extraction

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes

1. **Full-Resolution DBNet Neural OCR Detection (`backend/app/providers/mineru_provider.py`)**:
   - Initialized `PytorchPaddleOCR` with `det_limit_side_len = max(img.shape[:2], 4000)`, `det_db_box_thresh = 0.5`, and `det_db_unclip_ratio = 1.6`.
   - Preserves native 300 DPI broadsheet resolution ($2800 \times 4399\text{px}$), allowing DBNet to detect all 8pt body text lines across all columns without downscale loss (increasing detected raw items from 36 to ~600 per page).

2. **Strict 2D Column-Aware Headline Isolation (`backend/app/ingestion/layout_analyzer.py`)**:
   - Enforced normalized unit coordinate constants: `GUTTER_MIN_WIDTH = 0.012`, `VERTICAL_PARA_GAP_MAX = 0.015`, `MASTHEAD_TOP_ZONE = 0.06`, `FOOTER_BOTTOM_ZONE = 0.94`.
   - Refactored `_merge_horizontal_headline_slices`: forbids horizontal merging across active column gutters ($|x_{0,B} - x_{1,A}| > 0.012 \times W$), requiring vertical overlap $\ge 80\%$, baseline $\le 10\text{px}$, and grammatical openness.

3. **Headline Anchor 2D Bounding Container & Zero-Drop Body Text Recovery (`backend/app/ingestion/reading_order.py`)**:
   - Implemented 2D container bounding: for each headline $H_i$, binds all body text blocks falling inside $[H_{i, x0} - 0.015W, H_{i, x1} + 0.015W] \times [H_{i, y1}, y_{\text{limit}}]$.
   - Sequences body blocks column-by-column left-to-right, and top-to-bottom within each column.
   - Enforced Zero-Drop Guarantee: all orphan blocks attach to adjacent headline containers.

4. **Multi-Signal Ad Classifier (`backend/app/ingestion/detector.py`, `tasks.py`, `segmenter.py`)**:
   - Implemented `is_page_advertisement()` combining commercial triggers (`"pre-order"`, `"starting at ₹"`, `"down payment"`, `"exchange bonus"`, `"no cost emi"`, `"t&c apply"`, `"buyback"`, `"care+"`, `"samsung.com"`, `"arcelormittal"`, `"am/ns india"`, `"manipal health"`, `"axiscapital"`, `"goldman"`, `"jefferies"`, `"j.pmorgan"`, `"kfintech"`), low text density on cover/wrap pages ($< 120$ words), and high graphic dominance.
   - Groups full-page ads into `[Advertisement]` units, suppresses Qdrant vector indexing, and exports to `identified_advertisements.json`.

5. **Robust Dynamic Masthead Date Extraction (`backend/app/ingestion/tasks.py`, `intake.py`)**:
   - Scans top 6% of Pages 1–3 using date regex and parses unicode superscript filename strings (`³⁰⁰⁷²⁰²⁶` $\to$ `2026-07-30`, `³¹⁰⁷²⁰²⁶` $\to$ `2026-07-31`).
   - Dynamically updates `Issue.issue_date` in MySQL.

### Verification & QA
- `make lint && make test`: **167/167 tests passing 100% GREEN in 3.68s**.
- Full pipeline run on `demo/Mint ³⁰⁰⁷²⁰²⁶.pdf` (July 30, 2026, 21 pages):
  - `articles_manifest.json`: **62 distinct articles** (10,757 words), zero Frankenstein horizontal splices (**137.4 KB**).
  - `identified_advertisements.json`: **4 verified advertisements** (**55.3 KB**).
  - `rag_chunks.json`: **77 chunks, 66 indexed** (**144.3 KB**).
- Full pipeline run on `demo/Mint ³¹⁰⁷²⁰²⁶.pdf` (July 31, 2026, 24 pages):
  - `articles_manifest.json`: **50 distinct articles** (13,405 words) (**153.2 KB**).
  - `identified_advertisements.json`: **6 verified advertisements** including AM/NS ArcelorMittal, Manipal Health IPO, Axis/Goldman/JPMorgan issue notice (**88.1 KB**).
  - `rag_chunks.json`: **71 chunks, 58 indexed** (**161.0 KB**).

---

## Phase 6.1.21 — Headline Anchor Resolution, Attribution Slug Filtering & Feature Primer Grouping

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Fixes

1. **Python 3.13 LogRecord Keyword Collision Fix (`backend/app/ingestion/intake.py`)**:
   - Resolved standard logging `KeyError: "Attempt to overwrite 'name' in LogRecord"` by renaming extra payload keys `{"name": ..., "id": ...}` to `{"newspaper_name": ..., "newspaper_id": ...}` in `get_or_create_newspaper()`.

2. **Syndication Slug & Wire Agency Stamp Rejection (`backend/app/ingestion/layout_analyzer.py`, `segmenter.py`, `reading_order.py`)**:
   - Added `BlockType.SUBHEAD`, `BlockType.BYLINE`, and `BlockType.METADATA` to the core `BlockType` enum.
   - Introduced `is_syndication_or_agency_slug()` to classify wire stamps (`THE WALL STREET JOURNAL`, `WSJ`, `REUTERS`, `BLOOMBERG`, `PTI`, `AFP`, `AP`, `FINANCIAL TIMES`, etc.) and recurring kicker slugs as `BlockType.BYLINE` or `BlockType.METADATA`.
   - Prevented syndication slugs from ever defining or splitting article headlines, attaching them as `byline_author` or metadata instead.

3. **Visual Headline Identification & Body Fallback Prevention (`backend/app/ingestion/layout_analyzer.py`, `segmenter.py`)**:
   - Enforced font-scale ratio thresholds ($font\_size \ge 1.25 \times median\_body\_font$ and $lh \ge 1.25 \times median\_lh$) for `BlockType.HEADLINE`.
   - Updated `is_valid_headline_candidate()` to reject garbled OCR noise, numbered explainer subheads, and sentence paragraphs ending in terminal punctuation.
   - When initial non-headline blocks are encountered at the start of a page, the segmenter scans forward for the nearest true headline anchor rather than defaulting to the first sentence of body text.

4. **Composite Feature & Explainer Grouping (Mint Primer & Plain Facts) (`backend/app/ingestion/segmenter.py`)**:
   - Added `STANDALONE_FEATURE_KICKER_REGEX` and `is_numbered_feature_subhead()` to detect multi-part explainers (e.g. `mint primer`, `PLAIN FACTS`, `LONG STORY`).
   - Grouped all numbered Q&A items (`1 How...`, `2 Why...`, `3 What...`, `4...`, `5...`) into a single unified `SegmentedArticle` under the overarching master banner headline, preserving complete body text across columns.

### Verification & QA
- `make lint`: **0 errors across 66 source files**.
- `make test`: **170/170 tests passing 100% GREEN in 2.83s**.
- Added unit tests:
  - `test_syndication_slug_rejected_as_headline_and_actual_headline_preserved`
  - `test_mint_primer_grouped_into_single_article_with_all_questions`
  - `test_plain_facts_lead_economy_story_headline_preserved`

---

## Phase 6.1.22 — Native PDF Layout Parsing, Strict Column Reading Order, Font-Heuristic Headlines, Pre-Chunking Sanitizer & Drop Cap Reattachment

**Date**: 2026-08-22
**Status**: Completed ✅

### Problems Addressed
1. **Cross-Column Horizontal Bleeding**: Multi-column text blocks in digital PDFs were being sorted horizontally or merged across vertical column gutters, conflating distinct articles (e.g. Column 1 ISRO launch news vs Column 2 Cotton imports).
2. **False Mid-Paragraph Headlines**: Mid-paragraph bold emphasis spans were incorrectly tagged as `BlockType.HEADLINE`, while real headlines lacking font flags were missed.
3. **Standalone Drop Caps**: Single-letter initial uppercase drop caps (`"I"`, `"W"`, `"S"`, `"T"`, `"A"`) extracted by PyMuPDF were output as standalone 1-character blocks, causing fragmented sentences in downstream chunks.
4. **Noise / Promo Text Leakage**: UUID hashes, WhatsApp/Telegram promotional spam, and printer CMYK calibration marks were leaking into layout elements and vector chunks.

### Architectural Solutions & Implementations

1. **Pre-Chunking Regex Sanitizer (`backend/app/ingestion/detector.py`, `layout_analyzer.py`, `segmenter.py`, `chunker.py`)**:
   - Implemented `is_noise_or_promo_text()` and `sanitize_block_text()`:
     - UUIDs: `\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b`
     - Social/Promo Links: `(?i)(?:Join\s+FREE\s+(?:Whatsapp|Telegram)\s+Channel.*|https?://(?:t\.me|chat\.whatsapp\.com|wa\.me|bit\.ly|tinyurl\.com)/\S*|\bt\.me/\S*|\bwhatsapp\s+channel\b|\btelegram\s+group\b)`
     - Printer Marks: `(?i)(?:A\s*ND-NDE\s*C\s*M\s*Y\s*K|\bC\s*M\s*Y\s*K\b|\bcyan\s+magenta\s+yellow\s+black\b|epaper\s*[\.\-]\s*livemint|pdf\s*version\s*generated)`
   - Applied noise filtering across all stages: digital block extraction, spatial layout element creation, article segmentation, and hierarchical document chunking.

2. **Drop Cap Reattachment (`backend/app/ingestion/detector.py`, `layout_analyzer.py`)**:
   - Implemented `reattach_drop_caps()` for `DigitalTextBlock` and Pass 0 spatial drop cap consolidation in `LayoutAnalyzer._consolidate_elements()`.
   - Single-letter uppercase blocks (`len(text.strip()) == 1` and `text.strip().isupper()`) positioned adjacent to subsequent body paragraph blocks in the same column track are automatically merged (`"I"` + `"ndia is planning..."` $\to$ `"India is planning..."`), expanding bounding box envelopes and eliminating 1-letter orphan blocks.

3. **Font-Heuristic Headline Detection (`backend/app/ingestion/detector.py`, `layout_analyzer.py`)**:
   - Computed median body text font size ($dominant\_font\_size$) and line height ($median\_lh$).
   - Enforced strict criteria for `BlockType.HEADLINE`:
     - $font\_size \ge 1.25 \times dominant\_font\_size$
     - Title Case ($>55\%$ capitalized content words) or UPPERCASE (`text.isupper()`)
     - Minimum 2 words, not ending in terminal sentence punctuation (`.`, `;`) for long blocks
     - Rejected boilerplate stopwords and syndication/kicker slugs from becoming headlines.

4. **Strict Column-Bound Reading Order (`backend/app/ingestion/reading_order.py`, `layout_analyzer.py`)**:
   - In `ReadingOrderResolver`, strictly partitioned elements into discrete 2D column tracks under headline containers.
   - Enforced top-to-bottom traversal within Column 1 before moving to Column 2, preventing horizontal cross-column bleeding.
   - Prohibited horizontal merging across detected column gutters.

### Verification & QA
- `make lint`: **0 errors across 66 source files**.
- `make test`: **182/182 tests passing 100% GREEN in 2.89s**.
- Added new unit test suites:
  - `backend/tests/test_sanitizer.py`: 5 tests covering UUIDs, WhatsApp/Telegram promos, printer marks, and block sanitization.
  - `backend/tests/test_drop_cap.py`: 3 tests covering `DigitalTextBlock` lowercase continuation, capitalized continuation, and `LayoutAnalyzer` consolidation.
  - `backend/tests/test_reading_order.py::test_side_by_side_column_stories_never_bleed`: Verified independent side-by-side columns (ISRO in Col 1 vs Cotton Imports in Col 2) resolve in strict column-bound reading order.

---

## Phase 6.1.23 — Document Ingestion Migration to Docling (Layout, Reading Order, Table Extraction & MCP)

**Date**: 2026-08-22
**Status**: Completed ✅

### Problems Addressed & Motivations
1. **Fragile Custom Spatial Math**: Manual column-gutter calculation and horizontal lookahead routines in broadsheet newspapers were fragile on complex multi-column front pages.
2. **Deep Neural Layout & Table Extraction**: Replacing custom heuristics with IBM Docling (`docling==2.121.0`) provides state-of-the-art document layout parsing, pre-linearized human reading order, and table matrix extraction.
3. **Container-Friendly Local Execution**: Seamless execution locally on Apple Silicon (MPS / CPU) and NVIDIA CUDA without cloud dependencies.

### Architectural Solutions & Implementations

1. **Docling Ingestion Adapter (`backend/app/providers/docling_provider.py`)**:
   - Implemented `DoclingProvider` adhering to both `DocumentLayoutProvider` and `OCREngine` protocols.
   - Initialized `DocumentConverter` with `PdfPipelineOptions(do_ocr=True, do_table_structure=True)` and dynamic accelerator detection (MPS/CUDA/CPU).
   - Extracted pre-linearized reading order via `doc.iterate_items(page_no=p)`.
   - Converted Docling `BoundingBox` to standard top-left origin coordinates `(x0, y0, x1, y1)`.
   - Extracted structured table matrices (`headers`, `rows`, `raw_markdown`, `raw_html`) into `ExtractedTableData`.
   - Provided deterministic PyMuPDF fallback adapter for lightweight offline unit tests.

2. **Configuration & Registry Integration (`backend/app/providers/base.py`, `registry.py`, `model_config.yaml`)**:
   - Added `ProviderType.DOCLING = "docling"` in `base.py`.
   - Registered `DoclingProvider` in `ModelRegistry` for `provider_type in ("docling", "docling_parser")`.
   - Configured `docling_parser` in `model_config.yaml` and bound `layout_analysis`, `document_parser`, and `ocr` tasks to `docling_parser`.

3. **Pipeline Refactoring (`backend/app/ingestion/layout_analyzer.py`, `segmenter.py`)**:
   - Refactored `LayoutAnalyzer.analyze_page()` to consume Docling's structured document graph and pre-linearized reading order directly, bypassing fragile spatial lookahead math.
   - Preserved bounding box envelopes in `SegmentedArticle.bbox_list` and `ArticlePage.bbox_json` for frontend canvas overlays.

### Verification & QA
- `make lint`: **0 errors across 67 source files**.
- `make test`: **187/187 tests passing 100% GREEN in 12.82s**.
- Added unit test suite `backend/tests/test_docling_provider.py`:
  - `test_docling_provider_protocols`
  - `test_detect_device_mode`
  - `test_parse_markdown_table_to_matrix`
  - `test_parse_html_table_to_matrix`
  - `test_docling_parse_pdf_document_live_fixture`
  - `test_docling_parse_page_image_and_ocr`
  - `test_docling_model_registry_resolution`
  - `test_layout_analyzer_with_docling`

---

## Phase 6.1.24 — Multi-Category Universal Ad Detection Engine & Vector Suppression

**Date**: 2026-08-22
**Status**: Completed ✅

### Problems Addressed & Motivations
1. **Commercial Jacket Wraps & Retail Spreads Inadvertently Indexed**: Commercial cover wraps, smartphone launch spreads, real estate matrices, and retail display ads were slipping past single-keyword checks and being indexed into Qdrant as general news.
2. **Fragile Brand-Specific Rules**: Hardcoded brand strings failed on new campaigns and retail promotions.
3. **Docling Ingestion Latency (24s $\to$ <0.3s/page)**: Running full vision OCR per-page on 12-megapixel broadsheets was resolved with issue-wide vectorized pre-parsing.

### Architectural Solutions & Implementations

1. **Modular Functional Regex Patterns (`backend/app/ingestion/detector.py`)**:
   - `EXPLICIT_AD_HEADER_REGEX`: Masthead/banner markers (`advertisement`, `advertorial`, `sponsored feature`, `brand connect`, `special marketing feature`, `consumer connect`, `media marketing`).
   - `CTA_REGEX`: Calls to action (`book now`, `pre-order`, `buy now`, `call toll free`, `visit us at`, `apply now`, `scan to know more`, `download the app`, `toll free no`).
   - `PRICING_FINANCE_REGEX`: Pricing and purchase incentives (`starting at`, `special offer`, `limited period offer`, `flat \d+% off`, `down payment`, `no cost emi`, `zero processing fee`, `exchange bonus`, `t&c apply`).
   - `REAL_ESTATE_AUTO_REGEX`: Real estate/auto markers (`ready to move`, `possession soon`, `\d+\s*bhk`, `rera reg`, `ex-showroom price`, `test drive today`, `authorized dealership`).
   - `STATUTORY_TENDERS_REGEX`: Legal disclosures/tenders (`public notice`, `statutory notice`, `before the hon'ble`, `nclt`, `auction sale notice`, `tender notice`, `corrigendum`).
   - `IPO_FINANCIAL_REGEX`: Capital market notices (`initial public offering`, `price band`, `equity shares of face value`, `bid/issue opens`, `retail individual bidders`, `book running lead managers`, `red herring prospectus`, `asba`).

2. **Digital Discovery & Contact Footprint Scoring (`detector.py`)**:
   - `CONTACT_FOOTPRINT_REGEX`: Toll-free numbers (`1800|1860`), URLs (`https?://`, `www.`), and email addresses (+1.5 score).

3. **Editorial Contrast & Safety Guardrail (`detector.py`)**:
   - `EDITORIAL_MARKERS_REGEX`: Editorial markers (`bureau`, `correspondent`, `special correspondent`, `express news service`, `edited by`, `opinion`, `editorial`, `columns?`, `continued on page \d+`, `from page \d+`).
   - Protected threshold: Articles with $\ge 3$ editorial markers and $\ge 350$ words require a high commercial confidence threshold (`ad_score >= 6.0`) to avoid false positives on corporate M&A / finance news.

4. **Multi-Signal Scoring Heuristic (`is_page_advertisement()`, `check_is_advertisement_text()`)**:
   - Immediate match on top explicit ad header $\rightarrow$ `True`.
   - Distinct functional category matches $\rightarrow$ $+2.0$ per category.
   - Contact / URL footprint $\rightarrow$ $+1.5$.
   - Position weighting: Page 1 and back page $\rightarrow$ $+1.0$.
   - Decision logic:
     - `ad_score >= 4.5` $\rightarrow$ `True`.
     - `word_count < 250` and `ad_score >= 3.0` $\rightarrow$ `True`.
     - Editorial hits $\ge 3$ and `word_count >= 350` $\rightarrow$ require `ad_score >= 6.0`.

5. **Single-Unit Enveloping & Vector Suppression (`segmenter.py`, `tasks.py`)**:
   - Single-unit grouping into `SegmentedArticle` with `article_type = "advertisement"` and `section = "advertisement"`.
   - Complete bypass of Qdrant dense vector upsert (`is_indexed = False`).
   - Persisted into `identified_advertisements.json` debug artifact and MySQL.

### Verification & QA
- `make lint`: **0 errors across 67 source files**.
- `make test`: **192/192 tests passing 100% GREEN in 12.77s**.
- Added test cases in `test_detector.py`:
  - `test_retail_tech_launch_spread_detected_as_ad`
  - `test_real_estate_and_auto_ad_detected`
  - `test_statutory_and_ipo_financial_notices`
  - `test_editorial_contrast_safety_guardrail`
  - `test_corporate_news_article_never_flagged_as_advertisement`

---

## Phase 6.1.25 — Layout Semantics Redesign for Cross-Newspaper Compatibility (*The Hindu* & *Mint*)

**Date**: 2026-08-22
**Status**: Completed ✅

### Problems Addressed & Motivations
1. **Dense Commercial T&C Traps**: Tech and automotive cover wraps (e.g. Samsung Galaxy Z Fold8 cover wrap with 212 words) were bypassing static word ceilings due to extensive terms & conditions, pricing matrices, and legal disclaimers.
2. **Table of Contents / Front-Page Index Teaser Hallucinations**: Pipe-delimited index pointers (`"Global | Trump approves... >P14"`) were being stitched into fake news stories.
3. **Quote Attribution Headline Pollution**: ALL CAPS speaker designations (`"PENNY WONG AUSTRALIAN FOREIGN MINISTER"`, `"PENNYWONG AUSTRALIANFOREIGN MINISTER"`) were passing font heuristics and getting misclassified as standalone headlines.
4. **Vector Retrieval Preservation**: Ensuring ads are indexed into Qdrant with `section = "Advertisements & Notices"` and `article_type = "advertisement"` rather than being dropped, enabling retriever filtering.

### Architectural Solutions & Implementations

1. **The T&C Trapdoor Rule & Lexicon Density Scoring (`backend/app/ingestion/detector.py`)**:
   - `TC_LEGAL_PHRASES_PATTERNS`: Target commercial/legal strings (`t&c apply`, `terms and conditions apply`, `inclusive of all taxes`, `sole discretion`, `no cost emi`, `easy emi`, `cashback`, `damage protection`, `buyback`, `images simulated`, `screen simulated`, `optional accessories`, `emi options`, `exchange bonus`, `down payment`, `zero processing fee`, `annual percentage rate`).
   - `count_distinct_tc_phrases()` & `calculate_commercial_lexicon_density()`: Evaluates functional commercial density.
   - **The T&C Trapdoor Rule**: If a page or major block contains $\ge 3$ distinct commercial/legal phrases, it triggers `is_advertisement_page = True` **regardless of word count**.
   - Deprecated static `< 120 word` ceiling.

2. **Vector Inclusion Rule (`backend/app/ingestion/tasks.py`)**:
   - Routed advertisements explicitly to `section = "Advertisements & Notices"` and `article_type = "advertisement"`.
   - Enabled standard Qdrant vector indexing and chunking for advertisements (skipping only trivial sub-10-word fragments), allowing retriever payloads to filter `article_type` dynamically.

3. **Table of Contents (ToC) / Index Isolation (`backend/app/ingestion/reading_order.py`, `layout_analyzer.py`, `segmenter.py`)**:
   - Added `BlockType.TOC_INDEX = "toc_index"` to `BlockType` enum.
   - Implemented `is_toc_index_block(text)`: Scans for delimiter patterns (`|`), chevron/arrow page pointers (`>P14`, `-> P8`), and section slugs (`Global |`, `Money |`, `Views |`, etc.).
   - Updated `build_layout_from_parsed_nodes()` to classify ToC blocks as `BlockType.TOC_INDEX`.
   - In `ArticleSegmenter`, severed reading chains for `TOC_INDEX` blocks, isolating them from body text merging and article creation.

4. **Quote Attribution vs. Headline Discrimination (`layout_analyzer.py`, `segmenter.py`)**:
   - Added `BlockType.PULLQUOTE_AUTHOR = "pullquote_author"` to `BlockType` enum.
   - Implemented `is_pullquote_author_block(text, surrounding_text)` with `PULLQUOTE_TITLE_REGEX` and `HEADLINE_ACTION_VERBS` guardrail.
   - Rejects pullquote author attributions from `is_valid_headline_candidate()`, preventing speaker tags like `"PENNYWONG AUSTRALIANFOREIGN MINISTER"` from starting hallucinated articles.

### Verification & QA
- `make lint`: **0 errors across 67 source files**.
- `make test`: **195/195 tests passing 100% GREEN in 12.13s**.
- Added unit tests:
  - `test_tc_trapdoor_dense_ad_spread_flagged_regardless_of_word_count` in `test_detector.py`
  - `test_toc_index_block_isolated_and_severed` in `test_segmenter.py`
  - `test_pullquote_author_attribution_rejected_as_headline` in `test_segmenter.py`

---

## Phase 6.2 — Full-Stack Newspaper Intelligence & UI Implementation

**Date**: 2026-08-22
**Status**: Completed ✅

### What was built

1. **SVG Canvas Coordinate Engine & Spatial Overlays (`CanvasOverlay.jsx`)**:
   - Implemented native SVG `viewBox="0 0 width height"` scaling mapped directly over MinIO raster images (`/api/pages/{page_id}/image`), completely preventing coordinate drift on responsive window resizing.
   - Interactive, color-coded `<rect>` elements mapped by category:
     - 🟢 **News / Articles**: `rgba(34, 197, 94, 0.08)` stroke `rgb(34, 197, 94)`
     - 🔵 **Structured Tables**: `rgba(59, 130, 246, 0.1)` stroke `rgb(59, 130, 246)`
     - 🟣 **Photos & Infographics**: `rgba(168, 85, 247, 0.1)` stroke `rgb(168, 85, 247)`
     - 🟡 **Advertisements & Notices**: `rgba(245, 158, 11, 0.08)` stroke `rgb(245, 158, 11)`
   - Floating hover tooltips, click selection, and SVG filter glow pulse animation.

2. **Cross-Component Spatial Citation Bridge (`ActiveHighlightContext.jsx`)**:
   - Shared React Context managing `{ activeIssueId, activePageNumber, activeArticleId, activeBboxes, isPulsing }`.
   - `highlightArticle()` helper that updates target issue/page, triggers temporary 4-second pulse animation on source bounding boxes, and switches view to Broadsheet Reader.

3. **Interactive Broadsheet Reader (`BroadsheetReader.jsx`)**:
   - Split-screen workspace: Left canvas with Zoom in/out/reset, page carousel, folio indicators, and overlay toggles; Right inspector pane with serif headlines, kicker, byline, AI summary pill, prominence score gauge, full text, and multi-page continuation jump buttons.
   - Bidirectional hover and selection synchronization between list cards and SVG overlays.

4. **Real-Time Streaming Agentic Assistant (`AgentAssistant.jsx`)**:
   - Connected to `POST /api/query/stream` consuming SSE events (`stage`, `plan`, `tool_result`, `token`, `citations`, `done`).
   - Collapsible **Plan & Tool Telemetry** disclosure showing executed SQL and hybrid vector steps.
   - Clickable citation badges `[Newspaper, Page N]` wired directly to `ActiveHighlightContext`.
   - Dynamic LLM model provider switcher (`ollama_llama`, `anthropic_sonnet`, `openai_gpt4`, `groq_llama`).

5. **Faceted Archive Navigator & Ingestion Console (`ArchiveExplorer.jsx`, `UploadTrigger.jsx`, `RawDataViewer.jsx`)**:
   - `ArchiveExplorer`: Multi-dimensional filtering by publication, date, status, issue metrics, and 3-tier hard deletion.
   - `UploadTrigger`: Drag-and-drop PDF dropzone, pipeline stage tracker, and direct one-click "Read Issue" button upon completion.
   - `RawDataViewer`: Runtime dynamic task-provider binding manager (`PUT /api/settings/model-bindings`) and live JSON endpoint inspector.

6. **Application Shell & Styling (`App.jsx`, `index.css`, Tailwind v4)**:
   - Modern top navigation bar, newspaper-inspired typography, dark slate palette, and custom scrollbars.

### Verification & QA
- `npm run build`: **Frontend built in 628ms with 0 errors**.
- `make lint`: **0 errors across 67 backend source files**.
- `make test`: **195/195 tests passing 100% GREEN in 14.54s**.

---

## Phase 7 — Observability, Performance, Cost Control & Resilience

**Date**: 2026-08-22
**Status**: Completed ✅

### What was built

1. **Prometheus Metrics Engine & Middleware (`app/core/metrics.py`, `app/api/main.py`)**:
   - Integrated non-blocking Prometheus metrics tracking:
     - `newslens_http_requests_total(method, endpoint, status_code)`
     - `newslens_http_request_duration_seconds(method, endpoint)`
     - `newslens_agent_queries_total(archetype, status, model)`
     - `newslens_agent_query_duration_seconds(archetype)`
     - `newslens_llm_tokens_total(provider, model, direction)`
     - `newslens_llm_cost_usd_total(provider, model)`
     - `newslens_cache_events_total(cache_type, event)`
     - `newslens_cascade_fallbacks_total(primary_provider, fallback_provider, reason)`
     - `newslens_ingestion_pages_total(newspaper, extraction_mode)`
     - `newslens_ingestion_stage_duration_seconds(stage)`
     - `newslens_celery_active_tasks`
   - Mounted `GET /metrics` exposition endpoint and registered `PrometheusMiddleware` in FastAPI.

2. **Resilient Redis Cache Store with Graceful Degradation (`app/storage/cache_store.py`)**:
   - Deterministic SHA-256 query caching computed from normalized composite tuples: `compute_query_cache_key(query, model_id, date_filters, issue_ids)`.
   - Text embedding vector caching with `compute_embedding_cache_key(text, model)`.
   - Safe exception handling wrapping all Redis interactions — if Redis is down, unreachable, or times out, queries pass through seamlessly to underlying providers without raising unhandled exceptions.

3. **Token Cost Accountant & Budget Guardrails (`app/core/cost_tracker.py`)**:
   - Official pricing catalog across Anthropic (`claude-3-5-sonnet`, `claude-3-7-sonnet`, `claude-3-5-haiku`), OpenAI (`gpt-4o`, `gpt-4o-mini`, `text-embedding-3-large`), Groq (`llama-3.3-70b-versatile`), and \$0.00 local engines.
   - `calculate_cost_usd()`, `record_usage_and_cost()`, and `validate_query_budget(estimated_cost_usd, max_budget_usd)`.

4. **Fault-Tolerant Provider Cascade Manager (`app/providers/cascade.py`)**:
   - `CascadeChatProvider`: Prioritized fallback chains with automatic recovery from HTTP 429 rate limits, connection timeouts, and provider errors.
   - Structured JSON audit logging (`logger.warning("Provider cascade triggered", extra={...})`) and Prometheus fallback counter tracking.

5. **Extended Health Checks & Celery Telemetry (`app/api/routers/health.py`)**:
   - Added asynchronous worker ping inspection `_check_celery()` reporting active worker nodes and queue latency.

### Verification & QA
- `make lint`: **0 errors across 71 backend source files** (`ruff` + `mypy` strict).
- `make test`: **208/208 tests passing 100% GREEN in 17.75s**.
- Added unit & integration tests:
  - `tests/test_metrics.py`: Verifies `/metrics` endpoint and non-blocking metric collection.
  - `tests/test_cache_store.py`: Verifies deterministic key normalization and graceful degradation on Redis downtime.
  - `tests/test_cost_tracker.py`: Verifies pricing resolution, cost calculation, and budget guardrails.
  - `tests/test_cascade.py`: Verifies fallback chains, structured logging, and all-failed exceptions.

---

## Critical Hotfix — Conversational Query Condensation & Coreference Resolution

**Date**: 2026-08-22  
**Status**: Completed ✅

### Architectural Enhancements & Problem Solved
Follow-up conversational questions containing pronouns and deictic references (e.g. `"can you summarize it"`, `"tell me more about this"`, `"who was involved?"`) previously failed vector search because the literal query lacked named entities and semantic keywords. In addition, generic prompts submitted on turn 1 of clean sessions generated irrelevant searches.

### Key Changes
1. **Query Condenser Engine (`backend/app/agent/condenser.py`)**:
   - `needs_condensation(query, chat_history)`: Identifies pronouns and short follow-up phrases in multi-turn dialogues.
   - `is_ambiguous_standalone_query(query, chat_history)`: Detects ambiguous prompts on clean sessions (turn 1) with $< 6$ words and no context.
   - `condense_conversational_query(query, chat_history, provider)`: Rewrites follow-up questions into self-contained, entity-dense search queries using fast LLM completion without generating direct answers.
2. **LangGraph State Machine & Guardrails (`backend/app/agent/state.py`, `backend/app/agent/graph.py`)**:
   - Added `chat_history` and `original_query` to `AgentState`.
   - Integrated ambiguity guardrail into `_classify_and_plan_node`: Clean session ambiguous queries short-circuit immediately to return `"Please specify which article, topic, or newspaper issue you would like me to summarize."` without invoking vector or SQL tools.
   - Integrated query condensation before `QueryPlanner.plan_query` to ground tool scheduling in the rewritten context.
3. **API & Streaming Real-Time Events (`backend/app/api/routers/query.py`)**:
   - Added `chat_history` to `QueryRequest` schema.
   - Emitted `event: query_condensed` with `data: {"condensed_query": "..."}` and stage `'condensing_query'` in SSE stream.
4. **Frontend Context Sync & UI (`frontend/src/components/AgentAssistant.jsx`)**:
   - Wired `handleSend` to include recent conversational turns (`chat_history`) in the request body.
   - Rendered subtle `Resolved Context` pill indicating the reformulated search query.
   - Added stage indicator for coreference resolution.
5. **Comprehensive Unit & Integration Test Suite (`backend/tests/test_query_condenser.py`)**:
   - 9 test cases verifying pronoun detection, clean session guardrail short-circuiting, LLM query rewriting, state machine execution, and `/api/query` response handling.

### Verification
- `make lint`: **0 errors across 72 source files** (`ruff` + `mypy` strict).
- `make test`: **217/217 tests passing 100% GREEN in 22.27s**.
- `npm run build`: **Vite build succeeded with 0 errors**.

---

## Critical Hotfix — Thought Process & Reasoning Trace Separation

**Date**: 2026-08-22  
**Status**: Completed ✅

### Problem Solved
Reasoning models (such as Groq Qwen 3.6 / DeepSeek) emit internal chain-of-thought traces wrapped in `<think>...</think>` tags. Previously, these tokens streamed directly into the final chat response bubble, cluttering the synthesized answer with raw step-by-step reasoning notes.

### Key Changes
1. **Streaming Token Parser (`backend/app/api/routers/query.py`)**:
   - Dynamically parses incoming stream chunks for `<think>` and `</think>` tags.
   - Emits internal reasoning tokens via `event: thought` with stage `'thinking'`, keeping thought tokens strictly separated from the final answer stream.
   - Emits `event: thought_done` containing the full formatted thought trace and duration timing (`duration_sec`).
   - Only clean answer tokens (outside `<think>` tags) are emitted via `event: token`.
2. **Synthesis Sanitization (`backend/app/agent/synthesizer.py`)**:
   - Strips `<think>.*?</think>` tags in `synthesize()` so that stored database `QueryLog` answers and citation extractions are clean.
3. **Frontend Collapsible Thought Accordion (`frontend/src/components/AgentAssistant.jsx`)**:
   - Added collapsible `Thought for Xs >` accordion component styled after modern reasoning AI interfaces.
   - Displays reasoning trace cleanly outside the main answer text, collapsed by default once synthesis completes.
   - Renders only pure, verified answers (`ans`) in the main chat bubble.
   - Added client-side sanitizers (`sanitizeAnswerText`, `extractFallbackThought`) as an additional safety net.

### Verification
- `make lint`: **0 errors across 72 source files** (`ruff` + `mypy` strict).
- `make test`: **217/217 tests passing 100% GREEN in 25.21s**.
- `npm run build`: **Vite build succeeded with 0 errors**.
- **Frontend runtime**: Defined `sanitizeAnswerText` and `extractFallbackThought` at top level in `AgentAssistant.jsx` ensuring hot reload and clean page mounting without `ReferenceError`.

---

## Critical Bugfix — Robust `<think>` Tag Stream Parsing & 1-Based Physical PDF Page Number Standardization

**Date**: 2026-08-22  
**Status**: Completed ✅

### Problems Solved
1. **Thinking Token Leakage / Blank Final Response**: When using reasoning models (e.g. `Qwen 3.6`, `DeepSeek R1`), the response was trapped inside `<think>` or unclosed thinking blocks, rendering only in the Thought Process accordion and leaving the final answer blank.
2. **Page Number Disconnect**: Citations and deep links pointed to mismatched pages where extracted printed folios collided with physical PDF page numbers.

### Key Architectural Fixes
1. **Robust `<think>` Tag Stream & Answer Recovery (`synthesizer.py` & `query.py`)**:
   - Implemented `parse_thought_and_answer()` helper using regex section pattern matching (`\n\s*(?:#{1,4}\s+|Based on|According to|In conclusion|Summary:|Answer:)`) to separate reasoning notes from answer narrative even if `</think>` is omitted by the LLM.
   - In `stream_query`, if `answer_chunks` is empty at stream completion, dynamically parses `think_chunks` and emits `event: token` with the clean narrative so the response view is **never left blank**.
2. **Frontend Robust Decoupling (`AgentAssistant.jsx`)**:
   - Implemented `splitThoughtAndAnswer` and updated `sanitizeAnswerText` in `AgentAssistant.jsx`.
   - On SSE `done` event, recovers answer text if `msg.content` was trapped in `msg.thought`.
   - Collapsible `Thought for Xs >` accordion displays ONLY internal reasoning trace, while main message bubble displays clean markdown answer.
3. **Strict 1-Based Physical PDF Page Number Standardization**:
   - Grounded all retrieval results (`HybridSearchResult`, `EntitySearchResult`, `TimelineMilestone`), prompt context templates, and citation badges strictly on the physical 1-based PDF page index: `page_number` (1..$N$).
   - Updated `SYNTHESIZER_SYSTEM_PROMPT` to enforce citation format:
     `[{Newspaper Name}, {YYYY-MM-DD}, Page {PDF_Page_Number}, "{Headline}"]`
   - Updated `_build_evidence_context()` to explicitly state:
     `[Evidence: Newspaper Name, YYYY-MM-DD, Page {pdf_page} (PDF Page {pdf_page}), Headline: "..."]`
   - Populated `issue_id` and `bboxes` on all retrieval results and `AgentCitation` records.
   - Updated `ActiveHighlightContext.jsx` and `AgentAssistant.jsx` to pass `Number(pageNumber) || 1` directly to `BroadsheetReader` to load the exact PDF page image.

### Verification
- `make lint`: **0 errors across 72 source files** (`ruff` + `mypy` strict).
- `make test`: **217/217 tests passing 100% GREEN in 25.21s**.
- `npm run build`: **Vite production build completed with 0 errors**.

---

## Google Gemini Provider Integration

**Date**: 2026-08-23  
**Status**: Completed ✅

### What was built
1. **`GeminiProvider` ([`backend/app/providers/gemini_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/gemini_provider.py))**:
   - Implemented `ChatModelProvider` and `VisionModelProvider` for Google Gemini models via Google Generative Language REST API (`https://generativelanguage.googleapis.com/v1beta/models`).
   - Supports text generation, multimodal vision inputs (`analyze_image`), structured output parsing (`responseMimeType: application/json`), and real-time SSE token streaming (`streamGenerateContent`).
2. **Model Registry & Config Binding ([`backend/app/providers/registry.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/registry.py) & [`model_config.yaml`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/model_config.yaml))**:
   - Added `gemini_flash` (`gemini-3.7-flash`) and `gemini_pro` (`gemini-pro-latest`) configs.
   - Added `GEMINI_API_KEY` and `GOOGLE_API_KEY` to `Settings` in `backend/app/core/config.py` and `.env.example`.
3. **Frontend Selector ([`frontend/src/components/AgentAssistant.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/AgentAssistant.jsx))**:
   - Added `✨ Google / Gemini 3.7 Flash` and `✨ Google / Gemini Pro Latest` options to the LLM model selector.
4. **Unit Tests ([`backend/tests/test_providers.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_providers.py))**:
   - Added `TestGeminiProvider` suite verifying protocol conformance, API key validation, and chat completions.

### Verification
- `make lint`: **0 errors across 73 source files** (`ruff` + `mypy` strict).
- `make test`: **220/220 tests passing 100% GREEN in 69.18s**.
- `npm run build`: **Vite build completed with 0 errors in 701ms**.

---

## Frontend Resilience Hotfix & Multi-Engine Ingestion Selector

**Date**: 2026-08-23  
**Status**: Completed ✅

### Problems Solved
1. **Frontend Blank Black Screen**: When database queries or API endpoints returned non-array error payloads (e.g. `{error: "Internal server error"}` when MySQL is offline), components calling `issues.map(...)` threw unhandled `TypeError` exceptions that crashed the entire React tree on initial mount.
2. **Ingestion Engine Selection**: The user requested a dropdown in the Ingestion Console to choose between **Docling**, **MinerU**, **Google Gemini Vision**, and **Local VLM + Tesseract** during PDF upload.

### Key Architectural Fixes
1. **Global React Error Boundary & Defensive Array Checks (`App.jsx`, `BroadsheetReader.jsx`, `ArchiveExplorer.jsx`)**:
   - Added a top-level `ErrorBoundary` in `App.jsx` with a styled recovery screen and reload action.
   - Enforced `Array.isArray(data) ? data : []` across all fetch hooks and list state initializers so that transient backend connection errors or non-array payloads never crash the interface.
2. **Multi-Engine Ingestion Selector (`UploadTrigger.jsx`, `ingest.py`, `tasks.py`)**:
   - Added **Parsing & Layout Engine** selector in `UploadTrigger.jsx` with options:
     - `📄 Docling (Neural Layout & OCR) — Recommended`
     - `📐 MinerU / Magic-PDF (Academic & Tables)`
     - `✨ Google Gemini Vision (Multimodal Cloud)`
     - `💻 Local VLM + Tesseract`
   - Updated `/api/ingest/upload` and `run_ingestion_pipeline(issue_id, pdf_bytes, parser_engine)` to pass and execute the chosen neural layout engine.

### Verification
- `make lint`: **0 errors across 73 source files** (`ruff` + `mypy` strict).
- `make test`: **220/220 tests passing 100% GREEN in 19.71s**.
- `npm run build`: **Vite build succeeded with 0 errors in 688ms**.
- **Live Browser Verification via Puppeteer**: Successfully loaded `localhost:5173`, confirmed broadsheet viewer, interactive overlays, tab switching, and verified the live **Parsing & Layout Engine** dropdown in the Ingestion Console.

---

*Next phase: Phase 8 — Hosted Deployment & Production Orchestration (Multi-stage Dockerfiles, Docker Compose Prod, Helm/K8s)*

















