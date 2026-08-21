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

*Next phase: Phase 6.2 — Frontend UI Polish (Tailwind CSS, Radix UI, Reader UI, Visual Bounding-Box Overlays)*



