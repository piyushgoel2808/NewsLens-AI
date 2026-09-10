# NewsLens-AI Codebase Architecture & File Reference Guide

> **Document Version**: 1.0.0 (Production Reference)  
> **Target Audience**: Core Engineers, System Architects, Research Scientists, and AI Pair Programmers.  
> **Purpose**: A complete, highly structured blueprint of the NewsLens-AI codebase. Contains the exact directory hierarchy, the architectural purpose of each directory, and an exhaustive file-by-file specification covering components, responsibilities, external frameworks/tools, and LLM/VLM models used.

---

## Table of Contents

1. [Master Codebase Directory Structure](#1-master-codebase-directory-structure)
2. [Architectural Overview & Subsystem Responsibilities](#2-architectural-overview--subsystem-responsibilities)
3. [Root Configuration & Operational Infrastructure](#3-root-configuration--operational-infrastructure)
4. [Backend Core Application (`backend/app/`)](#4-backend-core-application-backendapp)
   - [4.1 `backend/app/agent/` — Conversational Planning & RAG Workflow](#41-backendappagent--conversational-planning--rag-workflow)
   - [4.2 `backend/app/api/` & `routers/` — FastAPI Server & REST/SSE Endpoints](#42-backendappapi--routers--fastapi-server--restsse-endpoints)
   - [4.3 `backend/app/core/` — System Primitives, Governance & Rulebooks](#43-backendappcore--system-primitives-governance--rulebooks)
   - [4.4 `backend/app/evaluation/` — Information Retrieval Benchmarks](#44-backendappevaluation--information-retrieval-benchmarks)
   - [4.5 `backend/app/ingestion/` — 12-Stage Broadsheet Processing Pipeline](#45-backendappingestion--12-stage-broadsheet-processing-pipeline)
   - [4.6 `backend/app/models/` — Relational SQLAlchemy 2.0 Async Schemas](#46-backendappmodels--relational-sqlalchemy-20-async-schemas)
   - [4.7 `backend/app/providers/` — Dynamic Model Provider Tier](#47-backendappproviders--dynamic-model-provider-tier)
   - [4.8 `backend/app/retrieval/` — Multi-Tool Search, Reranking & Auditing](#48-backendappretrieval--multi-tool-search-reranking--auditing)
   - [4.9 `backend/app/storage/` — Multi-Tier Persistence Clients](#49-backendappstorage--multi-tier-persistence-clients)
5. [Backend Test Suite (`backend/tests/`)](#5-backend-test-suite-backendtests)
6. [Frontend Client Application (`frontend/`)](#6-frontend-client-application-frontend)
   - [6.1 Frontend Root & Build Tooling](#61-frontend-root--build-tooling)
   - [6.2 `frontend/src/context/` — Global Synchronized State](#62-frontendsrccontext--global-synchronized-state)
   - [6.3 `frontend/src/components/` — Workspaces, Readers & Modals](#63-frontendsrccomponents--workspaces-readers--modals)
7. [Operations & Diagnostic Scripts (`scripts/`)](#7-operations--diagnostic-scripts-scripts)
8. [Documentation Suite (`docs/`)](#8-documentation-suite-docs)

---

## 1. Master Codebase Directory Structure

```text
NewsLens-AI/
├── docker-compose.local.yml         # Containerized services: MySQL 8.4 LTS, Qdrant, MinIO, Redis, Celery
├── model_config.yaml                # Unified LLM/VLM model provider registry and dynamic task bindings
├── service-account.json             # Google Cloud Vision API credentials for document OCR
├── README.md                        # Master repository overview, setup guide, and documentation links
│
├── backend/                         # Python 3.13+ backend application root
│   ├── pyproject.toml               # Poetry/Hatch/uv project configuration & locked dependency specs
│   ├── uv.lock                      # Deterministic locked dependency graph managed by uv
│   │
│   ├── app/                         # Core application package
│   │   ├── agent/                   # Conversational RAG agent, cognitive planner, state, synthesizer
│   │   ├── api/                     # FastAPI setup, lifespan management, middleware, and sub-routers
│   │   │   └── routers/             # Endpoint definitions (articles, query, ingest, models, settings, etc.)
│   │   ├── core/                    # Global settings, logging, cost tracker, Prometheus metrics, YAML rules
│   │   ├── evaluation/              # IR benchmark evaluation metrics (Recall@K, MRR, NDCG@K, Precision)
│   │   ├── ingestion/               # 12-stage industrial broadsheet PDF ingestion and VLM extraction engine
│   │   ├── models/                  # SQLAlchemy 2.0 async relational schemas and ORM entities
│   │   ├── providers/               # Abstract model providers (Ollama, Groq, Gemini, OpenAI, GCV)
│   │   ├── retrieval/               # Multi-tool retrieval engines (hybrid search, SQL analytics, reranking)
│   │   └── storage/                 # Persistence clients (MySQL FULLTEXT, Qdrant, MinIO S3, Redis Cache)
│   │
│   └── tests/                       # Over 55 pytest test suites (unit, integration, and regression)
│
├── frontend/                        # Modern Single Page Application (React 18, Vite, Tailwind CSS)
│   ├── index.html                   # HTML5 entrypoint with broadsheet typography & viewport
│   ├── package.json                 # Node dependencies, scripts, and build tooling
│   ├── vite.config.js               # Vite bundler configuration & local API reverse proxy
│   ├── tailwind.config.js           # Broadsheet typography and dark-mode aesthetic theme
│   ├── postcss.config.js            # PostCSS plugin declarations
│   └── src/                         # React application source code
│       ├── App.jsx                  # Master application shell, navigation tabs, and global status
│       ├── main.jsx                 # Application DOM bootstrap and context wrappers
│       ├── index.css                # Global stylesheet, custom scrollbars, and broadsheet typography
│       ├── context/                 # Context providers (ActiveHighlightContext)
│       └── components/              # Interactive UI components (BroadsheetReader, AgentAssistant, etc.)
│
├── scripts/                         # Command-line tools, verification tests, and diagnostics
│   ├── generate_sample_newspaper.py # Generates synthetic multi-column broadsheet PDFs for offline testing
│   ├── qa_diagnostic_test.py        # Automated test suite running multi-archetype conversational queries
│   ├── reclassify_articles.py       # Batch article re-classification against updated category rulebooks
│   ├── show_schema.py               # Database schema inspection and foreign key relationship printer
│   ├── verify_phase1.py .. 6_1.py   # Multi-phase automated verification gates
│   └── verify_providers.py          # Real-time health and latency checker for LLM/VLM providers
│
└── docs/                            # Architectural guides, schemas, logs, and data flow manuals
    ├── architecture.md              # System design, multi-tier storage, and data flow architecture
    ├── database_schema.md           # Database ER diagram, column types, foreign keys, indexes
    ├── data_flow.md                 # Visual sequence diagrams tracing queries and document ingestion
    ├── data_flow_architecture.md    # Detailed data flow diagrams with component boundaries
    ├── end_to_end_data_flow_guide.md# Live production-verified data flow walkthrough with real DB traces
    ├── engineering_log.md           # Chronological technical log of architectural fixes and milestones
    ├── features.md                  # Comprehensive inventory of user-facing intelligence capabilities
    └── codebase_directory_and_file_reference.md # Master codebase directory and file reference
```

---

## 2. Architectural Overview & Subsystem Responsibilities

NewsLens-AI is an agentic intelligence platform engineered specifically for **broadsheet newspapers**. Broadsheets possess unique challenges that defeat naive RAG pipelines:
1. **2D Complex Geometry**: Articles wrap across vertical columns, jump across intervening advertisement blocks, and continue onto distant pages (e.g. Page 1 continuing on Page 9).
2. **Visual & Quantitative Data**: Financial tables, stock candlestick charts, and editorial infographics carry vital intelligence that OCR alone cannot parse without vision-language reasoning.
3. **Conversational Multi-Turn Ambiguity**: Users naturally ask follow-up questions referencing past context ("list all those 11 articles", "what was the date?"), which require query condensation and strict publication isolation.
4. **Relational vs. Semantic Queries**: Structural questions ("Summarize Issue 93", "List articles on Page 1") require exact relational database queries, whereas conceptual questions require dense/sparse hybrid search and cross-attention neural reranking.

---

## 3. Root Configuration & Operational Infrastructure

### Directory: `/` (Repository Root)
* **Purpose / Reason**: Establishes global system configuration, multi-container orchestration, third-party cloud authentication credentials, and developer documentation.
* **Work It Is Doing**: Acts as the project control center. Configures Docker networking, volume mounting, environment variable declarations, and unified model task bindings.

#### Files in Root Directory:

##### [`model_config.yaml`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/model_config.yaml)
* **What It Has**: 
  - Complete provider connection dictionary: `ollama_nemotron`, `ollama_deepseek`, `ollama_llama3`, `ollama_chat`, `ollama_qwen3vl`, `ollama_vlm`, `ollama_gemma4_26b`, `ollama_gemma4_12b`, `ollama_embed`, `local_embed_bge`, `google_cloud_vision`, `gemini_vision`, `gemini_flash`, `gemini_pro`, `openai_gpt4o`, `openai_gpt4o_mini`, `openai_embed`.
  - Task bindings dictionary: `layout_analysis`, `document_parser`, `ocr`, `embedding`, `query_planner`, `answerer`, `metadata_extraction`, `classification`, `article_segmentation`, `visual_extraction`.
* **Work It Is Doing**: Decouples application code from specific vendors or model names. When the application requests the `"query_planner"`, the registry reads this file to dynamically invoke `ollama_gemma4_12b` (or whichever model is bound). Enables hot-reloading model changes at runtime.
* **Important Tools / Frameworks**: YAML 1.2, PyYAML parser.
* **LLM / VLM / Embedding Models**: Configures `gemma4:12b`, `qwen3-vl:latest`, `nemotron-3.5-lightning`, `llama3.1:8b`, `deepseek-r1:14b`, `BAAI/bge-m3`, `gemini-3.7-flash`, `gpt-4o`, `text-embedding-3-large`.

##### [`docker-compose.local.yml`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docker-compose.local.yml)
* **What It Has**: Container configurations for 5 core services: `mysql`, `qdrant`, `minio`, `redis`, and `celery_worker`.
* **Work It Is Doing**:
  - `mysql`: Deploys MySQL 8.4 LTS with utf8mb4 collation, 1GB buffer pool, and persistent storage in `mysql_data`.
  - `qdrant`: Deploys Qdrant vector database on port 6333 with WAL persistence in `qdrant_data`.
  - `minio`: Deploys high-performance S3-compatible object storage on ports 9000 (API) and 9001 (Console).
  - `redis`: Deploys Redis 7 Alpine as the task message broker for Celery and high-speed query cache.
* **Important Tools / Frameworks**: Docker Compose specification, Healthchecks, Volume Mounting.
* **LLM / VLM / Embedding Models**: None (Infrastructure).

##### [`service-account.json`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/service-account.json)
* **What It Has**: Google Cloud Platform Service Account private key credentials (`private_key_id`, `client_email`, `project_id`).
* **Work It Is Doing**: Provides OAuth2 machine-to-machine authentication credentials for Google Cloud Vision OCR API calls during document layout parsing and text layer recovery.
* **Important Tools / Frameworks**: Google Cloud IAM, OAuth2.
* **LLM / VLM / Embedding Models**: Google Cloud Vision Document Text Detection Engine.

---

## 4. Backend Core Application (`backend/app/`)

### 4.1 `backend/app/agent/` — Conversational Planning & RAG Workflow
* **Purpose / Reason**: Implements the autonomous decision-making core of NewsLens-AI. Manages conversational memory, analyzes research intent, constructs execution plans, executes tools, filters noise via Corrective RAG (CRAG), and streams structured executive intelligence briefs.
* **Work It Is Doing**: Resolves multi-turn conversational coreference, enforces publication isolation barriers, coordinates multi-tool retrieval, and enforces 100% citation precision.

#### Files in `backend/app/agent/`:

##### [`backend/app/agent/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/__init__.py)
* **What It Has**: Package initialization, re-exports for `AgentGraph`, `QueryPlanner`, `QueryCondenser`, `Synthesizer`.
* **Work It Is Doing**: Exposes clean interface boundaries for the agent module.

##### [`backend/app/agent/condenser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/condenser.py)
* **What It Has**: 
  - `QueryCondenser` class.
  - Helper functions: `is_in_context_meta_query()`, `extract_active_issue_from_history()`, `extract_active_article_context()`, `extract_active_exclusion_context()`.
  - `CONDENSATION_PROMPT` system template.
* **Work It Is Doing**:
  - **Conversational Meta-Query Short-Circuit**: Checks if a query is purely asking about previous turn metadata (e.g. "What was the date?", "Which paper was this from?") and bypasses heavy retrieval to answer instantly from chat context.
  - **Query-Aware Context Isolation**: Analyzes prior turns to extract active publication names, issue dates, and differential comparison states. Purges stale context when the user switches topics.
  - **Coreference Resolution**: Rewrites ambiguous follow-up questions (e.g. "list all those 11 articles") into complete, standalone search queries ("list all those 11 articles in The Goan but not in The Morning Standard dated 2026-08-01").
* **Important Tools / Frameworks**: Python AsyncIO, Regular Expressions (`re`), Pydantic.
* **LLM / VLM / Embedding Models**: Invokes the configured `query_planner` LLM (e.g. `gemma4:12b`, `llama3.1:8b`, or `gpt-4o-mini`).

##### [`backend/app/agent/planner.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/planner.py)
* **What It Has**: 
  - Data models: `PlanResult`, `PlannedToolCall`, `ToolCallSpec`, `AgentPlan` (with `QueryPlan` and `ExtractedToolArguments` aliases for backward compatibility).
  - `QueryPlanner` class with true direct tool calling and deterministic fallback.
  - Core functions: `resolve_tool_sequence()` (backward-compatible delegate), `extract_parameters_from_query()`.
  - Methods: `plan_query_async()`, `plan_query()`, `classify_archetype()`, `_plan_query_heuristic()`, `_build_plan_from_structured_model()`.
  - Lean `PLANNER_SYSTEM_PROMPT` with 3 canonical few-shot examples illustrating direct tool call generation.
* **Work It Is Doing**:
  - **True Agentic Direct Tool Planning (Option 2)**: Directly prompts LLMs to schedule ordered tool calls (`[ToolCallSpec(tool_name, arguments, purpose)]`) inside `AgentPlan`, eliminating procedural indirection and boolean soup.
  - **Lean Deterministic Heuristic Router**: Clean single-pass intent classifier mapping queries to 6 core archetypes:
    1. `thematic_timeline` (chronological progression across multiple dates)
    2. `entity_deep_dive` (multi-hop entity network search and profiling)
    3. `cross_newspaper_comparison` (differential coverage, omissions, framing differences across broadsheets)
    4. `quantitative_trend` (article counts, topic distributions, page-level article manifests, full issue overviews)
    5. `article_catalog` (fast listing and catalog manifest generation for specific dates/sections)
    6. `factual_lookup` (targeted semantic + keyword search for point-in-time facts and quotes)
  - **Transparent Legacy Adapter**: Translates older mock objects and test fixtures (`QueryPlan`, `primary_tool`, `ExtractedToolArguments`) to direct tool calls with ground-truth parameter reconciliation.
  - **Ground-Truth Reconciliation & Hallucination Pruning**: Extracted query parameters (brand names, issue IDs, dates, page numbers) override any LLM hallucinations.
  - **Live Archive Grounding**: Dynamically injects `get_archive_metadata()` into the planner prompt, grounding the LLM with live issue dates, active publications, and canonical categories.
  - **Query Preservation & Generic Filler Sanitization**: In `_build_plan_from_structured_model()`, detects generic filler phrases and restores substantive user domain queries.
  - **High-Throughput Cloud Failover**: Prioritizes `nvidia_nemotron` (<1s hosted inference with streaming reasoning) on cloud failover routes.
  - Employs typo-tolerant regex parameter extraction for newspaper names (e.g. "he Morning Standard" $\to$ "The Morning Standard") and publication dates.
  - Produces structured Chain-of-Thought reasoning traces and deterministically schedules 1 to 4 complementary tool calls (`sql_analytics`, `hybrid_search`, `entity_search`, `timeline_builder`, `coverage_analysis`, `web_search`).
  - Includes `_plan_query_heuristic()` for instantaneous zero-latency local fallback if LLM generation encounters timeouts.
* **Important Tools / Frameworks**: Pydantic v2 schemas, Structured Outputs (`response_schema`), Regular Expressions.
* **LLM / VLM / Embedding Models**: `nvidia_nemotron` (NVIDIA NIM), `ollama_gemma4_12b` (Ollama), `groq_llama` (Groq), or `gemini_flash` (Gemini).

##### [`backend/app/agent/state.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/state.py)
* **What It Has**: `AgentState` TypedDict, `EvidenceItem` dataclass, `AgentCitation` dataclass.
* **Work It Is Doing**:
  - Defines the global state container passed through the LangGraph state machine.
  - Stores: user query, condensed query, conversation history, query plan, raw tool execution outputs, filtered evidence, streaming tokens, `<think>` reasoning traces, and bounding-box citations.
* **Important Tools / Frameworks**: Python `typing.TypedDict`, `typing.Annotated`, Pydantic models.
* **LLM / VLM / Embedding Models**: None (State Definition).

##### [`backend/app/agent/graph.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/graph.py)
* **What It Has**: 
  - `AgentGraph` class.
  - LangGraph node implementations: `_plan_query_node()`, `_execute_tools_node()`, `_evaluate_evidence_node()`, `_synthesize_answer_node()`, `_corrective_fallback_node()`.
  - Workflow graph compilation with conditional edges.
* **Work It Is Doing**:
  - Orchestrates the state machine workflow: Plan $\to$ Execute Tools $\to$ Evaluate Evidence (CRAG) $\to$ Synthesize Answer.
  - **Concurrent Tool Execution**: Dispatches planned tool calls concurrently via `asyncio.gather(*tasks, return_exceptions=True)`, reducing multi-tool query latency by 40–60%.
  - **Adaptive Zero-Hit Fallback**: In `_execute_single_tool`, if `hybrid_search` or `sql_analytics(issue_summary)` with a category filter returns 0 articles, automatically retries without the category constraint to prevent empty retrieval.
  - **Corrective RAG (CRAG) Gate**: Scores evidence relevance against stemmed query tokens, strips irrelevant distractors, and triggers fallback searches (`entity_search` or `web_search`) if grounded evidence is empty.
  - **Macro Manifest Protection**: Grants relational SQL manifests an automatic relevance score of $1.0$, guaranteeing that comprehensive exclusion lists and article counts are never pruned.
* **Important Tools / Frameworks**: LangGraph, Python AsyncIO, SQLAlchemy Async Session Factory.
* **LLM / VLM / Embedding Models**: Orchestrates planning and synthesis models; executes deterministic stemming algorithms.

##### [`backend/app/agent/synthesizer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/synthesizer.py)
* **What It Has**: 
  - `Synthesizer` class.
  - Dynamic system prompts via `_build_synthesizer_system_prompt()`.
  - Helper functions: `parse_thought_and_answer()`, `_build_evidence_context()`.
  - Generator method `synthesize_stream()`.
* **Work It Is Doing**:
  - Generates authoritative, highly readable executive intelligence briefs.
  - **Domain-Adaptive Synthesis**: Dynamically adapts comparison table structures based on topic domain (`Key Findings & Medical Focus` for health, `Key Figures & Metrics` for finance, `Key Policy Decisions & Statements` for politics) and outputs dedicated table manifests for `article_catalog` queries.
  - **Archetype Preservation in Fallbacks**: Passes explicit `archetype` to `_generate_deterministic_summary()`, ensuring cross-newspaper comparisons preserve multi-edition publication tables without dropping scheduled editions or degrading into single-paper templates.
  - **Granular Domain Stem Budgeting**: Maps composite domain labels (e.g. `Health & Medicine`) to granular search token stems (`["health", "hospital", "pharma", "medicine", "doctor", ...]`), ensuring health articles receive top relevance ranking in evidence budget limits.
  - **Headline Cleansing & OCR Font Ligature Repair Integration**: Sanitizes author/doctor byline boxes into descriptive feature labels while protecting real all-caps news headlines, and runs `repair_text_ligatures()` across all evidence and headlines.
  - **Evidence Context Budgeting**: Slices evidence to Top 12 items and enforces context caps (up to 4,000 characters for manifests/matrices, 1,200 characters for standard articles).
  - **Critical Publication Scoping Barrier**: Injects explicit constraints listing verified available publications, forbidding the model from hallucinating or citing absent newspapers.
  - **Reasoning Stream Parsing**: Separates model reasoning traces (`<think>...</think>` or `<thought>...</thought>`) from the final response text.
  - **Strict 1-Shot Citation Enforcement**: Mandates bracketed inline citations on every factual assertion:
    `[{Newspaper Name}, {YYYY-MM-DD}, Page {P}, "{Headline}"]` or `[📊 Chart: ...]`.
  - Anti-repetition constraints preventing duplicate bullet points.
* **Important Tools / Frameworks**: Async Generators (`AsyncIterator`), Regex Parsing, Pydantic.
* **LLM / VLM / Embedding Models**: Bound to `answerer` task (`gemma4:12b`, `llama3.1:8b`, `deepseek-r1:14b`, `nemotron-3.5-lightning`, or `gpt-4o`).

---

### 4.2 `backend/app/api/` & `routers/` — FastAPI Server & REST/SSE Endpoints
* **Purpose / Reason**: Serves as the high-concurrency API gateway and real-time streaming layer of NewsLens-AI.
* **Work It Is Doing**: Handles HTTP/REST requests, file streaming, multipart PDF uploads, dependency injection of database sessions, CORS policies, and Server-Sent Events (SSE).

#### Files in `backend/app/api/`:

##### [`backend/app/api/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/__init__.py)
* **What It Has**: Package initialization.
* **Work It Is Doing**: Marks directory as a Python package.

##### [`backend/app/api/main.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/main.py)
* **What It Has**: FastAPI `app` instance, lifespan async context manager, CORS middleware setup, Prometheus middleware integration, router inclusion (`newspapers`, `articles`, `query`, `ingest`, `metadata`, `models`, `settings`, `health`).
* **Work It Is Doing**:
  - Serves as the master HTTP application entrypoint.
  - Startup lifespan: Initializes MySQL connection pools (`init_db`), ensures Qdrant collections exist, and launches background pre-warming for embedding (`BAAI/bge-m3`) and cross-encoder reranker models to absorb cold-start latency.
  - Shutdown lifespan: Gracefully closes database connection pools and client sessions.
  - Exposes standard `/metrics` endpoint for Prometheus scraping.
* **Important Tools / Frameworks**: FastAPI, Starlette CORS, Prometheus Client.
* **LLM / VLM / Embedding Models**: Pre-warms BGE-M3 and MS-MARCO MiniLM.

#### Files in `backend/app/api/routers/`:

##### [`backend/app/api/routers/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/__init__.py)
* **What It Has**: Sub-package initialization.
* **Work It Is Doing**: Marks routers directory as a Python package.

##### [`backend/app/api/routers/articles.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/articles.py)
* **What It Has**: 
  - Routes: `GET /articles`, `GET /articles/{id}`, `GET /articles/{id}/chunks`, `POST /photos/{photo_id}/analyze`.
* **Work It Is Doing**:
  - Provides article content retrieval, pagination, category filtering, and sub-chunk inspection.
  - `POST /photos/{photo_id}/analyze`: Implements on-demand VLM scene analysis. Downloads the photo crop from MinIO, executes `VisualDataExtractor.process_image_crop()`, extracts descriptions and markdown tables, and persists them into MySQL `photos.vlm_description`.
* **Important Tools / Frameworks**: FastAPI, SQLAlchemy Async, MinIO Python Client, Pydantic.
* **LLM / VLM / Embedding Models**: Bound to `visual_extraction` (Qwen-VL / `qwen3-vl:latest` or `qwen2.5vl:7b`).

##### [`backend/app/api/routers/health.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/health.py)
* **What It Has**: 
  - Routes: `GET /health`, `GET /health/detailed`.
* **Work It Is Doing**:
  - Real-time health monitoring of all platform dependencies.
  - Executes probe queries against MySQL (`SELECT 1`), Qdrant (`client.get_collections()`), Redis (`client.ping()`), MinIO (`client.bucket_exists()`), and active LLM providers.
* **Important Tools / Frameworks**: FastAPI, SQLAlchemy, Redis, Qdrant Client.
* **LLM / VLM / Embedding Models**: Probes active LLMs for responsiveness.

##### [`backend/app/api/routers/ingest.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/ingest.py)
* **What It Has**: 
  - Routes: `POST /ingest/upload`, `GET /ingest/jobs`, `GET /ingest/jobs/{job_id}`, `DELETE /ingest/issues/{issue_id}`.
* **Work It Is Doing**:
  - Accepts PDF broadsheet file uploads via `UploadFile`.
  - Calculates SHA256 checksums to detect duplicate uploads.
  - Dispatches background asynchronous processing via Celery (`process_issue_ingestion_task.delay()`).
  - Implements atomic cascading deletion via `DeletionService`, wiping relational rows, MinIO image objects, and Qdrant points simultaneously.
* **Important Tools / Frameworks**: FastAPI `UploadFile`, Celery task dispatch, DeletionService.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/api/routers/metadata.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/metadata.py)
* **What It Has**: 
  - Routes: `GET /metadata/entities`, `GET /metadata/topics`, `GET /metadata/entity-graph`.
* **Work It Is Doing**:
  - Computes entity mention frequencies and salience distributions.
  - Aggregates multi-article entity co-occurrences and transforms them into an interactive network graph (nodes & edges) for visualization.
* **Important Tools / Frameworks**: FastAPI, SQLAlchemy aggregation queries and group-by clauses.
* **LLM / VLM / Embedding Models**: None (Relational Graph Aggregation).

##### [`backend/app/api/routers/models.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/models.py)
* **What It Has**: 
  - Route: `GET /models/available`.
* **Work It Is Doing**: Inspects the active `ModelRegistry`, reporting all configured models, provider types (Ollama, Groq, Gemini, OpenAI), context windows, pricing, and active task bindings to the frontend client.
* **Important Tools / Frameworks**: FastAPI, ModelRegistry.
* **LLM / VLM / Embedding Models**: Queries metadata across all configured models.

##### [`backend/app/api/routers/newspapers.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/newspapers.py)
* **What It Has**: 
  - Routes: `GET /newspapers`, `POST /newspapers`, `GET /issues`, `GET /issues/{issue_id}`, `GET /pages/{page_id}/image`, `GET /issues/{issue_id}/inspect`.
* **Work It Is Doing**:
  - Manages publications, editions, and issues.
  - Streams high-resolution 300 DPI page images directly from storage.
  - `GET /issues/{issue_id}/inspect`: Exposes the deep layout debug payload (Docling 2D bounding boxes, raw OCR text blocks, article segment boundaries) for developer inspection.
* **Important Tools / Frameworks**: FastAPI `FileResponse`, SQLAlchemy 2.0 async select with eager joins.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/api/routers/query.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/query.py)
* **What It Has**: 
  - Routes: `POST /query`, `POST /query/stream`, `POST /query/plan`, `POST /query/timeline`, `GET /query/history`.
  - Pydantic models: `QueryRequest`, `QueryResponse`, `TimelineQueryRequest`.
* **Work It Is Doing**:
  - Main user query execution endpoint.
  - `POST /query/stream`: Pushes real-time Server-Sent Events (SSE) including pipeline stages (`stage`), internal thoughts (`thought`), narrative tokens (`token`), citation objects with bounding boxes (`citations`), and cost metrics (`done`).
  - `POST /query/plan`: Inspects the Query Planner Chain-of-Thought and scheduled tools without executing generation.
* **Important Tools / Frameworks**: FastAPI `StreamingResponse`, SSE Protocol (`text/event-stream`), AgentGraph.
* **LLM / VLM / Embedding Models**: Coordinates the full cognitive agent stack.

##### [`backend/app/api/routers/settings.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/api/routers/settings.py)
* **What It Has**: 
  - Routes: `GET /settings/bindings`, `POST /settings/bindings`, `POST /settings/bindings/reset`.
  - Pydantic models: `UpdateBindingsRequest`.
* **Work It Is Doing**:
  - Provides runtime task binding configuration.
  - Allows the user to switch active models (e.g. re-binding `query_planner` or `answerer` from `ollama_gemma4_12b` to `groq_llama` or `gemini_flash`) dynamically without restarting the server.
* **Important Tools / Frameworks**: FastAPI, ModelRegistry update methods.
* **LLM / VLM / Embedding Models**: Controls model routing for all tasks.

---

### 4.3 `backend/app/core/` — System Primitives, Governance & Rulebooks
* **Purpose / Reason**: Establishes global system primitives, operational telemetry, financial governance, structured logging, and newspaper-specific heuristics.
* **Work It Is Doing**: Validates environment variables and YAML settings, logs structured JSON traces, tracks token usage and financial cost in USD, exports Prometheus metrics, and maintains canonical alias mappings.

#### Files in `backend/app/core/`:

##### [`backend/app/core/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/core/__init__.py)
* **What It Has**: Package initialization.
* **Work It Is Doing**: Marks core as a Python package.

##### [`backend/app/core/config.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/core/config.py)
* **What It Has**: `Settings` class (Pydantic BaseSettings), `DatabaseSettings`, `QdrantSettings`, `MinioSettings`, `RedisSettings`, `CelerySettings`, `ModelConfig`, `get_settings()`.
* **Work It Is Doing**: Loads environment variables from `.env` and merges them with `model_config.yaml`. Enforces strict type validation on database connection URLs, API keys, and port numbers.
* **Important Tools / Frameworks**: Pydantic Settings v2, PyYAML, `lru_cache`.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/core/cost_tracker.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/core/cost_tracker.py)
* **What It Has**: `ModelPrice`, `BudgetExceededError`, `calculate_cost_usd()`, `record_usage_and_cost()`, `validate_query_budget()`.
* **Work It Is Doing**:
  - Maintains exact input/output pricing tables per million tokens for commercial cloud models (assigning $0.00 to local Ollama models).
  - Enforces per-query budget caps (e.g. $\le \$0.05$ per query) to prevent runaway recursive API billing.
* **Important Tools / Frameworks**: Python Decimal / Float math, Prometheus metric counters.
* **LLM / VLM / Embedding Models**: Tracks costs for all commercial cloud models.

##### [`backend/app/core/logging.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/core/logging.py)
* **What It Has**: `_NewsLensFormatter`, `setup_logging()`, `get_logger()`.
* **Work It Is Doing**: Configures unified structured logging across all processes (API, Celery workers, retrieval). Formats log records with timestamps, log levels, correlation IDs, module names, and custom extra attributes.
* **Important Tools / Frameworks**: Python `logging`, JSON formatting.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/core/metrics.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/core/metrics.py)
* **What It Has**: `PrometheusMiddleware`, `generate_prometheus_metrics()`, Prometheus Counters, Histograms, and Gauges (`agent_query_latency_seconds`, `llm_tokens_total`, `llm_cost_usd_total`, `cache_events_total`).
* **Work It Is Doing**: Records real-time performance metrics for every HTTP request, tool execution, LLM call, and cache event. Exposes standard Prometheus `/metrics` scraping format.
* **Important Tools / Frameworks**: `prometheus_client`, Starlette BaseHTTPMiddleware.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/core/category_aliases.yaml`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/core/category_aliases.yaml)
* **What It Has**: Canonical category taxonomy (`Front Page`, `National & Politics`, `Business & Markets`, `Corporate & Industry`, `Sports`, `Opinion & Editorial`, `World & Geopolitics`, `Science & Environment`, `Lifestyle`, `Arts & Culture`), category keywords, and user query synonyms.
* **Work It Is Doing**: Normalizes diverse broadsheet section labels (e.g. "Economy", "Markets", "Rupee & Bullion", "Companies" $\to$ "Business & Markets") during article segmentation and relational SQL filtering.
* **Important Tools / Frameworks**: YAML 1.2.
* **LLM / VLM / Embedding Models**: None (Deterministic Taxonomy).

##### [`backend/app/core/newspaper_rules.yaml`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/core/newspaper_rules.yaml)
* **What It Has**: Newspaper-specific layout heuristics for *The Economic Times*, *Mint*, *The Hindu*, *The Indian Express*, *The Goan*, *The Morning Standard*, *The Guardian*, *The New York Times*, *Business Standard*.
* **Work It Is Doing**: Defines masthead bounding box zones, date header coordinates, font size thresholds, and known advertising envelope patterns unique to each publication.
* **Important Tools / Frameworks**: YAML 1.2.
* **LLM / VLM / Embedding Models**: None (Broadsheet Rules).

---

### 4.4 `backend/app/evaluation/` — Information Retrieval Benchmarks
* **Purpose / Reason**: Provides quantitative evaluation of retrieval accuracy and citation precision against ground-truth benchmarks.
* **Work It Is Doing**: Computes standard information retrieval (IR) metrics to benchmark changes in embedding models, top-k parameters, and reranking thresholds.

#### Files in `backend/app/evaluation/`:

##### [`backend/app/evaluation/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/evaluation/__init__.py)
* **What It Has**: Package initialization.
* **Work It Is Doing**: Marks evaluation as a Python package.

##### [`backend/app/evaluation/metrics.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/evaluation/metrics.py)
* **What It Has**: `compute_recall_at_k()`, `compute_precision_at_k()`, `compute_mrr()`, `compute_ndcg_at_k()`, `compute_citation_precision()`.
* **Work It Is Doing**:
  - Evaluates retrieval quality by comparing retrieved article IDs against ground-truth relevance sets.
  - Implements Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG@K).
  - Verifies that $100\%$ of synthesized citations in brackets correspond to actual relational articles in MySQL.
* **Important Tools / Frameworks**: Python Math, NumPy.
* **LLM / VLM / Embedding Models**: None.

---

### 4.5 `backend/app/ingestion/` — 12-Stage Broadsheet Processing Pipeline
* **Purpose / Reason**: The industrial-grade newspaper parsing, layout decomposition, OCR, visual extraction, and indexing engine.
* **Work It Is Doing**: Takes raw, unstructured multi-megabyte broadsheet PDF issues and turns them into high-resolution page renders, column-ordered text, structured cross-page articles, transcribed infographics, photo scene descriptions, and vectorized chunks.

#### Files in `backend/app/ingestion/`:

##### [`backend/app/ingestion/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/__init__.py)
* **What It Has**: Ingestion package initialization.
* **Work It Is Doing**: Exports pipeline components.

##### [`backend/app/ingestion/celery_app.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/celery_app.py)
* **What It Has**: Celery app instance, broker configuration (`redis://localhost:6379/0`), result backend, task serializer settings.
* **Work It Is Doing**: Configures the asynchronous task queue for distributed background ingestion.
* **Important Tools / Frameworks**: Celery, Redis.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/tasks.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/tasks.py)
* **What It Has**: Celery tasks: `process_issue_ingestion_task()`, `run_ingestion_pipeline()`, `detect_masthead_and_date()`, `check_is_advertisement_text()`.
* **Work It Is Doing**:
  - The master pipeline coordinator orchestrating the 12 stages in strict sequence:
    1. Rasterization & Metadata Intake
    2. Docling 2D Layout & Vision OCR
    3. Masthead & Publication Date Extraction
    4. Reading Order & Column Geometry
    5. Article Segmentation & Kicker Extraction
    6. Cross-Page Continuation Assembly
    7. Advertisement Filtering
    8. LLM Enrichment & Topic Tagging
    9. Qwen-VL Visual Extraction & OCR Cross-Validation
    10. Sub-document Chunking with Context Headers
    11. BGE-M3 Dense Vector Indexing
    12. Debug Artifacts Export
  - Handles database transaction commits, rollback on error, and job status updates in MySQL.
* **Important Tools / Frameworks**: Celery task decorator (`@celery_app.task`), SQLAlchemy Async, Python AsyncIO bridge.
* **LLM / VLM / Embedding Models**: Orchestrates calls across Google Cloud Vision, Qwen-VL, Gemma4-12B, and BAAI/bge-m3.

##### [`backend/app/ingestion/rasterizer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/rasterizer.py)
* **What It Has**: `PDFRasterizer` class, `RasterizedPage` dataclass.
* **Work It Is Doing**: Renders 300 DPI high-resolution PNG page images from PDF broadsheets using PyMuPDF (yielding ~8,188 x 11,400 px images). Saves images to disk and MinIO.
* **Important Tools / Frameworks**: PyMuPDF (`fitz`), Pillow (PIL).
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/docling_parser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/docling_parser.py)
* **What It Has**: `DoclingLayoutParser` class, `DoclingParsedItem` dataclass, `CorruptedPdfTextLayerError`.
* **Work It Is Doing**:
  - Executes deep 2D layout analysis on pages using DocLayNet models.
  - Detects corrupted font CMap ligatures (e.g. `�` replacement character ratios >= 3%) and raises `CorruptedPdfTextLayerError` to trigger pure OCR fallback.
  - Normalizes bounding boxes and outputs structured layout element tokens (`title`, `section_header`, `text`, `caption`, `picture`, `table`).
* **Important Tools / Frameworks**: Docling library, PyMuPDF, NumPy.
* **LLM / VLM / Embedding Models**: DocLayNet Layout Analysis Models.

##### [`backend/app/ingestion/ocr_service.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/ocr_service.py)
* **What It Has**: `OCRService` class.
* **Work It Is Doing**: Dispatches cropped image regions to Google Cloud Vision API (`google-cloud-vision`) or local Tesseract OCR, performing contrast enhancement and deskewing.
* **Important Tools / Frameworks**: Google Cloud Vision SDK, PyTesseract, PIL ImageEnhance.
* **LLM / VLM / Embedding Models**: Google Cloud Vision Document Text Detection.

##### [`backend/app/ingestion/reading_order.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/reading_order.py)
* **What It Has**: `ReadingOrderResolver` class, `OrderedReadingBlock`, `BlockType`.
* **Work It Is Doing**: Solves multi-column broadsheet flow using XY-cut geometric clustering, preventing the fatal flaw of reading horizontally across vertical columns.
* **Important Tools / Frameworks**: Computational Geometry, Interval Math.
* **LLM / VLM / Embedding Models**: None (Geometric Algorithm).

##### [`backend/app/ingestion/segmenter.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/segmenter.py)
* **What It Has**: `ArticleSegmenter` class, `SegmentedArticle`, `extract_kicker_and_clean_headline()`, `is_valid_headline_candidate()`.
* **Work It Is Doing**: Groups titles, kickers, subheadlines, bylines, and narrative body paragraphs into distinct candidate articles on a single page. Filters syndication agency slugs (PTI, Reuters, ANI) and numbered feature subheads.
* **Important Tools / Frameworks**: Regular Expressions, Statistical heuristics on font size and line spacing.
* **LLM / VLM / Embedding Models**: None (Deterministic Rule-Based Parser).

##### [`backend/app/ingestion/cross_page_assembler.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/cross_page_assembler.py)
* **What It Has**: `CrossPageAssembler` class, `AssembledArticle`, `_headline_overlap_metrics()`, `_has_continuation_marker()`.
* **Work It Is Doing**: Detects jump-line markers (e.g. "Continued on Page 9", "From Page 1") and links split article blocks across different pages into unified database records with multi-page coordinate mapping.
* **Important Tools / Frameworks**: SequenceMatcher (fuzzy string matching), Regex.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/visual_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/visual_extractor.py)
* **What It Has**: `VisualDataExtractor` class, `VisualClassification`, `VisualExtractionResult`, `repair_and_parse_json()`, `cross_validate_with_ocr()`.
* **Work It Is Doing**:
  - Implements the 3-Stage Visual Intelligence Pipeline.
  - Stage 1: Triage Gate classifying into `data_chart`, `table`, `infographic`, `photo`, `logo`, `decorative`.
  - Stage 2: Prompts Qwen-VL with `STRUCTURED_EXTRACTION_PROMPT` to transcribe complex charts/infographics into executive summaries, GitHub-flavored Markdown tables, and key metrics.
  - Stage 3: Numerical Cross-Validation verifying extracted numbers against OCR tokens; triggers deterministic spatial OCR fallback if match ratio < 0.40.
* **Important Tools / Frameworks**: Qwen-VL Vision Provider, PyTesseract, PIL, JSON Repair algorithms.
* **LLM / VLM / Embedding Models**: Bound to `visual_extraction` (`qwen3-vl:latest` or `qwen2.5vl:7b`).

##### [`backend/app/ingestion/media_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/media_extractor.py)
* **What It Has**: `MediaExtractor` class, `extract_grounded_boxes_from_thinking()`, `parse_grounded_boxes()`.
* **Work It Is Doing**:
  - Intercepts Qwen-VL's native `<think>...</think>` spatial reasoning stream to extract coordinates `[xmin, ymin, xmax, ymax]` scaled to 0..1000 and converts them to pixel bounding boxes.
  - Crops photo regions, uploads them to MinIO, and records entries in MySQL `photos` and `tables`.
* **Important Tools / Frameworks**: PIL Image, MinIO Client, Regular Expressions.
* **LLM / VLM / Embedding Models**: Qwen-VL (`ollama_qwen3vl`).

##### [`backend/app/ingestion/chunker.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/chunker.py)
* **What It Has**: `NewspaperChunker` class, `DocumentChunk` dataclass.
* **Work It Is Doing**:
  - Segments article full text into retrieval chunks (target: 300 tokens, overlap: 50 tokens).
  - Prepends every chunk with a rich semantic header: `[Newspaper: {name} | Date: {YYYY-MM-DD} | Section: {sec} | Headline: {hl} | Page(s): {folios}]`.
  - Creates dedicated visual chunks (`chunk_type="visual"`) for transcribed charts and tables.
* **Important Tools / Frameworks**: TikToken / Token-based length estimators, Recursive Character Splitting.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/embedder.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/embedder.py)
* **What It Has**: `ArticleEmbedder` class.
* **Work It Is Doing**: Vectorizes sub-document chunks using the registered embedding provider (`local_embed_bge` / `BAAI/bge-m3`), creates 1024-dimensional vectors, and upserts them into Qdrant collection `newslens_articles`.
* **Important Tools / Frameworks**: Qdrant Async Client, LocalEmbeddingProvider (SentenceTransformers).
* **LLM / VLM / Embedding Models**: `BAAI/bge-m3` (1024 dimensions) or OpenAI `text-embedding-3-large`.

##### [`backend/app/ingestion/classifier.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/classifier.py)
* **What It Has**: `ArticleClassifier` class, `ClassificationResult`.
* **Work It Is Doing**: Analyzes article headlines and excerpts using LLM to assign canonical category IDs (e.g. `Business & Markets`, `Politics`) and confidence scores.
* **Important Tools / Frameworks**: Pydantic, Structured Prompts.
* **LLM / VLM / Embedding Models**: `ollama_gemma4_12b` (Ollama) or `gpt-4o-mini`.

##### [`backend/app/ingestion/metadata_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/metadata_extractor.py)
* **What It Has**: `MetadataExtractor` class, `ExtractedEntity`, `ExtractedTopic`, `ArticleMetadataResult`.
* **Work It Is Doing**: Extracts named entities (persons, organizations, locations), topical tags, and salience scores ($0.0$ to $1.0$) for every article.
* **Important Tools / Frameworks**: Pydantic, Structured Outputs.
* **LLM / VLM / Embedding Models**: Bound to `metadata_extraction` (`gemma4:12b` or `gemini_flash`).

##### [`backend/app/ingestion/folio_detector.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/folio_detector.py)
* **What It Has**: `FolioDetector` class, `strip_dates_and_metadata()`.
* **Work It Is Doing**: Scans the top and bottom margins of broadsheet pages to locate and parse true printed folio strings (e.g. "Page 9", "Page 10") distinguishing them from physical PDF page indices.
* **Important Tools / Frameworks**: Regex patterns, Coordinate geometry.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/masthead_verifier.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/masthead_verifier.py)
* **What It Has**: `MastheadVerifier` class.
* **Work It Is Doing**: Inspects the Page 1 header banner to detect the publication name (e.g. *The Economic Times*, *The Hindu*, *The Goan*) and exact issue date string.
* **Important Tools / Frameworks**: Fuzzy string matching, Date parsing (`datetime`).
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/geometry.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/geometry.py)
* **What It Has**: `BBox` dataclass, spatial operations (`intersects`, `contains`, `iou`, `scale`, `to_dict`).
* **Work It Is Doing**: Implements fundamental 2D geometric operations for bounding boxes `[x0, y0, x1, y1]`, coordinate transforms, and overlap ratios.
* **Important Tools / Frameworks**: Pure Python Math.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/debug_exporter.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/debug_exporter.py)
* **What It Has**: `DebugArtifactsExporter` class.
* **Work It Is Doing**: Exports 5 structured JSON debug files for every ingested issue:
  1. `articles_manifest.json` (all articles and metadata)
  2. `rag_chunks.json` (all vector chunks and headers)
  3. `ocr_extracted_text.json` (complete OCR token stream)
  4. `identified_advertisements.json` (detected ad envelopes)
  5. `ingestion_summary.json` (timing and stage metrics)
* **Important Tools / Frameworks**: JSON serialization, Pathlib.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/deletion_service.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/deletion_service.py)
* **What It Has**: `DeletionService` class.
* **Work It Is Doing**: Handles safe, atomic cascading deletion of issues across all storage tiers: removes MySQL relational rows, deletes Qdrant vector points, purges MinIO page images and photo crops, and invalidates Redis cache keys.
* **Important Tools / Frameworks**: SQLAlchemy AsyncSession, Qdrant Client, MinIO Client, Redis.
* **LLM / VLM / Embedding Models**: None.

##### Other Ingestion Support Modules:
- [`backend/app/ingestion/detector.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/detector.py): Fast heuristic page triage detector (identifies digital vs scanned pages, drop caps, and text noise).
- [`backend/app/ingestion/compressor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/compressor.py): Downsamples heavy high-res PDF pages when memory limits are constrained.
- [`backend/app/ingestion/intake.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/intake.py): Validates uploaded files, checks MIME types, verifies PDF headers, and creates `IngestionJob` tracking records.
- [`backend/app/ingestion/layout_analyzer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/layout_analyzer.py): Detects advertisement bounding boxes, syndication slugs, and table of contents index blocks.
- [`backend/app/ingestion/page_reingestion.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/page_reingestion.py): Re-runs pipeline stages over single pages without re-processing entire multi-page issues.
- [`backend/app/ingestion/unified_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/unified_extractor.py): Unified extractor combining DocLayNet layout items and OCR spans.
- [`backend/app/ingestion/consensus_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/consensus_extractor.py): Cross-validates publication dates across multiple pages to achieve consensus.
- [`backend/app/ingestion/extraction_schemas.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/extraction_schemas.py): Pydantic schemas for intermediate pipeline artifacts (`ArticleSkeleton`, `PageLayoutExtraction`, `ExtractedTable`).

---

### 4.6 `backend/app/models/` — Relational SQLAlchemy 2.0 Async Schemas
* **Purpose / Reason**: Defines the relational database schemas, relationships, constraints, and indexes for the NewsLens-AI system of record in MySQL.
* **Work It Is Doing**: Provides type-safe async ORM access, enforces referential integrity through foreign key cascades, and sets up high-performance indexes.

#### Files in `backend/app/models/`:

##### [`backend/app/models/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/models/__init__.py)
* **What It Has**: Package initialization, re-exporting all ORM models.
* **Work It Is Doing**: Ensures all models are imported so Alembic and SQLAlchemy metadata can discover all tables.

##### [`backend/app/models/base.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/models/base.py)
* **What It Has**: `Base` (DeclarativeBase), `init_db()`, `get_session_factory()`, `get_db()`, `close_db()`.
* **Work It Is Doing**: Initializes the asynchronous SQLAlchemy engine (`create_async_engine`) with connection pooling, statement caching, and provides FastAPI dependency injection for `AsyncSession`.
* **Important Tools / Frameworks**: SQLAlchemy 2.0 Async, `aiomysql`.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/models/newspaper.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/models/newspaper.py)
* **What It Has**: `Newspaper`, `Issue`, `Page` ORM models.
* **Work It Is Doing**:
  - `newspapers`: Stores newspaper metadata (name, code, country, language, publisher).
  - `issues`: Stores publication issues (date, edition, total pages, status, total articles count).
  - `pages`: Stores individual broadsheet pages (page number, printed folio, width, height, image path).
* **Important Tools / Frameworks**: SQLAlchemy Mapped, ForeignKey, Relationships.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/models/article.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/models/article.py)
* **What It Has**: `Article`, `ArticlePage`, `ArticleChunk`, `Photo`, `ArticleTable`, `ArticleCategory` ORM models.
* **Work It Is Doing**:
  - `articles`: Primary article table (headline, subheadline, byline, section, prominence score, full text).
  - `article_pages`: Junction table mapping articles to pages with exact bounding box JSON arrays.
  - `article_chunks`: Sub-document chunks with token counts and Qdrant vector UUID string pointers.
  - `photos`: Cropped visual elements, captions, MinIO object keys, visual types, and VLM descriptions.
* **Important Tools / Frameworks**: MySQL LONGTEXT, JSON columns, Enum types.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/models/entity.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/models/entity.py)
* **What It Has**: `Entity`, `ArticleEntity`, `Topic`, `ArticleTopic`, `Event` ORM models.
* **Work It Is Doing**: Relational knowledge graph storing named entities (`person`, `org`, `location`, `misc`), mention counts, and salience scores.
* **Important Tools / Frameworks**: Many-to-many junction tables with cascade deletes.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/models/ingestion.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/models/ingestion.py)
* **What It Has**: `IngestionJob` ORM model.
* **Work It Is Doing**: Tracks background Celery ingestion jobs, file paths, progress percentages, and error stack traces.
* **Important Tools / Frameworks**: SQLAlchemy Mapped.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/models/query.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/models/query.py)
* **What It Has**: `QueryLog` ORM model.
* **Work It Is Doing**: Audits every user query, execution time, token usage, cost in USD, and retrieved citation IDs.
* **Important Tools / Frameworks**: SQLAlchemy Mapped.
* **LLM / VLM / Embedding Models**: None.

---

### 4.7 `backend/app/providers/` — Dynamic Model Provider Tier
* **Purpose / Reason**: Hardware-agnostic abstraction layer isolating model vendors and hosting environments from core application logic.
* **Work It Is Doing**: Standardizes chat completions, vision understanding, and embedding generation across Ollama, Groq, Google Gemini, OpenAI, Anthropic, OpenRouter, and local SentenceTransformers.

#### Files in `backend/app/providers/`:

##### [`backend/app/providers/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/__init__.py)
* **What It Has**: Package initialization.
* **Work It Is Doing**: Re-exports `ModelRegistry`, `get_registry()`, base classes.

##### [`backend/app/providers/base.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/base.py)
* **What It Has**: `ChatModelProvider`, `VisionModelProvider`, `EmbeddingModelProvider`, `Message`, `ToolDefinition`, `ProviderType`, `ProviderCapability`.
* **Work It Is Doing**: Abstract Base Classes defining unified contracts for `chat_complete()`, `chat_stream()`, `analyze_image()`, and `embed_text()`.
* **Important Tools / Frameworks**: Python `abc`, Pydantic.
* **LLM / VLM / Embedding Models**: Interface definition for all models.

##### [`backend/app/providers/registry.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/registry.py)
* **What It Has**: `ModelRegistry` class, `get_registry()`, `reset_registry()`.
* **Work It Is Doing**: Reads `model_config.yaml`, instantiates concrete provider classes, and resolves task bindings (`get_provider("query_planner")`). Supports dynamic runtime updates.
* **Important Tools / Frameworks**: Singleton pattern, YAML parsing.
* **LLM / VLM / Embedding Models**: Manages the complete lifecycle of all configured models.

##### Concrete Provider Implementations:
- [`backend/app/providers/ollama_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/ollama_provider.py): Local inference via Ollama HTTP API (`/api/chat`, `/api/generate`, `/api/embeddings`). Supports Qwen-VL, Gemma4-12B, Nemotron, Llama 3.1.
- [`backend/app/providers/groq_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/groq_provider.py): Ultra-low-latency LPU inference via Groq SDK (`llama-3.3-70b-versatile`, `qwen-2.5-32b`).
- [`backend/app/providers/gemini_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/gemini_provider.py): Google Gemini API (`gemini-3.7-flash`, `gemini-pro-latest`) supporting multimodal vision and structured JSON outputs.
- [`backend/app/providers/google_vision_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/google_vision_provider.py): Google Cloud Vision API integration for OCR and document text detection.
- [`backend/app/providers/openai_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/openai_provider.py): OpenAI API (`gpt-4o`, `gpt-4o-mini`, `text-embedding-3-large`).
- [`backend/app/providers/nvidia_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/nvidia_provider.py): NVIDIA NIM hosted inference via OpenAI-compatible endpoint (`https://integrate.api.nvidia.com/v1`). Supports `nvidia/nemotron-3.5-lightning-30b-a3b` with progressive `<think>` reasoning streaming and `meta/llama-3.2-11b-vision-instruct` for multimodal vision.
- [`backend/app/providers/openrouter_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/openrouter_provider.py): OpenRouter unified gateway with multi-key rotation and rate-limit handling.
- [`backend/app/providers/anthropic_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/anthropic_provider.py): Anthropic Claude API (`claude-3-5-sonnet`).
- [`backend/app/providers/local_embedding_provider.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/local_embedding_provider.py): PyTorch & SentenceTransformers client for local `BAAI/bge-m3` embedding computation.
- [`backend/app/providers/cascade.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/providers/cascade.py): Automatic fallback cascade (e.g. Groq -> Ollama -> Gemini) handling rate limits and timeouts transparently.

---

### 4.8 `backend/app/retrieval/` — Multi-Tool Search, Reranking & Auditing
* **Purpose / Reason**: Houses the specialized retrieval engines called by the Query Planner to gather grounded evidence.
* **Work It Is Doing**: Executes dense vector search, sparse keyword search, relational aggregation queries, entity traversal, timeline generation, negative coverage auditing, and neural cross-attention reranking.

#### Files in `backend/app/retrieval/`:

##### [`backend/app/retrieval/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/__init__.py)
* **What It Has**: Package initialization, re-exports for `HybridSearchEngine`, `SQLAnalyticsEngine`, `CoverageAnalyzer`, `TimelineBuilder`, `EntitySearchEngine`, `CrossEncoderReranker`, and `repair_text_ligatures`.

##### [`backend/app/retrieval/sanitizer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/sanitizer.py)
* **What It Has**: `repair_text_ligatures()`, `_LIGATURE_REPLACEMENTS`, `_REGEX_LIGATURE_REPAIRS`.
* **Work It Is Doing**:
  - Decomposes typographic Unicode ligatures (`\ufb00` = `ff`, `\ufb01` = `fi`, `\ufb02` = `fl`, `\ufb03` = `ffi`, `\ufb04` = `ffl`, `\ufb05` = `ft`, `\ufb06` = `st`).
  - Repairs broadsheet OCR font dropout patterns where ligature glyphs were dropped or mapped to replacement characters or multiple spaces (e.g. `e \ufffd orts` / `e   orts` $\to$ `efforts`, `in \ufffd ation` $\to$ `inflation`, `di \ufffd erent` $\to$ `different`, `sta\ufffd` $\to$ `staff`, `o\ufffd cial` $\to$ `official`).
* **Important Tools / Frameworks**: Regular Expressions (`re`).
* **LLM / VLM / Embedding Models**: None (Deterministic Typographic Text Repair).

##### [`backend/app/retrieval/hybrid_search.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/hybrid_search.py)
* **What It Has**: `HybridSearchEngine` class, `SearchFilter`, `HybridSearchResult`.
* **Work It Is Doing**:
  - Implements dense vector search (Qdrant) and sparse keyword search (MySQL FULLTEXT).
  - Merges results using Reciprocal Rank Fusion ($k=60$):
    RRF Score = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)
  - Passes the top candidate pool (up to 75 items) through the neural Cross-Encoder reranker.
  - **Category Post-Filtering**: Filters candidates across `category.name`, `section`, and `printed_section` with dynamic candidate pool expansion (`max(50, top_k * 4)`).
  - Retrieves visual chunks (`has_visual_data=True`) containing Markdown tables transcribed by Qwen-VL.
* **Important Tools / Frameworks**: Qdrant Async, MySQL FULLTEXT, CrossEncoderReranker.
* **LLM / VLM / Embedding Models**: `BAAI/bge-m3` (dense embeddings), `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker).

##### [`backend/app/retrieval/sql_analytics.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/sql_analytics.py)
* **What It Has**: `SQLAnalyticsEngine` class, `sanitize_headline()`, `get_archive_metadata()`.
* **Work It Is Doing**:
  - Executes deterministic, parameterized SQL aggregation queries.
  - `get_archive_metadata()`: Fast cached (<20ms) extraction of available issue dates, active newspapers, and canonical database categories.
  - `sanitize_headline()`: Cleanses headlines where doctor/author profile names were mistakenly extracted as the headline, preserving genuine all-caps headlines and bylines.
  - `get_issue_summary()`: Retrieves the full article manifest for a newspaper issue with economic domain bridging (`Business & Markets` + `Economy & Policy`).
  - `get_newspaper_coverage_difference()`: Computes verified exclusive articles between two publications on a given date (e.g. The Goan vs The Morning Standard).
  - `get_entity_mention_trends()`: Computes monthly/daily mention trajectories.
* **Important Tools / Frameworks**: SQLAlchemy Core & ORM async select queries.
* **LLM / VLM / Embedding Models**: None (Deterministic Relational Grounding).

##### [`backend/app/retrieval/reranker.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/reranker.py)
* **What It Has**: `CrossEncoderReranker`, `HeuristicReranker`, `_detect_best_device()`.
* **Work It Is Doing**: Computes full cross-attention interaction scores between query and candidate snippets on MPS (Apple Silicon), CUDA, or CPU.
* **Important Tools / Frameworks**: `sentence_transformers.CrossEncoder`, PyTorch.
* **LLM / VLM / Embedding Models**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

##### [`backend/app/retrieval/coverage_analyzer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/coverage_analyzer.py)
* **What It Has**: `CoverageStatus`, `PublicationCoverageReport`, `CoverageMatrix`, `CoverageAnalyzer`.
* **Work It Is Doing**: Enforces the Coverage Invariant via a 3-tier audit. Scopes negative audits strictly to publications with active issues on the target date, and uses calibrated cross-encoder logit thresholds (`>= -5.0`) to avoid false `PROCESSING_ERROR` classifications.
* **Important Tools / Frameworks**: SQLAlchemy AsyncSession, HybridSearchEngine.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/retrieval/entity_filter.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/entity_filter.py)
* **What It Has**: `EntitySearchResult`, `EntitySearchEngine`.
* **Work It Is Doing**: Searches articles by entity name, entity type, and salience score threshold.
* **Important Tools / Frameworks**: SQLAlchemy AsyncSession.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/retrieval/timeline_builder.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/timeline_builder.py)
* **What It Has**: `NewspaperPerspective`, `TimelineMilestone`, `NarrativeTrajectoryResponse`, `TimelineBuilder`.
* **Work It Is Doing**: Reconstructs chronological event progression across editions and dates.
* **Important Tools / Frameworks**: SQLAlchemy AsyncSession, Date grouping.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/retrieval/web_search.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/web_search.py)
* **What It Has**: `WebSearchResult`, `WebSearchEngine`.
* **Work It Is Doing**: DuckDuckGo live web search fallback for recent/unarchived events.
* **Important Tools / Frameworks**: `duckduckgo_search` library.
* **LLM / VLM / Embedding Models**: None.

---

### 4.9 `backend/app/storage/` — Multi-Tier Persistence Clients
* **Purpose / Reason**: Unified data access layer encapsulating interactions with MySQL, Qdrant, MinIO, and Redis.
* **Work It Is Doing**: Manages connection pooling, index creation, vector similarity search, object storage uploads/downloads, and query cache keys.

#### Files in `backend/app/storage/`:

##### [`backend/app/storage/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/storage/__init__.py)
* **What It Has**: Package initialization, re-exports for `QdrantStore`, `MinioStore`, `CacheStore`, `MySQLFullTextSearch`.

##### [`backend/app/storage/base.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/storage/base.py)
* **What It Has**: `VectorPoint`, `VectorSearchResult`, `FullTextSearchResult`, `VectorStore`, `ObjectStore` abstract protocols.
* **Work It Is Doing**: Defines interfaces for vector stores and object storage.

##### [`backend/app/storage/qdrant_store.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/storage/qdrant_store.py)
* **What It Has**: `QdrantStore` class.
* **Work It Is Doing**: Manages the `newslens_articles` Qdrant collection with 1024-dimension Cosine distance vectors. Executes filtered similarity queries matching publication, date, section, and entity filters.
* **Important Tools / Frameworks**: `qdrant_client.AsyncQdrantClient`.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/storage/mysql_fulltext.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/storage/mysql_fulltext.py)
* **What It Has**: `MySQLFullTextSearch` class.
* **Work It Is Doing**: Executes boolean mode FULLTEXT search on MySQL `articles` table using `MATCH(headline, full_text) AGAINST(:query IN BOOLEAN MODE)`.
* **Important Tools / Frameworks**: SQLAlchemy AsyncSession, MySQL FULLTEXT index.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/storage/minio_store.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/storage/minio_store.py)
* **What It Has**: `MinioStore` class.
* **Work It Is Doing**: Stores and retrieves raw newspaper PDFs, high-resolution 300 DPI page PNG renders, and cropped photo image assets in S3-compatible MinIO buckets.
* **Important Tools / Frameworks**: `miniopy_async` / MinIO Python Client.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/storage/cache_store.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/storage/cache_store.py)
* **What It Has**: `CacheStore` class, `compute_query_cache_key()`, `compute_embedding_cache_key()`.
* **Work It Is Doing**: Caches embeddings and identical query responses in Redis with configurable TTLs to eliminate redundant computation.
* **Important Tools / Frameworks**: `redis.asyncio`.
* **LLM / VLM / Embedding Models**: None.

---

## 5. Backend Test Suite (`backend/tests/`)

* **Purpose / Reason**: Guarantees system correctness, data integrity, anti-hallucination guardrails, and deterministic tool execution across releases.
* **Work It Is Doing**: Executes over 55 comprehensive pytest suites covering every subsystem.

#### Key Test Suites in `backend/tests/`:
- [`conftest.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/conftest.py): Pytest fixtures for async database sessions, mock model providers, and temporary test storage.
- [`test_planner.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_planner.py): Verifies archetype classification, typo tolerance, and tool argument extraction.
- [`test_condenser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_condenser.py): Validates multi-turn pronoun resolution and context isolation.
- [`test_graph.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_graph.py): Tests the LangGraph workflow, CRAG relevance gate, and fallback triggers.
- [`test_synthesizer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_synthesizer.py): Verifies 4-tier response formatting, `<think>` tag stripping, and 100% citation precision.
- [`test_docling_parser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_docling_parser.py): Tests 2D layout bounding box extraction and font CMap corruption detection.
- [`test_visual_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_visual_extractor.py): Tests Qwen-VL infographic transcription and OCR cross-validation.
- [`test_vlm_grounding.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_vlm_grounding.py): Tests coordinate parsing from Qwen-VL's native `<think>` stream.
- [`test_hybrid_search.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_hybrid_search.py): Tests dense/sparse RRF fusion and Cross-Encoder reranking.
- [`test_sql_analytics.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_sql_analytics.py): Tests issue manifests and cross-newspaper coverage difference counts.

---

## 6. Frontend Client Application (`frontend/`)

### 6.1 Frontend Root & Build Tooling
* **Purpose / Reason**: Houses build scripts, dependency manifests, and style pre-processors for the user-facing web application.
* **Work It Is Doing**: Bundles React components, compiles Tailwind CSS utilities, and proxies API requests to the backend.

#### Files in `frontend/`:
- [`package.json`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/package.json): React 18, Vite, Lucide React icons, Tailwind CSS, PostCSS.
- [`vite.config.js`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/vite.config.js): Local dev server port (5173), reverse proxy forwarding `/api` to `http://127.0.0.1:8000`.
- [`tailwind.config.js`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/tailwind.config.js): Custom broadsheet typography, dark-mode slate/emerald theme.
- [`postcss.config.js`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/postcss.config.js): PostCSS configuration.
- [`frontend/index.html`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/index.html): HTML entrypoint with viewport & fonts.
- [`frontend/src/App.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/App.jsx): Main layout with sticky header, navigation tabs, active model tier indicator, and active workspace switcher.
- [`frontend/src/main.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/main.jsx): React entrypoint rendering `<App />` within `ActiveHighlightProvider`.
- [`frontend/src/index.css`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/index.css): Global styles, custom scrollbars, broadsheet typography rules.

---

### 6.2 `frontend/src/context/` — Global Synchronized State

##### [`frontend/src/context/ActiveHighlightContext.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/context/ActiveHighlightContext.jsx)
* **What It Has**: `ActiveHighlightProvider`, `useActiveHighlight()` hook, state for active newspaper, selected issue, active page, highlighted article bounding boxes, and active citation focus.
* **Work It Is Doing**: Enables cross-component synchrony: clicking an inline citation in `AgentAssistant` automatically switches tabs to `BroadsheetReader`, zooms to the referenced page, and highlights the exact bounding box polygons on the canvas.

---

### 6.3 `frontend/src/components/` — Workspaces, Readers & Modals

##### [`frontend/src/components/AgentAssistant.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/AgentAssistant.jsx)
* **What It Has**: Conversational chat interface, SSE event listener, collapsible `<think>` reasoning accordion, inline citation pills, follow-up prompt pills, export buttons.
* **Work It Is Doing**: Consumes the `/api/query/stream` SSE wire protocol in real-time, rendering thinking steps and markdown responses progressively. Links citations directly to the broadsheet canvas.
* **Important Tools / Frameworks**: Fetch API EventStream reader, Lucide React, Markdown rendering.

##### [`frontend/src/components/BroadsheetReader.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/BroadsheetReader.jsx)
* **What It Has**: High-resolution pan/zoom canvas reader, page navigation carousel, article text inspection drawer.
* **Work It Is Doing**: Displays 300 DPI broadsheet pages with smooth zoom controls; overlays interactive SVG bounding boxes for articles, photos, and infographics.
* **Important Tools / Frameworks**: Canvas / SVG overlay rendering, CSS Transforms.

##### [`frontend/src/components/CanvasOverlay.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/CanvasOverlay.jsx)
* **What It Has**: Dynamic SVG overlay layer.
* **Work It Is Doing**: Maps normalized or pixel bounding boxes `[x0, y0, x1, y1]` over the rendered broadsheet page image, providing hover highlights and click selection.

##### [`frontend/src/components/ArchiveExplorer.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/ArchiveExplorer.jsx)
* **What It Has**: Multi-newspaper grid, issue calendar browser, page thumbnail gallery.
* **Work It Is Doing**: Allows users to explore ingested broadsheet editions across publications and dates, view issue metadata, and trigger deletion.

##### [`frontend/src/components/EntityGraphWorkspace.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/EntityGraphWorkspace.jsx)
* **What It Has**: Interactive entity relationship network and co-occurrence explorer.
* **Work It Is Doing**: Visualizes connections between political figures, corporations, and locations extracted across news issues.

##### [`frontend/src/components/TimelineWorkspace.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/TimelineWorkspace.jsx)
* **What It Has**: Chronological timeline visualizer with date milestones and article cards.
* **Work It Is Doing**: Renders story trajectory responses from `/api/query/timeline` across publications.

##### [`frontend/src/components/InspectionViewer.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/InspectionViewer.jsx)
* **What It Has**: Developer debug inspection console.
* **Work It Is Doing**: Displays raw DocLayNet bounding boxes, raw OCR text blocks, and parsed layout elements for any page to audit pipeline extraction quality.

##### [`frontend/src/components/ModelSelector.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/ModelSelector.jsx)
* **What It Has**: Model configuration modal and task binding selector.
* **Work It Is Doing**: Calls `/api/settings/bindings` to let users switch active models (e.g. from Ollama to Groq or Gemini) in real time.

##### [`frontend/src/components/RawDataViewer.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/RawDataViewer.jsx)
* **What It Has**: JSON tree inspector.
* **Work It Is Doing**: Renders raw JSON manifests (`articles_manifest.json`, `rag_chunks.json`, `ingestion_summary.json`) for deep inspection.

##### [`frontend/src/components/UploadTrigger.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/UploadTrigger.jsx)
* **What It Has**: Broadsheet PDF drag-and-drop modal, publication selector, real-time Celery ingestion progress bar.
* **Work It Is Doing**: Posts PDF files to `/api/ingest/upload` and polls `/api/ingest/jobs/{id}` until ingestion completes.

---

## 7. Operations & Diagnostic Scripts (`scripts/`)

* **Purpose / Reason**: Standalone operational tools, verification tests, diagnostic runners, and maintenance utilities.
* **Work It Is Doing**: Runs automated multi-phase system validation gates, tests provider latency, inspects schemas, and generates synthetic test broadsheets.

#### Files in `scripts/`:

##### [`scripts/verify_providers.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/verify_providers.py)
* **What It Has**: Connectivity tests for all registered model providers.
* **Work It Is Doing**: Pings Ollama, Groq, Gemini, OpenAI, and Google Cloud Vision; verifies that embeddings and text completions function correctly.

##### [`scripts/qa_diagnostic_test.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/qa_diagnostic_test.py)
* **What It Has**: Automated test suite running multi-archetype queries.
* **Work It Is Doing**: Tests differential queries ("List news in The Goan but not Morning Standard"), verifies tool choices, and validates citation formatting.

##### [`scripts/show_schema.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/show_schema.py)
* **What It Has**: SQLAlchemy database schema inspector.
* **Work It Is Doing**: Connects to MySQL and prints table definitions, foreign keys, and indexes.

##### [`scripts/reclassify_articles.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/reclassify_articles.py)
* **What It Has**: Batch article re-classification utility.
* **Work It Is Doing**: Re-runs the `ArticleClassifier` over existing database articles to update categories against modified alias rulebooks.

##### Verification Scripts:
- [`scripts/verify_phase1.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/verify_phase1.py): Verifies PDF rasterization and Docling layout analysis.
- [`scripts/verify_phase2.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/verify_phase2.py): Verifies reading order and headline segmentation.
- [`scripts/verify_phase3.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/verify_phase3.py): Verifies cross-page continuation assembly and MySQL persistence.
- [`scripts/verify_phase4.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/verify_phase4.py): Verifies BGE-M3 embedding generation and Qdrant indexing.
- [`scripts/verify_phase5.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/verify_phase5.py): Verifies hybrid search, RRF fusion, and Cross-Encoder reranking.
- [`scripts/verify_phase6_1.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/verify_phase6_1.py): Verifies the autonomous agent graph, planner, and synthesizer.
- [`scripts/generate_sample_newspaper.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/scripts/generate_sample_newspaper.py): Generates multi-column sample newspaper PDFs for offline testing.

---

## 8. Documentation Suite (`docs/`)

* **Purpose / Reason**: Serves as the authoritative knowledge base for the platform architecture, data structures, and operational procedures.
* **Work It Is Doing**: Documents system capabilities, schemas, wire protocols, and verified runtime data flows.

#### Files in `docs/`:
- [`docs/end_to_end_data_flow_guide.md`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docs/end_to_end_data_flow_guide.md): The flagship production-verified data flow guide with real database IDs, table rows, Qdrant vectors, Qwen-VL infographic reasoning, and live tool execution traces.
- [`docs/database_schema.md`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docs/database_schema.md): Complete database schema reference, entity diagrams, and index descriptions.
- [`docs/architecture.md`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docs/architecture.md): Deep-dive into subsystem architecture, component interactions, and scalability.
- [`docs/data_flow.md`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docs/data_flow.md): Sequence diagrams illustrating ingestion and agentic query handling.
- [`docs/data_flow_architecture.md`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docs/data_flow_architecture.md): Visual block diagrams tracing the flow of data across storage tiers.
- [`docs/features.md`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docs/features.md): Inventory of user-facing intelligence features.
- [`docs/engineering_log.md`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docs/engineering_log.md): Chronological engineering log recording bug fixes and optimizations.
- [`docs/codebase_directory_and_file_reference.md`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/docs/codebase_directory_and_file_reference.md): THIS FILE — Master codebase directory, file, framework, and model reference.

---
*End of NewsLens-AI Codebase Architecture & File Reference Guide.*
