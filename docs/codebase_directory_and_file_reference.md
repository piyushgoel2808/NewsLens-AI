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
│   │   ├── agent/                   # Conversational RAG agent (planner, archive_context, executor, sql_dispatcher, evaluator, answer_verifier, synthesizer)
│   │   ├── api/                     # FastAPI setup, lifespan management, middleware, and sub-routers
│   │   │   └── routers/             # Endpoint definitions (articles, query, ingest, models, settings, etc.)
│   │   ├── core/                    # Global settings, logging, cost tracker, Prometheus metrics, YAML rules
│   │   ├── evaluation/              # IR benchmark evaluation metrics (Recall@K, MRR, NDCG@K, Precision)
│   │   ├── ingestion/               # Consolidated 12-stage broadsheet PDF ingestion, layout analysis & VLM engine
│   │   │   ├── layout/              # Spatial column analysis, reading order & cross-page story assembly
│   │   │   ├── parsers/             # Document extraction engines (Docling neural, multimodal VLM, OCR)
│   │   │   ├── metadata.py          # Consolidated header, folio, masthead verification & issue consensus
│   │   │   └── storage.py           # Consolidated stream deflation, 3-tier hard deletion & debug exporter
│   │   ├── models/                  # SQLAlchemy 2.0 async relational schemas and ORM entities
│   │   ├── providers/               # Abstract model providers (Ollama, Groq, Gemini, OpenAI, GCV)
│   │   ├── retrieval/               # Multi-tool retrieval engines (hybrid search, visual inspection, asset resolution, SQL analytics, reranking)
│   │   └── storage/                 # Persistence clients (MySQL FULLTEXT, Qdrant, MinIO S3, Redis Cache)
│   │
│   └── tests/                       # Over 55 pytest test suites (411 unit, integration, and regression tests)
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
* **What It Has**: Package initialization, re-exports for `AgentGraph`, `QueryPlanner`, `QueryCondenser`, `SQLAnalyticsDispatcher`, `Synthesizer`, `AnswerVerifier`.
* **Work It Is Doing**: Exposes clean interface boundaries for the agent module and multi-node state machine.

##### [`backend/app/agent/condenser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/condenser.py)
* **What It Has**: 
  - `QueryCondenser` class implementing a 3-Tier conversational query condensation architecture.
  - Helper functions: `parse_inline_citation()`, `is_in_context_meta_query()`, `extract_active_issue_from_history()`, `extract_active_article_context()`, `extract_active_exclusion_context()`.
  - Reader asset context integration: `resolve_attached_asset_context()`, `resolve_authoritative_article_id()`.
  - `CONDENSATION_PROMPT` system template with 4 core intent rules.
* **Work It Is Doing**:
  - **Tier 1: Deterministic Gatekeeper & Fast Bypasses**: Instantly short-circuits clean sessions without history, detects in-context meta-queries ("What was the date?", "Which paper was this from?") to answer directly from chat context, and parses inline citations (`[4] Newspaper, YYYY-MM-DD, Page X, Headline: "..."`).
  - **Tier 2: Structured Context Assembly & LLM Few-Shot Rewriting**: Integrates active reader attached assets (`attached_article_id`, `attached_photo_id`), resolves coreferences and ambiguous pronouns ("it", "those 11 articles"), and strictly evicts conflicting dates/publications when the user initiates a temporal or brand pivot.
  - **Tier 3: Lightweight Normalizer & Safe Fallback**: Enforces zero destructive string mutations, falling back safely to the raw query if rewriting is unnecessary.
* **Important Tools / Frameworks**: Python AsyncIO, Regular Expressions (`re`), Pydantic.
* **LLM / VLM / Embedding Models**: Invokes the configured `query_planner` LLM (e.g. `gemma4:12b`, `llama3.1:8b`, or `gpt-4o-mini`).

##### [`backend/app/agent/models.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/models.py)
* **What It Has**: 
  - Dynamic Presentation Schemas: `SectionSpec` (`title`, `format_type`, `content_guideline`, `is_optional`), `AnswerBlueprint` (`archetype`, `executive_framing`, `sections`, `target_word_count`, `table_columns`, `prohibited_elements`, `tone_and_style`).
  - Reflexive Evaluation Schemas: `EvaluationVerdict` (`is_sufficient`, `quality_score`, `gap_reason`, `recommended_action`, `corrective_hints`).
  - Domain Data Models: `ToolName` (including `INSPECT_VISUAL_ASSET` and `DYNAMIC_ANALYSIS`), `QueryArchetype` (7 archetypes: `factual_lookup`, `quantitative_trend`, `thematic_timeline`, `cross_newspaper_comparison`, `entity_deep_dive`, `negative_coverage_audit`, `article_catalog`), `PlannedToolCall`, `PlanResult`, `ToolCallSpec`, `AgentPlan`.
  - Backward compatibility aliases and containers: `QueryPlan`, `ExtractedToolArguments`.
* **Work It Is Doing**:
  - Defines the core type-safe schema contracts for agentic query planning, dynamic answer blueprints, and reflexive CRAG evaluation.
  - Encapsulates tool argument contracts for visual inspection and dynamic tool synthesis.
  - Decouples Pydantic models and dataclasses from orchestration logic for zero-dependency reuse across retrieval and graph nodes.

##### [`backend/app/agent/extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/extractor.py)
* **What It Has**: 
  - Regex patterns: `_KNOWN_BRANDS_PATTERNS`, `_SECTION_PATTERNS`, Month + Year date patterns (`_MONTH_YEAR_PATTERNS`).
  - Core functions: `extract_parameters_from_query()`, `get_brand_patterns()`, `build_targeted_web_query()`, `is_archive_wide_newspaper_query()`.
  - Dynamic cache: `_DYNAMIC_PATTERNS_CACHE`.
* **Work It Is Doing**:
  - **Dynamic Brand Pattern Resolution (`get_brand_patterns`)**: Combines predefined brand patterns with dynamically discovered publications from `get_known_publications()`, compiling and caching regex patterns with zero redundant recompilations.
  - **Named Entity Recognition (NER) & Parameter Extraction**: Deterministically extracts publication brands, publication dates (ISO, DMY, and named months), issue IDs, page filters, and categories from natural language queries.
  - **Month + Year Date Range Parsing**: Automatically identifies month-level queries (e.g., "August 2026") and maps them to canonical ranges (`date_from: 2026-08-01`, `date_to: 2026-08-31`) while setting `issue_date: None` to query across whole editions.
  - **Brand-Masked Categorization**: Masks brand tokens to prevent brand names (e.g. "The Economic Times") from falsely triggering section categories (e.g. "Economy & Policy").
  - **Conversational Prefix Stripping**: Cleans user queries for high-precision search.

##### [`backend/app/agent/tool_factory.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/tool_factory.py)
* **What It Has**: 
  - Canonical Tool Builders: `build_sql_summary_tool()`, `build_sql_difference_tool()`, `build_sql_coverage_comparison_tool()`, `build_hybrid_search_tool()`, `build_coverage_analysis_tool()`, `build_timeline_tool()`, `build_entity_search_tool()`, `build_web_search_tool()`, `build_inspect_visual_asset_tool()`, `build_dynamic_analysis_tool()`.
  - Reconcilers: `reconcile_and_sanitize_arguments()`, `sanitize_generic_filler_query()`.
* **Work It Is Doing**:
  - **Single Source of Truth for Tool Construction**: Centralizes the generation of `PlannedToolCall` objects with clean parameter filtering.
  - **Visual Asset & Dynamic Tool Construction**: Builds `inspect_visual_asset` and `dynamic_analysis` tool calls with normalized parameters.
  - **Hallucination Pruner**: Checks LLM-generated arguments against query ground truth and prunes hallucinated brand names, dates, or page numbers.
  - **Date Reconciliation**: Prevents stale attached asset dates from overriding explicit query dates.
  - **Filler Sanitization**: Detects few-shot prompt contamination and restores substantive user domain queries.

##### [`backend/app/agent/archive_context.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/archive_context.py)
* **What It Has**:
  - `STATIC_BROADSHEET_SCHEMA`: 100% declarative in-memory schema catalog of broadsheet MySQL tables, columns, and relationships (zero DB network calls).
  - `ARCHIVE_SCHEMA`: Frozen column sets per table for compile-time AST and tool critic validation.
  - `KNOWN_COLUMN_HALLUCINATIONS`: Map correcting hallucinated columns (e.g. `published_at` $\to$ `issues.issue_date`, `newspaper` $\to$ `newspapers.name`, `category` $\to$ `article_categories.name`).
  - `ArchiveMetadata` dataclass: Typed container for `min_date`, `max_date`, `publications`, `categories`, and `context_str`.
  - `get_archive_metadata()`: Asynchronously introspects MySQL with 5-minute TTL cache, returning structured `ArchiveMetadata`.
  - `get_known_publications()`: Dynamically returns active broadsheet brands, merging database records with static canonical fallbacks.
  - `get_archive_and_schema_context()`: Asynchronously retrieves live archive bounds, delegating to `get_archive_metadata()`.
  - `get_fallback_archive_metadata()`: Graceful offline fallback providing canonical static defaults (`STATIC_CANONICAL_PUBLICATIONS`, `STATIC_ARCHIVE_DATE_MIN`, `STATIC_ARCHIVE_DATE_MAX`, `STATIC_CANONICAL_CATEGORIES`).
* **Work It Is Doing**:
  - **Softly Decoupled Grounding**: Eliminates fragile database coupling from query planning. If MySQL is unreachable, cold-starting, or under migration, planning proceeds instantaneously with static archive bounds rather than throwing 500 errors.
  - **Relational Schema Awareness**: Equips `QueryPlanner`, `ToolMaker`, and `ToolCritic` with accurate relational table definitions, foreign keys, and column names.
* **Important Tools / Frameworks**: Python `dataclass`, SQLAlchemy AsyncSession, In-memory TTL cache.
* **LLM / VLM / Embedding Models**: None (Declarative Metadata & Invariant Catalog).

##### [`backend/app/agent/planner.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/planner.py)
* **What It Has**: 
  - `QueryPlanner` class coordinating direct tool planning, blueprint generation, and adaptive re-planning.
  - `plan_query_async()` producing `PlanResult` containing both `tool_calls` and `AnswerBlueprint`.
  - `replan_with_feedback_async()` implementing closed-loop adaptive re-planning with anti-repetition guard.
  - Re-exports of `models`, `extractor`, and `tool_factory` symbols via `__all__` for 100% backward compatibility.
  - Lean `PLANNER_SYSTEM_PROMPT` with canonical few-shot examples including visual asset queries, dynamic analysis, and layout blueprints.
* **Work It Is Doing**:
  - **True Agentic Tool & Blueprint Planning**: Directly prompts LLMs to schedule ordered tool calls alongside a tailored `AnswerBlueprint` (defining section formats, word counts, and prohibited elements).
  - **Closed-Loop Adaptive Re-Planner**: `replan_with_feedback_async()` consumes diagnosed gaps from the reflexive evaluator, widening date bounds or expanding `top_k` while strictly blocking the agent from repeating identical failing calls.
  - **Lean Deterministic Heuristic Router**: Clean single-pass intent classifier mapping queries to 7 core archetypes with heuristic blueprint defaults:
    1. `thematic_timeline` (chronological progression across multiple dates)
    2. `entity_deep_dive` (multi-hop entity network search and profiling)
    3. `cross_newspaper_comparison` (differential coverage, omissions, framing differences across broadsheets)
    4. `quantitative_trend` (macro statistics, topic distributions, section volume summaries)
    5. `negative_coverage_audit` (unreported news verification and omission analysis)
    6. `article_catalog` (fast listing and catalog manifest generation for specific dates/sections)
    7. `factual_lookup` (targeted semantic + keyword search for point-in-time facts, quotes, and visual graphics)
  - **Transparent Legacy Adapter**: Translates older mock objects and test fixtures (`QueryPlan`, `primary_tool`, `ExtractedToolArguments`) to direct tool calls via `tool_factory`.
  - **Live Archive Grounding**: Dynamically injects `get_archive_and_schema_context()` into the planner prompt, grounding the LLM with live issue dates, active publications, and canonical categories.
  - **High-Throughput Cloud Failover**: Prioritizes `nvidia_nemotron` (<1s hosted inference with streaming reasoning) on cloud failover routes.
* **Important Tools / Frameworks**: Pydantic v2 schemas, Structured Outputs (`response_schema`), Regular Expressions.
* **LLM / VLM / Embedding Models**: `nvidia_nemotron` (NVIDIA NIM), `ollama_gemma4_12b` (Ollama), `groq_llama` (Groq), or `gemini_flash` (Gemini).

##### [`backend/app/agent/state.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/state.py)
* **What It Has**: `AgentState` TypedDict, `EvidenceItem` dataclass, `AgentCitation` dataclass.
* **Work It Is Doing**:
  - Defines the global state container passed through the LangGraph state machine.
  - Stores: user query, condensed query, conversation history, query plan, raw tool execution outputs, filtered evidence, streaming tokens, `<think>` reasoning traces, and bounding-box citations (including `is_visual_asset` and image URLs).
* **Important Tools / Frameworks**: Python `typing.TypedDict`, `typing.Annotated`, Pydantic models.
* **LLM / VLM / Embedding Models**: None (State Definition).

##### [`backend/app/agent/executor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/executor.py)
* **What It Has**: `ToolExecutor` coordinator class, `execute_plan()`, `execute_dynamic_tool()`, `execute_inspect_visual_asset()`.
* **Work It Is Doing**:
  - Encapsulates isolated, concurrent tool dispatch for all planned tool calls (`hybrid_search`, `sql_analytics`, `inspect_visual_asset`, `entity_search`, `web_search`, `dynamic_analysis`) via `asyncio.gather(*tasks, return_exceptions=True)`.
  - **Decoupled Relational SQL Dispatch**: Delegates pre-compiled relational SQL analytics routines (`issue_summary`, `coverage_difference`, `shared_coverage`, `coverage_comparison`, `count_issues`, `count_articles`, `count_ads`, `photo_count_per_section`, `topic_distribution`, `frontpage_ratio`, `entity_trends`) to `SQLAnalyticsDispatcher` ([`backend/app/agent/sql_dispatcher.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/sql_dispatcher.py)), keeping the tool coordinator lean.
  - **Layer 1 Dynamic Fallback**: Intercepts unsupported parameters (e.g. `analysis_type="word_count_variance"` in `sql_analytics`) and automatically delegates to `dynamic_analysis`.
  - **Modular Retrieval Delegation**:
    - Delegates visual inspection cascade, crop enrichment, and chart transcription to `VisualInspectionEngine` ([`retrieval/visual_inspector.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/visual_inspector.py)).
    - Delegates broadsheet manifest and matrix rendering to presentation formatters ([`retrieval/formatters.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/formatters.py)).
    - Uses database asset context resolver ([`retrieval/asset_resolver.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/asset_resolver.py)) for attached asset metadata reconciliation.
  - **Safe Logging Telemetry**: Sanitizes extra logging dictionaries to avoid colliding with Python stdlib `logging.LogRecord` reserved attributes.
* **Important Tools / Frameworks**: Python AsyncIO, SQLAlchemy Async Engine, Retrieval Engine Tools, MinIO Client, SQLAnalyticsDispatcher, VisualInspectionEngine, SandboxedExecutor.
* **LLM / VLM / Embedding Models**: Bound to `visual_extraction` (via `VisualInspectionEngine`) for on-demand VLM enrichment.

##### [`backend/app/agent/sql_dispatcher.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/sql_dispatcher.py)
* **What It Has**:
  - `SQLAnalyticsDispatcher` class: `dispatch()`, `_exec_sql_entity_trends()`, `_exec_sql_issue_summary()`, `_exec_sql_count_ads()`, `_exec_sql_count_issues()`, `_exec_sql_count_articles()`, `_exec_sql_photo_counts()`, `_exec_sql_topic_distribution()`, `_exec_sql_frontpage_ratio()`, `_exec_sql_coverage_comparison()`, `_exec_sql_coverage_difference()`, `_exec_sql_shared_coverage()`.
  - Integration with `CoverageAnalyzer` and `formatters` (`format_issue_manifest`, `format_coverage_difference_snippet`, `format_coverage_matrix_snippet`, `format_shared_coverage_snippet`).
* **Work It Is Doing**:
  - **Relational Analytics Dispatching**: Encapsulates all 11 pre-compiled relational analytical routines, cleanly separating database reporting from tool orchestration lifecycle.
  - **Dynamic Parameter & Alias Normalization**: Maps diverse aliases (e.g. `count_ads`, `advertisements`, `newspaper_availability`, `photos_by_section`, `common_stories`) to canonical database routines.
  - **Presentation Formatting**: Uses standardized formatters to render structured markdown manifests, difference reports, and comparison matrices for evidence state injection.
* **Important Tools / Frameworks**: Python AsyncIO, SQLAlchemy Async Engine, SQLAnalyticsEngine, CoverageAnalyzer, formatters.
* **LLM / VLM / Embedding Models**: None (Deterministic Relational Database Execution).

##### [`backend/app/agent/evaluator.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/evaluator.py)
* **What It Has**: `EvidenceEvaluator` class, `evaluate_evidence_async()`, `filter_evidence()`, `_stem()`, `_stem_phrase()`, `is_structural_or_relevant_evidence()`.
* **Work It Is Doing**:
  - Implements the **Reflexive CRAG Evaluator** grading evidence completeness and retrieval sufficiency.
  - **Hybrid Fast-Floor Evaluation (<5ms)**: Evaluates evidence against hard minimum bounds; immediately approves $\ge 1$ high-confidence article with $\ge 100$ words of clean body text without invoking LLM evaluation.
  - **Reflexive LLM-as-Judge**: When below the fast floor, invokes an LLM judge returning a typed `EvaluationVerdict` (`is_sufficient`, `quality_score`, `gap_reason`, `recommended_action`, `corrective_hints`).
  - **Semantic Hit Protection**: Protects high-confidence dense vector hits (`prominence_score >= 0.65`) from naive token stem pruning.
  - **Structural Archetype Protection**: Safeguards macro manifests and cross-newspaper comparison tables with 1.0 relevance scores.
* **Important Tools / Frameworks**: Fast-floor checks, Reflexive LLM Judge, Pydantic `EvaluationVerdict`.
* **LLM / VLM / Embedding Models**: Configured `evaluator` / `query_planner` LLM for ambiguous or zero-hit cases.

##### [`backend/app/agent/graph.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/graph.py)
* **What It Has**: 
  - `AgentWorkflow` class coordinating LangGraph `StateGraph`.
  - Nodes: `classify_and_plan`, `execute_tools`, `evaluate_and_fallback`, `execute_adaptive_replan`, `execute_dynamic_code`, `synthesize_answer`, `log_query`.
  - Conditional Edge Routers: `_route_after_planning()`, `_route_after_evaluation()`.
* **Work It Is Doing**:
  - Orchestrates the full conversational RAG lifecycle with closed-loop adaptive recovery.
  - **Blueprint Propagation**: Passes `AnswerBlueprint` from planner through `AgentState` to synthesizer and MySQL `QueryLog.plan_json`.
  - **Closed-Loop Reflexive Recovery**: Evaluator verdicts branch to `execute_adaptive_replan` (for missing static tools) or `execute_dynamic_code` (for ad-hoc tool synthesis), capped by a strict 1-cycle ceiling (`recovery_attempts < 1`).
  - **Attached Asset Conflict Eviction**: Compares attached asset date and publication against query entities, evicting mismatched assets before planning.
  - **Concurrent Tool Dispatch**: Executes all scheduled tools in parallel via `ToolExecutor`.
  - **Direct State Context Propagation**: Reads active issue context directly from `AgentState`.
* **Important Tools / Frameworks**: LangGraph `StateGraph`, Python AsyncIO.
* **LLM / VLM / Embedding Models**: Orchestrates planning, evaluation, and synthesis models.

##### [`backend/app/agent/tool_maker.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/tool_maker.py)
* **What It Has**: 
  - `ToolMaker` class, `ToolMakerResult` dataclass, `ensure_standard_imports()`.
  - `TOOL_MAKER_SYSTEM_PROMPT` containing full broadsheet MySQL schema definitions and few-shot analytical coding patterns.
* **Work It Is Doing**:
  - Implements the **LLM-as-Tool-Maker** pattern for synthesizing ad-hoc Python/SQL analysis tools on demand.
  - **Auto-Import Pre-Injection**: Automatically detects unimported standard module calls (`re.`, `math.`, `statistics.`, `json.`, `pd.`, `np.`, `text(`) and injects missing import headers before safety verification and execution.
  - **Closed-Loop Self-Refinement with ToolCritic**: Generates Python analysis code, runs it in `SandboxedExecutor`, audits results with `ToolCritic`, and if defects are detected, re-prompts the LLM with structured diagnostic critique over token-budgeted 4-message conversation history (up to 3 retry attempts).
  - Populates descriptive error messages upon retry exhaustion for transparent agent logging.
* **Important Tools / Frameworks**: Python AST, Regular Expressions, ToolCritic integration.
* **LLM / VLM / Embedding Models**: Invokes configured `query_planner` or `coder` model.

##### [`backend/app/agent/tool_critic.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/tool_critic.py)
* **What It Has**: 
  - `ToolCritic` class, `EvaluationScorecard` dataclass.
  - `ARCHIVE_SCHEMA` broadsheet schema map, `KNOWN_COLUMN_HALLUCINATIONS` mapping.
* **Work It Is Doing**:
  - Implements the **5-Dimension Quality Critic** auditing generated tools:
    1. **SASC** (Syntactic & AST Security Compliance): 1.0 or 0.0, zero tolerance for forbidden modules or attributes.
    2. **SRF** (SQL Relational & Schema Fidelity): AST SQL extraction; flags hallucinated columns (`published_at` -> `issues.issue_date`); requires `DISTINCT` on multi-table joins to prevent Cartesian explosion; supports `AS category` aliases and `article_categories` joins.
    3. **REH** (Runtime Execution Health): Subprocess error checking, type errors, timeouts.
    4. **DSF** (Data-to-Summary Faithfulness): Distinguishes **Legitimate Absence** (score 1.0 when `data: []` and summary acknowledges absence) from **Narrative Hallucination** (score 0.1/0.4 when `data: []` but summary claims positive counts); recognizes **Aggregate Computations** where `data: []` is supplemented with markdown tables and `metadata` counts; performs numerical consistency checks.
    5. **RPS** (Intent Alignment & Filter Plausibility): Checks for date formatting defects (un-normalized `2/8/2026` vs ISO `2026-08-02`), newspaper naming aliases (`goan` vs `The Goan`), and legitimate out-of-range dates.
* **Important Tools / Frameworks**: Python AST, AST visitor, Regex schema analyzer.
* **LLM / VLM / Embedding Models**: Deterministic Code & Schema Critic.

##### [`backend/app/agent/sandbox.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/sandbox.py)
* **What It Has**: 
  - `ASTSafetyScanner` class: `scan_code()`, `ALLOWED_MODULES`, `FORBIDDEN_BUILTINS`, `FORBIDDEN_MODULES`, `FORBIDDEN_ATTRS`.
  - `SandboxedExecutor` class: `execute_code()`, `ExecutionResult`.
  - `SecurityError`, `SandboxExecutionError`, `SandboxTimeoutError` custom exceptions.
* **Work It Is Doing**:
  - Performs pre-execution static AST analysis of dynamically synthesized Python code to reject unsafe operations (file I/O, OS commands, network calls, dynamic code evaluation, dunder traversal).
  - Spawns an isolated worker subprocess executing `sandbox_runner.py` with resource constraints (15s timeout, 512MB RAM cap).
  - Manages read-only database connections with unconditional rollback in a `finally` block to protect against data mutation.
* **Important Tools / Frameworks**: Python `ast`, `subprocess`, `resource` (Unix memory limits), SQLAlchemy.
* **LLM / VLM / Embedding Models**: None (Security & Execution Engine).

##### [`backend/app/agent/sandbox_runner.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/sandbox_runner.py)
* **What It Has**: Standalone subprocess runner script invoked by `SandboxedExecutor`.
* **Work It Is Doing**:
  - Reads JSON payload containing Python code, arguments, and DB connection credentials from stdin.
  - Sets up OS-level resource limits (`RLIMIT_AS` memory ceiling).
  - Pre-populates execution namespace `exec_globals` with pre-imported standard modules (`re`, `math`, `statistics`, `json`, `datetime`, `pd`, `np`, `text`) to ensure user scripts execute without `NameError`.
  - Establishes a read-only database transaction with autocommit disabled.
  - Executes the verified `analyze(db, query, context)` function and outputs structured JSON results to stdout.
* **Important Tools / Frameworks**: Python `sys`, `json`, `traceback`, `pymysql`, `SQLAlchemy`.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/agent/taxonomy.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/taxonomy.py)
* **What It Has**:
  - Centralized `DOMAIN_TAXONOMY` mapping for 6 domains (`Economics & Finance`, `Health & Medicine`, `Sports`, `Politics & Governance`, `Crime & Law`, `Technology & AI`).
  - Domain helpers: `detect_domain_from_query()`, `get_domain_terms()`, `is_domain_match()`, `score_evidence_item()`.
* **Work It Is Doing**:
  - Serves as the single source of truth for domain classification across the agent layer.
  - Encapsulates domain regexes, keyword stems, column headers, and negative exclusion rules with required positive overrides.
  - Provides scoring functions for token-budget sorting and relevance filtering.
* **Important Tools / Frameworks**: Python `re`, typing.

##### [`backend/app/agent/prompt_context.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/prompt_context.py)
* **What It Has**:
  - Evidence formatting helpers: `clean_snippet()`, `sanitize_evidence_item()`, `build_evidence_context()`, `build_synthesizer_user_prompt()`.
* **Work It Is Doing**:
  - Cleans retrieval artifacts, chunk tags, and visual asset brackets via single-pass regex and repairs font ligatures (`repair_text_ligatures()`).
  - Implements strict token budgeting, deduplicating evidence items and budgeting up to 12 top-ranked items (including photos and VLM scene descriptions).
  - Injects strict publication boundaries and domain guidance into the user prompt.
* **Important Tools / Frameworks**: Regex, Font Ligature Sanitizer.

##### [`backend/app/agent/fallback_presenter.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/fallback_presenter.py)
* **What It Has**:
  - `generate_deterministic_summary()`, `has_valid_evidence()`, `EMPTY_EVIDENCE_RESPONSE`.
  - Modular static renderers: `render_comparison_matrix()`, `render_front_page_comparison()`, `render_broadsheet_perspectives()`, `render_explore_further()`.
* **Work It Is Doing**:
  - Delivers a structured, publication-grade markdown brief when all cloud and local LLM providers are offline.
  - Implements archetype preservation, rendering cross-newspaper comparison matrices, front-page comparisons, and zero-coverage reporting (`No standalone [Domain] reporting in this edition`).
* **Important Tools / Frameworks**: Markdown generation, domain filtering.

##### [`backend/app/agent/synthesizer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/synthesizer.py)
* **What It Has**: 
  - `AnswerSynthesizer` coordinator class, `compile_structure_from_blueprint()`, `clean_robotic_catalog_tables()`, and `parse_thought_and_answer()`.
  - Dynamic system prompt composition (`_build_synthesizer_system_prompt()`, `COMMON_ANALYTICAL_GUIDELINES`, `COMMON_MEMORY_AND_CONSTRAINTS`).
  - Citation extraction and provenance helper: `extract_citations()`, `_make_citation()`.
  - Failover loops and streaming method: `synthesize()`, `synthesize_stream()`.
  - Transparent delegation to `taxonomy.py`, `prompt_context.py`, and `fallback_presenter.py`.
* **Work It Is Doing**:
  - Generates authoritative, highly readable executive intelligence briefs by orchestrating provider failovers and assembling structured prompts.
  - **Dynamic Blueprint Compilation**: Translates `AnswerBlueprint` into structured prompt guidelines at runtime, honoring user formatting constraints (e.g. word counts, bullet points, table exclusions) without rigid static templates.
  - **Robotic Catalog Table Elimination**: `clean_robotic_catalog_tables()` deterministically detects and strips mechanical tables from single-article narrative or summary answers.
  - **Single-Article Full-Text Context**: Preserves up to 7,500 characters of parent article text in prompt context, ensuring complete narrative coverage.
  - **Quantitative & Statistical Metric Absence Hard-Stop**: Strictly prohibits estimating or fabricating statistical figures, variances, standard deviations, or category breakdown tables when not explicitly present in verified tool evidence. Truthfully reports when statistical calculations could not be computed.
  - **Reasoning Stream Parsing**: Separates model reasoning traces (`<think>...</think>`) from the final response text.
  - **Strict 1-Shot Citation Enforcement**: Mandates bracketed inline citations on every factual assertion.
* **Important Tools / Frameworks**: Async Generators (`AsyncIterator`), Provider Registry, Cost Tracker.
* **LLM / VLM / Embedding Models**: Bound to `answerer` task (`nemotron-3.5-lightning`, `gemma4:12b`, `llama3.1:8b`, `deepseek-r1:14b`, or `gpt-4o`).

##### [`backend/app/agent/answer_verifier.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/answer_verifier.py)
* **What It Has**:
  - `AnswerVerifier` editorial auditor class and `AnswerVerificationResult` dataclass.
  - `ANSWER_VERIFIER_SYSTEM_PROMPT`: Rigorous 4-dimension audit instructions for groundedness, absence faithfulness, fluff elimination, and evidence gaps.
  - `verify_answer()`: Two-tier auditing executing fast deterministic rule-based checks (<5ms) before engaging the reflexive LLM auditor.
  - `_verify_publication_scope()`: Enforces publication boundary isolation, detecting if an answer covers out-of-scope broadsheets.
  - `_verify_date_alignment()`: Verifies that cited dates match the requested temporal boundary and retrieved evidence dates.
* **Work It Is Doing**:
  - **Reflective Post-Synthesis Auditor**: Acts as a peer editorial fact-checker reviewing synthesized responses before delivery to the user.
  - **4-Dimension Audit Framework**:
    1. *Faithfulness & Truthfulness*: Ensures assertions, counts, and dates are grounded in evidence. Flags positive claims when evidence demonstrates absence (e.g. 0 records).
    2. *Freedom from Hallucination & Corporate Fluff*: Strips speculative consulting boilerplate and fake citations.
    3. *Publication Scope & Date Alignment*: Detects cross-newspaper and cross-date boundary violations.
    4. *Evidence Gap Detection & Closed-Loop Routing*: If an evidentiary gap is diagnosed and dynamic code execution is required, emits `recommended_action="fallback_to_dynamic_tool"` with `dynamic_tool_hint`, triggering the LangGraph state machine to loop back to `execute_dynamic_code`.
* **Important Tools / Frameworks**: Pydantic v2 schemas, Structured JSON outputs, Provider Registry.
* **LLM / VLM / Embedding Models**: Bound to `query_planner` or `answerer` task (`nemotron-3.5-lightning`, `gemma4:12b`, `gpt-4o`).

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
  - Pydantic models: `QueryRequest` (including `attached_article_id` and `attached_photo_id`), `QueryResponse`, `TimelineQueryRequest`.
* **Work It Is Doing**:
  - Main user query execution endpoint.
  - `POST /query/stream`: Pushes real-time Server-Sent Events (SSE) including pipeline stages (`stage` supporting `planning`, `searching`, `inspecting_visual_asset`, `evaluating_evidence`, `synthesizing`), internal thoughts (`thought`), narrative tokens (`token`), citation objects with bounding boxes and visual flags (`citations` with `is_visual_asset`, `image_url: /api/photos/{id}/image`), and cost metrics (`done`).
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

#### 4.5 `backend/app/ingestion/` — Consolidated 12-Stage Broadsheet Processing Pipeline
* **Purpose / Reason**: The industrial-grade newspaper parsing, layout decomposition, OCR, visual extraction, and indexing engine.
* **Work It Is Doing**: Takes raw, unstructured multi-megabyte broadsheet PDF issues and turns them into high-resolution page renders, column-ordered text, structured cross-page articles, transcribed infographics, photo scene descriptions, and vectorized chunks. Architecturally consolidated from 28 micro-modules into cohesive architectural boundaries (`metadata.py`, `storage.py`, `layout/`, `parsers/`) while maintaining 100% backward compatibility via proxy shims.

#### Architecture & Subpackage Hierarchy:
- **`metadata.py`**: Consolidated Header, Folio, Masthead Verification & Multi-Page Issue Consensus.
- **`storage.py`**: Consolidated Stream Deflation, 3-Tier Hard Deletion & Diagnostic Debug Artifacts Export.
- **`layout/` Subpackage**: Spatial column analysis, reading order resolution, and cross-page article continuation assembly.
- **`parsers/` Subpackage**: Specialized document extraction engines (Docling 2D neural layout, multimodal VLM, and OCR).
- **Core Pipeline Services**: Master Celery workflow (`tasks.py`), single-page re-ingestion (`page_reingestion.py`), intake validation (`intake.py`), rasterization (`rasterizer.py`), visual extraction (`visual_extractor.py`), classification (`classifier.py`), chunking (`chunker.py`), and embedding (`embedder.py`).
- **Backward-Compatible Proxy Shims**: Lightweight re-export forwarders for all legacy module paths.

---

#### Consolidated Header & Storage Modules:

##### [`backend/app/ingestion/metadata.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/metadata.py)
* **What It Has**: 
  - `FolioDetector`: Running header, section folio, and page numeral extractor.
  - `MastheadVerifier`: RapidOCR Page 1 masthead banner scanner and publication date verifier.
  - `ConsensusExtractor` & `extract_newspaper_and_date_consensus()`: Multi-page majority voting on brand and issue date.
  - Unified Date Registry: Single canonical source of truth for `_DATE_PATTERNS`, `_MONTH_MAP`, and `parse_extracted_date()`.
  - Dataclasses: `HeaderCandidate`, `MastheadMatch`, `FolioMetadata`, `ConsensusMetadata`.
* **Work It Is Doing**:
  - Unifies previously fragmented header, folio, and masthead extraction into a single, cohesive metadata engine.
  - Eliminates duplicate date regexes across the codebase, ensuring consistent date normalization across all publications.
  - Resolves issue publication date and newspaper identity with multi-page majority voting.
* **Important Tools / Frameworks**: RapidOCR (ONNX PP-OCRv6), PyMuPDF (`fitz`), Regular Expressions, Dataclasses.
* **LLM / VLM / Embedding Models**: None (Deterministic Computer Vision & Heuristics).

##### [`backend/app/ingestion/storage.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/storage.py)
* **What It Has**: 
  - `compress_pdf_bytes()`, `compress_pdf()`: PyMuPDF PDF stream deflation and downsampling.
  - `DeletionService`: 3-tier cascade hard deletion across MySQL, Qdrant, and MinIO.
  - `DebugArtifactsExporter`: Diagnostic bounding box drawings and manifest exports.
* **Work It Is Doing**:
  - Consolidates all storage, deflation, and lifecycle maintenance utilities into a single module.
  - Executes atomic cascading deletions: removes relational rows in MySQL, deletes vector points in Qdrant, purges page rasters and visual crops in MinIO, and invalidates Redis cache keys.
  - Exports 5 structured JSON debug manifests (`articles_manifest.json`, `rag_chunks.json`, `ocr_extracted_text.json`, `identified_advertisements.json`, `ingestion_summary.json`).
* **Important Tools / Frameworks**: PyMuPDF (`fitz`), Qdrant Async Client, MinIO Client, SQLAlchemy AsyncSession, Redis.
* **LLM / VLM / Embedding Models**: None.

---

#### `backend/app/ingestion/layout/` — Spatial Layout & Article Assembly Subpackage:

##### [`backend/app/ingestion/layout/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/layout/__init__.py)
* **What It Has**: Public package facade exporting `LayoutAnalyzer`, `ArticleSegmenter`, `ReadingOrderResolver`, `CrossPageAssembler`, `BlockType`, `LayoutElement`, `OrderedReadingBlock`, `AssembledArticle`, `PageBBoxMapping`.
* **Work It Is Doing**: Exposes a clean, unified public interface for spatial broadsheet analysis and article segmentation.

##### [`backend/app/ingestion/layout/slugs.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/layout/slugs.py)
* **What It Has**: 
  - Constants: `WIRE_AGENCIES`, `DATELINE_CITIES`, `SECTION_HEADER_BLACKLIST`, `SYNDICATION_SLUGS`, `JUMP_PHRASE_PATTERNS`.
  - Utility Functions: `is_syndication_or_agency_slug()`, `is_numbered_feature_subhead()`, `clean_ocr_text_artifacts()`.
* **Work It Is Doing**:
  - Serves as the single source of truth for broadsheet text heuristics, wire service identification, and section header filtering.
  - Eliminates ~220 lines of duplicate regexes and filter functions previously duplicated across `layout_analyzer.py` and `cross_page_assembler.py`.

##### [`backend/app/ingestion/layout/analyzer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/layout/analyzer.py)
* **What It Has**: 
  - `LayoutAnalyzer` class.
  - `ReadingOrderResolver` class, `BlockType`, `LayoutElement`, `OrderedReadingBlock`.
* **Work It Is Doing**:
  - Slices complex broadsheet pages into vertical column tracks, horizontal headline bands, and advertisement envelopes.
  - Solves multi-column broadsheet reading order using XY-cut geometric clustering, preventing column bleeding.
  - Re-attaches drop-caps (e.g. large initial "T") and repairs hyphenated line breaks.
  - Merges horizontal multi-column headline slices across column gutters.
* **Important Tools / Frameworks**: Computational Geometry, Interval Math, PyMuPDF.
* **LLM / VLM / Embedding Models**: None (Deterministic Spatial Algorithms).

##### [`backend/app/ingestion/layout/segmenter.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/layout/segmenter.py)
* **What It Has**: 
  - `ArticleSegmenter` class, `SegmentedArticle`.
  - `CrossPageAssembler` class, `AssembledArticle`, `PageBBoxMapping`.
  - Helper functions: `extract_kicker_and_clean_headline()`, `is_valid_headline_candidate()`.
* **Work It Is Doing**:
  - Groups headlines, kickers, sub-decks, bylines, and narrative paragraphs into coherent candidate articles on a single page.
  - Detects jump-line continuation markers (*"Continued on Page 9"*, *"From Page 1"*) and stitches split stories across distant pages into unified database records with multi-page coordinate mapping.
  - De-bundles multi-story summary columns (*Mint Shorts*, *Briefs*) into separate articles.
* **Important Tools / Frameworks**: SequenceMatcher (fuzzy string matching), Regular Expressions, Geometric containment.
* **LLM / VLM / Embedding Models**: None.

---

#### `backend/app/ingestion/parsers/` — Document Extraction Subpackage:

##### [`backend/app/ingestion/parsers/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/parsers/__init__.py)
* **What It Has**: Public package facade exporting `DoclingLayoutParser`, `UnifiedExtractor`, `OCRService`, `ArticleSkeleton`, `PageLayoutExtraction`, `ExtractedTable`, `ExtractedPicture`, `VisualCropData`.
* **Work It Is Doing**: Exposes a unified interface for all document parsing and OCR engines.

##### [`backend/app/ingestion/parsers/schemas.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/parsers/schemas.py)
* **What It Has**: Pydantic v2 schemas: `ArticleSkeleton`, `PageLayoutExtraction`, `ExtractedTable`, `ExtractedPicture`, `VisualCropData`, `ArticleEnrichment`, `ArticleGenre`, `ProminenceTier`.
* **Work It Is Doing**:
  - Defines the structured data interchange schemas for document parsing and VLM extraction.
  - Provides type safety across parser engines with Pydantic v2 keyword defaults.

##### [`backend/app/ingestion/parsers/docling.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/parsers/docling.py)
* **What It Has**: `DoclingLayoutParser` class, `DoclingParsedItem` dataclass, `CorruptedPdfTextLayerError`.
* **Work It Is Doing**:
  - Executes deep 2D layout analysis on broadsheet pages using DocLayNet neural models.
  - Detects corrupted font CMap ligatures (e.g. `` replacement character ratios $\ge 3\%$) and raises `CorruptedPdfTextLayerError` to trigger pure image OCR fallback.
  - Normalizes bounding boxes and outputs structured layout element tokens (`title`, `section_header`, `text`, `caption`, `picture`, `table`).
* **Important Tools / Frameworks**: Docling library, PyMuPDF, NumPy.
* **LLM / VLM / Embedding Models**: DocLayNet Layout Analysis Models.

##### [`backend/app/ingestion/parsers/vlm.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/parsers/vlm.py)
* **What It Has**: `UnifiedExtractor` class, structured extraction prompts (`EXTRACTION_SYSTEM_PROMPT`).
* **Work It Is Doing**:
  - Executes single-pass multimodal extraction on full-page images via Gemini, Gemma, or Qwen-VL.
  - Extracts article boundaries, headlines, body text, and visual metadata in a single inference call.
* **Important Tools / Frameworks**: ModelRegistry, VisionModelProvider, Pydantic Structured Outputs.
* **LLM / VLM / Embedding Models**: Configured vision provider (`gemini_flash`, `gemma4_26b`, or `qwen3-vl`).

##### [`backend/app/ingestion/parsers/ocr.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/parsers/ocr.py)
* **What It Has**: `OCRService` class.
* **Work It Is Doing**:
  - Dispatches cropped image regions to Google Cloud Vision API (`google-cloud-vision`), RapidOCR, or local Tesseract OCR.
  - Performs image contrast enhancement, deskewing, and coordinate normalization.
* **Important Tools / Frameworks**: Google Cloud Vision SDK, RapidOCR, PyTesseract, PIL ImageEnhance.
* **LLM / VLM / Embedding Models**: Google Cloud Vision Document Text Detection or RapidOCR PP-OCRv6.

---

#### Core Orchestration & Processing Pipeline Services:

##### [`backend/app/ingestion/__init__.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/__init__.py)
* **What It Has**: Subsystem entrypoint with canonical public exports.
* **Work It Is Doing**: Exports canonical pipeline symbols (`IntakeService`, `PDFRasterizer`, `LayoutAnalyzer`, `ArticleSegmenter`, `DoclingLayoutParser`, `UnifiedExtractor`, `ArticleClassifier`, `NewspaperChunker`, `ArticleEmbedder`, `DeletionService`, `run_ingestion_pipeline`).

##### [`backend/app/ingestion/celery_app.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/celery_app.py)
* **What It Has**: Celery app instance, broker configuration (`redis://localhost:6379/0`), result backend, task serializer settings.
* **Work It Is Doing**: Configures the asynchronous task queue for distributed background ingestion.
* **Important Tools / Frameworks**: Celery, Redis.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/tasks.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/tasks.py)
* **What It Has**: Celery tasks: `process_issue_ingestion_task()`, `run_ingestion_pipeline()`, `detect_masthead_and_date()`, `check_is_advertisement_text()`.
* **Work It Is Doing**:
  - Master pipeline coordinator orchestrating the 12 stages in strict sequence:
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

##### [`backend/app/ingestion/page_reingestion.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/page_reingestion.py)
* **What It Has**: `PageReingestionService` class.
* **Work It Is Doing**:
  - Coordinates on-demand single-page re-processing via `POST /api/issues/{issue_id}/pages/{page_number}/reingest`.
  - Atomically purges previous page-exclusive articles, entities, topics, chunks, photos, and Qdrant vector points.
  - Re-runs rasterization, layout parsing, OCR, photo harvesting, and embedding for that specific page without corrupting or re-processing the entire 24+ page issue.
* **Important Tools / Frameworks**: SQLAlchemy AsyncSession, MinIO, Qdrant Client.
* **LLM / VLM / Embedding Models**: Docling, Qwen-VL, BAAI/bge-m3.

##### [`backend/app/ingestion/rasterizer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/rasterizer.py)
* **What It Has**: `PDFRasterizer` class, `RasterizedPage` dataclass.
* **Work It Is Doing**:
  - Renders 300 DPI high-resolution PNG page images from PDF broadsheets using PyMuPDF (yielding ~8,188 x 11,400 px images).
  - Implements both multi-page document rasterization (`rasterize_pdf_bytes()`) and targeted single-page rasterization (`rasterize_single_page()`).
  - Uploads images to MinIO (`newslens-pages`) and records database metadata.
* **Important Tools / Frameworks**: PyMuPDF (`fitz`), Pillow (PIL), MinIO Client.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/intake.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/intake.py)
* **What It Has**: `IntakeService` class, `IntakeResult` dataclass.
* **Work It Is Doing**: Validates uploaded files, computes SHA-256 checksums, checks MIME types, verifies PDF headers, prevents duplicate ingestion, and creates `IngestionJob` tracking records.
* **Important Tools / Frameworks**: Hashlib (SHA-256), Pathlib, SQLAlchemy AsyncSession.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/ingestion/detector.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/detector.py)
* **What It Has**: `PDFPageDetector` class, `check_is_advertisement_text()`.
* **Work It Is Doing**: Analyzes native PDF text layers to identify digital vs scanned pages, drop-caps, text noise ratios, and statutory commercial advertisement blocks.
* **Important Tools / Frameworks**: PyMuPDF, Regular Expressions.
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

##### [`backend/app/ingestion/classifier.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/classifier.py)
* **What It Has**: `ArticleClassifier` class, `ClassificationResult`.
* **Work It Is Doing**: Analyzes article headlines, decks, and body text using multi-signal scoring and LLM classification to assign canonical category IDs (e.g. `Business & Markets`, `Politics & Governance`) and confidence scores.
* **Important Tools / Frameworks**: Pydantic, Structured Prompts.
* **LLM / VLM / Embedding Models**: `ollama_gemma4_12b` (Ollama) or `gpt-4o-mini`.

##### [`backend/app/ingestion/metadata_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/metadata_extractor.py)
* **What It Has**: `MetadataExtractor` class, `ExtractedEntity`, `ExtractedTopic`, `ArticleMetadataResult`.
* **Work It Is Doing**: Extracts named entities (persons, organizations, locations), topical tags, and salience scores ($0.0$ to $1.0$) for every article.
* **Important Tools / Frameworks**: Pydantic, Structured Outputs.
* **LLM / VLM / Embedding Models**: Bound to `metadata_extraction` (`gemma4:12b` or `gemini_flash`).

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

##### [`backend/app/ingestion/geometry.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/geometry.py)
* **What It Has**: `BBox` dataclass, spatial operations (`intersects`, `contains`, `iou`, `scale`, `to_dict`).
* **Work It Is Doing**: Implements fundamental 2D geometric operations for bounding boxes `[x0, y0, x1, y1]`, coordinate transforms, and overlap ratios.
* **Important Tools / Frameworks**: Pure Python Math.
* **LLM / VLM / Embedding Models**: None.

---

#### Backward-Compatibility Re-Export Shims:
To maintain zero breakage across external tools, legacy endpoints, and all 411 tests, the following files serve as transparent re-export forwarding shims:
- [`backend/app/ingestion/folio_detector.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/folio_detector.py) $\to$ Forwarded to `metadata.py`
- [`backend/app/ingestion/masthead_verifier.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/masthead_verifier.py) $\to$ Forwarded to `metadata.py`
- [`backend/app/ingestion/consensus_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/consensus_extractor.py) $\to$ Forwarded to `metadata.py`
- [`backend/app/ingestion/compressor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/compressor.py) $\to$ Forwarded to `storage.py`
- [`backend/app/ingestion/deletion_service.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/deletion_service.py) $\to$ Forwarded to `storage.py`
- [`backend/app/ingestion/debug_exporter.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/debug_exporter.py) $\to$ Forwarded to `storage.py`
- [`backend/app/ingestion/extraction_schemas.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/extraction_schemas.py) $\to$ Forwarded to `parsers.schemas`
- [`backend/app/ingestion/docling_parser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/docling_parser.py) $\to$ Forwarded to `parsers.docling`
- [`backend/app/ingestion/unified_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/unified_extractor.py) $\to$ Forwarded to `parsers.vlm`
- [`backend/app/ingestion/ocr_service.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/ocr_service.py) $\to$ Forwarded to `parsers.ocr`
- [`backend/app/ingestion/reading_order.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/reading_order.py) $\to$ Forwarded to `layout.analyzer`
- [`backend/app/ingestion/cross_page_assembler.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/cross_page_assembler.py) $\to$ Forwarded to `layout.segmenter`
- [`backend/app/ingestion/layout_analyzer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/layout_analyzer.py) $\to$ Forwarded to `layout.analyzer`
- [`backend/app/ingestion/segmenter.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/ingestion/segmenter.py) $\to$ Forwarded to `layout.segmenter`

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
* **What It Has**: `ModelRegistry` class, `get_registry()`, `reset_registry()`, `get_chat_failover_candidates()`.
* **Work It Is Doing**:
  - Reads `model_config.yaml`, instantiates concrete provider classes, and resolves task bindings (`get_provider("query_planner")`).
  - Supports dynamic runtime updates and task cache invalidation.
  - **Prioritized Failover Routing (`get_chat_failover_candidates`)**: Computes ordered lists of configured chat-capable provider IDs for resilient fallbacks across cloud (`nvidia_nemotron`, `openrouter_nemotron`, `openrouter_gemma4_26b`, `gemini_flash`, `groq_compound`, `openai_gpt4o_mini`, `groq_qwen`, `openai_gpt4o`, `gemini_pro`) and sovereign local endpoints (`ollama_llama3`, `ollama_deepseek`, `ollama_nemotron`), respecting `prefer_local` flags.
* **Important Tools / Frameworks**: Singleton pattern, YAML parsing, Dynamic failover lists.
* **LLM / VLM / Embedding Models**: Manages the complete lifecycle and failover resolution of all configured models.

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
* **What It Has**: Package initialization, re-exports for `HybridSearchEngine`, `SQLAnalyticsEngine`, `CoverageAnalyzer`, `TimelineBuilder`, `EntitySearchEngine`, `CrossEncoderReranker`, `repair_text_ligatures`, `VisualInspectionEngine`, `resolve_attached_asset_context`, `resolve_authoritative_article_id`, `resolve_conversation_working_context`, `ConversationWorkingContext`, `format_issue_manifest`, `format_coverage_matrix_snippet`, `format_coverage_difference_snippet`, and `format_shared_coverage_snippet`.

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
* **What It Has**: `SQLAnalyticsEngine` class, `sanitize_headline()`, `get_archive_metadata()`, `normalize_date_to_iso()`.
* **Work It Is Doing**:
  - Executes deterministic, parameterized SQL aggregation queries.
  - `get_archive_metadata()`: Fast cached (<20ms) extraction of available issue dates, active newspapers, and canonical database categories.
  - `sanitize_headline()`: Cleanses headlines where doctor/author profile names were mistakenly extracted as the headline, preserving genuine all-caps headlines and bylines.
  - `get_photo_counts_by_section()`: High-speed relational photo count analytics grouping by `COALESCE(Article.section, 'Unassigned')` across `Issue` and `Newspaper` with ISO date and date range normalization (`date_from`, `date_to`).
  - `count_articles()`: Computes exact article counts supporting publication, section, article type, single date, or date ranges (`date_from`, `date_to`).
  - `get_issue_summary()`: Retrieves the full article manifest for a newspaper issue with economic domain bridging (`Business & Markets` + `Economy & Policy`).
  - `get_newspaper_coverage_difference()`: Computes verified exclusive articles between two publications on a given date (e.g. The Goan vs The Morning Standard).
  - `get_entity_mention_trends()`: Computes monthly/daily mention trajectories.
* **Important Tools / Frameworks**: SQLAlchemy Core & ORM async select queries.
* **LLM / VLM / Embedding Models**: None (Deterministic Relational Grounding).

##### [`backend/app/retrieval/reranker.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/reranker.py)
* **What It Has**: `CrossEncoderReranker`, `HeuristicReranker`, `_detect_best_device()`, `predict()` synchronous method.
* **Work It Is Doing**:
  - Computes full cross-attention interaction scores between query and candidate snippets.
  - **macOS CPU Optimization**: On Apple Silicon (macOS), explicitly selects CPU over MPS for `ms-marco-MiniLM-L-6-v2` because the lightweight 6-layer model executes in **~80ms** on CPU, completely avoiding the 10–15 second Metal shader compilation lag and GPU buffer synchronization overhead of MPS.
  - **Synchronous Pair Scoring**: Provides `predict(pairs: list[tuple[str, str]]) -> list[float]` for zero-overhead batch scoring in timeline verification and retrieval nodes.
  - **Graceful Fallback**: Automatically degrades to `HeuristicReranker` if PyTorch or sentence-transformers dependencies are unavailable.
* **Important Tools / Frameworks**: `sentence_transformers.CrossEncoder`, PyTorch, AsyncIO threadpool execution.
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
* **Work It Is Doing**: Multi-tier live internet retrieval cascading across Tier 1 NewsData.io (accredited journalistic press & newspapers), Tier 2 Serper (Google Search API), Tier 3 Tavily (AI research search), and Tier 4 DuckDuckGo HTML scraping fallback. Formats live web results into structured citations with publisher names, article URLs, and publication dates.
* **Important Tools / Frameworks**: `httpx` (async HTTP client), NewsData.io API, Serper API, Tavily API, DuckDuckGo HTML parser.
* **LLM / VLM / Embedding Models**: None.

##### [`backend/app/retrieval/asset_resolver.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/asset_resolver.py)
* **What It Has**:
  - `resolve_attached_asset_context()`: Resolves ground truth database metadata for attached workspace assets (photos, articles, issue dates, headlines, captions).
  - `resolve_authoritative_article_id()`: Fast DB lookup to find exact `article_id` for a known or quoted headline.
  - `ConversationWorkingContext` dataclass: Consolidated Ground Truth context and conflict flags for a conversational turn.
  - `resolve_conversation_working_context()`: Authoritative reconciliation of attached assets, query parameters, headline bindings, and active issue context.
* **Work It Is Doing**:
  - **Ground Truth Context Authority**: Decouples asset resolution from agent execution logic.
  - **Conflict Detection**: Automatically detects cross-date, cross-publication, and headline conflicts between query text and attached assets.
* **Important Tools / Frameworks**: SQLAlchemy AsyncSession, Regular Expressions (`re`).
* **LLM / VLM / Embedding Models**: None (Deterministic Database Grounding).

##### [`backend/app/retrieval/visual_inspector.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/visual_inspector.py)
* **What It Has**:
  - `VisualInspectionEngine` class.
  - `inspect_visual_asset()`: Deep multimodal visual inspection, on-demand VLM extraction, and table transcription.
  - `resolve_visual_target_params()`: Resolves target photo/article IDs, dates, newspapers, and headlines from args, state, citations, and history.
  - `resolve_photos_for_inspection()`: 5-strategy discovery cascade:
    - *Strategy A*: Explicit Photo ID + companion charts/tables.
    - *Strategy B*: Target headline resolved from query citation or context.
    - *Strategy C*: Explicit Article ID + companion charts/tables.
    - *Strategy D*: Multi-criteria DB search (newspaper, date, page, query tokens).
    - *Strategy E*: Scoped caption and VLM description search.
  - `enrich_photo_vlm_descriptions()`: On-demand MinIO image crop fetch and Gemini/Qwen VLM transcription.
  - `format_visual_asset_item()`: Formats resolved `Photo` into standardized broadsheet evidence item.
* **Work It Is Doing**:
  - Houses the complete multimodal visual intelligence and broadsheet visual crop inspection engine.
  - Enriches default placeholder descriptions lazily by streaming raw image bytes from MinIO and executing VLM extraction.
* **Important Tools / Frameworks**: SQLAlchemy AsyncSession, MinIO Store, VisualDataExtractor.
* **LLM / VLM / Embedding Models**: Bound to `visual_extraction` task (`gemini_vision`, `qwen3vl`, or `openai_gpt4o`).

##### [`backend/app/retrieval/formatters.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/retrieval/formatters.py)
* **What It Has**:
  - `format_issue_manifest()`: Renders unified, human-readable broadsheet manifest string from structured issue summaries.
  - `format_coverage_matrix_snippet()`: Renders unified 3-tier coverage reconciliation matrix string.
  - `format_coverage_difference_snippet()`: Renders verified exclusive coverage difference manifest string.
  - `format_shared_coverage_snippet()`: Renders verified shared syndicated wire coverage manifest string.
* **Work It Is Doing**:
  - Consolidates all human-readable markdown snippet generation for retrieval evidence.
  - Integrates `repair_text_ligatures()` to guarantee clean typographic presentation in manifests.
* **Important Tools / Frameworks**: Python string formatting, Typographic Ligature Repair.
* **LLM / VLM / Embedding Models**: None (Deterministic Presentation Formatting).

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
- [`test_dynamic_answer_blueprint.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_dynamic_answer_blueprint.py): Verifies `AnswerBlueprint` schema, `SectionSpec` validation, planner blueprint generation, dynamic prompt compilation, and state machine integration.
- [`test_condenser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_condenser.py): Validates multi-turn pronoun resolution, headline conflict detection, and context isolation.
- [`test_graph.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_graph.py): Tests the LangGraph workflow, conditional edge routing, adaptive re-plan, and dynamic code execution nodes.
- [`test_reflexive_evaluator.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_reflexive_evaluator.py): Tests hybrid Fast-Floor bypass (<5ms), reflexive LLM-as-Judge evaluation producing `EvaluationVerdict`, gap diagnosis, and corrective hints.
- [`test_evaluator_crag_dynamic.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_evaluator_crag_dynamic.py): Tests closed-loop CRAG evaluation, adaptive re-planning with anti-repetition guard, and dynamic code fallback integration.
- [`test_synthesizer.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_synthesizer.py): Verifies dynamic blueprint synthesis, single-article context budgeting, robotic catalog table stripping, and 100% citation precision.
- [`test_sandbox.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_sandbox.py): Verifies AST safety scanning, forbidden module/builtin detection, timeout enforcement, memory capping, and read-only rollback transactions.
- [`test_tool_maker.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_tool_maker.py): Validates dynamic tool synthesis, prompt formatting with broadsheet schema, auto-import injection, and closed-loop self-refinement.
- [`test_tool_critic.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_tool_critic.py): Verifies the 5-metric evaluation scorecard (SASC, SRF, REH, DSF, RPS), AST SQL query extraction, column hallucination detection, legitimate absence vs hallucination distinction, and aggregate table acceptance.
- [`test_cross_date_contamination.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_cross_date_contamination.py): Tests cross-date anti-leakage shield, ensuring query dates override attached asset dates across condenser, router, and executor.
- [`test_docling_parser.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_docling_parser.py): Tests 2D layout bounding box extraction and font CMap corruption detection.
- [`test_visual_extractor.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_visual_extractor.py): Tests Qwen-VL infographic transcription and OCR cross-validation.
- [`test_vlm_grounding.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_vlm_grounding.py): Tests coordinate parsing from Qwen-VL's native `<think>` stream.
- [`test_hybrid_search.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_hybrid_search.py): Tests dense/sparse RRF fusion and Cross-Encoder reranking.
- [`test_sql_analytics.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/tests/test_sql_analytics.py): Tests issue manifests, photo counts by section, advertisement counts, and cross-newspaper coverage difference counts.

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
* **What It Has**: Conversational chat interface, SSE event listener, collapsible `<think>` reasoning accordion, active attached asset banner (`attachedAsset` state with headline, date, page, and dismiss button), visual citation cards with thumbnail rendering (`/api/photos/{id}/image`), inline citation pills, follow-up prompt pills, export buttons.
* **Work It Is Doing**: Consumes the `/api/query/stream` SSE wire protocol in real-time, rendering thinking steps, live stage indicators (including `inspecting_visual_asset`), and markdown responses progressively. Supports attaching photos and articles directly from the Broadsheet Reader and links citations directly to the broadsheet canvas.
* **Important Tools / Frameworks**: Fetch API EventStream reader, Lucide React, Markdown rendering.

##### [`frontend/src/components/BroadsheetReader.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/BroadsheetReader.jsx)
* **What It Has**: High-resolution pan/zoom canvas reader, page navigation carousel, article text inspection drawer, `"Ask Agent About This Infographic / Photo"` action buttons on visual asset cards.
* **Work It Is Doing**: Displays 300 DPI broadsheet pages with smooth zoom controls; overlays interactive SVG bounding boxes for articles, photos, and infographics. Equips every visual asset with deep-linking controls that open the Agent Assistant with the asset pre-attached for multimodal inquiry.
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
* **What It Has**: Quick model selection modal and compact task binding switcher.
* **Work It Is Doing**: Calls `/api/settings/bindings` to let users switch active models in real time.

##### [`frontend/src/components/ModelSettingsStudio.jsx`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/frontend/src/components/ModelSettingsStudio.jsx)
* **What It Has**: Comprehensive AI provider management studio and task binding workbench.
* **Work It Is Doing**: Provides a full-featured interface for configuring hosted and local model providers (NVIDIA NIM, Ollama, Anthropic, OpenAI, Gemini, Groq), managing API keys, running real-time connectivity and latency health tests, inspecting model capabilities (chat, reasoning, multimodal vision, embeddings), and dynamically updating task assignments (`query_planner`, `synthesizer`, `vlm_extractor`, `embedding`, `ocr`) with hot reload.
* **Important Tools / Frameworks**: React Hooks, Lucide Icons, REST API integration.

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
