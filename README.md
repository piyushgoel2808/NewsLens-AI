# 📰 NewsLens-AI: Newspaper Intelligence Agentic RAG Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.11+-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![MySQL 8](https://img.shields.io/badge/MySQL-8.0-4479A1.svg?logo=mysql&logoColor=white)](https://www.mysql.com/)
[![Redis](https://img.shields.io/badge/Redis-7.4-DC382D.svg?logo=redis&logoColor=white)](https://redis.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NewsLens-AI** is an advanced, enterprise-grade **Agentic RAG & Broadsheet Intelligence Platform** engineered for analyzing scanned historical and modern newspaper archives. It turns multi-page, multi-column print broadsheets into structured, searchable intelligence with interactive spatial citations, cross-publication story timelines, and deep conversational reasoning.

---

## 🌟 Key Features

- **📑 Docling Broadsheet Neural Layout & 2D Article Segmentation**: Uses **IBM Docling (DocLayNet)** with **RapidOCR** and 2D spatial sorting to cleanly coalesce multi-line headlines and subheadline/decks, extract inline bylines (`BY <NAME>`), and segment complex multi-story broadsheets without column bleeding.
- **📷 2D Spatial Photo/Infographic Extraction & Multimodal Data Chunks**: Automatically isolates editorial photos, graphics, and composite photo galleries, pairs them with shared multi-column captions using 2D proximity scoring, and indexes infographic data tables into Qdrant via **Qwen3-VL** (`qwen3-vl:latest`).
- **🏛️ Multi-Page Consensus Masthead Verification**: Robustly extracts newspaper brand and publication dates across global (*The New York Times*, *The Wall Street Journal*, *Financial Times*, *The Washington Post*, *The Guardian*) and national broadsheets with RapidOCR visual verification.
- **⚡ Two-Stage Neural Retrieval Cascade (Cross-Encoder)**: Stage 1 RRF hybrid search (Qdrant `BAAI/bge-m3` + MySQL `FULLTEXT`) expands candidates to $N=75$, followed by Stage 2 Cross-Encoder neural reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with macOS CPU optimization (avoiding MPS Metal shader lag) and candidate pool capping ($K=20$), returning the Top 10 high-precision hits in sub-second latency (~1.0s vs 30.4s).
- **📊 3-Tier Negative Coverage Engine**: Performs relational audits in MySQL to identify zero-coverage publications, semantic validation, and multi-newspaper editorial reconciliation matrices (`POST /api/query/coverage`).
- **🤖 Autonomous Modular LangGraph Agent with Live Archive Grounding & Concurrency**: Pydantic structured Chain-of-Thought (CoT) planning grounded in live database metadata (`get_archive_metadata`). Built on a decoupled clean architecture: planning (`models.py`, `extractor.py`, `tool_factory.py`, `planner.py`), execution state machine (`executor.py`, `evaluator.py`, `graph.py`), and synthesis (`synthesizer.py` with centralized `DOMAIN_TAXONOMY`). Autonomously routes across 7 broadsheet archetypes (including sub-200ms `article_catalog`), executes planned tools concurrently via `asyncio.gather`, and performs real-time adaptive fallbacks on zero-hit category queries.
- **🕸️ Interactive Multi-Hop Entity Knowledge Graph**: Interactive visualizer mapping entity relationships, co-occurrences, and hop depths across shared stories and event clusters with Corrective RAG (CRAG) self-reflection.
- **⚡ Domain-Adaptive Structured Executive Briefs & Strict Domain Purity**:
  - `⚡ Executive Summary`: Crisp journalistic analysis with zero query-echoing.
  - `📊 Cross-Newspaper Comparison Matrix`: Adaptive domain headers (`Key Findings & Medical Focus`, `Key Figures & Metrics`, `Key Policy Decisions`) with strict Zero-Coverage reporting for papers lacking coverage in target topics.
  - `📌 Key Verified Facts & Highlights`: Precise figures, index moves, corporate actions, and 100% verified inline citations.
  - `📰 Broadsheet Perspectives`: Distinct editorial framing (*Mint*, *Business Standard*, *The Hindu*, *The Goan*).
  - `🔍 Explore Further`: Clickable exploration pills in the UI for instant drill-down queries.
- **🌐 Dual-Mode Retrieval (Archive + Live News Web Grounding)**: Multi-tier live internet retrieval cascading across **NewsData.io** (accredited journalistic press & newspapers), **Serper** (Google Search API), **Tavily** (AI research search), and **DuckDuckGo** (HTML scraping fallback) with verified publisher attribution and distinct visual citation badges separating printed archive folios from live web sources.
- **📈 Cross-Newspaper Narrative Trajectory & Story Timelines**: Reconstructs evolving stories across calendar dates with 4-tier anti-hallucination gates, tracking reporting phases (`Breaking`, `Development`, `Financial Impact`, `Regulatory/Outcome`) and editorial discrepancies with Redis caching.
- **🔄 Interactive Single-Page Re-Ingestion Engine**: On-demand re-processing for specific broadsheet pages via UI (`POST /api/issues/{issue_id}/pages/{page_number}/reingest`). Atomically purges previous page-exclusive articles, entities, and Qdrant vectors, re-running Docling OCR, photo harvesting, and semantic embedding without re-processing the entire 24+ page issue.
- **👁️ VLM Spatial Grounding & Sub-Photo Crop Recovery**: Uses Qwen-VL native visual grounding (`detect_subphotos_via_vlm_grounding`) to identify and crop discrete editorial portraits, insets, and standalone charts on composite display pages where heuristic boundary detection misses them.
- **💬 Conversational Context Condenser & In-Context Meta-Queries**: Seamlessly resolves pronouns (*"its"*, *"they"*, *"them"*) across dialogue turns while short-circuiting in-context meta-queries (*"what was the date"*, *"which newspaper"*, *"show citations"*) directly from chat history without triggering unnecessary retrieval cascades.
- **📑 Comprehensive Schema Documentation & CLI Inspector**: Built-in `make schema` and `make schema-list` commands coupled with full documentation in `docs/database_schema.md` detailing all 17 MySQL tables, relational invariants, and query JSON payloads.
- **🛡️ Intelligent Context Budgeting & Multi-Date Routing**: Dynamically resolves newspaper brand names to IDs, detects multi-date comparative queries (e.g., comparing Aug 1 and Aug 2 editions of the same publication) to schedule targeted SQL issue summaries, and caps evidence context tokens to prevent local LLM context overflow (4,096 tokens) or pre-training cutoff date hallucinations.
- **🔤 Corrupted Font CMap Recovery & Image OCR Fallback**: Automatically detects missing/broken `ToUnicode` CMaps or replacement character (`\ufffd`) dominance in PDF streams and escalates to pure image OCR via `GoogleCloudVisionOCR` and `LayoutAnalyzer`, recovering verified text and articles on complex broadsheets without character corruption.
- **⚖️ Deterministic Cross-Newspaper Differential Coverage Engine**: Computes exact article differences between publications on the same date (*"In Newspaper A but not in Newspaper B"* via `sql_analytics.get_newspaper_coverage_difference`), performing headline token overlap scoring to segregate regional/hyperlocal exclusives from shared wire stories with exact page folios and sections.
- **🔒 Dynamic Publication & Date Isolation**: Prevents cross-turn conversation context contamination through query-aware guardrails in `extract_active_issue_from_history()`, strict active publication prompt scoping (`Verified Available Publications for this Query`), and complete client-side storage resets.
- **🔢 Conversational Follow-Up Enumeration**: Seamlessly resolves multi-turn follow-ups (e.g. *"list all those articles"*) by preserving the differential comparison context and rendering complete, un-truncated article manifests from the relational database.
- **🎯 Interactive Scan Reader & Visual Asset Inspector**: High-resolution 300 DPI broadsheet reader with spatial bounding-box pulses, visual sidebar badges (`📷 Photo`, `📊 Infographic`, `🔢 Table`), on-demand VLM photo analysis, and single-page re-ingest button with live status banners.
- **🔍 Deep Multimodal Visual Asset Inspection (`inspect_visual_asset`)**: Equips the agent with a 5-tier Strategy Cascade (A: `photo_id` + companion charts; B: headline lookup; C: `article_id` + companion charts; D: multi-criteria DB search; E: scoped caption/VLM search) and lazy on-demand VLM extraction streaming raw crops directly from MinIO `bucket_pages` to enrich placeholder descriptions into structured Markdown tables and quantitative metrics.
- **🔗 Broadsheet Reader & Agent Assistant Visual Deep-Linking**: Interactive `"Ask Agent About This Infographic / Photo"` buttons on visual asset overlays and inspector cards, opening the Agent Assistant with an active attached asset pill (`attachedAsset`), automatically binding asset IDs for immediate multimodal analysis and streaming visual citation cards with image thumbnails (`/api/photos/{id}/image`).
- **🛡️ Conversational Citation Parsing & Anti-Leakage Guardrails**: Features `parse_inline_citation()` supporting broadsheet bracketed formats (`[4] Newspaper, Date, Page, Headline: "..."`) combined with 3 strict cross-turn parameter eviction guardrails purging stale `article_id`, `photo_id`, `headline`, and `target_newspapers` when switching dates, publications, or topics.
- **🛠️ Dynamic Tool Synthesis & Closed-Loop ToolCritic**: When broadsheet research queries exceed the capabilities of predefined tools, the agent synthesizes ad-hoc Python/SQL analysis tools on demand (`backend/app/agent/tool_maker.py`). Generated tools are audited by **`ToolCritic`** across 5 quantitative dimensions (SASC, SRF, REH, DSF, RPS) with automated retry refinement and executed inside an isolated subprocess AST Sandbox (`backend/app/agent/sandbox.py`) with a 15-second timeout, 512MB RAM cap, and read-only rollback transactions.
- **🔄 Reflexive CRAG Evaluator & Closed-Loop Adaptive Re-Planning**: Features a hybrid Fast-Floor check (<5ms) that immediately approves high-confidence broadsheet evidence ($\ge 1$ article, $\ge 100$ words) and a reflexive LLM judge (`evaluate_evidence_async`) producing a typed `EvaluationVerdict` (`quality_score`, `gap_diagnosis`, `recommended_action`, `corrective_hints`). When gaps are diagnosed, the adaptive re-planner modifies date scopes, expands `top_k`, and reformulates searches with an anti-repetition guard across LangGraph recovery nodes (`execute_adaptive_replan`, `execute_dynamic_code`) under a strict 1-cycle ceiling.
- **📐 Dynamic Answer Blueprint Architecture**: Decouples presentation planning from response generation. The planner synthesizes a tailored `AnswerBlueprint` (`SectionSpec` formats: narrative, bullet lists, markdown tables, metric cards, timelines; target word counts, table columns, prohibited elements) which the synthesizer dynamically compiles into prompt guidelines (`compile_structure_from_blueprint()`), replacing rigid static prompt cascades while strictly preserving broadsheet citations `[Newspaper, YYYY-MM-DD, Page N, "Headline"]`.
- **📰 Single-Article Full-Text Context & Robotic Table Elimination**: Preserves up to 7,500 characters of full parent article text for focused inquiries while suppressing bulky visual annotation noise. Deterministically detects and strips repetitive metadata tables (`clean_robotic_catalog_tables`) from single-article summaries.
- **⚡ Native Fast-Path Relational SQL Analytics & Dual Evidence Grounding**: Fast-path SQL handlers (~10ms) for photo section distributions, advertisement counts, and issue summaries. Generates dual evidence (macro overview + individual article records) enabling interactive, clickable citation badges that highlight source articles on the broadsheet canvas.
- **📅 Cross-Date Context Isolation & Anti-Leakage Shield**: Eliminates cross-turn date and asset contamination across multi-turn sessions. Explicit user query dates strictly override stale attached visual assets or previous-turn metadata, with automatic conflict eviction in `condenser.py`, `graph.py`, `query.py`, and non-overwriting date invariants in `executor.py`.
- **🎛️ Dynamic Model Settings Studio (`ModelSettingsStudio.jsx`)**: Comprehensive interactive settings studio for managing hosted and local AI providers (NVIDIA NIM, Ollama, Anthropic, OpenAI, Gemini, Groq), real-time API key configuration, live connectivity/latency health checks, visual capability indicators, and hot-swappable task bindings (`query_planner`, `synthesizer`, `vlm_extractor`, `embedding`, `ocr`) without server restarts.
- **📦 Consolidated Modular Ingestion Architecture**: Engineered along 4 cohesive architectural boundaries: dedicated subpackages for spatial layout (`layout/slugs.py`, `layout/analyzer.py`, `layout/segmenter.py`) and document parsers (`parsers/schemas.py`, `parsers/docling.py`, `parsers/vlm.py`, `parsers/ocr.py`), plus unified header metadata (`metadata.py`) and storage maintenance (`storage.py`), eliminating ~220 LOC duplicated regexes while guaranteeing 100% backward compatibility via proxy shims.

---

## 🏗️ System Architecture

```
                                 ┌──────────────────────────────────────────────────────────┐
                                 │                   React 18 + Vite SPA                    │
                                 │  • Newspaper Scan Reader with 300 DPI Bounding-Box Overlay│
                                 │  • Interactive Visual Asset Inspector (Photos/Infographics│
                                 │  • Real-Time Agentic Assistant with Reasoning Trace (SSE)│
                                 │  • Interactive Multi-Hop Entity Knowledge Graph UI       │
                                 │  • Cross-Newspaper Narrative Trajectory Explorer         │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │ REST / SSE Streaming
                                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FastAPI Backend Server                                    │
├─────────────────────────────┬───────────────────────────────┬─────────────────────────────┤
│   Document Ingestion Flow   │    Retrieval Toolbelt & DB    │   Agentic Reasoning Flow    │
│  • Celery Async Ingestion   │  • MySQL 8 (System of Record) │  • LangGraph State Machine  │
│  • IBM Docling (DocLayNet)  │  • Qdrant Dense Vector Store  │  • Dynamic Query Planner    │
│  • Multi-Page Consensus     │  • MinIO Object Storage (S3)  │  • Dynamic Tool Synthesis   │
│  • 2D Spatial Photo Binding │  • Redis 7 (Cache & Lock)     │  • Subprocess AST Sandbox   │
│  • Qwen3-VL Visual Extract  │  • RRF (Dense + Sparse Fusion)│  • Corrective RAG (CRAG)    │
└─────────────────────────────┴───────────────────────────────┴─────────────────────────────┘
```

---

## 🚀 Quick Start

NewsLens-AI can be deployed in three ways: **Full-Stack Docker Compose** (zero host dependencies), **Makefile Developer Automation** (fastest for active development), or **Manual Setup**.

### Prerequisites
- [Docker & Docker Compose](https://docs.docker.com/get-docker/) (v2.20+)
- *(Optional for host development)*: Python 3.12+ with [`uv`](https://docs.astral.sh/uv/) and Node.js 20+

---

### Option 1: One-Command Production Boot (Docker Compose — Recommended)

Run the entire application stack in containers with a single command:

```bash
# 1. Clone the repository
git clone https://github.com/piyushgoel2808/NewsLens-AI.git
cd NewsLens-AI

# 2. Copy the environment configuration template
cp .env.example .env

# 3. Build images and start all 8 services
docker compose up -d --build
```

#### Running Service Endpoints

| Service | Endpoint | Credentials / Details |
| :--- | :--- | :--- |
| **Frontend Web App** | [http://localhost:5173](http://localhost:5173) | Interactive broadsheet viewer, search & chat UI |
| **Backend API & Swagger** | [http://localhost:8000/api/docs](http://localhost:8000/api/docs) | Interactive OpenAPI / Swagger documentation |
| **Backend Health Check** | [http://localhost:8000/api/health](http://localhost:8000/api/health) | Live service connectivity status (DB, Qdrant, Redis, MinIO) |
| **MinIO Console** | [http://localhost:9001](http://localhost:9001) | `minioadmin` / `minioadmin` (broadsheet PDF & visual crops storage) |
| **Qdrant Vector Dashboard** | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | Broadsheet neural dense vector index |
| **Ollama Local LLM** | [http://localhost:11434](http://localhost:11434) | Local inference engine |
| **MySQL 8** | `localhost:3306` | `newslens` / `newslens_pass` (`newslens_db`) |
| **Redis 7** | `localhost:6379` | Query cache & Celery message broker |

> [!TIP]
> To follow live logs across all containers:
> ```bash
> docker compose logs -f
> ```
> To stop all containers and preserve volumes:
> ```bash
> docker compose down
> ```

---

### Option 2: One-Command Developer Boot (Makefile)

If you have `uv` and `npm` installed locally, initialize dependencies, database migrations, and development containers with one command:

```bash
# 1. Clone and enter directory
git clone https://github.com/piyushgoel2808/NewsLens-AI.git
cd NewsLens-AI

# 2. Initialize environment and start infra containers
cp .env.example .env
make setup

# 3. Start backend & frontend in development mode
make dev
```

---

### Option 3: Manual Step-by-Step Setup

```bash
# 1. Start backing infrastructure in Docker
docker compose up -d mysql qdrant minio redis ollama

# 2. Backend setup with uv
cd backend
uv sync --all-extras
uv run alembic upgrade head
uv run uvicorn app.api.main:create_app --factory --host 0.0.0.0 --port 8000 --reload

# 3. Frontend setup with npm (in a new terminal tab)
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser!

---

## ⚙️ Model Provider Configuration (`model_config.yaml`)

NewsLens-AI supports declarative provider bindings without changing application code:

```yaml
providers:
  groq_compound:
    provider: groq
    model: groq/compound
    context_window: 128000
    supports_tool_use: true

  gemini_flash:
    provider: gemini
    model: gemini-3.7-flash
    context_window: 1000000
    supports_vision: true
    supports_tool_use: true

  ollama_llama3:
    provider: ollama
    model: llama3.1:8b
    base_url: http://localhost:11434

  ollama_qwen3vl:
    provider: ollama
    model: qwen3-vl:latest
    base_url: http://localhost:11434
    supports_vision: true

  local_embed_bge:
    provider: local_sentence_transformers
    model: BAAI/bge-m3
    embedding_dim: 1024

task_bindings:
  query_planner: groq_compound
  answerer: groq_compound
  layout_analysis: docling_parser
  article_segmentation: ollama_llama3
  visual_extraction: ollama_qwen3vl
  embedding: local_embed_bge
```

---

## 🧪 Testing & Code Quality

```bash
# Run backend linter & type checks
cd backend
uv run ruff check .
uv run mypy app/

# Run complete test suite (512 unit & integration tests — 100% passing)
uv run pytest tests/ -v

# Verify frontend production build
cd ../frontend
npm run build
```
---

## 📚 Documentation & Technical Deep-Dives

- **[End-to-End Data Flow & Data Structure Guide](docs/end_to_end_data_flow_guide.md)**: Exhaustive walkthrough of every stage from raw PDF ingestion, Docling layout parsing, and 12+ SQL tables to query condensation, cognitive planning, all 8 tool executions (including `dynamic_analysis`), CRAG evaluation, Synthesizer prompt budgeting, SSE streaming, and IR evaluation metrics.
- **[System Architecture & Subsystems](docs/architecture.md)**: Full architecture breakdown covering the 10 core subsystems, LangGraph agent workflows, AST sandbox execution, cross-date anti-leakage shield, and performance profiles.
- **[Complete End-to-End Data Flow & System Architecture](docs/data_flow_architecture.md)**: Unified master technical specification tracing the broadsheet intake pipeline, visual VLM failover, query planner flowchart, all 8 tools, AST sandbox execution, CRAG evaluation, and resilience matrices.
- **[Tools Reference Guide & Dynamic Top-K](docs/tools_reference_guide.md)**: Comprehensive manual for all 8 agentic tools, argument schemas, execution invariants, and dynamic `top_k` scaling flowcharts.
- **[Hallucination Prevention, CRAG & Self-Correction](docs/hallucination_prevention_and_crag.md)**: Deep dive into the 6-layer hallucination defense, ToolCritic 5-metric scorecards, legitimate absence discrimination, and closed-loop fallback pathways.
- **[Feature Matrix & Capabilities](docs/features.md)**: Complete guide to all broadsheet capabilities, negative coverage audits, VLM extraction, and on-demand tool synthesis.
- **[Codebase Architecture & File Reference Guide](docs/codebase_directory_and_file_reference.md)**: Complete directory tree, folder responsibilities, and file-by-file technical reference detailing classes, functions, external tools/frameworks, and LLM/VLM models used.
- **[Relational Database Schema & Manifest Reference](docs/database_schema.md)**: Deep dive into all 17 MySQL tables, relational invariants, foreign keys, spatial bounding boxes, and manifest queries.
- **[Engineering & Incident Log](docs/engineering_log.md)**: Chronological engineering log documenting architectural decisions, performance milestones, and bug resolutions (Phases 1 through 18).

---

## 🔧 Troubleshooting & FAQ

### 1. Ollama Models Not Found
If running with local Ollama models (`llama3.1:8b` or `qwen3-vl:latest`), pull them into the container:
```bash
docker exec -it newslens-ollama ollama pull llama3.1:8b
docker exec -it newslens-ollama ollama pull qwen3-vl:latest
```
Alternatively, switch the primary task bindings in `backend/app/core/model_config.yaml` to hosted providers (e.g. Groq, Gemini, OpenRouter, NVIDIA NIM) by setting their respective API keys in your `.env`.

### 2. Database Connection Errors / Migrations
If the backend cannot connect to MySQL on startup, ensure the MySQL container has reached healthy status:
```bash
docker compose ps mysql
```
To manually apply migrations:
```bash
make migrate
# Or: cd backend && uv run alembic upgrade head
```

### 3. MinIO Storage Buckets
The backend automatically initializes required buckets (`bucket_pages`, `bucket_crops`, `bucket_issues`, `bucket_articles`) on startup. You can access the visual MinIO console at [http://localhost:9001](http://localhost:9001) using credentials `minioadmin` / `minioadmin`.

### 4. Port Conflicts
If default ports are already bound on your host:
- MySQL: `3306`
- Qdrant: `6333`
- Redis: `6379`
- MinIO: `9000` (API) / `9001` (Console)
- Ollama: `11434`
- Backend: `8000`
- Frontend: `5173`

Update the corresponding port mappings in `.env` and `docker-compose.yml`.

---

## 🤝 Community & Contributing

We welcome contributions from the community! Whether reporting bugs, proposing new features, or improving broadsheet parsing algorithms:

- 📖 Review the **[Contributing Guidelines](CONTRIBUTING.md)** for local development, code style, and PR processes.
- 🛡️ Review our **[Security Policy](SECURITY.md)** for responsible vulnerability reporting.
- 📜 Check out the **[Changelog](CHANGELOG.md)** for recent feature additions and release history.

---

## 📄 License

NewsLens-AI is open-source software licensed under the **[MIT License](LICENSE)**. © 2026 Piyush Goel.
