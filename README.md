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

- **📑 Dual-Mode Broadsheet Neural Layout & 2D Article Segmentation**: Uses **IBM Docling (DocLayNet)** with **RapidOCR** and 2D spatial sorting to cleanly coalesce multi-line headlines and subheadline/decks, extract inline bylines (`BY <NAME>`), and segment complex multi-story broadsheets without column bleeding. Supports **Dual Parsing Modes**: local containerized Docling (`docling_parser`) for offline/on-prem deployments, and **IBM Cloud Docling SaaS API** (`docling_cloud`) to offload heavy neural layout workloads with zero container compute overhead.
- **📷 2D Spatial Photo/Infographic Extraction & Multimodal Data Chunks**: Automatically isolates editorial photos, graphics, and composite photo galleries, pairs them with shared multi-column captions using 2D proximity scoring, and indexes infographic data tables into Qdrant via **Qwen3-VL** (`qwen3-vl:latest`).
- **🏛️ Multi-Page Consensus Masthead Verification**: Robustly extracts newspaper brand and publication dates across global (*The New York Times*, *The Wall Street Journal*, *Financial Times*, *The Washington Post*, *The Guardian*) and national broadsheets with RapidOCR visual verification.
- **⚡ Dual-Mode Vector Embeddings & Strict Dual Qdrant Collection Architecture**: Employs an intelligent dual vector pipeline:
  - **Cloud Mode (`gemini-embedding-001`)**: 768-dimensional embeddings utilizing Matryoshka Representation Learning (MRL) via Google Vertex AI / AI Studio with asymmetric retrieval task types (`RETRIEVAL_DOCUMENT` for chunk ingestion, `RETRIEVAL_QUERY` for search queries), indexing into **`article_chunks_v2`** with **zero container RAM footprint** in Cloud Run.
  - **Local/Hybrid Mode (`BAAI/bge-m3`)**: 1024-dimensional dense vectors via PyTorch / SentenceTransformers indexing into **`article_chunks`**.
  - **Auto-Dimension Collection Routing**: `QdrantStore` dynamically inspects vector dimensionality (768d vs 1024d) to isolate search and upserts to the matching collection, while `delete_by_filter()` operates across both collections to ensure clean issue re-ingestion.
- **⚡ 10x Fast Query Engine & Sub-2s Streaming TTFT**: Engineered an ultra-responsive query pipeline slashing Time-to-First-Token (TTFT) from 25–45s down to **1.8s–2.4s** and full streamed synthesis in **3.5s–5.0s**. Combines direct Google Vertex AI model routing (`gemini-2.5-flash`), calibrated thinking budgets (`thinking_budget: 0` for structured tool planning, CRAG evaluation, and streaming synthesis), adaptive CPU Cross-Encoder candidate capping ($K=8\dots16$ candidates), and CRAG dual-signal confidence fast-flooring while **strictly preserving 100% LLM comprehension of impure/conversational user queries and dynamic `AnswerBlueprint` output formatting**.
- **⚡ Two-Stage Neural Retrieval Cascade (Cross-Encoder)**: Stage 1 RRF hybrid search (Qdrant `article_chunks` / `article_chunks_v2` + MySQL `FULLTEXT`) expands candidates to $N=75$, followed by Stage 2 Cross-Encoder neural reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with macOS CPU optimization (avoiding MPS Metal shader lag) and adaptive candidate pool capping ($K=8\dots16$), returning the Top 10 high-precision hits in sub-second latency (~450ms vs 2.8s).
- **🔄 Background Vector Re-Indexing CLI (`scripts/reindex_embeddings.py`)**: Production-ready migration utility for backfilling existing MySQL chunks into `article_chunks_v2` using `gemini-embedding-001` with chunked batching, rate-limit backoff, and live progress indicators.
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
- **🚀 Dynamic Model-Aware Token Budgeting & Reasoning Headroom (`resolve_dynamic_token_budget`)**: Eliminates static archetype token caps in favor of dynamic per-model output envelopes derived from `ProviderCapability` (`max_output_tokens` 4,096 or 8,192). Auto-detects reasoning models (Gemini 2.5, DeepSeek-R1, Nemotron, QwQ, thinking models) with `reasoning_headroom: 2048`, preventing internal chain-of-thought tokens from starving final answers.
- **🧠 Authoritative Planner LLM Cognitive Routing**: Removed fast-path deterministic bypasses from LangGraph's classification node; the Planner LLM is the authoritative cognitive router operating with live relational schema, date bounds, and conversational memory, reserving heuristic planning solely as an emergency failover.
- **🧮 Statistical & Analytical Computation Archetype (`analytical_computation`)**: Formally decouples mathematical inquiries (*"calculate average word count"*, *"average length"*, *"correlation"*) from scalar archive counts, routing them directly to `dynamic_analysis` without artificial 80-word count ceilings.
- **🔍 OCR Noise Tolerance & Intelligent Semantic Reconstruction**: Dedicated synthesis directives authorizing the model to phonetically and semantically reconstruct words corrupted during broadsheet OCR scanning (*"Reconstruction is not hallucination"*), strictly prohibiting raw OCR typographical errors in final responses.
- **🛡️ Isolated Single-Article Prompt Context**: Enforces strict chunk filtering for article explanation and deep-dive queries, eliminating prompt context contamination from unrelated advertisements or outside page stories.

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

## 🚀 Quick Start & Deployment Options

NewsLens-AI supports dual-track execution out-of-the-box:
1. **Track A: Local Sovereign Mode (100% Free / On-Premise)**: Runs completely offline using Ollama, local Docling, and BAAI/bge-m3 embeddings. Zero external API keys required.
2. **Track B: Full Cloud Mode (GCP Production / Cloud Run)**: Offloads all heavy parsing to IBM Docling Cloud SaaS and vector embeddings to Google Gemini 001 MRL (768d), achieving zero container RAM overhead (~350 MB) and instant sub-2s cold starts.

---

### Option 1: One-Command Boot (Docker Compose)

Run the complete 8-service application stack with Docker Compose:

```bash
# 1. Clone the repository
git clone https://github.com/piyushgoel2808/NewsLens-AI.git
cd NewsLens-AI

# 2. Copy the environment configuration template
cp .env.example .env

# For Local Sovereign Mode (No API keys needed):
# Leave .env defaults as-is. It will use Ollama, local Docling, and local BGE-M3.

# For Full Cloud Mode (Google Cloud / Docling Cloud):
# Add your keys in .env:
#   GEMINI_API_KEY=AIzaSy...
#   DOCLING_API_KEY=azI6...
#   DOCLING_SERVICE_URL=https://api.aws-c1.dcls.saas.ibm.com/...

# 3. Build images and start all 8 services
docker compose up -d --build
```

#### Running Service Endpoints

| Service | Endpoint | Credentials / Details |
| :--- | :--- | :--- |
| **Frontend (Docker Stack)** | [http://localhost:3000](http://localhost:3000) | Production Nginx SPA with real-time SSE streaming |
| **Frontend (Vite Dev)** | [http://localhost:5173](http://localhost:5173) | Development server with instant HMR (`make frontend-dev`) |
| **Backend API & Swagger** | [http://localhost:8000/api/docs](http://localhost:8000/api/docs) | Interactive OpenAPI / Swagger documentation |
| **Backend Health Check** | [http://localhost:8000/api/health](http://localhost:8000/api/health) | Live service connectivity status (DB, Qdrant, Redis, MinIO) |
| **MinIO Console** | [http://localhost:9001](http://localhost:9001) | `minioadmin` / `minioadmin123` (broadsheet PDF & visual crops storage) |
| **Qdrant Vector Dashboard** | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | Broadsheet neural dense vector index |
| **Ollama Local LLM** | [http://localhost:11434](http://localhost:11434) | Local inference engine |
| **MySQL 8** | `localhost:3306` | `newslens` / `newslens_pass` (`newslens`) |
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

### Option 4: Production Cloud Deployment (Google Cloud Platform)

NewsLens-AI runs natively in production on **Google Cloud Platform (GCP)** in region `asia-south1` (Mumbai) across serverless Cloud Run services, Cloud SQL, Google Cloud Storage, Qdrant Cloud, and Upstash Redis.

#### Live Production Endpoints

| Component | Status | Production URL / Identifier |
| :--- | :--- | :--- |
| **Frontend UI** | **LIVE (200 OK)** | [https://newslens-frontend-679327043786.asia-south1.run.app](https://newslens-frontend-679327043786.asia-south1.run.app) |
| **Backend API** | **LIVE (200 OK)** | [https://newslens-backend-679327043786.asia-south1.run.app](https://newslens-backend-679327043786.asia-south1.run.app) |
| **Health Check** | **HEALTHY** | [https://newslens-backend-679327043786.asia-south1.run.app/health](https://newslens-backend-679327043786.asia-south1.run.app/health) |
| **Celery Worker** | **CONNECTED** | `newslens-worker` on Cloud Run (`--no-cpu-throttling`, 2Gi RAM, 1 vCPU, min-instances=1) |
| **Cloud SQL MySQL 8.0** | **MANAGED** | `newslens-ai-prod:asia-south1:newslens-mysql` |
| **Vector DB (Qdrant Cloud)** | **MANAGED** | `australia-southeast1-0.gcp.cloud.qdrant.io:6333`<br/>• `article_chunks` (1024d Cosine - BGE-M3)<br/>• `article_chunks_v2` (768d Cosine - Gemini 001 MRL) |
| **Object Storage (GCS)** | **ACTIVE** | `gs://newslens-ai-prod-pages` & `gs://newslens-ai-prod-originals` |
| **Redis & Message Broker** | **MANAGED** | Upstash Redis TLS (`rediss://...`) |
| **CI/CD Pipeline** | **AUTOMATED** | GitHub Actions with Workload Identity Federation (Zero permanent keys) |

#### GCP Architecture & Cost Optimization Highlights
1. **Cloud Run Serverless Services & Request-Based Billing**:
   - `newslens-frontend`: Lightweight Nginx 1.27 Alpine reverse proxy container serving the React SPA bundle, dynamic runtime environment substitution (`$PORT`), and proxying `/api/*` to the backend with unbuffered SSE streaming.
   - `newslens-backend`: FastAPI running under Python 3.12 with Gunicorn/Uvicorn workers, Cloud SQL Unix domain socket connectivity, configured with `--cpu-throttling`, 1 vCPU, and 2Gi RAM (CPU is billed only during active HTTP requests; idle CPU is free).
   - `newslens-worker`: Background Celery task consumer configured with `--no-cpu-throttling`, 2Gi RAM, 1 vCPU, `--min-instances=1`, and an embedded HTTP health server on `$PORT` to satisfy Cloud Run service liveness probes while processing ingestion queues 24/7 without Redis broken pipe drops.
   - `newslens-migrate`: Cloud Run Job running Alembic database migrations (`alembic upgrade head`) before revisions are deployed.
2. **Vertex AI Native Routing & GCP Credit Preservation**:
   - In production on Cloud Run, the Gemini provider automatically discovers Application Default Credentials (ADC) from the attached `newslens-runner` Service Account.
   - Inference routes directly through Google Cloud Vertex AI (`https://aiplatform.googleapis.com/v1/publishers/google/models`) using OAuth2 Bearer tokens.
   - This keeps 100% of LLM tokens and VLM OCR operations covered under **GCP Promotional Credits**, avoiding out-of-pocket credit card charges on Google AI Studio.
   - Local developers can seamlessly run against Google AI Studio (`AIza...` key) or Google Cloud Vertex AI (`USE_VERTEX_AI=true`).
3. **85–90% Cost Reduction Architecture**:
   - **Compute**: Moving the API backend from always-allocated CPU to request-based `--cpu-throttling` slashes continuous Cloud Run compute spend by ~85% (~₹800/day savings).
   - **Inference**: High-volume extraction uses `gemini-2.5-flash` with 150 DPI page rendering (reducing OCR token volume by 50% vs 300 DPI).
   - **In-Memory Model Caching**: Module-level process-wide `CrossEncoder` caching prevents 30–40s CPU spikes and redundant disk weight reloads per search query.
   - **Client Request Deduplication**: Frontend `apiDeduplicator` prevents simultaneous bursts of duplicate HTTP calls on initial page load.
4. **Zero Container RAM & Native Cloud Provider Offload**:
   - In production cloud mode (`cloud_full` preset), heavy neural compute is fully offloaded to managed cloud APIs: `gemini-embedding-001` eliminates loading 2.4 GB of PyTorch model weights inside the container, and `docling_cloud` offloads layout parsing to IBM Cloud.
5. **Dynamic Object Storage Abstraction**:
   - Production uses native `google-cloud-storage` (`GoogleCloudStorageStore`) against GCS buckets, while local development seamlessly uses MinIO (`MinioStore`) via `get_object_store()`.
6. **Automated CI/CD**:
   - Every push to `main` triggers `.github/workflows/deploy-gcp.yml`, which executes the test suite against an ephemeral MySQL 8 service container, authenticates to GCP via Workload Identity Federation, builds and pushes multi-arch images to Google Artifact Registry, runs database migrations, and updates Cloud Run revisions with zero downtime.

---

## ⚙️ Model Provider Configuration (`model_config.yaml`)

NewsLens-AI supports declarative provider bindings and dynamic presets without changing application code:

```yaml
providers:
  gemini_flash:
    provider: gemini
    model: gemini-2.5-flash
    context_window: 1000000
    supports_vision: true
    supports_tool_use: true

  gemini_embedding:
    provider: gemini_embedding
    model: gemini-embedding-001
    embedding_dim: 768  # 768-dim via Matryoshka Representation Learning (MRL)

  local_embed_bge:
    provider: local_sentence_transformers
    model: BAAI/bge-m3
    embedding_dim: 1024

  docling_cloud:
    provider: docling_cloud
    model: ibm-docling-saas

  docling_parser:
    provider: docling
    model: docling-local

task_bindings:
  query_planner: gemini_flash
  synthesizer: gemini_flash
  visual_extraction: gemini_flash
  layout_analysis: docling_cloud  # Or docling_parser for offline/local
  embedding: gemini_embedding    # Or local_embed_bge for offline/local
```

### Dynamic Preset Profiles (Model Settings Studio)

The UI's **Model Settings Studio** allows one-click switching between deployment archetypes:

| Preset Profile | Planning & Synthesis | Layout Parsing | Embeddings & Target Collection | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Full Cloud (`cloud_full`)** | `gemini_flash` | `docling_cloud` | `gemini_embedding` &rarr; `article_chunks_v2` (768d) | Production Cloud Run (Zero container RAM) |
| **Cloud Hybrid (`cloud_hybrid`)** | `gemini_flash` | `docling_parser` | `local_embed_bge` &rarr; `article_chunks` (1024d) | Cloud LLMs + Local PyTorch embeddings |
| **Local Offline (`local_offline`)** | `ollama_llama3` / `deepseek` | `docling_parser` | `local_embed_bge` &rarr; `article_chunks` (1024d) | 100% offline air-gapped workstations |

### Background Vector Re-Indexing CLI

When migrating existing broadsheet archives between vector embedding models, run the background re-indexer:

```bash
# Re-index all existing MySQL article chunks into Qdrant article_chunks_v2 (768d)
cd backend
python ../scripts/reindex_embeddings.py --target-model gemini_embedding --batch-size 32

# Or dry-run / inspect without modifying vectors:
python ../scripts/reindex_embeddings.py --target-model gemini_embedding --dry-run
```

---

## 🧪 Testing & Code Quality

```bash
# Run backend linter & type checks
cd backend
uv run ruff check .
uv run mypy app/

# Run complete test suite (574 unit & integration tests — 100% passing)
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
- **[Engineering & Incident Log](docs/engineering_log.md)**: Chronological engineering log documenting architectural decisions, performance milestones, and bug resolutions (Phases 1 through 29).

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
