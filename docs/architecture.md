# NewsLens-AI Architecture & System Design Specification

NewsLens-AI is an **Enterprise-Grade Agentic Intelligence Platform** purpose-built for historical broadsheet and modern newspaper archives. It provides end-to-end multi-column PDF layout segmentation, multimodal visual infographic extraction, deterministic OCR matrix reconstruction, hybrid dense/sparse vector retrieval, conversational multi-turn query synthesis with Corrective RAG (CRAG), cross-publication narrative trajectory tracking, and visual broadsheet transparency.

---

## 1. High-Level System Architecture

```
                                 ┌──────────────────────────────────────────────────────────┐
                                 │                 React 18 + Vite SPA Client               │
                                 │  • Newspaper Scan Reader with 300 DPI Bounding-Box Overlay│
                                 │  • Interactive Visual Asset Inspector (Photos/Infographics│
                                 │  • Real-Time Agentic Assistant with Reasoning Trace (SSE)│
                                 │  • Interactive Multi-Hop Entity Knowledge Graph UI       │
                                 │  • Cross-Newspaper Narrative Trajectory Explorer         │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │ REST / SSE Streaming
                                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FastAPI Backend Server                                  │
├─────────────────────────────┬───────────────────────────────┬─────────────────────────────┤
│   Document Ingestion Flow   │    Retrieval Toolbelt & DB    │   Agentic Reasoning Flow    │
│  • Celery Async Ingestion   │  • MySQL 8 (System of Record) │  • LangGraph State Machine  │
│  • PyMuPDF + Docling Layout │  • Qdrant Dense Vector Store  │  • Dynamic Query Planner    │
│  • Multi-Page Consensus     │  • MinIO Storage (S3 Crops)   │  • 7-Tool Multimodal Dispatch│
│  • Spatial Column De-bundle │  • Redis 7 (Cache & Lock)     │  • Corrective RAG (CRAG)    │
│  • Probabilistic 12-Domain  │  • RRF (Dense + Sparse Fusion)│  • 4-Tier Synthesizer       │
│  • Spatial Matrix OCR Parser│  • Fulltext MySQL Indexing    │  • Strict Provenance Citator│
└─────────────────────────────┴───────────────────────────────┴─────────────────────────────┘
```

---

## 2. Technology Stack & Framework Comparison

| Layer / Component | Chosen Technology | Evaluated Alternatives | Why Chosen & Key Architectural Trade-offs |
|---|---|---|---|
| **Language & Runtime** | **Python 3.12 / 3.13** | Node.js, Go, Rust | First-class ecosystem for AI/ML (PyTorch, PyMuPDF, Sentence-Transformers, LangGraph, Docling, PIL, Tesseract) and high-performance async I/O. |
| **Package Management** | **`uv`** (Astral) | Poetry, Pipenv, Conda | **10–100x faster** dependency resolution and installation. Standard PEP 621 `pyproject.toml` support without vendor lock-in; ideal for CI and container builds. |
| **Web Framework** | **FastAPI + Uvicorn** | Flask, Django, Express | Native async/await concurrency, automatic OpenAPI/Swagger documentation, Pydantic v2 data validation, and first-class Server-Sent Events (SSE) streaming. |
| **Task Queue** | **Celery + Redis** | Celery+RabbitMQ, RQ, Dramatiq | Distributed background processing for multi-page broadsheet OCR and VLM extraction with Redis serving dual roles (Celery broker and query cache). |
| **System of Record** | **MySQL 8** (`aiomysql` + `pymysql`) | PostgreSQL / pgvector | Strict relational schema, battle-tested `FULLTEXT` indexing on broadsheet text, native JSON payload columns, and high-throughput async connections via `aiomysql`. |
| **Vector Database** | **Qdrant** | Pinecone, Milvus, Chroma, Weaviate | Self-hostable, rust-powered vector search with rich payload filtering (newspaper, issue_date, section, article_type, prominence), cosine similarity, and low memory footprint. |
| **Object Store** | **MinIO** (S3-compatible) | Local Filesystem, AWS S3 only | Local S3-compliant distributed object storage for 300 DPI page scans and high-res image crops, allowing seamless transition to AWS S3/GCS without code changes. |
| **PDF Extraction & Layout** | **PyMuPDF + Docling** | PDFMiner, Poppler, naive OCR | PyMuPDF provides ultra-fast digital text/font extraction and high-res rasterization; IBM Docling (DocLayNet) provides 2D spatial layout and reading-order tree analysis. |
| **OCR Engines** | **Tesseract + RapidOCR** | Tesseract alone, Cloud Vision only | RapidOCR (ONNX) and Tesseract provide high-speed local character transcription, token coordinate bounding boxes, and multi-language support (English + Indic scripts). |
| **Frontend Framework** | **React 18 + Vite** | Next.js, Nuxt, Angular | Lightweight client-side Single Page Application (SPA), instant HMR development with Vite, zero unnecessary server-rendering overhead for desktop analytical tools. |
| **Styling & Icons** | **Tailwind CSS + Lucide** | Material UI, Ant Design, Bootstrap | Utility-first styling for complex responsive broadsheet canvas layouts, crisp typography, dark mode support, and comprehensive icons. |

---

## 3. Model Evolution, Provider Strategy & Selection Rationale

NewsLens-AI employs a **Hot-Swappable Provider Registry Architecture** (`model_config.yaml`), decoupling high-level application workflows from specific LLM vendors.

### Evolution & Model Transitions
1. **Local vs. Hosted Provider Flexibility**:
   - **Local Inference (Ollama & Sentence-Transformers)**: Supports privacy-conscious, offline deployments using `llama3.1:70b` / `llama3.2:3b` for planning and answer synthesis, `qwen2.5-vl` / `qwen3-vl` for visual layout triage, and `BAAI/bge-m3` for local dense embeddings.
   - **Hosted Production Models (NVIDIA NIM, Anthropic, OpenAI, Google, Groq, OpenRouter)**: Supports NVIDIA NIM (`nvidia/nemotron-3.5-lightning-30b-a3b` with native CoT reasoning streaming, `meta/llama-3.2-11b-vision-instruct` for multimodal layout analysis), `claude-sonnet-4-5`, `gpt-4o`, `gemini-3.7-flash`, and Groq LPU inference for ultra-fast response times.
2. **Why `BAAI/bge-m3` as Default Embedding**:
   - 1024-dimensional dense representation.
   - 8,192-token context window (accommodates lengthy long-form newspaper articles without aggressive truncation).
   - Multi-lingual cross-lingual alignment (handles English, Hindi, and regional vernacular broadsheets).
   - Zero external API call costs and zero latency fluctuations.
3. **The Necessity of Deterministic Fallbacks**:
   - Local vision models (e.g. running on Apple Silicon or consumer GPUs) occasionally return empty responses (`''`) or encounter memory timeouts when parsing high-density financial matrices.
   - **The Deterministic Spatial OCR Matrix Reconstruction Engine** was engineered as a zero-failure fallback: when VLM structured extraction returns empty, the spatial matrix algorithm reconstructs tabular data directly from OCR bounding boxes with confidence $\ge 0.85$.

---

## 4. Core Subsystem Architecture

### A. Document Ingestion & Layout Analysis Pipeline (`backend/app/ingestion/`)

```
   ┌─────────────────────────────────────────────────────────────┐
   │                     Broadsheet PDF Scan                     │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │          PyMuPDF Lossless Intake & 300 DPI Raster           │
   │  • Extracts embedded digital text, font sizes & bbox boxes  │
   │  • Rasterizes high-res page image to MinIO Storage          │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │       Multi-Page Consensus Masthead & Date Extractor        │
   │  • Extracts publication name, volume, issue, and date       │
   │  • Cross-validates across first 3 pages to resolve consensus│
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │          Spatial Layout Analysis & Element Consolidation    │
   │  • Slices page into columns, headline bands, and text boxes │
   │  • Re-attaches drop-caps and repairs hyphenated line breaks │
   │  • Merges horizontal headline slices across column tracks   │
   │  • Detects statutory ad envelopes & injects barrier headers │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│ Article Linearization & Debundle │      │ Visual Asset Harvest & Matrix    │
│ • Column de-bundling (Shorts)    │      │ • Crops photos, charts & tables  │
│ • Kicker extraction & bylines    │      │ • Spatial photo-article binding  │
│ • Cross-page jump-line stitching │      │ • Dual VLM + Spatial OCR Matrix  │
└────────────────┬─────────────────┘      └────────────────┬─────────────────┘
                 │                                         │
                 └────────────────────────┬────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────┐
   │        Probabilistic 12-Domain Newsroom Classification       │
   │  • Multi-signal scoring: Headline (3x), Deck (2x), Body (1x)│
   │  • Context Anchor Dampening for financial/political idioms  │
   │  • Secondary Topic Extraction & MySQL Junction Persistence  │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │      Hierarchical Contextual Chunking & Qdrant Indexing     │
   │  • Prepends metadata headers [Newspaper|Date|Sec|HL|Page]   │
   │  • Generates dedicated visual data chunks for tables/charts │
   │  • Embeds via BGE-M3 (1024-dim) into Qdrant Vector DB       │
   └─────────────────────────────────────────────────────────────┘
```

> [!NOTE]
> **Subsystem Modularization & Clean Boundaries**:
> The broadsheet ingestion pipeline is organized into 4 cohesive architectural modules:
> - **`metadata.py`**: Consolidated Folio detection, RapidOCR masthead verification, and multi-page majority voting on issue date and brand consensus.
> - **`storage.py`**: Consolidated PDF stream deflation, 3-tier cascade hard deletion across storage tiers, and diagnostic debug artifact exports.
> - **`layout/` Subpackage**: Broadsheet spatial geometry, column slicing, reading order (`analyzer.py`), multi-page jump stitching (`segmenter.py`), and shared agency/dateline heuristics (`slugs.py`).
> - **`parsers/` Subpackage**: Extraction schemas (`schemas.py`), DocLayNet 2D neural parsing (`docling.py`), single-pass multimodal VLM (`vlm.py`), and OCR orchestrator (`ocr.py`).

---

### B. Dual-Engine Visual Infographic & Table Intelligence (`visual_extractor.py`)

Visual elements in broadsheets contain high-value quantitative data (e.g. IPO subscription matrices, stock indices, budget allocations). NewsLens-AI handles these via a 3-stage visual intelligence pipeline:

1. **Stage 1: Triage Classification**:
   - Fast evaluation via lightweight VLM / OCR numerical token density heuristic.
   - Categorizes crops into `data_chart`, `table`, `infographic`, `photo`, or `decorative`.
2. **Stage 2: Structured Extraction & Spatial OCR Matrix**:
   - **Primary**: Multimodal VLM structured prompt returning JSON containing `summary`, `markdown_table`, `key_metrics`, and `confidence`.
   - **Deterministic Fallback**: If VLM returns empty or errors, `extract_table_via_spatial_ocr()` executes:
     - Clusters OCR tokens into horizontal rows by vertical coordinate proximity.
     - Detects column centers and horizontal alignment lanes.
     - Transcribes clean GitHub-flavored Markdown tables and computes key metrics.
3. **Stage 3: Numerical Cross-Validation**:
   - Validates numerical tokens in the table against OCR ground truth to adjust final confidence scores.
4. **Stage 4: On-Demand VLM Extraction During Agent Query Execution**:
   - Visual assets ingested with placeholder descriptions (or fast-path crops) are enriched lazily during conversational query execution when `inspect_visual_asset` targets them.
   - The tool fetches raw crop bytes directly from MinIO `bucket_pages`, passes them to `VisualDataExtractor.process_image_crop()`, transcribes rich markdown metrics and summaries, and dynamically persists the synthesized extraction back into MySQL `article_photos.vlm_description`.
   - Guarantees zero cold-start latency during bulk PDF ingestion while delivering high-fidelity quantitative grounding whenever an agent or reader inspects a chart or infographic.

---

### C. Universal 12-Domain Newsroom Taxonomy & Metaphor Disambiguation (`classifier.py`)

Broadsheet language is heavily idiomatic. Financial and political articles frequently borrow sports, military, and entertainment metaphors (*"Bulls hit market for a six"*, *"Political chess in cabinet reshuffle"*).

1. **12 Canonical Newsroom Desks**:
   `Business & Markets`, `Economy & Policy`, `Politics & Governance`, `National`, `World & International`, `Corporate & Industry`, `Technology & Startups`, `Sports`, `Entertainment & Culture`, `Science & Environment`, `Health & Medicine`, `Opinion & Editorial`.
2. **Multi-Signal Probabilistic Scoring**:
   $$\text{Score}(D) = 3.0 \times \sum_{w \in \text{HL}} \text{tf}(w) + 2.0 \times \sum_{w \in \text{Deck}} \text{tf}(w) + 1.0 \times \sum_{w \in \text{Body}} \text{tf}(w)$$
3. **Domain Context Anchor Dampening**:
   If domain anchors for finance/politics (*Sensex, Nifty, RBI, Revenue, Cabinet, Parliament, FDA*) are present, metaphorical sports/war keywords receive a dampening penalty ($0.15\times - 0.25\times$), preventing misclassification.
4. **Multi-Topic Secondary Tagging**:
   Articles exceeding a secondary score threshold ($\ge 3.0$ and within 40% of top score) are stored in `article_topics` junction table, allowing cross-desk multi-facet retrieval.

---

### D. Geometric Advertisement Barrier Isolation (`layout/analyzer.py`, `layout/segmenter.py`)

Statutory and commercial disclosures (*QIP announcements, IPO prospectus summaries, tender notices*) often lack standard news headlines and occupy multi-column rectangular zones.

1. **Envelope Detection**: Detects clusters of statutory keywords (`QUALIFIED INSTITUTIONS PLACEMENT`, `BOOK RUNNING LEAD MANAGERS`, `ISSUE PRICE`, `REGISTRAR TO THE ISSUE`) and constructs convex bounding envelopes.
2. **Synthetic Boundary Injection**: Injects synthetic barrier headline elements (`[Advertisement] <Ad Title>`) at the top of the envelope.
3. **Reading Order Isolation**: `ArticleSegmenter` isolates the advertisement into a dedicated `[Advertisement]` article, preventing adjacent editorial news columns from absorbing the advertisement copy.
4. **Marketing Slogan Byline Suppression**: `MARKETING_SLOGAN_REGEX` rejects commercial taglines (*"By Innovation I Built For The Future"*, *"Backed by Trust"*) from being parsed as journalist bylines.

---

### E. Agentic Broadsheet Reasoning Lifecycle (`backend/app/agent/`)

```
   ┌─────────────────────────────────────────────────────────────┐
   │                     User Natural Query                      │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                   Redis Query Cache Check                   │
   │  • SHA-256 hash lookup of normalized query string           │
   │  • Immediate sub-millisecond return on cached hits          │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ (Cache Miss)
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │           Conversational Context Condensation Node          │
   │  • Resolves pronouns ("its", "they", "this newspaper")      │
   │  • Binds active newspaper issue and date from chat history  │
   │  • Propagates active reader attached assets (article/photo) │
   │  • Short-circuits ambiguous initial queries with guidance   │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │           Cognitive Query Planner & Intent Router            │
   │  • Grounded with live archive metadata (dates, papers, cats) │
   │  • Classifies into 1 of 7 Broadsheet Query Archetypes        │
   │  • Dynamic Answer Blueprint Generation:                      │
   │    - Formulates SectionSpec (narrative, table, metric_card)  │
   │    - Target word counts, table columns, prohibited elements  │
   │  • Dispatches optimal tool execution sequence:               │
   │    - sql_analytics (issue manifest, photo counts, ads, stats)│
   │    - hybrid_search (dense Qdrant + sparse MySQL RRF)         │
   │    - inspect_visual_asset (multimodal charts, tables, photos)│
   │    - entity_search (knowledge graph & salience lookups)      │
   │    - timeline_builder (thematic chronological progression)   │
   │    - coverage_analyzer (cross-newspaper comparison)          │
   │    - web_search (real-time live internet grounding)          │
   │    - dynamic_analysis (on-demand synthesized Python/SQL tool)│
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │      Concurrent Tool Dispatch & Adaptive Execution Engine   │
   │  • Executes planned tools in parallel via asyncio.gather    │
   │  • Native fast-paths: photo section counts, ad audits (~10ms)│
   │  • Dual evidence generation for interactive citation pills  │
   │  • Deep Visual Inspection Cascade (Strategies A through E)  │
   │  • Cross-Date Invariant: Query date strictly overrides asset│
   │  • Real-time adaptive fallback on 0-hit category filters    │
   │  • Resilient multi-tier issue ID fallback in sql_analytics  │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │      Reflexive CRAG Evaluator & LLM-as-Judge Engine         │
   │  • Fast-Floor Check (<5ms): immediate approval for >=100    │
   │    words of clean broadsheet editorial evidence             │
   │  • Reflexive LLM Judge: generates EvaluationVerdict         │
   │    (quality_score, gap_diagnosis, recommended_action)       │
   │  • Conditional LangGraph Routing with 1-Cycle Ceiling:      │
   │    ├─► Sufficient ───────────────► Synthesize Answer        │
   │    ├─► Replan Needed ────────────► execute_adaptive_replan  │
   │    └─► Dynamic Tool Needed ──────► execute_dynamic_code     │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │     Dynamic Blueprint-Driven Broadsheet Synthesizer         │
   │  • Dynamic Prompt Compilation from AnswerBlueprint           │
   │  • Single-Article Full-Text Budgeting (up to 7,500 chars)   │
   │  • Deterministic Robotic Catalog Table Stripping            │
   │  • Photo Visual Scene Annotation Noise Sanitization         │
   │  • Quantitative Metric Absence Hard-Stop                    │
   │  • Broadsheet Ligature Repair & Author Box Cleansing        │
   │  • Strict 1-Shot Citations: [Paper, YYYY-MM-DD, Page, Title]│
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │            SSE Streaming Delivery & Audit Logging           │
   │  • Streams response tokens, reasoning trace & tool metrics  │
   │  • Logs execution latency, cost, and query audit in MySQL   │
   │  • Stores plan and AnswerBlueprint in QueryLog.plan_json    │
   └─────────────────────────────────────────────────────────────┘
```

---

### F. Dynamic Tool Generation, Closed-Loop ToolCritic, and AST Sandbox Engine (`tool_maker.py`, `tool_critic.py`, `sandbox.py`, `sandbox_runner.py`)

When broadsheet analytical queries cannot be satisfied by static tools (e.g. "What is the variance of word counts across categories?", "How many pages are in the Aug 1 edition?", "Compare average article length across editions"), NewsLens-AI dynamically synthesizes, audits, and executes ad-hoc tools via the **Closed-Loop LLM-as-Tool-Maker** pattern with automated self-refinement.

```
┌─────────────────────────────────────────────────────────────┐
│                 User Query / Fallback Event                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 ToolMaker (tool_maker.py)                   │
│ • Schema Prompt with MySQL broadsheet tables & relationships │
│ • Auto-Import Pre-Injection (ensure_standard_imports):       │
│   Pre-injects re, math, statistics, json, datetime, pd, np   │
│ • Synthesizes async function: `async def analyze(db, ...)`  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Generated Python Code
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           AST Safety Scanner (sandbox.py)                   │
│ • Validates AST against whitelist (math, re, json, datetime) │
│ • Blocks dangerous calls: os, sys, subprocess, eval, exec   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Safe Code
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Subprocess Sandbox Runner (sandbox_runner.py)      │
│ • Runs in isolated subprocess with RLIMIT_AS (512MB RAM cap) │
│ • 15s execution timeout; read-only DB transaction rollback   │
│ • Pre-imported exec_globals (re, math, pd, np, text)         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Raw Execution Output / Error
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           Diagnostic ToolCritic (tool_critic.py)            │
│ • 5-Dimension Automated Quality Audit:                      │
│   1. SASC: Syntactic & AST Security Compliance (1.0/0.0)    │
│   2. SRF:  SQL Relational & Schema Fidelity (0.0-1.0)       │
│   3. REH:  Runtime Execution Health (1.0/0.0)               │
│   4. DSF:  Data-to-Summary Faithfulness (0.0-1.0)           │
│   5. RPS:  Intent Alignment & Filter Plausibility (0.0-1.0) │
└──────────────────────────────┬──────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
[Audit Passes: Score >= 0.70]            [Audit Fails: Issues Detected]
         │                                           │
         ▼                                           ▼
┌─────────────────────────────────┐      ┌─────────────────────────────┐
│ Inject into Agent Evidence Pool │      │ Closed-Loop Self-Refinement │
│ • Injects ToolExecutionRecord   │      │ • Re-prompts LLM with       │
│ • High-Confidence Evidence(1.0) │      │   structured critique fixes │
│ • Emits SSE telemetry           │      │ • Up to 3 retry attempts    │
└─────────────────────────────────┘      └──────────────┬──────────────┘
                                                        │ Retry Loop
                                                        └───────► ToolMaker
```

#### The 5-Metric Evaluation Scorecard

1. **Metric 1: Syntactic & AST Security Compliance (SASC)**
   - Score: `1.0` or `0.0` (Hard gate).
   - Audits code against `ASTSafetyScanner` to ensure zero invocation of forbidden built-ins (`eval`, `exec`, `open`), banned dunders (`__subclasses__`), or blacklisted modules (`os`, `sys`, `subprocess`, `socket`).

2. **Metric 2: SQL Relational & Schema Fidelity (SRF)**
   - Score: `0.0` to `1.0` (Threshold: $\ge 0.70$).
   - AST-based SQL extraction parses SQL queries without assuming rigid `text("""...""")` formatting.
   - Detects hallucinated columns (e.g. `articles.published_at` -> remapped to `issues.issue_date`).
   - Prevents multi-table Cartesian multipliers (requires `DISTINCT` when joining `articles` with `pages`).
   - Validates category joins (`article_categories c ON a.category_id = c.id`) and permits valid `AS category` aliases.

3. **Metric 3: Runtime Execution Health (REH)**
   - Score: `1.0` or `0.0` (Hard gate).
   - Validates that the subprocess exited with returncode 0, without timeouts or unhandled exceptions.

4. **Metric 4: Internal Data-to-Summary Faithfulness (DSF)**
   - Score: `0.0` to `1.0` (Threshold: $\ge 0.70$).
   - **Legitimate Absence vs. Narrative Hallucination**: If a query returns 0 rows (`data: []`) and the summary truthfully acknowledges that no records were found, DSF awards a perfect `1.0`. If `data: []` but the summary fabricates positive numbers or extensive narratives, it is heavily penalized (`0.1`–`0.4`).
   - **Aggregate Computation Support**: For statistical queries (variances, distributions, photo counts), `data` is `[]` while results are stored in `metadata` and markdown tables in `summary`. DSF recognizes these structures, preventing false-positive hallucination rejections.
   - **Numerical Consistency Checks**: Audits cited numbers in the summary against exact metadata metrics (e.g., page counts, photo counts).

5. **Metric 5: Intent Alignment & Filter Plausibility (RPS)**
   - Score: `0.0` to `1.0` (Threshold: $\ge 0.70$).
   - Audits whether input dates (e.g., `2/8/2026`) were normalized to ISO-8601 (`2026-08-02`) before querying MySQL `DATE` fields.
   - Audits publication naming aliases (e.g., `goan` matching `The Goan`).
   - Grants immunity to legitimate out-of-range date queries.

#### Two-Layer Reactive Dynamic Fallback Architecture

1. **Layer 1: Unsupported Parameter Handoff (`executor.py`)**:
   - If the Planner dispatches `sql_analytics` with an unsupported `analysis_type` (such as `"word_count_variance"`, `"author_frequency"`, or custom multi-table aggregations not built into static methods), the executor automatically intercepts the call.
   - It delegates execution to `dynamic_analysis`, synthesizing a custom Python/SQL query tool on-the-fly with closed-loop `ToolCritic` auditing.

2. **Layer 2: CRAG Zero-Evidence Dynamic Fallback (`evaluator.py`)**:
   - When primary retrieval tools (e.g. `hybrid_search`) return zero hits or insufficient evidence (score $< 0.4$) on analytical queries, the Corrective RAG (CRAG) Evaluator intercepts the failure.
   - It invokes `ToolMaker` asynchronously to generate, audit, and execute a targeted ad-hoc analysis tool.
   - The recovered telemetry is injected into `AgentState["tool_executions"]` as high-confidence evidence ($1.0$), ensuring the Synthesizer has verified database facts to ground the final response.

---

### G. Cross-Date Context Isolation & Anti-Leakage Shield

In multi-turn broadsheet research, conversational context leakage poses a major hallucination hazard—specifically when users transition from exploring an attached visual asset on one date (e.g., Aug 5, 2026) to asking a general or specific question about another date (e.g., Aug 1, 2026). 

NewsLens-AI implements a **4-Tier Cross-Date Anti-Leakage Shield**:

1. **Query Condenser Isolation (`condenser.py`)**:
   - `extract_active_issue_from_history()` scans the new user query for explicit dates and newspaper brands.
   - If the user specifies a date that differs from the issue date stored in chat history, the condenser automatically evicts the stale history issue context, preventing date drift.

2. **Attached Asset Eviction Gate (`query.py`, `graph.py`)**:
   - When a user query includes an attached asset (e.g., an infographic from Page 12 on Aug 5), the router compares the asset's metadata against any explicit date mentioned in the query text.
   - If a conflict is detected (e.g., query specifies `Aug 1, 2026` while asset is `Aug 5, 2026`), the asset is pruned before entering the planning and execution graph.

3. **Sanitizer Date Reconciliation (`tool_factory.py`)**:
   - `reconcile_and_sanitize_arguments()` checks planned tool inputs against user query dates.
   - Any stale date filters inherited from attached asset parameters are sanitized to match the explicit query intention.

4. **Executor Date Non-Overwriting Invariant (`executor.py`)**:
   - In Strategy A and Strategy C of visual inspection and analytical dispatch, the executor enforces an immutable rule:
   ```python
   # Explicit query date strictly overrides attached asset date
   if explicit_query_date:
       effective_date = explicit_query_date
   else:
       effective_date = asset_date or default_date
   ```
   - Prevents stale asset metadata from poisoning database queries.

---

### H. Reflexive LLM-as-Judge Evidence Evaluation & Closed-Loop Agentic Re-Planning (`evaluator.py`, `planner.py`, `graph.py`)

When broadsheet queries encounter ambiguous, incomplete, or borderline retrieval evidence, NewsLens-AI employs a **Reflexive Closed-Loop CRAG Architecture**:

```
 ┌─────────────────────────────────────────────────────────────┐
 │                Retrieved Evidence Pool                      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │           Hybrid Fast-Floor Evaluation Bypass               │
 │ • Evidence has >= 1 high-confidence broadsheet hit AND      │
 │   >= 100 words of clean editorial body text                 │
 │ • Sub-5ms instant pass (is_sufficient=True, score=1.0)      │
 └──────────────┬──────────────────────────────┬───────────────┘
                │ (Passes Fast-Floor)          │ (Below Fast-Floor)
                ▼                              ▼
 ┌─────────────────────────────┐ ┌─────────────────────────────┐
 │      Proceed to Answer      │ │   Reflexive LLM-as-Judge    │
 │         Synthesizer         │ │ • Evaluates evidence depth  │
 └─────────────────────────────┘ │ • Emits EvaluationVerdict:  │
                                 │   - is_sufficient (bool)    │
                                 │   - quality_score (0.0-1.0) │
                                 │   - gap_diagnosis (str)     │
                                 │   - recommended_action      │
                                 │   - corrective_hints (list) │
                                 └──────────────┬──────────────┘
                                                │
          ┌─────────────────────────────────────┴─────────────────────────────────────┐
          ▼                                     ▼                                     ▼
 [action: proceed]                 [action: replan_static_tools]          [action: synthesize_dynamic_tool]
          │                                     │                                     │
          ▼                                     ▼                                     ▼
┌──────────────────┐                ┌───────────────────────────┐         ┌───────────────────────────┐
│ synthesize_answer│                │  execute_adaptive_replan  │         │   execute_dynamic_code    │
└──────────────────┘                │ • replan_with_feedback    │         │ • ToolMaker generation    │
                                    │ • Anti-repetition guard   │         │ • ToolCritic 5-D audit    │
                                    │ • Widens dates / top_k    │         │ • AST Subprocess Sandbox  │
                                    └─────────────┬─────────────┘         └─────────────┬─────────────┘
                                                  │                                     │
                                                  └──────────────────┬──────────────────┘
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │  Synthesize Final Answer    │
                                                      │  (1-Cycle Recovery Ceiling) │
                                                      └─────────────────────────────┘
```

1. **Hybrid Fast-Floor Bypass (`evaluator.py`)**:
   - Evaluates retrieved articles against hard minimum bounds (<5ms execution). If primary retrieval satisfies broadsheet editorial requirements ($\ge 1$ article with $\ge 100$ words of clean body text), bypasses LLM evaluation, saving latency and token overhead.
2. **Reflexive LLM-as-Judge (`evaluate_evidence_async`)**:
   - For complex, borderline, or zero-evidence cases, prompts an LLM judge with `EVALUATOR_SYSTEM_PROMPT` to analyze semantic relevance, entity completeness, and factual coverage.
   - Outputs a typed `EvaluationVerdict` diagnosing precise evidentiary gaps (e.g. *"Missing specific casualty figures from Southern edition"*).
3. **Closed-Loop Adaptive Re-Planner with Anti-Repetition Guard (`planner.py`)**:
   - `replan_with_feedback_async()` analyzes the gap diagnosis and previous tool execution records.
   - Dynamically widens date bounds, expands `top_k`, or rephrases query keywords while strictly prohibiting repeating tool calls that already failed.
4. **LangGraph Recovery Branches & 1-Cycle Ceiling (`graph.py`)**:
   - Implements conditional branching from `evaluate_and_fallback` to `execute_adaptive_replan` or `execute_dynamic_code`.
   - Bounded by a strict 1-cycle ceiling (`recovery_attempts < 1`), guaranteeing predictable response times.

---

### I. Dynamic Answer Blueprint Architecture (`models.py`, `planner.py`, `synthesizer.py`)

Prior implementations relied on a rigid 300-line static `if/elif/else` prompt cascade. Any user preference (e.g. *"summarize in 150 words"*, *"bullet points only"*, *"no tables"*) was vulnerable to being overridden by hardcoded archetype templates.

The **Dynamic Answer Blueprint Architecture** separates presentation planning from text generation:

1. **Pydantic Blueprint Schemas (`models.py`)**:
   - `SectionSpec`: Defines title, format type (`narrative`, `bullet_list`, `markdown_table`, `metric_card`, `timeline`), content guidelines, and optionality.
   - `AnswerBlueprint`: Encapsulates archetype, executive framing instructions, ordered `sections`, target word count, explicit table column schemas, prohibited elements (e.g. "no speculative prose", "no markdown tables"), and tone/style directives.
2. **Cognitive Blueprint Generation in Planner (`planner.py`)**:
   - During query planning, the planner evaluates both retrieval needs and presentation requirements, generating a tailored `AnswerBlueprint` in `PlanResult`.
   - Features constraint-aware heuristic defaults for all 7 archetypes with automatic adaptations for length limits, comparative tables, and chronological timelines.
3. **Dynamic Prompt Compilation (`synthesizer.py`)**:
   - `compile_structure_from_blueprint()` dynamically compiles the `AnswerBlueprint` into structured prompt instructions.
   - Enforces non-negotiable broadsheet citation formats `[Newspaper, YYYY-MM-DD, Page N, "Headline"]` and factual invariants while granting the LLM flexibility in presentation layout.
4. **Single-Article Budgeting & Robotic Table Elimination**:
   - Preserves up to 7,500 characters of full parent article text for single-article deep dives.
   - `clean_robotic_catalog_tables()` deterministically detects and purges mechanical metadata tables (`| # | Headline | Section | Page | Words |`) from single-article narrative answers.

---

## 5. Database Schema & Data Model (MySQL 8)

The system maintains **16 interconnected relational tables**:

```
 ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
 │  newspapers  │1       *│    issues    │1       *│    pages     │
 ├──────────────┤─────────├──────────────┤─────────├──────────────┤
 │ id (PK)      │         │ id (PK)      │         │ id (PK)      │
 │ name         │         │ newspaper_id │         │ issue_id     │
 │ publisher    │         │ issue_date   │         │ page_number  │
 │ country      │         │ edition      │         │ raster_key   │
 └──────────────┘         │ total_pages  │         │ width/height │
                          └──────┬───────┘         └──────┬───────┘
                                 │1                       │1
                                 │*                       │*
                          ┌──────┴───────┐         ┌──────┴───────┐
                          │   articles   │1       *│ article_pages│
                          ├──────────────┤─────────├──────────────┤
                          │ id (PK)      │         │ article_id   │
                          │ issue_id     │         │ page_number  │
                          │ headline     │         │ bbox_json    │
                          │ byline_author│         └──────────────┘
                          │ category_id  │
                          │ full_text    │
                          │ word_count   │
                          │ prominence   │
                          └──────┬───────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │1                      │1                      │1
         │*                      │*                      │*
  ┌──────┴───────┐        ┌──────┴───────┐        ┌──────┴───────┐
  │article_chunks│        │    photos    │        │article_topics│
  ├──────────────┤        ├──────────────┤        ├──────────────┤
  │ id (PK)      │        │ id (PK)      │        │ article_id   │
  │ article_id   │        │ article_id   │        │ topic_id     │
  │ qdrant_point │        │ object_key   │        │ relevance    │
  │ chunk_text   │        │ visual_type  │        └──────────────┘
  │ token_count  │        │ vlm_markdown │
  └──────────────┘        │ confidence   │
                          └──────────────┘
```

---

## 6. Key Learnings & Empirical Discoveries Across Iterations

| Phase / Iteration | Empirical Discovery & Challenge | Engineering Solution Implemented |
|---|---|---|
| **Phase 1: Coordinates** | Raster image pixels (300 DPI) differed from PDF digital coordinate points (72 DPI), misaligning bounding box overlays. | Implemented bidirectional coordinate scaling normalizers (`BBox.scale()`) standardizing all spatial polygons. |
| **Phase 2: Mint Shorts** | Compact summary columns contained 6–10 short stories under one banner, causing segmenters to merge them into giant articles. | Engineered multi-story column de-bundling that detects horizontal divider rules, bold uppercase lead-ins, and distinct bbox tracks. |
| **Phase 3: Typographic Ligatures** | Drop-caps (e.g. large initial "T") and OCR font ligatures created severed words (*"T he", "Ol estimates"*). | Added Pass 0 Drop-Cap Re-attachment and deterministic OCR headline repair dictionary. |
| **Phase 4: ToC Noise** | Front-page Table of Contents boxes and pull-quote author names (*"PENNY WONG AUSTRALIAN FOREIGN MINISTER"*) formed fake articles. | Implemented regex patterns and coordinate filters to detect and suppress index teasers and pull-quote attributions. |
| **Phase 5: Metaphors** | Financial papers used sports/war idioms (*"Bulls hit for a six"*), polluting sports desks with business articles. | Developed Domain Anchor Dampening ($0.15\times - 0.25\times$ penalty) when corporate/market anchor entities are detected. |
| **Phase 6: Empty VLMs** | Local VLMs occasionally returned empty outputs on dense tabular crops, leaving empty table records. | Built Deterministic Spatial OCR Matrix Reconstruction Engine to assemble Markdown tables directly from OCR coordinates. |
| **Phase 7: Ad Bleed** | Half-page commercial ads without standard headlines bled into bottom editorial stories (*Retail Investors Skip IPOs*). | Engineered convex ad-envelope detection, injecting barrier delimiter headers (`[Advertisement] ...`) to isolate ad units. |
| **Phase 8: Slogan Bylines** | Corporate marketing taglines (*"By Innovation I Built For The Future"*) matched byline regexes and became author names. | Introduced `MARKETING_SLOGAN_REGEX` to validate author candidates against commercial buzzword filters. |
| **Phase 9: Strict Issue IDs** | Users typing `"issue 84"` when the database stored `"Issue #88"` caused `sql_analytics` to return 0 results and poison chat context. | Upgraded `sql_analytics.py` with multi-tier fallback resolution matching by `(newspaper_name, issue_date)` when IDs mismatch. |
| **Phase 9.17: Autonomous Reasoning & Concurrency** | Blind planning without live archive metadata caused tool hallucination; serial tool execution caused latency spikes; 0-hit category filters starved evidence; author boxes became fake headlines. | Grounded planner with `get_archive_metadata()`; added `article_catalog` archetype (<200ms); ran tools concurrently via `asyncio.gather`; added adaptive zero-hit category fallback; sanitized author/doctor boxes with `sanitize_headline()`. |
| **Phase 9.18: NVIDIA NIM Hosted Acceleration** | Local LLMs struggled with 90s latency on long syntheses and lacked real-time thinking traces. | Integrated `NvidiaProvider` supporting `nvidia/nemotron-3.5-lightning` (<1s response, live `<think>` streaming) and `meta/llama-3.2-11b-vision`. |
| **Phase 9.19: Topic Integrity & Font Ligature Recovery** | Few-shot contamination overwrote user topics with generic phrases; 25s unconstrained coverage audits delayed domain queries; synthesizer fallbacks dropped multi-newspaper tables; OCR dropped font ligatures (`e   orts`). | Dynamic filler sanitization in planner; conditional coverage analysis; explicit `archetype` preservation in deterministic synthesizer; granular domain token stem budgeting; and dedicated Unicode ligature decomposition engine (`repair_text_ligatures`). |
| **Phase 9.23: Clean Modular Planner Architecture** | Monolithic 1,593-line `planner.py` combined NER extraction, tool construction, hallucination pruning, and routing, causing test fragility and tight coupling. | Refactored into 4 single-responsibility modules: `models.py` (schemas), `extractor.py` (NER & parameters), `tool_factory.py` (canonical tool builders & sanitizers), and `planner.py` (lean coordinator). |
| **Phase 9.24: Heuristic Disambiguation & Context Retention** | Heuristic fallback misclassified topic manifests as `quantitative_trend` instead of `article_catalog`; multi-turn follow-ups pruned active brand and date parameters. | Added structural disambiguation for `article_catalog` in heuristic fallback; guarded active multi-turn working context in `reconcile_and_sanitize_arguments()`. |
| **Phase 9.25: Agent State Machine Decoupling** | Monolithic 1,153-line `graph.py` combined ORM logic, execution, and CRAG evaluation; in-place imports inside coroutines caused `_ModuleLock` micro-stalls; CRAG discarded semantic vector hits. | Modularized into `executor.py` (tool dispatch & manifests) and `evaluator.py` (CRAG & semantic hit protection); reduced `graph.py` to 267 lines with native LangGraph conditional edge routing. |
| **Phase 9.26: Date Normalization & Multi-Edition Audit** | Slash-formatted dates (`1/8/2026`) failed database queries, omitting publications from comparative matrices. | Added universal ISO-8601 normalization (`normalize_date_to_iso`) in `sql_analytics.py`; integrated complete multi-edition manifests into comparative prompts. |
| **Phase 9.27: CrossEncoder Latency & Synthesizer De-Bloating** | Apple Silicon MPS backend caused 30.4s hybrid search latency spikes; off-domain articles polluted topical comparisons; synthesizer was 1,182 lines with quadruple-duplicated domain maps. | Forced `device="cpu"` on macOS for `CrossEncoderReranker` (80ms execution) and capped candidate pool at 20 (dropping latency from 30.4s to 1.0s); filtered domain noise in SQL analytics; enforced strict zero-coverage reporting; de-bloated `synthesizer.py` with centralized `DOMAIN_TAXONOMY` and modular static renderers. |
| **Phase 9.28: Decoupled Modular Synthesizer Architecture** | Monolithic 1,073-line `synthesizer.py` conflated 6 responsibilities (domain classification, context formatting, token budgeting, prompt templating, LLM streaming, and deterministic report generation). | Decoupled into single-responsibility modules: `taxonomy.py` (domain classification & scoring), `prompt_context.py` (evidence context & budgeting), and `fallback_presenter.py` (deterministic report engine), shrinking `synthesizer.py` to a lean coordinator (~320 lines) while preserving 100% backward compatibility and test coverage. |
| **Phase 10: Ingestion Subsystem Consolidation & Remediation** | Ingestion pipeline sprawl across ad-hoc modules; ad bleed into editorial text; phantom publication dates; false-positive article segmentation. | Consolidated ingestion into structured `layout/` and `parsers/` subpackages; implemented geometric ad barrier isolation; added multi-page majority voting for masthead dates; refined newsroom taxonomy with geopolitical domain anchors and contextual dampening. |
| **Phase 11: Multimodal Visual Intelligence & Conversational Guardrails** | Complex broadsheet charts and infographics were ignored during agent QA; conversational multi-turn context leaked outdated article/issue metadata; dual-page citation noise degraded credibility; hardcoded provider configs caused lock-in. | Introduced `inspect_visual_asset` tool with 5-tier Strategy Cascade A–E and on-demand MinIO VLM extraction; added Broadsheet Reader `"Ask Agent About This Infographic / Photo"` deep-link integration; engineered `parse_inline_citation` with strict cross-turn parameter eviction guardrails; eliminated dual-page citation formatting; added dynamic `ModelSettingsStudio.jsx`. |
| **Phase 12: Dynamic Tool Synthesis & AST Sandbox** | Unforeseen user analytics queries (page counts, edition size distributions, ad-hoc aggregations) failed on static tool definitions; CRAG had no recovery path for zero-evidence computational queries; running arbitrary LLM code posed security and data mutation risks. | Engineered LLM-as-Tool-Maker pattern (`tool_maker.py`) generating ad-hoc Python/SQL tools; built subprocess AST Sandbox (`sandbox.py`) with strict module/builtin whitelisting, 15s timeout, 512MB RAM cap, and read-only rollback transactions; integrated Two-Layer Dynamic Fallback (Layer 1 in `executor.py` for unsupported parameters; Layer 2 in `evaluator.py` for zero-evidence CRAG recovery). |
| **Phase 13: Cross-Date Context Isolation & Anti-Leakage Shield** | In multi-turn sessions with attached assets, asking a question about a different date (e.g. Aug 1 vs Aug 5) caused the agent to leak the attached asset date or overwrite user queries, querying the wrong newspaper edition. | Implemented 4-Tier Cross-Date Anti-Leakage Shield: conflict-aware date eviction in `condenser.py`, asset eviction gate in `query.py` and `graph.py`, tool argument reconciliation in `tool_factory.py`, and strict date non-overwriting invariant in `executor.py`. |
| **Phase 14: Closed-Loop Dynamic Tool Critic & Self-Refinement** | Syntactically valid generated dynamic tools hallucinated MySQL column names (`published_at`), introduced Cartesian multipliers in multi-table joins, or fabricated positive summaries when 0 rows were returned. | Engineered 5-dimension `ToolCritic` (`tool_critic.py`) evaluating SASC, SRF, REH, DSF, and RPS; AST SQL query parsing; legitimate absence distinction; closed-loop retry refinement with token-budgeted critiques; and auto-import pre-injection (`ensure_standard_imports`). |
| **Phase 15: Temporal Range Parsing & Native Relational Photo Analytics** | Month-wide queries (`"August 2026"`) locked to single dates; photo queries by section triggered 100s dynamic tool timeouts due to DSF aggregate table false-positives. | Added regex Month + Year extraction (`extractor.py`) mapping to `date_from`/`date_to`; added native `get_photo_counts_by_section` in `sql_analytics.py` and fast-path dispatch in `executor.py` (~10ms); updated ToolCritic to recognize markdown tables and metadata metrics in aggregate computations; enforced `QUANTITATIVE & STATISTICAL METRIC ABSENCE HARD-STOP` in `synthesizer.py`. |
| **Phase 16: Reflexive CRAG Evaluator & Closed-Loop Adaptive Re-Planning** | Static token overlap discarded valid dynamic tool evidence; retrieval dead-ends had no recovery path without repeating failing queries. | Built hybrid Fast-Floor (<5ms) + Reflexive LLM-as-Judge (`evaluate_evidence_async`) emitting structured `EvaluationVerdict`; added adaptive re-planner with anti-repetition guard; wired LangGraph conditional branches (`execute_adaptive_replan`, `execute_dynamic_code`) with strict 1-cycle ceiling. |
| **Phase 17: Context Full-Text Budgeting & Robotic Table Elimination** | Truncated chunks starved single-article synthesis; visual annotations added prompt noise; single-article narrative answers rendered robotic metadata tables. | Allocated up to 7,500 chars of `parent_article_text` for single-article queries; purged visual noise; added `clean_robotic_catalog_tables()` deterministic cleaner; added headline conflict detection in `condenser.py`. |
| **Phase 18: Dynamic Answer Blueprint Architecture** | Rigid 300-line static `if/elif/else` prompt cascade ignored explicit user formatting constraints (e.g. word counts, bullet points, table exclusions). | Designed Pydantic `SectionSpec` and `AnswerBlueprint`; dynamically synthesized blueprints in `planner.py`; compiled prompts via `compile_structure_from_blueprint()` in `synthesizer.py`, decoupling layout from generation while guaranteeing broadsheet citations. |

---

## 7. Future Work & Roadmap

1. **Temporal Lineage & Storyline Delta Tracking**:
   - Automated entity sentiment and financial valuation evolution across decades of archived issues.
2. **Vernacular Multi-Lingual Broadsheets**:
   - Extension of layout analysis and spatial OCR matrix reconstruction to Hindi (Dainik Bhaskar, Jagran), Tamil, Bengali, and Arabic scripts.
3. **Cross-Publication Bias Radar**:
   - Comparative framing analytics evaluating editorial sentiment and headline divergence on identical news events across multiple broadsheets.
4. **Real-Time Live E-Paper Ingestion**:
   - Automated S3 bucket watcher / Webhook pipeline for real-time dawn ingestion of daily PDF editions as they go to print.
5. **Multimodal Audio Briefings**:
   - Automated podcast-style audio news generation synthesizing verified facts from the daily executive summary.
