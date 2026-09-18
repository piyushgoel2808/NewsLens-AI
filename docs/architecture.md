# NewsLens-AI Architecture & System Design Specification

NewsLens-AI is an **Enterprise-Grade Agentic Intelligence Platform** purpose-built for historical broadsheet and modern newspaper archives. It provides end-to-end multi-column PDF layout segmentation, multimodal visual infographic extraction, deterministic OCR matrix reconstruction, hybrid dense/sparse vector retrieval, conversational multi-turn query synthesis with Corrective RAG (CRAG), cross-publication narrative trajectory tracking, and visual broadsheet transparency.

---

## 1. High-Level System Architecture

```
                                 ┌──────────────────────────────────────────────────────────┐
                                 │                 React 18 + Vite SPA Client               │
                                 │  • Newspaper Scan Reader with 150 DPI Bounding-Box Overlay│
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
| **Object Store** | **MinIO** (S3-compatible) | Local Filesystem, AWS S3 only | Local S3-compliant distributed object storage for 150 DPI page scans and high-res image crops, allowing seamless transition to AWS S3/GCS without code changes. |
| **PDF Extraction & Layout** | **PyMuPDF + Docling** | PDFMiner, Poppler, naive OCR | PyMuPDF provides ultra-fast digital text/font extraction and high-res rasterization; IBM Docling (DocLayNet) provides 2D spatial layout and reading-order tree analysis. |
| **OCR Engines** | **Tesseract + RapidOCR** | Tesseract alone, Cloud Vision only | RapidOCR (ONNX) and Tesseract provide high-speed local character transcription, token coordinate bounding boxes, and multi-language support (English + Indic scripts). |
| **Frontend Framework** | **React 18 + Vite** | Next.js, Nuxt, Angular | Lightweight client-side Single Page Application (SPA), instant HMR development with Vite, zero unnecessary server-rendering overhead for desktop analytical tools. |
| **Styling & Icons** | **Tailwind CSS + Lucide** | Material UI, Ant Design, Bootstrap | Utility-first styling for complex responsive broadsheet canvas layouts, crisp typography, dark mode support, and comprehensive icons. |

---

## 3. Model Evolution, Provider Strategy & Selection Rationale

NewsLens-AI employs a **Hot-Swappable Provider Registry Architecture** (`model_config.yaml`), decoupling high-level application workflows from specific LLM vendors.

### Evolution & Model Transitions
1. **Local vs. Hosted Provider Flexibility**:
   - **Local Inference (Ollama & Sentence-Transformers)**: Supports privacy-conscious, offline deployments using `llama3.1:8b` / `deepseek-r1:14b` for planning and answer synthesis, `qwen2.5-vl` / `qwen3-vl` for visual layout triage, IBM Docling for 2D geometry parsing, and `BAAI/bge-m3` for local dense embeddings.
    - **Primary Hosted Cloud Engine (Google AI Studio / Gemini Cloud)**:
     - **`gemini-3.8-flash`** (Canonical Workhorse): 1,048,576-token context window, ultra-low latency, native multimodal visual extraction without downscaling, and zero-failure adherence to Pydantic JSON schemas. Bound to `query_planner`, `answerer`, `visual_extraction`, `query_condenser`, and `article_segmentation`.
     - **`gemini-3.8-live`** (Real-time Audio/Stream): Native audio-to-audio multimodal interaction via Gemini Live API.
     - **`gemini-3.5-flash`** (High-Speed Fallback): 1,048,576-token context window serving as zero-downtime fallback candidate.
     - **`gemini-3.1-pro-preview` / `gemini-3.1-flash-lite`**: Specialized reasoning and conversational lightweight models.
     - **Transparent Candidate Failover**: `GeminiProvider` incorporates an automatic multi-model candidate failover (`[gemini-3.8-flash, gemini-3.5-flash, gemini-3.6-flash, gemini-3.7-flash]`), ensuring continuous zero-downtime execution across varying Google API account access tiers without encountering deprecated 404 models.
   - **Hosted Gateways & Alternative Endpoints (Groq, OpenAI, NVIDIA NIM, OpenRouter)**: Supports Groq LPU inference (`groq_compound`, `groq_qwen`), OpenAI (`gpt-4o`, `gpt-4o-mini`), NVIDIA NIM (`nvidia/nemotron-3.5-lightning`), and optional multi-provider routing via OpenRouter.
2. **Why `BAAI/bge-m3` as Default Embedding**:
   - 1024-dimensional dense representation.
   - 8,192-token context window (accommodates lengthy long-form newspaper articles without aggressive truncation).
   - Multi-lingual cross-lingual alignment (handles English, Hindi, and regional vernacular broadsheets).
   - Zero external API call costs and zero latency fluctuations.
3. **The Necessity of Deterministic Fallbacks**:
   - Local vision models (e.g. running on Apple Silicon or consumer GPUs) occasionally return empty responses (`''`) or encounter memory timeouts when parsing high-density financial matrices.
   - **The Deterministic Spatial OCR Matrix Reconstruction Engine** was engineered as a zero-failure fallback: when VLM structured extraction returns empty, the spatial matrix algorithm reconstructs tabular data directly from OCR bounding boxes with confidence $\ge 0.85$.
4. **Resilient Dynamic Provider Failover (`ModelRegistry.get_chat_failover_candidates`)**:
   - Computes an ordered, filtered candidate list of active providers capable of chat/tool-calling.
   - Automatically prioritizes cloud endpoints (`gemini_flash`, `gemini_pro`, `groq_compound`, `openai_gpt4o_mini`, `groq_qwen`, `openrouter_nemotron`, etc.) or sovereign local Ollama instances based on `prefer_local` policy, ensuring uninterrupted agent execution during external rate limits or transient outages.

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
   │          PyMuPDF Lossless Intake & 150 DPI Raster           │
   │  • Extracts embedded digital text, font sizes & bbox boxes  │
   │  • Rasterizes high-res page image (150 DPI) to MinIO Storage│
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
│ Article Linearization & Debundle │      │ Single-Pass Visual Intelligence  │
│ • Column de-bundling (Shorts)    │      │ • Normalized region manifest     │
│ • Kicker extraction & bylines    │      │ • Gemini-3.8-Flash single pass   │
│ • Cross-page jump-line stitching │      │ • Qwen3-VL Semaphore(2) per-crop │
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
> - **`parsers/` Subpackage**: Extraction schemas (`schemas.py`), DocLayNet 2D neural parsing (`docling.py`), single-pass multimodal VLM (`single_pass_extractor.py`), and OCR orchestrator (`ocr.py`).

---

### B. High-Throughput Visual Intelligence & Single-Pass Extractor (`single_pass_extractor.py`, `visual_extractor.py`)

Visual elements in broadsheets contain high-value quantitative and journalistic data (e.g. corporate financial charts, stock matrices, infographics, editorial photojournalism). NewsLens-AI employs an adaptive visual extraction pipeline engineered for both enterprise throughput and zero hallucination:

1. **Adaptive Dual-Execution Strategy**:
   - **Cloud Vision (`gemini-3.8-flash`) — Unified Single-Pass**: Instead of making 30 separate sequential network calls for individual image crops, `SinglePassVisualExtractor` passes the entire master 150 DPI page image along with a normalized JSON manifest of target regions (`[x0, y0, x1, y1]` in float coordinates $0.0 \dots 1.0$). Gemini processes all visual elements concurrently in a single LLM turn (10–30s per page), returning an indexed JSON array of structured analyses.
   - **Local VLMs (`qwen3-vl:latest` via Ollama) — Concurrent Per-Crop**: Because local models can experience context saturation or monologue loops when fed multi-element broadsheet manifests, the pipeline automatically routes local models to concurrent per-crop extraction gated by `asyncio.Semaphore(2)` to balance CPU/GPU load while keeping memory consumption bounded.
2. **Deterministic Fallback & Zero Empty Description Guarantee**:
   - Every candidate region from the Docling layout harvest is tracked by `region_id`.
   - If the VLM response omits an item or returns a blank description, `_fallback_extract_region()` triggers automated PIL/OCR heuristics to populate guaranteed non-empty descriptions (`[News Visual Asset: Page X, Region Y]`).
3. **Preamble & Thinking Token Sanitization (`clean_vlm_text`)**:
   - Unclosed `<think>` reasoning tags, conversational preambles (*"Got it, let's analyze..."*), and extraneous markdown fences are purged using regex patterns before persisting descriptions to MySQL or Qdrant.
4. **Stage 2 Structured Extraction & Spatial OCR Matrix**:
   - **Primary**: Multimodal VLM structured prompt returning JSON containing `summary`, `markdown_table`, `key_metrics`, and `confidence`.
   - **Deterministic Fallback**: If VLM returns empty or errors, `extract_table_via_spatial_ocr()` executes:
     - Clusters OCR tokens into horizontal rows by vertical coordinate proximity.
     - Detects column centers and horizontal alignment lanes.
     - Transcribes clean GitHub-flavored Markdown tables and computes key metrics.
5. **Stage 3: Numerical Cross-Validation**:
   - Validates numerical tokens in the table against OCR ground truth to adjust final confidence scores.
6. **Stage 4: On-Demand VLM Extraction During Agent Query Execution**:
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

### D. Geometric Advertisement Barrier Isolation & Neural Layout Guards (`parsers/docling.py`, `layout/analyzer.py`, `layout/segmenter.py`)

Statutory, commercial disclosures (*QIP announcements, IPO prospectus summaries, tender notices*), and full-page advertisements often lack standard news headlines and occupy multi-column or whole-page zones that can corrupt editorial reading order.

1. **Picture Bounding Box Stripping from Article Envelopes**:
   - In `parsers/docling.py`, elements labeled as `picture` or visual containers are strictly excluded when computing article text bounding envelopes (`[x0, y0, x1, y1]`).
   - Prevents embedded or adjacent advertisements and full-width photos from artificially inflating article text envelopes across unrelated columns.
2. **Docling Neural Ad Container Isolation**:
   - Neural layout analysis inspects cluster tokens, keywords (`ADVERTISEMENT`, `PUBLIC NOTICE`, `TENDER NOTICE`, `TENDER`), and spatial bounding ratios.
   - Isolated advertisement blocks are detached from editorial article reading trees and segregated into discrete advertisement containers.
3. **Spatial Discontinuity & Column Boundary Guards**:
   - If consecutive text lines exhibit an abnormal vertical jump ($>200\text{px}$) or crossing of physical column gutters without continuation indicators, segmenters treat the break as a hard boundary.
   - Jump-pointer patterns (`▶ P2`, `Continued on Page...`) trigger clean continuation state tracking rather than inline column bleed.
4. **Envelope Detection & Synthetic Boundary Injection**:
   - Detects clusters of statutory keywords (`QUALIFIED INSTITUTIONS PLACEMENT`, `BOOK RUNNING LEAD MANAGERS`, `ISSUE PRICE`, `REGISTRAR TO THE ISSUE`) and constructs convex bounding envelopes.
   - Injects synthetic barrier headline elements (`[Advertisement] <Ad Title>`) at the top of the envelope.
5. **Reading Order Isolation**:
   - `ArticleSegmenter` isolates the advertisement into a dedicated `[Advertisement]` article, preventing adjacent editorial news columns from absorbing the advertisement copy.
6. **Marketing Slogan Byline Suppression**:
   - `MARKETING_SLOGAN_REGEX` rejects commercial taglines (*"By Innovation I Built For The Future"*, *"Backed by Trust"*) from being parsed as journalist bylines.
7. **Photo-Article Spatial Binding & Ad Penalty**:
   - In `media_extractor.py`, photos are bound to parent articles using convex spatial envelope containment and Euclidean proximity, but candidate articles marked as `[Advertisement]` or possessing giant envelopes ($>40\%$ canvas) are penalized so editorial photos attach to valid editorial articles.

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
   ┌─────────────────────────────────────────────────────────────┐
   │           Cognitive Query Planner & Intent Router            │
   │  • Grounded with live archive metadata (dates, papers, cats) │
   │  • Authoritative Planner LLM decision-making (sole authority)│
   │  • Classifies into 1 of 8 Broadsheet Query Archetypes        │
   │    (including dedicated analytical_computation archetype)   │
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
   │  • Deterministic heuristic router reserved strictly as fallbk│
   └──────────────────────────────┬──────────────────────────────┘
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
   │  • Dynamic Model-Aware Token Budgeting (resolve_budget)     │
   │  • ProviderCapability Envelopes (8,192 / 4,096 max tokens)  │
   │  • Reasoning Auto-Detection & 2,048 CoT Headroom            │
   │  • OCR Noise Tolerance & Intelligent Semantic Reconstruction│
   │  • Isolated Single-Article Evidence Context (ad filtering)  │
   │  • Dynamic Prompt Compilation from AnswerBlueprint           │
   │  • Single-Article Full-Text Budgeting (up to 7,500 chars)   │
   │  • Deterministic Robotic Catalog Table Stripping            │
   │  • Photo Visual Scene Annotation Noise Sanitization         │
   │  • Strict 1-Shot Citations: [Paper, YYYY-MM-DD, Page, Title]│
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │      Reflective Answer Verifier & Fact-Checking Critic      │
   │  • 4-Dimension Editorial Audit (Faithfulness, Absence,      │
   │    Fluff Elimination, Publication Scope & Date Alignment)   │
   │  • Fast-Gate Check (<5ms) for deterministic invariants      │
   │  • Conditional LangGraph Routing with 1-Cycle Ceiling:      │
   │    ├─► Valid / Refined ──────────► Log & Deliver via SSE    │
   │    └─► Evidence Gap Detected ────► execute_dynamic_code     │
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

### J. Reflective LLM Answer Verifier & Editorial Fact-Checking Critic (`answer_verifier.py`)

Synthesized broadsheet briefs are audited prior to client delivery by an editorial verification critic that acts as a post-generation safeguard:

1. **Two-Tier Verification Architecture**:
   - **Tier 1 (Deterministic Fast Gates, <5ms)**: Runs fast regex and string checks against retrieved evidence:
     - `_verify_publication_scope()`: Scans the draft for mentions of un-retrieved or out-of-scope publications.
     - `_verify_date_alignment()`: Verifies that cited dates match the requested query range and verified evidence.
     - `_verify_absence_faithfulness()`: If evidence documents 0 records/omissions, checks that the draft does not hallucinate positive counts or availability.
   - **Tier 2 (Reflexive LLM Auditor)**: For complex analytical and multi-source answers, prompts an LLM with `ANSWER_VERIFIER_SYSTEM_PROMPT` to grade four core dimensions:
     1. *Faithfulness & Truthfulness*: Ensures every assertion, number, and quote is grounded in evidence.
     2. *Contradiction Detection*: Flags positive claims when evidence demonstrates absence.
     3. *Speculative Fluff Elimination*: Strips ungrounded corporate consulting advice and dummy placeholder citations.
     4. *Evidence Gap Detection*: Diagnoses missing facts and routes to dynamic tools when code execution is required.

2. **Typed Audit Verdict (`AnswerVerificationResult`)**:
   ```python
   @dataclass
   class AnswerVerificationResult:
       is_valid: bool
       has_hallucination: bool = False
       has_contradiction: bool = False
       evidence_gap_detected: bool = False
       quality_score: float = 1.0
       factual_errors: list[str] = field(default_factory=list)
       critique: str = ""
       recommended_action: str = "accept"  # "accept" | "refine_answer" | "fallback_to_dynamic_tool"
       refined_answer: str | None = None
       dynamic_tool_hint: str | None = None
       latency_ms: int = 0
   ```

3. **Closed-Loop Dynamic Tool Rollback Loop (`graph.py`)**:
   - If the verifier diagnoses an evidentiary gap requiring ad-hoc code (`recommended_action="fallback_to_dynamic_tool"`), the state machine executes conditional edge `_route_after_verification`, looping back to `execute_dynamic_code`.
   - Bounded by a strict 1-cycle ceiling (`verification_attempts < 1` and `recovery_attempts < 1`), ensuring predictable response latency.

---

### K. Broadsheet Visual Asset Inspection, Companion Resolution & Presentation Formatters (`retrieval/`)

Retrieval logic is cleanly decoupled from state orchestration into specialized engines:

1. **`visual_inspector.py` (`VisualInspectionEngine`)**:
   - Houses the complete multimodal broadsheet visual asset discovery and transcription engine.
   - Implements the 5-strategy discovery cascade (A: explicit photo_id + companion charts; B: target headline; C: explicit article_id + companion charts; D: multi-criteria DB search; E: scoped caption/VLM search).
   - Features on-demand MinIO image crop streaming and Gemini/Qwen VLM transcription when placeholder descriptions are encountered.

2. **`asset_resolver.py`**:
   - Manages ground truth database metadata for attached workspace assets (`resolve_attached_asset_context`).
   - Fast lookup for exact article IDs by quoted headline (`resolve_authoritative_article_id`).
   - Authoritative reconciliation of attached assets, query parameters, and active issue context (`resolve_conversation_working_context`).

3. **`formatters.py`**:
   - Consolidates all human-readable markdown snippet generation: `format_issue_manifest()`, `format_coverage_matrix_snippet()`, `format_coverage_difference_snippet()`, `format_shared_coverage_snippet()`.
   - Integrates typographic ligature repair (`repair_text_ligatures()`) across all manifest text.

---

### L. Softly-Decoupled Broadsheet Schema & In-Memory Archive Context (`archive_context.py`)

Prior architectures tightly coupled query planning to live MySQL database connections. If the database was slow or offline, planning stalled.

1. **Declarative In-Memory Schema Catalog (`STATIC_BROADSHEET_SCHEMA`)**:
   - 100% in-memory representation of MySQL tables (`newspapers`, `issues`, `articles`, `article_categories`, `pages`, `photos`), columns, foreign keys, and relationships. Requires zero network calls.
2. **Soft Decoupling with Graceful Fallback (`get_archive_metadata`, `get_archive_and_schema_context`)**:
   - Retrieves live archive bounds and active publications asynchronously, cached with a 5-minute TTL via structured `ArchiveMetadata`.
   - If database access errors or times out, immediately falls back to `get_fallback_archive_metadata()` (`STATIC_CANONICAL_PUBLICATIONS`, `STATIC_ARCHIVE_DATE_MIN`, `STATIC_ARCHIVE_DATE_MAX`, `STATIC_CANONICAL_CATEGORIES`).
   - Guarantees that query planning and dynamic tool synthesis never fail due to database transient unavailability.

---

### M. Decoupled Relational SQL Analytics Dispatcher (`sql_dispatcher.py`)

To achieve single-responsibility modularity in the execution pipeline, all 11 pre-compiled relational analytical routines were cleanly extracted from `executor.py` into [`backend/app/agent/sql_dispatcher.py`](file:///Users/piyushgoel/Downloads/Projects/NewsLens-AI/backend/app/agent/sql_dispatcher.py):

1. **Analytical Routine Dispatch (`SQLAnalyticsDispatcher.dispatch`)**:
   - Manages: `entity_trends`, `issue_summary`, `count_ads`, `count_issues`, `count_articles`, `photo_counts`, `topic_distribution`, `frontpage_ratio`, `coverage_comparison`, `coverage_difference`, `shared_coverage`.
2. **Dynamic Alias Normalization**:
   - Maps synonyms (e.g. `count_advertisements` $\to$ `count_ads`, `newspaper_availability` $\to$ `count_issues`, `photos_by_section` $\to$ `photo_counts`) to canonical database routines with zero tool caller friction.
3. **Structured Presentation Formatting**:
   - Integrates `CoverageAnalyzer` and `formatters.py` to produce standardized, ligature-repaired Markdown manifests and coverage matrices injected directly into agent evidence state.

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
| **Phase 19: Reflective Answer Verification & Publication Scope Integrity** | Synthesized answers occasionally fabricated corporate boilerplate or cited out-of-scope broadsheets when evidence documented absence. | Built `AnswerVerifier` (`answer_verifier.py`) with 4-dimension audit (faithfulness, absence integrity, fluff removal, scope/date alignment), deterministic fast gates (<5ms), and closed-loop fallback to `execute_dynamic_code`. |
| **Phase 20: Retrieval Engine Modularization & Clean Separation of Concerns** | Monolithic `executor.py` conflated tool dispatch, visual inspection cascades, database asset reconciliation, and presentation formatting in 3,000+ lines. | Extracted `visual_inspector.py` (`VisualInspectionEngine` with Strategies A-E and MinIO crop enrichment), `asset_resolver.py` (ground truth database asset lookup and conflict analysis), and `formatters.py` (manifest and coverage matrix snippets). |
| **Phase 21: Softly-Decoupled Broadsheet Schema & In-Memory Archive Context** | Planner depended on live MySQL database connections for archive metadata; database connection delays stalled agent planning. | Engineered `archive_context.py` providing `STATIC_BROADSHEET_SCHEMA`, in-memory TTL caching with fallback defaults (`get_archive_and_schema_context`), eliminating hard database dependencies during planning. |
| **Phase 23: Google Gemini Full Cloud Architecture & Production Containerization** | OpenRouter dual-key rate limits and legacy model deprecations created operational friction; onboarding required manual multi-service orchestration without container guarantees. | Transitioned primary cloud engine to Google AI Studio Gemini (`gemini-3.8-flash`, `gemini-3.8-live`, `gemini-3.5-flash`); engineered multi-candidate failover cascade (`GeminiProvider._get_model_candidates`) and Pydantic schema title cleaner; built 8-service Docker Compose specification (`docker-compose.yml`), multi-stage Dockerfiles (`backend/Dockerfile`, `frontend/Dockerfile`, `frontend/nginx.conf` with SSE reverse proxy), unified developer `Makefile`, and open-source governance standard (`LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`). |
| **Phase 24: High-Throughput Ingestion & Single-Pass Multimodal Extraction** | Monolithic 300 DPI broadsheet rasters incurred heavy memory pressure and slow rendering (6–8s/page); serial per-crop visual extraction led to 30+ network trips per page; embedded advertisements and photo bounding boxes contaminated article text envelopes and continuation reading trees. | Standardized on 150 DPI rasterization ($4\times$ memory reduction, ~1.5s/page); engineered `SinglePassVisualExtractor` supporting unified single-pass for cloud vision (`gemini-3.8-flash` with normalized coordinate manifests) and concurrent per-crop execution (`qwen3-vl:latest` via `asyncio.Semaphore(2)`); stripped `picture` bboxes from Docling article text envelopes; added ad container isolation & spatial discontinuity guards ($>200\text{px}$ jumps); built `clean_vlm_text` token sanitizer; added page-aware photo filtering and page badges to `BroadsheetReader.jsx`. |
| **Phase 25: Serverless Cloud Migration to GCP & Resilient Production Architecture** | Self-hosted infrastructure on local developer machines or single VMs created compute bottlenecks; ephemeral containers lacked persistence for vectors and broadsheet scans; background Celery workers suffered CPU starvation on standard serverless tiers; permanent cloud credentials posed security risks. | Migrated to Google Cloud Platform (`asia-south1`): deployed `newslens-frontend` (Nginx SPA reverse proxy), `newslens-backend` (FastAPI with Cloud SQL Unix Socket), and `newslens-worker` (`--no-cpu-throttling` + embedded HTTP health server on `$PORT`) to Cloud Run; migrated storage from MinIO to Google Cloud Storage (`gs://newslens-ai-prod-pages`, `gs://newslens-ai-prod-originals`) via polymorphic `ObjectStore` factory (`GoogleCloudStorageStore`); provisioned Cloud SQL MySQL 8.0 with automated migrations via Cloud Run Job `newslens-migrate`; integrated Qdrant Cloud managed vector cluster and Upstash Redis TLS with `ssl_cert_reqs=required`; secured all secrets in Secret Manager and established Workload Identity Federation (WIF) for zero-permanent-credential GitHub Actions CI/CD. |
| **Phase 26: Planner Cognitive Authority, Dynamic Token Budgeting & OCR Semantic Reconstruction** | Hardcoded token caps caused reasoning-token starvation (numerical answers truncated at `**2`); fast-path bypasses skipped Planner LLM reasoning; OCR typographical noise was copied into final answers; single-article queries suffered prompt contamination from outside ads. | Eliminated static token caps with `resolve_dynamic_token_budget()` deriving allocations from `ProviderCapability` (`max_output_tokens` 4,096/8,192); added reasoning model auto-detection and 2,048-token CoT headroom; removed deterministic fast-path bypass from LangGraph classification node making Planner LLM the authoritative router; introduced `analytical_computation` archetype; added OCR noise tolerance & semantic reconstruction guidelines in `synthesizer.py`; isolated single-article prompt context from irrelevant ads. |

---

## 7. Production Deployment & Containerization Architecture

NewsLens-AI provides a dual-tier deployment infrastructure: an enterprise-ready **Google Cloud Platform (GCP) Serverless Topology** for live global availability, and an isolated **Docker Compose** stack for local developer agility.

### A. Google Cloud Platform (GCP) Serverless Production Topology

In production, NewsLens-AI runs on Google Cloud Platform in region `asia-south1` (Mumbai), leveraging serverless Cloud Run services, managed databases, object storage, and zero-trust IAM authentication.

```
                                      ┌──────────────────────────────────────────────────────────┐
                                      │                      Client Browser                      │
                                      │  • Newspaper Scan Reader (300/150 DPI WebP Canvas)       │
                                      │  • Real-Time Agentic Assistant (SSE Streaming)           │
                                      └────────────────────────────┬─────────────────────────────┘
                                                                   │ HTTPS (Port 443)
                                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       GOOGLE CLOUD PLATFORM (asia-south1: Mumbai)                                      │
│                                                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                             Cloud Run Serverless Compute                                         │  │
│  │                                                                                                                  │  │
│  │  ┌─────────────────────────────────────┐      ┌───────────────────────────────────┐                              │  │
│  │  │ newslens-frontend                   │      │ newslens-backend                  │                              │  │
│  │  │ • Nginx 1.27 Alpine Container       ├─────►│ • FastAPI (Python 3.12, Uvicorn)  │                              │  │
│  │  │ • Static React 18 SPA Bundle        │/api/ │ • Cloud SQL Unix Domain Socket    │                              │  │
│  │  │ • Unbuffered SSE Streaming Proxy    │      │ • Dynamic GCS Storage Factory     │                              │  │
│  │  └─────────────────────────────────────┘      └─────────────────┬─────────────────┘                              │  │
│  │                                                                 │                                                │  │
│  │  ┌─────────────────────────────────────┐                        │ Celery Task RPC                                │  │
│  │  │ newslens-worker                     │                        │ via Redis TLS                                  │  │
│  │  │ • Celery Asynchronous Consumer      │◄───────────────────────┤                                                │  │
│  │  │ • Cloud Run --no-cpu-throttling     │                        │                                                │  │
│  │  │ • Embedded HTTP Health Check ($PORT)│                        │                                                │  │
│  │  │ • 6Gi RAM, 2 vCPU, min-instances=1  │                        │                                                │  │
│  │  └─────────────────────────────────────┘                        │                                                │  │
│  │                                                                 │                                                │  │
│  │  ┌─────────────────────────────────────┐                        │                                                │  │
│  │  │ newslens-migrate (Cloud Run Job)    ├────────────────────────┼────────────────────────────────┐               │  │
│  │  │ • Alembic Database Migrations       │                        │                                │               │  │
│  │  └─────────────────────────────────────┘                        │                                │               │  │
│  └─────────────────────────────────────────────────────────────────┼────────────────────────────────┼───────────────┘  │
│                                                                    │                                │                  │
│  ┌───────────────────────────────────────────────────┐             │                                │                  │
│  │ Google Cloud Storage (GCS)                        │             │                                │                  │
│  │ • gs://newslens-ai-prod-pages (Rasterized WebP)   │◄────────────┤                                │                  │
│  │ • gs://newslens-ai-prod-originals (Source PDFs)   │             │                                │                  │
│  └───────────────────────────────────────────────────┘             │                                │                  │
│                                                                    │                                │                  │
│  ┌───────────────────────────────────────────────────┐             │                                │                  │
│  │ Cloud SQL for MySQL 8.0                           │             │                                │                  │
│  │ • newslens-ai-prod:asia-south1:newslens-mysql     │◄────────────┴────────────────────────────────┘                  │
│  │ • Unix Domain Socket: /cloudsql/...               │                                                                 │
│  │ • FULLTEXT Indexes + utf8mb4_unicode_ci           │                                                                 │
│  └───────────────────────────────────────────────────┘                                                                 │
│                                                                                                                        │
│  ┌───────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────────┐  │
│  │ Google Secret Manager                             │   │ IAM & Workload Identity Federation (WIF)                 │  │
│  │ • 11 Production Secrets (DB, API Keys, Tokens)    │   │ • Service Account: newslens-runner                       │  │
│  │ • Dynamic Cloud Run Volume / Env Injections       │   │ • Keyless GitHub Actions Authentication (OIDC Pool)      │  │
│  └───────────────────────────────────────────────────┘   └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                                     │ External Secure Connectors
                                                                     ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                               EXTERNAL MANAGED CLOUD SERVICES                                          │
│                                                                                                                        │
│  ┌──────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────────┐   │
│  │ Qdrant Cloud (Managed Vector DB)                 │   │ Upstash Redis TLS (Managed Broker & Cache)               │   │
│  │ • australia-southeast1-0.gcp.cloud.qdrant.io:6333│   │ • hopeful-octopus-283720.upstash.io:6379                 │   │
│  │ • Collection: article_chunks (1024-dim BGE-M3)   │   │ • TLS Enforced (rediss://...ssl_cert_reqs=required)      │   │
│  └──────────────────────────────────────────────────┘   └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────────┐   │
│  │ Google AI Studio (Gemini 3.8 Flash)              │   │ Google Cloud Vision API                                  │   │
│  │ • Query Planning, Reasoning & Multimodal VLM     │   │ • Pure Broadsheet OCR & Layout Analysis Fallback         │   │
│  └──────────────────────────────────────────────────┘   └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Production Specifications
1. **Dynamic Storage Factory (`GoogleCloudStorageStore`)**:
   - Implements the polymorphic `ObjectStore` interface in `backend/app/storage/gcs_store.py` backed by `google-cloud-storage`.
   - The factory `get_object_store()` dynamically instantiates `GoogleCloudStorageStore` in production (`STORAGE_BACKEND=gcs`) and `MinioStore` in local development (`STORAGE_BACKEND=minio`).
2. **Cloud Run Celery Worker (`newslens-worker`)**:
   - Standard Cloud Run services throttle CPU to zero when not handling incoming HTTP requests. `newslens-worker` is deployed with `--no-cpu-throttling` and `--min-instances=1`, ensuring the Celery consumer maintains 100% CPU capacity 24/7.
   - Embeds a lightweight background HTTP health server (`app/run_worker.py`) on `$PORT` (8080) that responds `200 OK` to Cloud Run startup and liveness probes while running `celery worker` in the foreground.
3. **Database Migration Job (`newslens-migrate`)**:
   - Cloud Run Job configured to run `alembic upgrade head` over the Cloud SQL Unix domain socket before updating backend or worker revisions, guaranteeing zero-downtime schema evolution.
4. **Keyless CI/CD via Workload Identity Federation**:
   - `.github/workflows/deploy-gcp.yml` uses Google Cloud Workload Identity Federation to exchange GitHub Actions OIDC tokens for short-lived Google Cloud access tokens, completely eliminating static service account JSON keys.
   - Automatically runs Alembic migrations on ephemeral test MySQL containers, executes the 574-test suite, builds and pushes multi-arch images to Google Artifact Registry, and rolls out updates to Cloud Run.

---

### B. Local Development Containerization (Docker Compose)

For offline development, NewsLens-AI provides a containerized infrastructure orchestrated via **Docker Compose** (`docker-compose.yml`) and supported by a unified developer `Makefile`.

```
                                  ┌──────────────────────────────────────────────────────────┐
                                  │                  Nginx Reverse Proxy                     │
                                  │           (frontend container - Port 5173)               │
                                  │  • Serves React 18 / Vite SPA Single Page Application   │
                                  │  • Reverse-proxies /api/ with SSE buffering disabled     │
                                  └────────────┬─────────────────────────────┬───────────────┘
                                               │ /api/                       │ Static SPA
                                               ▼                             ▼
                                  ┌──────────────────────────┐   ┌───────────────────────────┐
                                  │   FastAPI Backend Server │   │   Compiled React UI       │
                                  │    (backend:8000)        │   │   (HTML, JS, CSS, Assets) │
                                  └──────┬────────────┬──────┘   └───────────────────────────┘
                                         │            │
                         Celery Task RPC │            │ Direct Connection
                                         ▼            ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 INFRASTRUCTURE SERVICES CLUSTER                             │
├──────────────────────────────┬───────────────────────────────┬─────────────────────────────┤
│ MySQL 8.0 (Relational SoR)   │ Qdrant (Vector Engine)        │ MinIO (S3 Object Storage)   │
│ • Port 3306                  │ • Port 6333 (HTTP & Dashboard)│ • Port 9000 (S3 API)        │
│ • Fulltext & JSON columns    │ • 1024-dim BGE-M3 collections │ • Port 9001 (Web Console)   │
├──────────────────────────────┼───────────────────────────────┼─────────────────────────────┤
│ Redis 7 (Cache & Broker)     │ Celery Worker (worker)        │ Ollama (Local AI Engine)    │
│ • Port 6379                  │ • Distributed ingestion queue │ • Port 11434                │
│ • In-memory cache & pub/sub  │ • Multi-page PDF rasterization│ • Sovereign local models    │
└──────────────────────────────┴───────────────────────────────┴─────────────────────────────┘
```

### Local Service Specifications
1. **`frontend` (Nginx + React 18 SPA)**:
   - Multi-stage Docker build (`node:20-alpine` $\to$ `nginx:1.27-alpine`).
   - Custom `nginx.conf` featuring SPA fallback (`try_files $uri $uri/ /index.html`), gzip compression, and reverse-proxy for `/api/` with `proxy_buffering off` and `proxy_read_timeout 300s` for real-time Server-Sent Events (SSE) streaming.
2. **`backend` (FastAPI + Python 3.12)**:
   - Multi-stage Docker build utilizing `astral-sh/uv:latest` for ultra-fast dependency resolution.
   - Pre-configured system libraries (`tesseract-ocr`, `libgl1`, `curl`, `build-essential`).
   - Automatically executes database migrations (`alembic upgrade head`) before launching Uvicorn workers.
3. **`worker` (Celery Async Ingestion Engine)**:
   - Reuses backend container image to process asynchronous broadsheet PDF ingestion, OCR rasterization, and vector indexing.
4. **`mysql`, `qdrant`, `minio`, `redis`, `ollama`**:
   - Production Docker images configured with health checks, persistent volumes, and custom network bridging (`newslens-network`).

### Unified Developer Interface (`Makefile`)
- `make setup`: Single command onboarding (checks `.env`, spins up database & storage, runs `uv sync` & `alembic upgrade`, installs frontend).
- `make dev` / `make run-all`: Concurrent development server runner.
- `make prod-up` / `make prod-down` / `make prod-logs`: Docker production lifecycle management.
- `make secrets-check`: Automated pre-flight security scan ensuring zero API keys or credentials enter git.

---

## 8. Future Work & Roadmap

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
