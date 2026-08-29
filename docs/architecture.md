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
│  • Multi-Page Consensus     │  • MinIO Object Storage (S3)  │  • Multi-Tool Dispatcher    │
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
   - **Hosted Production Models (Anthropic, OpenAI, Google, Groq)**: Supports `claude-sonnet-4-5`, `gpt-4o`, `gemini-2.5-flash`, and Groq LPU inference for ultra-fast response times.
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

### D. Geometric Advertisement Barrier Isolation (`layout_analyzer.py`, `segmenter.py`)

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
   │  • Short-circuits ambiguous initial queries with guidance   │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │           Cognitive Query Planner & Intent Router           │
   │  • Classifies into 1 of 6 Broadsheet Query Archetypes       │
   │  • Dispatches optimal tool execution sequence:              │
   │    - sql_analytics (issue manifest, stats, section lists)   │
   │    - hybrid_search (dense Qdrant + sparse MySQL RRF)        │
   │    - entity_search (knowledge graph & salience lookups)     │
   │    - timeline_builder (thematic chronological progression)  │
   │    - coverage_analyzer (cross-newspaper comparison)         │
   │    - web_search (real-time live internet grounding)         │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │         Multi-Tool Dispatch & Execution Engine              │
   │  • Executes planned tool calls asynchronously in parallel   │
   │  • Resilient multi-tier issue ID fallback in sql_analytics  │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │          Corrective RAG (CRAG) Retrieval Evaluator          │
   │  • Grades keyword relevance & density of retrieved evidence │
   │  • Triggers broadened fallback query if confidence is low   │
   │  • Enforces anti-hallucination hard stops on empty evidence │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │         4-Tier Broadsheet Grounded Synthesizer              │
   │  • Formulates structured response:                          │
   │    1. Executive Summary                                     │
   │    2. Key Verified Facts & Highlights                       │
   │    3. Broadsheet Perspectives (Front Page vs Inside)        │
   │    4. Explore Further Follow-up Suggestions                 │
   │  • Generates strict markdown citations: [Paper, Date, Page] │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │            SSE Streaming Delivery & Audit Logging           │
   │  • Streams response tokens, reasoning trace & tool metrics  │
   │  • Logs execution latency, cost, and query audit in MySQL   │
   └─────────────────────────────────────────────────────────────┘
```

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
