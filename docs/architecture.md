# NewsLens-AI Architecture & System Design

NewsLens-AI is an **Enterprise-Grade Agentic Intelligence Platform** engineered for historical broadsheet and modern newspaper archives. It provides end-to-end multi-column PDF layout segmentation, multimodal OCR, hybrid dense/sparse vector retrieval, conversational multi-turn query synthesis, cross-publication narrative trajectory tracking, and visual asset transparency.

---

## 1. High-Level System Architecture

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
│  • Multi-Page Consensus     │  • MinIO Object Storage (S3)  │  • 2-Stage Rerank Cascade   │
│  • 2D Spatial Photo Binding │  • Redis 7 (Cache & Lock)     │  • 4-Tier Synthesizer       │
│  • Qwen3-VL Visual Extract  │  • RRF (Dense + Sparse Fusion)│  • Corrective RAG (CRAG)    │
└─────────────────────────────┴───────────────────────────────┴─────────────────────────────┘
```

---

## 2. Technology Stack & Infrastructure

| Layer | Technology | Function |
|---|---|---|
| **Frontend** | React 18, Vite, Tailwind CSS, Lucide Icons, Canvas API | High-resolution PDF/image reader, interactive bbox overlays, visual asset inspector, SSE streaming consumer |
| **Backend API** | Python 3.13 / 3.12, FastAPI, Uvicorn | Async HTTP APIs, Server-Sent Events (SSE) streaming, streaming media proxies |
| **Task Queue** | Celery + Redis 7 | Distributed background ingestion pipeline and PDF rasterization |
| **System of Record** | MySQL 8 (`aiomysql` async + `pymysql` sync) | 16 relational tables with `FULLTEXT` indexing on headlines and full text |
| **Vector Database** | Qdrant | Dense vector search with cosine similarity and metadata payload filtering |
| **Object Store** | MinIO / AWS S3 | 300 DPI page PNG scans, cropped photo/infographic assets, and original PDF archives |
| **Cache & Lock** | Redis 7 | Query caching, heavy-LLM timeline caching (1-hr TTL), rate limiting |
| **Document Parsers** | IBM Docling (DocLayNet), MinerU, Google Cloud Vision, Gemini VLM, PyMuPDF | Multimodal document layout analysis, table extraction, and 2D reading order |
| **OCR Engines** | RapidOCR (ONNX), Google Cloud Vision API (`DOCUMENT_TEXT_DETECTION`), Tesseract | High-precision newspaper text and bbox transcription with 98%+ confidence |
| **VLM & Visual Extraction**| Qwen3-VL (`qwen3-vl:latest`), Gemini 2.5/Flash, Ollama | Infographic chart-to-table transcription, visual triage, and photo captioning |
| **LLM Reasoning** | Groq (Compound, Qwen), Gemini, OpenAI, Ollama (Gemma 4, Llama 3) | Multi-provider failover chain for planning, condensation, and synthesis |
| **Embedding Models** | BAAI/bge-m3 (1024-dim), nomic-embed-text, OpenAI | Dense semantic representation for hybrid search |

---

## 3. Core Subsystems

### A. Document Ingestion & Visual Pipeline (`backend/app/ingestion/`)

```
   ┌──────────────────┐
   │ Broadsheet PDF   │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────────────────────────────┐
   │ Intake & Lossless Compression (PyMuPDF)  │
   └────────┬─────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────────────┐
   │ 300 DPI Rasterization & MinIO Archive    │
   └────────┬─────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────────────┐
   │ Multi-Page Consensus Masthead & Date     │
   │ (ConsensusExtractor + MastheadVerifier)  │
   └────────┬─────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────────────┐
   │ 2D Broadsheet Layout Analysis (Docling)  │
   │ • DocLayNet + RapidOCR                   │
   │ • Title / Deck Coalescence & Bylines     │
   │ • Multi-Article Page Separation          │
   └────────┬─────────────────────────────────┘
            │
            ├──────────────────────────────────────────────────────┐
            │                                                      │
            ▼                                                      ▼
   ┌──────────────────────────────────────────┐           ┌──────────────────────────────────────────┐
   │ Article Text & Structure Assembly        │           │ 2D Spatial Photo & Infographic Harvest   │
   │ • 2D Reading Order Linearization         │           │ • Spatial Caption Proximity Matching     │
   │ • Cross-Page Jump Line Stitching         │           │ • Convex Envelope Article Binding        │
   │ • Minimum Structural Noise Filtering     │           │ • Qwen3-VL Visual Triage & Extraction    │
   └────────┬─────────────────────────────────┘           └────────┬─────────────────────────────────┘
            │                                                      │
            └──────────────────────────┬───────────────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ Hierarchical Chunking & Embedding │
                     │ • BAAI/bge-m3 (1024-dim) vectors  │
                     │ • Visual Infographic Data Chunks  │
                     │ • Dual Sync: Qdrant + MySQL FT    │
                     └───────────────────────────────────┘
```

1. **Intake & Rasterization**: Uploads broadsheet PDFs, verifies SHA-256 idempotency, optimizes lossless compression via PyMuPDF (`fitz`), stores original PDFs in MinIO `newslens-originals`, and renders 300 DPI high-resolution page scans into `newslens-pages`.
2. **Multi-Page Consensus Masthead & Date Recognition**:
   - Inspects headers, folios, and footers across up to 15 pages to extract authoritative brand and publication date.
   - Comprehensive signature database covering national and global broadsheets (*The New York Times*, *The Wall Street Journal*, *Financial Times*, *The Washington Post*, *The Guardian*, *Mint*, *Business Standard*, *The Hindu*, *The Times of India*, *The Economic Times*, etc.) and acronyms (`NYT`, `WSJ`, `FT`, `WAPO`, `ET`, `TOI`, `BS`).
   - Fallback to visual `MastheadVerifier` utilizing RapidOCR on Page 1 masthead zone (top 20%).
3. **Docling Neural Layout Parser (`docling_parser.py`)**:
   - Executes IBM Docling with DocLayNet and RapidOCR to segment complex broadsheet grids into semantic blocks (`title`, `section_header`, `paragraph`, `picture`, `caption`, `table`, `chart`).
   - **Headline & Subheadline/Deck Coalescence**: Gathers multi-line headlines, summary decks, and inline bylines (`BY <NAME>`) without creating fragmented 8-word orphan articles.
   - **Multi-Article Separation**: Accurately flushes and separates distinct articles on dense multi-story broadsheet pages.
4. **2D Spatial Photo & Infographic Extraction**:
   - Identifies distinct pictures and figures, filtering out full-page canvas background scans (`w >= 90%` and `h >= 90%`).
   - Pairs multi-photo composite galleries (left, center, right photos) with shared captions using 2D vertical distance and horizontal overlap scoring.
   - Generates high-res image crops in MinIO (`photos/{page_id}/photo_{n}.png`).
   - Automatically binds photos to parent articles via convex spatial envelope containment and Euclidean proximity fallbacks.
5. **VLM Visual Data Extraction (`visual_extractor.py`)**:
   - **Stage 1 (Visual Triage)**: Uses aspect ratio heuristics + VLM classification to differentiate data infographics from editorial photos.
   - **Stage 2 (Structured Extraction)**: Transcribes charts, graphs, and tables into Markdown tables, metric bullet points, and executive summaries via `qwen3-vl:latest`.
   - **Stage 3 (Numerical Cross-Validation)**: Compares VLM-extracted figures against local OCR tokens to prevent visual hallucination.
   - Injects dedicated visual data chunks into Qdrant (`chunk_type="visual"`, `has_visual_data=True`).
6. **Resilient Object Storage**:
   - `MinioStore.get()` gracefully catches `S3Error` (`NoSuchKey`, `NoSuchBucket`) returning empty bytes, ensuring missing assets return clean HTTP 404 responses instead of crashing the ASGI application with 500 errors.

---

### B. Agentic Retrieval & Reasoning Engine (`backend/app/agent/` & `backend/app/retrieval/`)

1. **Query Condenser (`condenser.py`)**: Resolves coreferences and conversational context across multi-turn dialogues. Detects in-context meta-queries (`"which newspaper was this from?"`) to bypass vector search and answer directly from citations.
2. **LLM Agentic Query Planner (`planner.py`)**: Pydantic structured Chain-of-Thought (CoT) reasoning model with explicit tool boundaries. Dynamically classifies query archetypes (`factual_lookup`, `thematic_timeline`, `quantitative_trend`, `cross_newspaper_comparison`, `entity_deep_dive`) and routes macro-level queries (whole-issue summaries, full page manifests, article counts) to `sql_analytics` and fine-grained queries to `hybrid_search` without hardcoded regexes.
3. **Two-Stage Neural Retrieval Cascade (`hybrid_search.py` & `reranker.py`)**:
   - **Stage 1 (High Recall Candidate Pool)**: Retrieves Top $N=75$ candidates using Reciprocal Rank Fusion (RRF $k=60$) across Qdrant dense vector search (`BAAI/bge-m3`) and MySQL `FULLTEXT` keyword search.
   - **Stage 2 (Cross-Encoder Neural Reranker)**: Evaluates candidate relevance via `sentence_transformers.CrossEncoder` (`BAAI/bge-reranker-v2-m3` / `ms-marco-MiniLM-L-6-v2`) with Apple Silicon `mps` auto-acceleration, producing the final Top 10 reranked context.
4. **Parent-Document (Small-to-Big) Context Hydration**: Retains exact chunk bounding-box anchors while packaging surrounding parent article context (`[Exact Chunk Match]` + `[Article Parent Context]`) to prevent truncated factual synthesis.
5. **Multi-Hop Knowledge Graph Traversal (`entity_filter.py`)**: Discovers multi-hop entity co-occurrence relations across shared newspaper stories and event clusters.
6. **Corrective RAG (CRAG) Self-Reflection (`graph.py`)**: Evaluates retrieved evidence quality and automatically triggers entity taxonomy and web search fallback if initial archive retrieval is weak.
7. **3-Tier Negative Coverage Engine (`coverage_analyzer.py`)**:
   - **Tier 1 (Relational Audit Invariant)**: Queries MySQL system-of-record to identify newspapers with zero articles on the requested topic.
   - **Tier 2 (Vector/Semantic Negative Verification)**: Audits low semantic similarity candidates to confirm negative reporting.
   - **Tier 3 (Multi-Newspaper Reconciliation Matrix)**: Generates comparative editorial matrices detailing perspective differences and omitted coverage.
8. **Synthesizer (`synthesizer.py`)**: Produces structured 4-tier briefs (`Executive Summary`, `Key Verified Facts`, `Broadsheet Perspectives`, `Explore Further`) with visual citations (`[📊 Chart: ...]` and `[📷 Photo: ...]`) and resilient multi-provider failover (`groq_compound` $\rightarrow$ `gemini_flash` $\rightarrow$ `groq_qwen` $\rightarrow$ `ollama_llama3`).

---

### C. Cross-Newspaper Narrative Trajectory (`backend/app/retrieval/timeline_builder.py`)

1. **Story Lifecycle Reconstruction**: Tracks narrative trajectories across multiple broadsheets and dates with reporting phases (`Breaking`, `Development`, `Financial Impact`, `Regulatory/Outcome`).
2. **4-Tier Anti-Hallucination Defense Pipeline**:
   - **Gate 1 (Salience Edge Filtering)**: Filters 2-hop connected entities by `salience_score >= 0.50`.
   - **Gate 2 (Temporal Windowing Constraint)**: Bounds multi-hop article retrieval to the calendar event range $[\text{min\_date}, \text{max\_date}]$.
   - **Gate 3 (Neural Cross-Encoder Verification)**: Reranks candidate pairs against the query topic using `CrossEncoderReranker`.
   - **Gate 4 (Empty-Evidence Hard Stop & Attribution)**: Zero-hallucination hard stop when no grounded evidence exists.
3. **Milestone Active Entity Evolution**: Captures central protagonist entities, their type, and centrality scores at each milestone.

---

## 4. Provider Abstraction & Model Registry

All model providers implement structural subtyping via Python `Protocol` interfaces:
- `ChatModelProvider`: `GroqProvider`, `GeminiProvider`, `OpenAIProvider`, `OllamaProvider`
- `EmbeddingProvider`: `LocalEmbeddingProvider` (bge-m3), `OpenAIProvider`
- `VisionModelProvider`: `GoogleCloudVisionOCR`, `GeminiProvider`, `OllamaProvider` (qwen3-vl, gemma4:26b, qwen2.5vl)
- `DocumentLayoutProvider` & `OCREngine`: `DoclingLayoutParser`, `MinerUProvider`, `GoogleCloudVisionOCR`, `GeminiProvider`, `TesseractOCR`

Configured declaratively in `model_config.yaml` with runtime zero-downtime swapping via `/api/settings/model-bindings`.
