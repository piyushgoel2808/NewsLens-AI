# NewsLens-AI Architecture & System Design

NewsLens-AI is an **Enterprise-Grade Agentic Intelligence System** for historical broadsheet and modern newspaper archives. It provides end-to-end multi-column PDF layout segmentation, multimodal OCR, hybrid dense/sparse vector retrieval, conversational multi-turn query synthesis, and cross-publication narrative trajectory tracking.

---

## 1. High-Level System Architecture

```
                                 ┌──────────────────────────────────────────────────────────┐
                                 │                   React 18 + Vite SPA                    │
                                 │  • Newspaper Scan Reader with Bounding-Box Overlay       │
                                 │  • Real-Time Agentic Assistant with Reasoning Trace      │
                                 │  • Interactive Cross-Newspaper Narrative Trajectory UI   │
                                 │  • Visual Citation Badge Deep-Linking                    │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │ REST / SSE Streaming
                                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FastAPI Backend Server                                    │
├─────────────────────────────┬───────────────────────────────┬─────────────────────────────┤
│   Document Ingestion Flow   │    Retrieval Toolbelt & DB    │   Agentic Reasoning Flow    │
│  • Celery Async Workers     │  • MySQL 8 (FULLTEXT Index)   │  • LangGraph State Graph    │
│  • Docling / MinerU Layout  │  • Qdrant Dense Vector Store  │  • Query Condenser & Memory │
│  • Gemini / RapidOCR Engine │  • MinIO Object Storage (S3)  │  • Multi-Tier Synthesizer   │
│  • 2D Reading Order Logic   │  • Redis 7 (Cache & Broker)   │  • Multi-Provider Failover  │
└─────────────────────────────┴───────────────────────────────┴─────────────────────────────┘
```

---

## 2. Technology Stack & Infrastructure

| Layer | Technology | Function |
|---|---|---|
| **Frontend** | React 18, Vite, Tailwind CSS, Lucide Icons, Canvas API | PDF/Image reader, interactive bbox overlays, SSE stream consumer |
| **Backend API** | Python 3.13 / 3.12, FastAPI, Uvicorn | Async HTTP APIs, Server-Sent Events (SSE) streaming |
| **Task Queue** | Celery + Redis 7 | Distributed background ingestion pipeline and PDF rasterization |
| **System of Record** | MySQL 8 (`aiomysql` async + `pymysql` sync) | 16 relational tables with `FULLTEXT` indexing on headlines and text |
| **Vector Database** | Qdrant | Dense vector search with cosine similarity and metadata payload filtering |
| **Object Store** | MinIO / AWS S3 | Rasterized page PNGs, cropped photo crops, and original PDF storage |
| **Cache & Lock** | Redis 7 | Query caching, heavy-LLM timeline caching (1-hr TTL), rate limiting |
| **Document Parsers** | IBM Docling, MinerU, Google Cloud Vision, Google Gemini VLM, PyMuPDF | Multimodal document layout analysis, table extraction, and 2D reading order |
| **OCR Engines** | Google Cloud Vision API (`DOCUMENT_TEXT_DETECTION`), Gemini 2.5/Flash, RapidOCR, Tesseract | High-precision newspaper text and bbox transcription with 98%+ confidence |
| **LLM Reasoning** | Groq (Compound, Qwen), Gemini, OpenAI, Ollama (Gemma 4, Llama 3) | Multi-provider failover chain for planning, condensation, and synthesis |
| **Embedding Models** | BAAI/bge-m3 (1024-dim), nomic-embed-text, OpenAI | Dense semantic representation for hybrid search |

---

## 3. Core Subsystems

### A. Document Ingestion Pipeline (`backend/app/ingestion/`)
1. **Intake & Rasterization**: Uploads broadsheet PDFs, optimizes lossless compression via PyMuPDF (`fitz`), stores originals in MinIO, renders high-DPI page images.
2. **Multi-Page Consensus Metadata**: Extracts and cross-validates masthead title and publication date across up to 15 pages to eliminate OCR anomalies.
3. **Layout & OCR Extraction**:
   - **Google Cloud Vision OCR**: High-accuracy `DOCUMENT_TEXT_DETECTION` (single API call per page) with spatial block segmentation and word-level bounding box extraction.
   - **Local VLM (Ollama Gemma 4 / Qwen)**: Grammar-constrained structured JSON layout parsing with automatic reasoning-tag pruning.
   - **Self-Healing Fallback**: Automatically cascades from local/hosted VLM failures to Google Cloud Vision OCR layout.
4. **2D Reading Order & Debundling**: Topological geometric sorting ensures column continuation without text bleeding, merges multi-page jumps, and rejects noisy boilerplate.
5. **Embedding & Indexing**: Generates 1024-dim dense vectors via BAAI/bge-m3 into Qdrant with dual-index sync in MySQL `FULLTEXT`.

### B. Agentic Retrieval & Reasoning Engine (`backend/app/agent/` & `backend/app/retrieval/`)
- **Query Condenser (`condenser.py`)**: Coreference resolution across dialog turns. Detects in-context meta-queries (`"which newspaper was this from?"`) to bypass vector search and answer directly from citations.
- **LLM Agentic Query Planner (`planner.py`)**: Pydantic structured Chain-of-Thought (CoT) reasoning model with explicit tool boundaries. Dynamically classifies query archetypes (`factual_lookup`, `thematic_timeline`, `quantitative_trend`, `cross_newspaper_comparison`, `entity_deep_dive`) and routes macro-level queries (whole-issue summaries, full page manifests, article counts) to `sql_analytics` and fine-grained queries to `hybrid_search` without hardcoded regexes.
- **Two-Stage Neural Retrieval Cascade (`hybrid_search.py` & `reranker.py`)**:
  - **Stage 1 (High Recall Candidate Pool)**: Retrieves Top $N=75$ candidates using Reciprocal Rank Fusion (RRF) across Qdrant dense vector search (`BAAI/bge-m3`) and MySQL `FULLTEXT` keyword search.
  - **Stage 2 (Cross-Encoder Neural Reranker)**: Evaluates candidate relevance via `sentence_transformers.CrossEncoder` (`BAAI/bge-reranker-v2-m3` / `ms-marco-MiniLM-L-6-v2`) with Apple Silicon `mps` auto-acceleration, producing the final Top 10 reranked context.
- **3-Tier Negative Coverage Engine (`coverage_analyzer.py`)**:
  - **Tier 1 (Relational Audit Invariant)**: Queries MySQL system-of-record to identify newspapers with zero articles on the requested topic.
  - **Tier 2 (Vector/Semantic Negative Verification)**: Audits low semantic similarity candidates to confirm negative reporting.
  - **Tier 3 (Multi-Newspaper Reconciliation Matrix)**: Generates comparative editorial matrices detailing perspective differences and omitted coverage.
- **Synthesizer (`synthesizer.py`)**: Produces structured 4-tier briefs (`Executive Summary`, `Key Verified Facts`, `Broadsheet Perspectives`, `Explore Further`) with resilient multi-provider failover (`groq_compound` $\rightarrow$ `gemini_flash` $\rightarrow$ `groq_qwen` $\rightarrow$ `ollama_llama3`).
- **IR Evaluation & Metrics Suite (`backend/app/evaluation/metrics.py`)**: Automated information retrieval evaluation benchmarks computing `Recall@K`, `Precision@K`, `MRR`, `NDCG@K`, `Faithfulness`, and `Coverage F1`.

### C. Cross-Newspaper Narrative Trajectory (`backend/app/retrieval/timeline_builder.py`)
- Reconstructs story lifecycles across multiple broadsheets and dates.
- Identifies reporting phases (`Breaking`, `Development`, `Financial Impact`, `Regulatory/Outcome`).
- Detects editorial discrepancies and reporting tensions across rival newspapers with Redis query caching.

---

## 4. Provider Abstraction & Model Registry

All model providers implement structural subtyping via Python `Protocol` interfaces:
- `ChatModelProvider`: `GroqProvider`, `GeminiProvider`, `OpenAIProvider`, `OllamaProvider`
- `EmbeddingProvider`: `LocalEmbeddingProvider` (bge-m3), `OpenAIProvider`
- `VisionModelProvider`: `GoogleCloudVisionOCR`, `GeminiProvider`, `OllamaProvider` (gemma4:26b, qwen2.5vl)
- `DocumentLayoutProvider` & `OCREngine`: `GoogleCloudVisionOCR`, `DoclingProvider`, `MinerUProvider`, `GeminiProvider`, `TesseractOCR`

Configured declaratively in `model_config.yaml` with runtime zero-downtime swapping via `/api/settings/model-bindings`.

