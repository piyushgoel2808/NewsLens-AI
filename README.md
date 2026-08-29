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
- **⚡ Two-Stage Neural Retrieval Cascade (Cross-Encoder)**: Stage 1 RRF hybrid search (Qdrant `BAAI/bge-m3` + MySQL `FULLTEXT`) expands candidates to $N=75$, followed by Stage 2 Cross-Encoder neural reranking (`BAAI/bge-reranker-v2-m3` with Apple Silicon `mps` acceleration) returning the Top 10 high-precision hits.
- **📊 3-Tier Negative Coverage Engine**: Performs relational audits in MySQL to identify zero-coverage publications, semantic validation, and multi-newspaper editorial reconciliation matrices (`POST /api/query/coverage`).
- **🤖 Autonomous LangGraph Agent Workflow with Agentic Query Routing**: Pydantic structured Chain-of-Thought (CoT) planning without regex rules. Autonomously routes macro-level queries (whole-issue summaries, page manifests, counts) to `sql_analytics` and fine-grained queries to `hybrid_search`.
- **🕸️ Interactive Multi-Hop Entity Knowledge Graph**: Interactive visualizer mapping entity relationships, co-occurrences, and hop depths across shared stories and event clusters with Corrective RAG (CRAG) self-reflection.
- **⚡ 4-Tier Structured Executive Intelligence Briefs**:
  - `⚡ Executive Summary`: High-impact core answer and market reaction.
  - `📌 Key Verified Facts & Financials`: Precise figures, index moves (Sensex, Nifty), FPI/FII net inflows (in ₹ crore), and inline citations.
  - `📰 Broadsheet Perspectives`: Distinct editorial angles (*Mint*, *Business Standard*, *The Hindu*, *The New York Times*).
  - `🔍 Explore Further`: Clickable exploration pills in the UI for instant drill-down queries.
- **🌐 Dual-Mode Retrieval (Archive + Live Web)**: Toggle live Google/Tavily/DuckDuckGo web grounding with distinct visual citation badges.
- **📈 Cross-Newspaper Narrative Trajectory & Story Timelines**: Reconstructs evolving stories across calendar dates with 4-tier anti-hallucination gates, tracking reporting phases (`Breaking`, `Development`, `Financial Impact`, `Regulatory/Outcome`) and editorial discrepancies with Redis caching.
- **🔄 Multi-Provider Resilience & Failover**: Hot-swappable provider support across **Groq**, **Google Gemini**, **OpenAI**, and local **Ollama** models with automated failover chains.
- **🎯 Interactive Scan Reader & Visual Asset Inspector**: High-resolution 300 DPI broadsheet reader with spatial bounding-box pulses, visual sidebar badges (`📷 Photo`, `📊 Infographic`, `🔢 Table`), and high-res cropped photo inspection.

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
│  • Multi-Page Consensus     │  • MinIO Object Storage (S3)  │  • 2-Stage Rerank Cascade   │
│  • 2D Spatial Photo Binding │  • Redis 7 (Cache & Lock)     │  • 4-Tier Synthesizer       │
│  • Qwen3-VL Visual Extract  │  • RRF (Dense + Sparse Fusion)│  • Corrective RAG (CRAG)    │
└─────────────────────────────┴───────────────────────────────┴─────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/) (fast Python package manager)
- **Node.js 18+** and `npm`
- **Docker & Docker Compose**

### 1. Clone & Set Up Infrastructure
```bash
# Clone the repository
git clone https://github.com/piyushgoel2808/NewsLens-AI.git
cd NewsLens-AI

# Copy environment variables
cp .env.example .env

# Spin up local services (MySQL, Qdrant, MinIO, Redis, Ollama)
docker compose -f docker-compose.local.yml up -d
```

### 2. Backend Setup
```bash
cd backend

# Install dependencies using uv
uv sync --all-extras

# Run database migrations
uv run alembic upgrade head

# Start the FastAPI server
uv run uvicorn app.api.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd ../frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
Open **`http://localhost:5173`** in your browser to access the NewsLens-AI platform!

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

# Run complete test suite (293 unit & integration tests)
uv run pytest tests/ -v

# Verify frontend production build
cd ../frontend
npm run build
```

---

