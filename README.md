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

- **📑 Multi-Column Layout Segmentation & 2D Reading Order**: Uses **IBM Docling**, **MinerU**, and **Google Gemini Vision** to segment complex broadsheet layouts into discrete articles, preventing column bleeding.
- **🔍 Hybrid Dense/Sparse Vector Search**: Combines **Qdrant** dense vector embeddings (`BAAI/bge-m3`, 1024-dim) with **MySQL FULLTEXT** keyword matching for maximum precision.
- **🤖 Autonomous LangGraph Agent Workflow**: Dynamic query archetype classification (`factual_lookup`, `thematic_timeline`, `entity_deep_dive`, `comparative_analysis`) with conversational coreference resolution and in-context memory.
- **⚡ 4-Tier Structured Executive Intelligence Briefs**:
  - `⚡ Executive Summary`: High-impact core answer and market reaction.
  - `📌 Key Verified Facts & Financials`: Precise figures, index moves (Sensex, Nifty), FPI/FII net inflows (in ₹ crore), and inline citations.
  - `📰 Broadsheet Perspectives`: Distinct editorial angles (*Mint*, *Business Standard*, *The Hindu*).
  - `🔍 Explore Further`: Clickable exploration pills in the UI for instant drill-down queries.
- **🌐 Dual-Mode Retrieval (Archive + Live Web)**: Toggle live Google/Tavily/DuckDuckGo web grounding with distinct visual citation badges.
- **📈 Cross-Newspaper Narrative Trajectory & Story Timelines**: Reconstructs evolving stories across calendar dates, tracks reporting phases (`Breaking`, `Development`, `Financial Impact`, `Regulatory/Outcome`), and detects editorial reporting discrepancies with Redis caching.
- **🔄 Multi-Provider Resilience & Failover**: Hot-swappable provider support across **Groq**, **Google Gemini**, **OpenAI**, and local **Ollama** models with automated failover chains.
- **🎯 Interactive Scan Reader with Spatial Bounding-Box Pulses**: Clicking any broadsheet citation navigates directly to the exact PDF page and pulses the bounding boxes of the referenced article.

---

## 🏗️ System Architecture

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

  local_embed_bge:
    provider: local_sentence_transformers
    model: BAAI/bge-m3
    embedding_dim: 1024

task_bindings:
  query_planner: groq_compound
  answerer: groq_compound
  layout_analysis: docling_parser
  article_segmentation: ollama_llama3
  embedding: local_embed_bge
```

---

## 🧪 Testing & Code Quality

```bash
# Run backend linter & type checks
cd backend
uv run ruff check .
uv run mypy app/

# Run complete test suite (246 unit & integration tests)
uv run pytest tests/ -v

# Verify frontend production build
cd ../frontend
npm run build
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
