# Changelog

All notable changes to **NewsLens-AI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-09-16

### Added
- **Core Broadsheet Layout Analysis & Neural Parsing**:
  - Multi-column article segmentation with IBM Docling (DocLayNet) and RapidOCR fallback.
  - 2D spatial headline-deck coalescing and drop-cap normalization.
  - Multi-page consensus masthead and publication date verification.
- **Multimodal Visual Asset Intelligence**:
  - Automatic isolation of editorial photos, infographics, and standalone figures.
  - 2D proximity scoring for caption-to-photo binding.
  - VLM visual grounding and on-demand infographic analysis via Qwen-VL and Gemini.
- **Agentic RAG & LangGraph State Machine**:
  - Dynamic Chain-of-Thought (CoT) query planner grounded in live relational archive metadata.
  - 8 specialized retrieval tools (hybrid search, issue inspection, entity network, cross-newspaper timeline, SQL analytics, visual inspection, dynamic tool synthesis).
  - CRAG (Corrective RAG) reflexive evaluator and adaptive re-planning loop.
  - Subprocess AST sandbox for safe execution of dynamic analytical tools.
- **Hybrid Retrieval & RRF**:
  - Dense vector retrieval with Qdrant and BAAI/bge-m3 embeddings.
  - Sparse relational retrieval with MySQL FULLTEXT search.
  - Stage-2 neural reranking with Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
- **Interactive React SPA Frontend**:
  - 300 DPI high-resolution Broadsheet Reader with interactive spatial bounding boxes.
  - Real-time Server-Sent Events (SSE) streaming reasoning trace and assistant chat.
  - Multi-hop Entity Knowledge Graph visualizer.
  - Cross-Newspaper Narrative Trajectory and Timeline explorer.
  - Dynamic Model Settings Studio for hot-swappable provider configurations.
- **Production Readiness & DevOps**:
  - Full Docker Compose stack (`docker-compose.yml`) for one-command containerized deployment.
  - Multi-stage Dockerfiles for backend and frontend.
  - Unified `make setup` and developer tooling in Makefile.
  - GitHub Actions CI workflow covering linting, type-checking, backend test suite, and frontend production builds.
