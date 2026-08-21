# NewsLens-AI Architecture

This document is the living architecture reference for the NewsLens-AI system.

## Overview

NewsLens-AI is an **Advanced Agentic RAG** system for newspaper intelligence — enabling semantic search, summarization, entity tracking, timeline reconstruction, and comparative analysis across scanned historical and digital newspaper archives.

## Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Backend | Python 3.12 + FastAPI | Async, uv for deps |
| Task queue | Celery + Redis | Background ingestion |
| System of record | MySQL 8 | 16 tables, FULLTEXT index |
| Vector store | Qdrant | Cosine similarity, payload filters |
| Object storage | MinIO (local) / S3 (hosted) | Page rasters, cropped photos |
| Cache + broker | Redis 7 | Celery backend + query cache |
| LLM/VLM | Ollama (local) / Anthropic / OpenAI (hosted) | Config-swappable |
| Embeddings | bge-m3 (local) / OpenAI (hosted) | Config-swappable |
| OCR | Tesseract | Async via thread pool |
| Agent framework | LangGraph | Phase 5+ |
| Frontend | Next.js 14+ | Phase 6+ |
| Deployment | docker-compose (local), Helm/K8s (Phase 8+) | |

## Module Map

| Module | Path | Phase |
|--------|------|-------|
| Config | `backend/app/core/config.py` | 0 |
| Logging | `backend/app/core/logging.py` | 0 |
| Providers | `backend/app/providers/` | 0 |
| Storage | `backend/app/storage/` | 0 |
| ORM Models | `backend/app/models/` | 0 |
| API | `backend/app/api/` | 0 |
| Ingestion pipeline | `backend/app/ingestion/` | 1–4 |
| Retrieval toolbelt | `backend/app/retrieval/` | 5 |
| Agent (LangGraph) | `backend/app/agent/` | 5 |
| Frontend | `frontend/` | 6 |

## Provider Abstraction

All model providers implement Python `Protocol` interfaces:

```
ChatModelProvider  → OllamaProvider, AnthropicProvider, OpenAIProvider
EmbeddingProvider  → LocalEmbeddingProvider, OpenAIProvider
VisionModelProvider → OllamaProvider (vl models), AnthropicProvider
OCREngine          → TesseractOCR
```

Task→provider binding is configured in `model_config.yaml`. No code change required to swap providers.

## Build Phases

| Phase | Focus |
|-------|-------|
| **0** ✅ | Foundations: config, providers, models, migrations, API skeleton |
| 1 | Intake, rasterization, digital-PDF text extraction |
| 2 | Layout analysis, reading order, OCR |
| 3 | Article segmentation, type classification |
| 4 | Metadata extraction, embedding, indexing |
| 5 | Retrieval toolbelt, agentic query engine (LangGraph) |
| 6 | API hardening, Next.js frontend |
| 7 | Observability, cost control, resilience |
| 8 | Hosted deployment (Helm/K8s) |
| 9 | Evaluation, documentation, handoff |
