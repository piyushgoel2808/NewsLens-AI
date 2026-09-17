# Changelog

All notable changes to **NewsLens-AI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-18

### Added
- **Production Serverless Deployment on Google Cloud Platform (`asia-south1`)**:
  - Live deployment of `newslens-frontend` (Nginx 1.27 Alpine SPA reverse proxy) and `newslens-backend` (FastAPI) to Google Cloud Run.
  - Live deployment of `newslens-worker` Celery background ingestion consumer configured with `--no-cpu-throttling` and `--min-instances=1` to guarantee uninterrupted background processing.
  - Dedicated worker entrypoint (`backend/app/run_worker.py`) with embedded background HTTP server on `$PORT` to satisfy Cloud Run liveness and readiness health probes while running Celery in the foreground.
  - Dedicated Cloud Run Job `newslens-migrate` executing Alembic migrations (`alembic upgrade head`) over the Cloud SQL Unix socket before container revision rollouts.
- **Polymorphic Cloud Storage Abstraction**:
  - `GoogleCloudStorageStore` (`backend/app/storage/gcs_store.py`) implementing the `ObjectStore` protocol with threadpool executor offloading for non-blocking async execution.
  - Dynamic storage factory `get_object_store()` (`backend/app/storage/factory.py`) supporting seamless switching between Google Cloud Storage (`STORAGE_BACKEND=gcs`) and MinIO S3 (`STORAGE_BACKEND=minio`).
  - Automated bucket initialization for `gs://newslens-ai-prod-pages` and `gs://newslens-ai-prod-originals`.
- **Managed Production Databases & Broker**:
  - Managed Cloud SQL for MySQL 8.0 with Unix Domain Socket connectivity (`/cloudsql/...`), utf8mb4 encoding, and FULLTEXT search indexes.
  - Managed Qdrant Cloud vector cluster on GCP with TLS encryption and API key authentication for 1024-dim BGE-M3 embeddings.
  - Upstash Managed Redis with TLS support (`rediss://...ssl_cert_reqs=required`) powering Celery task queues and sub-millisecond query caching.
- **Enterprise Keyless CI/CD Automation**:
  - GitHub Actions deployment pipeline (`.github/workflows/deploy-gcp.yml`) leveraging Google Cloud Workload Identity Federation (WIF) with OIDC authentication, eliminating all static JSON service account keys.
  - Automated quality gate running linting, type-checking, ephemeral MySQL migrations, and the full 574-test suite before builds.
  - Automated multi-arch container image builds pushed to Google Artifact Registry.
- **Production Model Configuration**:
  - `model_config.prod.yaml` binding Google Gemini Flash (`gemini-3.8-flash` with multi-candidate failover) across all core agentic reasoning, VLM, and extraction pipelines.
  - Centralized Google Secret Manager integration managing 11 production secrets.

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
