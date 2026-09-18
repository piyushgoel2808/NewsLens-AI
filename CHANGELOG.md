# Changelog

All notable changes to **NewsLens-AI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-09-18

### Added
- **Dynamic Model-Aware Token Budgeting (`resolve_dynamic_token_budget`)**:
  - Eliminated static archetype token caps in favor of dynamic per-model output envelopes derived from `ProviderCapability` (`max_output_tokens` 4,096 or 8,192).
  - Added reasoning model auto-detection (`is_reasoning_model: true`) and `reasoning_headroom = 2048` across Gemini 2.5 Flash, DeepSeek-R1, NVIDIA Nemotron, and OpenAI reasoning models.
  - Formulated dynamic token scaling for user-requested word limits: $\max(1024, \text{target\_words} \times 4) + \text{reasoning\_headroom}$.
  - Dynamically calculates `effective_max_tokens` inside provider loops for both `synthesize()` and `synthesize_stream()`.
- **Authoritative Planner LLM Cognitive Routing**:
  - Removed fast-path deterministic bypass from `_classify_and_plan_node` in LangGraph (`graph.py`); the Planner LLM is now the authoritative cognitive router receiving live database schema and archive boundaries.
  - Deterministic heuristic router is reserved strictly as an emergency resilience fallback in `QueryPlanner.plan_query_async`.
- **Statistical & Analytical Computation Archetype (`analytical_computation`)**:
  - Introduced `analytical_computation` archetype in Planner prompt and Blueprints, routing statistical/mathematical queries ("average word count", "average length", "correlation") to `dynamic_analysis`.
  - Decoupled analytical computations from scalar counts, eliminating the 80-word ceiling.
- **OCR Noise Tolerance & Semantic Reconstruction**:
  - Added Section 4 (`OCR NOISE TOLERANCE & INTELLIGENT RECONSTRUCTION`) to `COMMON_ANALYTICAL_GUIDELINES` in `synthesizer.py`.
  - Authorizes the LLM to phonetically and semantically reconstruct words corrupted by broadsheet scanning ("Reconstruction is not hallucination"), strictly prohibiting copying raw OCR errors into answers.
- **Prompt Context Isolation & Contamination Elimination**:
  - Single-article queries isolate evidence strictly to target article chunks, eliminating irrelevant advertisements and unrelated page stories from contaminating prompt context.
- **Conversational Follow-Up Condenser Disambiguation**:
  - Enhanced history extraction to parse citations and quoted headlines from previous turns.
  - Normalizer catches truncated outputs and safely reformulates follow-ups like `"summarise it"` into standalone article queries.
  - Emergency LLM fallback before Python string slicing in streaming API.

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
