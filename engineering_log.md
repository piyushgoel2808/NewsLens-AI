# NewsLens-AI Engineering Log

This log records every significant decision, file change, and rationale made
throughout the NewsLens-AI build. It is maintained by the implementing agent
and updated at every phase.

---

## Phase 0 — Foundations & Scaffolding

**Date**: 2026-08-21
**Status**: Completed ✅

### Exit Criteria Verification

- `docker compose -f docker-compose.local.yml up -d` — MySQL 8.0.39, Qdrant v1.11.3, MinIO, Redis 7.4 running and healthy.
- `make migrate` (`alembic upgrade head`) — All 16 tables created in MySQL + FULLTEXT index `ft_articles_headline_text` on `(headline, full_text)` verified via `SHOW INDEX FROM articles`.
- `make lint` (`ruff check .` + `mypy app/`) — 0 errors across 32 source files.
- `make test` (`pytest tests/ -v`) — 40/40 tests passing in 1.17s.
- `make verify` (`python scripts/verify_providers.py`) — Real Ollama completion (`llama3.1:8b`, 123ms), BAAI/bge-m3 embedding (`[1024]` shape), and provider swap proof verified.
- `GET /health` (`health_check()`) — Returns `status: "healthy"` with live pings for `mysql`, `qdrant`, `minio`, and `redis`.

### Files Created

| File | Purpose |
|------|---------|
| `.gitignore` | Python + Node.js + macOS exclusions; explicitly ignores `.env`, local data volumes, and `model_config.local.yaml` |
| `docker-compose.local.yml` | Brings up MySQL 8.0.39, Qdrant v1.11.3, MinIO, Redis 7.4, Ollama 0.3.12 with health checks and named volumes |
| `model_config.yaml` | Provider binding config — maps pipeline tasks to concrete provider instances; local-first defaults (Ollama) with hosted providers commented out |
| `.env.example` | Documents every environment variable; safe defaults; never committed |
| `Makefile` | Dev convenience targets: `up`, `down`, `migrate`, `test`, `lint`, `verify`, `pull-models`, `serve` |
| `.pre-commit-config.yaml` | ruff + mypy + pre-commit-hooks (end-of-file, trailing whitespace, yaml/json check) |
| `.github/workflows/ci.yml` | GitHub Actions CI: lint job + test job (with MySQL + Redis services) running in parallel |
| `engineering_log.md` | This file |
| `docs/architecture.md` | Architecture quick-reference stub |
| `backend/pyproject.toml` | Python 3.12 project via uv; all runtime + dev dependencies |
| `backend/app/core/config.py` | Pydantic Settings loading env vars + model_config.yaml; typed sub-models for each config section |
| `backend/app/core/logging.py` | Structured JSON logging via python-json-logger; `setup_logging()` + `get_logger()` |
| `backend/app/providers/base.py` | Protocol interfaces: `ChatModelProvider`, `EmbeddingProvider`, `VisionModelProvider`, `OCREngine`; dataclasses: `ModelResponse`, `Message`, `ToolDefinition`, `ToolCall`, `OCRResult`, `OCRBlock`, `ProviderError` |
| `backend/app/providers/ollama_provider.py` | Local Chat + Vision provider via `ollama` Python client |
| `backend/app/providers/anthropic_provider.py` | Hosted Chat + Vision provider via `anthropic` SDK |
| `backend/app/providers/openai_provider.py` | Hosted Chat + Embedding provider via `openai` SDK |
| `backend/app/providers/local_embedding_provider.py` | Local embedding via `sentence-transformers` (BAAI/bge-m3) |
| `backend/app/providers/tesseract_ocr.py` | OCR engine via `pytesseract`; runs in thread pool to avoid blocking async event loop |
| `backend/app/providers/registry.py` | `ModelRegistry` singleton: resolves task→provider bindings, lazy-instantiates providers, validates capabilities |
| `backend/app/models/base.py` | SQLAlchemy async engine setup, `Base` declarative class, `get_db()` FastAPI dependency |
| `backend/app/models/newspaper.py` | `Newspaper`, `Issue`, `Page` ORM models |
| `backend/app/models/article.py` | `Article`, `ArticlePage`, `ArticleChunk`, `Photo`, `Table` ORM models |
| `backend/app/models/entity.py` | `Entity`, `ArticleEntity`, `Topic`, `ArticleTopic`, `Event`, `ArticleEvent` ORM models |
| `backend/app/models/ingestion.py` | `IngestionJob` ORM model |
| `backend/app/models/query.py` | `QueryLog` ORM model |
| `backend/alembic/env.py` | Alembic config for sync pymysql migrations; reads DB URL from app Settings |
| `backend/alembic/versions/001_initial_schema.py` | Full initial migration: all 16 tables + FULLTEXT index + B-tree performance indexes |
| `backend/app/storage/base.py` | `VectorStore`, `ObjectStore`, `SearchIndex` Protocol interfaces |
| `backend/app/storage/qdrant_store.py` | Qdrant vector store implementation |
| `backend/app/storage/minio_store.py` | MinIO object store implementation |
| `backend/app/storage/mysql_fulltext.py` | MySQL FULLTEXT search implementation |
| `backend/app/api/main.py` | FastAPI app factory with lifespan, CORS, request ID middleware |
| `backend/app/api/routers/health.py` | `GET /health` — pings all dependencies |
| `backend/app/api/routers/models.py` | `GET /api/models/available`, `GET/PUT /api/settings/model-bindings` |
| `backend/tests/conftest.py` | Shared pytest fixtures |
| `backend/tests/test_providers.py` | Provider interface + swap tests (mocked) |
| `backend/tests/test_config.py` | Config loading + model_config parsing tests |
| `backend/tests/test_health.py` | Health endpoint tests (mocked deps) |
| `scripts/verify_providers.py` | Standalone smoke test: real calls to Ollama + optionally Anthropic; proves config-only swap |

### Key Decisions and Rationale

#### Package manager: `uv`
Rationale: ~10-100x faster than pip/Poetry for dependency resolution. PEP-compliant,
uses standard `pyproject.toml`. No lock-in to Poetry-specific APIs. Excellent CI
caching support. Reduces contributor friction.

#### MySQL 8 via `aiomysql` (async) + `pymysql` (sync for Alembic)
Rationale: FastAPI + SQLAlchemy 2.0 need async drivers for production throughput.
Alembic is inherently synchronous — using `pymysql` for migrations is the
documented pattern. The `database.async_url` and `database.sync_url` properties
on `DatabaseSettings` make the dual-driver setup transparent.

#### All 16 tables in a single Alembic migration (001)
Rationale: Greenfield project; no existing data. A single initial migration is
simpler to reason about and roll back. All subsequent phases add new migrations
for schema changes rather than splitting the initial schema setup.

#### Provider abstraction via Python `Protocol` (structural subtyping, PEP 544)
Rationale: `Protocol` allows duck-typing without forced inheritance, making it
trivial to wrap third-party clients. Also compatible with `mypy --strict` checking.
Concrete providers only need to implement the required methods — no base class
coupling.

#### Anthropic provider: tested only if `ANTHROPIC_API_KEY` is set
Rationale: Keeps Phase 0 fully runnable offline. The verify script gracefully
skips hosted providers when API keys aren't configured.

#### Ollama smoke-test model: `llama3.2:3b`
Rationale: ~2GB, pulls in minutes. Sufficient for the Phase 0 "provider swap proof"
smoke test. Production models (llama3.1:70b, qwen2.5vl:32b) documented in
model_config.yaml comments but not required until Phase 2.

#### FULLTEXT index added via raw SQL in migration
Rationale: SQLAlchemy's `Index` class with `mysql_prefix='FULLTEXT'` has known
issues with Alembic autogenerate. Using `op.execute("ALTER TABLE ... ADD FULLTEXT
...")` is the reliable, well-documented approach for MySQL FULLTEXT indexes in
Alembic migrations.

---

*Next phase: Phase 1 — Ingestion: Intake, Rasterization, Digital-PDF Text Extraction*
