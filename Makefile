# =============================================================================
# NewsLens-AI — Makefile
# =============================================================================
# Convenience targets for local development.
# Run 'make help' to see all available commands.
# =============================================================================

.PHONY: help up down logs migrate migrate-down test test-cov lint lint-fix \
        install verify pull-models shell-mysql shell-redis

# Default target
.DEFAULT_GOAL := help

COMPOSE := docker compose -f docker-compose.local.yml
BACKEND  := cd backend &&

help: ## Show this help message
	@echo ""
	@echo "NewsLens-AI Development Commands"
	@echo "================================="
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ \
		{ printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# --- Infrastructure ---

up: ## Start all infrastructure services (MySQL, Qdrant, MinIO, Redis, Ollama)
	$(COMPOSE) up -d
	@echo ""
	@echo "Services started. Run 'make logs' to tail logs."
	@echo "MinIO console: http://localhost:9001 (minioadmin / minioadmin123)"
	@echo ""

down: ## Stop all infrastructure services
	$(COMPOSE) down

logs: ## Tail logs from all services
	$(COMPOSE) logs -f

# --- Database ---

migrate: ## Run Alembic migrations (upgrade to head)
	$(BACKEND) uv run alembic upgrade head

migrate-down: ## Rollback the last Alembic migration
	$(BACKEND) uv run alembic downgrade -1

migrate-history: ## Show Alembic migration history
	$(BACKEND) uv run alembic history --verbose

# --- Development ---

install: ## Install all Python dependencies via uv
	$(BACKEND) uv sync --all-extras

test: ## Run pytest test suite
	$(BACKEND) uv run pytest tests/ -v

test-cov: ## Run tests with coverage report
	$(BACKEND) uv run pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

lint: ## Run ruff + mypy
	$(BACKEND) uv run ruff check .
	$(BACKEND) uv run mypy app/

lint-fix: ## Auto-fix ruff lint issues
	$(BACKEND) uv run ruff check . --fix
	$(BACKEND) uv run ruff format .

verify: ## Run provider verification smoke test (proves config-only swap works)
	uv run --project backend python scripts/verify_providers.py

verify-phase1: ## Run Phase 1 end-to-end ingestion pipeline verification
	uv run --project backend python scripts/verify_phase1.py

verify-phase2: ## Run Phase 2 OCR & layout extraction pipeline verification
	uv run --project backend python scripts/verify_phase2.py

# --- Ollama model management ---

pull-models: ## Pull required Ollama models (run once after 'make up')
	@echo "Pulling Ollama models (this may take a few minutes)..."
	docker exec newslens-ollama ollama pull llama3.2:3b
	docker exec newslens-ollama ollama pull nomic-embed-text
	docker exec newslens-ollama ollama pull qwen2.5vl:7b
	@echo "Models ready."

pull-models-prod: ## Pull production-scale Ollama models (large download)
	docker exec newslens-ollama ollama pull llama3.1:70b
	docker exec newslens-ollama ollama pull qwen2.5vl:32b

# --- Debug shells ---

shell-mysql: ## Open a MySQL shell
	$(COMPOSE) exec mysql mysql -u newslens -pnewslens_pass newslens

shell-redis: ## Open a Redis CLI shell
	$(COMPOSE) exec redis redis-cli

# --- Backend server (local, outside Docker) ---

serve: ## Run the FastAPI dev server (requires 'make up' first)
	$(BACKEND) uv run uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Run a Celery worker (requires 'make up' first)
	$(BACKEND) uv run celery -A app.workers.celery_app worker --loglevel=info
