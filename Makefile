# =============================================================================
# NewsLens-AI — Makefile
# =============================================================================
# Convenience targets for local development, production deployment, and testing.
# Run 'make help' to see all available commands.
# =============================================================================

.PHONY: help setup dev up down logs migrate migrate-down test test-cov lint lint-fix \
        install frontend-install frontend-dev frontend-build prod-up prod-down \
        prod-logs verify pull-models shell-mysql shell-redis secrets-check

.DEFAULT_GOAL := help

COMPOSE := docker compose -f docker-compose.local.yml
COMPOSE_PROD := docker compose -f docker-compose.yml
BACKEND  := cd backend &&
FRONTEND := cd frontend &&
FRONTEND_V2 := cd frontend-v2 &&

help: ## Show this help message
	@echo ""
	@echo "NewsLens-AI Commands"
	@echo "===================="
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z0-9_-]+:.*?##/ \
		{ printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# --- One-Step Onboarding ---

setup: ## One-command project setup (checks .env, starts services, installs dependencies, migrates DB)
	@echo "=== 1. Checking Environment File ==="
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo " Created .env from .env.example. Please review and add any hosted API keys."; \
	else \
		echo " .env already exists."; \
	fi
	@echo ""
	@echo "=== 2. Starting Infrastructure Services (MySQL, Qdrant, MinIO, Redis, Ollama) ==="
	$(COMPOSE) up -d
	@echo "Waiting for MySQL to accept connections..."
	@until docker exec newslens-mysql mysqladmin ping -h localhost -u root --password=newslens_root --silent > /dev/null 2>&1; do \
		echo " Waiting for MySQL database..."; \
		sleep 2; \
	done
	@echo " MySQL is ready."
	@echo ""
	@echo "=== 3. Installing Python Backend Dependencies ==="
	$(BACKEND) uv sync --all-extras
	@echo ""
	@echo "=== 4. Running Database Migrations ==="
	$(BACKEND) uv run alembic upgrade head
	@echo ""
	@echo "=== 5. Installing Frontend Dependencies ==="
	$(FRONTEND) npm install
	@echo ""
	@echo "=================================================================="
	@echo " Setup complete!"
	@echo "   To start in dev mode: make dev"
	@echo "   Or run individual services:"
	@echo "     make serve        (FastAPI backend on port 8000)"
	@echo "     make frontend-dev (Vite frontend on port 5173)"
	@echo "     make worker       (Celery ingestion worker)"
	@echo "   Or run full-stack containerized: make prod-up"
	@echo "=================================================================="

dev: ## Start backend and frontend development servers concurrently
	@echo "Starting NewsLens-AI backend (port 8000) and frontend (port 5173)..."
	@make -j2 serve frontend-dev

# --- Local Infrastructure (Docker) ---

up: ## Start local infra services (MySQL, Qdrant, MinIO, Redis, Ollama)
	$(COMPOSE) up -d
	@echo ""
	@echo "Services started. Run 'make logs' to tail logs."
	@echo "MinIO console: http://localhost:9001 (minioadmin / minioadmin123)"
	@echo ""

down: ## Stop local infra services
	$(COMPOSE) down

logs: ## Tail logs from local infra services
	$(COMPOSE) logs -f

# --- Full-Stack Containerized Production (Root Compose) ---

prod-up: ## Start the entire stack in containers (backend, frontend, worker, db, etc.)
	$(COMPOSE_PROD) up -d --build
	@echo ""
	@echo "Full stack running!"
	@echo "Frontend: http://localhost:3000 (or configured FRONTEND_PORT)"
	@echo "Backend API: http://localhost:8000"
	@echo "MinIO console: http://localhost:9001"

prod-down: ## Stop full-stack containers
	$(COMPOSE_PROD) down

prod-logs: ## Tail logs from full-stack containers
	$(COMPOSE_PROD) logs -f

# --- Database ---

migrate: ## Run Alembic migrations (upgrade to head)
	$(BACKEND) uv run alembic upgrade head

migrate-down: ## Rollback the last Alembic migration
	$(BACKEND) uv run alembic downgrade -1

migrate-history: ## Show Alembic migration history
	$(BACKEND) uv run alembic history --verbose

schema: ## View MySQL database schema (e.g. 'make schema' or 'make schema table=articles')
	$(BACKEND) uv run python ../scripts/show_schema.py $(table)

schema-list: ## List all MySQL database tables
	$(BACKEND) uv run python ../scripts/show_schema.py --list

# --- Python Backend ---

install: ## Install all Python dependencies via uv
	$(BACKEND) uv sync --all-extras

serve: ## Run FastAPI development server with auto-reload (requires 'make up' first)
	$(BACKEND) uv run uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Run Celery ingestion worker (requires 'make up' first)
	$(BACKEND) uv run celery -A app.workers.celery_app worker --loglevel=info

test: ## Run pytest test suite
	$(BACKEND) uv run pytest tests/ -v

test-cov: ## Run tests with coverage report
	$(BACKEND) uv run pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

lint: ## Run ruff and mypy static analysis
	$(BACKEND) uv run ruff check .
	$(BACKEND) uv run mypy app/

lint-fix: ## Auto-fix ruff lint issues and format code
	$(BACKEND) uv run ruff check . --fix
	$(BACKEND) uv run ruff format .

# --- Frontend ---

frontend-install: ## Install frontend npm dependencies
	$(FRONTEND) npm install

frontend-dev: ## Run Vite frontend development server
	$(FRONTEND) npm run dev

frontend-build: ## Build frontend for production
	$(FRONTEND) npm run build

# --- Frontend V2 (Active Redesign) ---

frontend-v2-install: ## Install frontend-v2 npm dependencies
	$(FRONTEND_V2) npm install

frontend-v2-dev: ## Run Vite frontend-v2 development server (port 5174)
	$(FRONTEND_V2) npm run dev

frontend-v2-build: ## Build frontend-v2 for production
	$(FRONTEND_V2) npm run build

dev-v2: ## Start backend (port 8000) and frontend-v2 (port 5174) concurrently
	@echo "Starting NewsLens-AI backend (port 8000) and redesigned frontend-v2 (port 5174)..."
	@make -j2 serve frontend-v2-dev

# --- Ollama Models ---

pull-models: ## Pull default Ollama models (llama3.2:3b, nomic-embed-text, qwen2.5vl:7b)
	@echo "Pulling Ollama models (this may take a few minutes)..."
	docker exec newslens-ollama ollama pull llama3.2:3b
	docker exec newslens-ollama ollama pull nomic-embed-text
	docker exec newslens-ollama ollama pull qwen2.5vl:7b
	@echo "Models ready."

pull-models-prod: ## Pull production-scale Ollama models (llama3.1:70b, qwen2.5vl:32b)
	docker exec newslens-ollama ollama pull llama3.1:70b
	docker exec newslens-ollama ollama pull qwen2.5vl:32b

# --- Verification & Security ---

secrets-check: ## Verify no secret files or credentials are tracked by git
	@echo "Checking for tracked secret files..."
	@git ls-files --error-unmatch .env *service-account*.json 2>/dev/null && \
		(echo "❌ ERROR: Sensitive files are tracked by git!" && exit 1) || \
		echo "✅ No sensitive files are tracked by git."

verify: ## Run provider verification smoke test
	uv run --project backend python scripts/verify_providers.py

verify-phase1: ## Run Phase 1 end-to-end ingestion pipeline verification
	uv run --project backend python scripts/verify_phase1.py

verify-phase2: ## Run Phase 2 OCR & layout extraction pipeline verification
	uv run --project backend python scripts/verify_phase2.py

verify-phase3: ## Run Phase 3 article segmentation & cross-page assembly verification
	uv run --project backend python scripts/verify_phase3.py

verify-phase4: ## Run Phase 4 metadata extraction, embedding & vector indexing verification
	uv run --project backend python scripts/verify_phase4.py

verify-phase5: ## Run Phase 5 agentic retrieval engine & LangGraph workflow verification
	uv run --project backend python scripts/verify_phase5.py

verify-phase6-1: ## Run Phase 6.1 API hardening & functional data flow verification
	uv run --project backend python scripts/verify_phase6_1.py

# --- Debug Shells ---

shell-mysql: ## Open an interactive MySQL shell
	$(COMPOSE) exec mysql mysql -u newslens -pnewslens_pass newslens

shell-redis: ## Open an interactive Redis CLI shell
	$(COMPOSE) exec redis redis-cli
