# Contributing to NewsLens-AI

Thank you for your interest in contributing to **NewsLens-AI**! We welcome bug reports, feature requests, documentation improvements, and code contributions.

---

## 🛠️ Development Setup

### 1. Prerequisites
- **Python 3.12+**
- [`uv`](https://docs.astral.sh/uv/) (fast Python package manager)
- **Node.js 20+** and `npm`
- **Docker & Docker Compose** (v2+)

### 2. Fork & Clone
```bash
git clone https://github.com/<your-username>/NewsLens-AI.git
cd NewsLens-AI
git checkout -b feature/your-feature-name
```

### 3. Quick Environment Setup
```bash
# Copy the environment file template
cp .env.example .env

# Run one-command setup (starts local containers, syncs python packages, applies migrations)
make setup
```

### 4. Running the Dev Servers
In terminal 1:
```bash
make serve
```

In terminal 2:
```bash
make frontend-dev
```

In terminal 3 (if testing asynchronous ingestion tasks):
```bash
make worker
```

---

## 🧪 Code Quality & Testing Guidelines

Before opening a pull request, ensure all linters, type checks, and tests pass:

```bash
# 1. Format & Lint code
make lint-fix
make lint

# 2. Run backend test suite
make test

# 3. Verify frontend production build
make frontend-build
```

### Python Standards
- We use **Ruff** for code formatting and linting (`ruff check` and `ruff format`).
- We enforce strict type checking with **Mypy** on `app/`.
- All database migrations must be tracked via **Alembic** (`uv run alembic revision --autogenerate -m "..."`).

### Frontend Standards
- React 18 + Vite SPA with TailwindCSS.
- Keep components modular under `frontend/src/components/`.
- Avoid hardcoded API endpoints; all calls use relative `/api` paths.

---

## 🔀 Pull Request Process

1. Create a descriptive branch: `feature/xyz` or `fix/issue-123`.
2. Commit your changes with clear, semantic commit messages (e.g. `feat: add OCR fallback for corrupted PDFs`, `fix: prevent CORS rejection in production`).
3. Ensure CI passes on your fork.
4. Open a Pull Request pointing to `main`. Include:
   - A description of the problem solved or feature added.
   - Any testing steps or screenshots/recordings if UI changes were made.
   - Reference any related issues (e.g. `Closes #12`).

---

## 💬 Community & Questions

For questions or discussions, please open an issue or GitHub Discussion thread on the repository.
