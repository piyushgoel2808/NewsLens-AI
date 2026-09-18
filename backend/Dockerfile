# Multi-stage Dockerfile for NewsLens-AI Backend
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install system dependencies (OpenCV runtime, Tesseract OCR, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

# Copy dependency specifications first for layer caching
COPY pyproject.toml ./

# Install python dependencies into system environment
RUN uv pip install --system -r pyproject.toml

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 10001 appuser \
    && mkdir -p /app/debug_output /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser

# Copy backend application code and configurations
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser model_config.prod.yaml /app/model_config.prod.yaml

USER appuser

EXPOSE 8000 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -sf http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
