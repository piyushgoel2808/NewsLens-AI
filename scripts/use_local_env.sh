#!/usr/bin/env bash
# Switch NewsLens-AI configuration to Local Infrastructure (Docker MinIO, Qdrant, Redis, MySQL)
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env.local ]; then
  cp .env.local .env
  touch backend/app/api/main.py
  echo "✅ Switched to LOCAL environment (.env.local -> .env)"
  echo "• Qdrant: http://localhost:6333/dashboard"
  echo "• MinIO Console: http://localhost:9001 (minioadmin / minioadmin123)"
  echo "• MinIO S3 API: http://localhost:9000"
  echo "• Redis: localhost:6379"
  echo "• MySQL: localhost:3306"
else
  echo "❌ .env.local not found"
  exit 1
fi
