#!/usr/bin/env bash
# Switch NewsLens-AI configuration to Cloud Infrastructure (Upstash, Cloud Qdrant, GCS, Cloud SQL)
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env.cloud ]; then
  cp .env.cloud .env
  touch backend/app/api/main.py
  echo "☁️ Switched to CLOUD environment (.env.cloud -> .env)"
  echo "• Qdrant: Cloud Cluster (Australia)"
  echo "• Redis: Upstash Cloud"
  echo "• Storage: MinIO/GCS"
else
  echo "❌ .env.cloud not found"
  exit 1
fi
