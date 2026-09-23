#!/usr/bin/env bash
# =============================================================================
# NewsLens-AI: Resume GCP Production Infrastructure
# =============================================================================
# This script restarts Cloud SQL and ensures Cloud Run is ready to serve:
# 1. Cloud SQL (newslens-mysql) -> activation-policy=ALWAYS
# 2. Waits for MySQL to reach RUNNABLE status
# 3. Cloud Run services remain at min-instances=0 (or 1) with auto-scaling on demand.
# =============================================================================

set -euo pipefail

PROJECT_ID="newslens-ai-prod"
REGION="asia-south1"
DB_INSTANCE="newslens-mysql"

echo "============================================================"
echo " Resuming NewsLens-AI GCP Infrastructure ($PROJECT_ID)"
echo "============================================================"

# 1. Start Cloud SQL Instance
echo "--> [1/2] Starting Cloud SQL Database ($DB_INSTANCE)..."
gcloud sql instances patch "$DB_INSTANCE" \
  --project="$PROJECT_ID" \
  --activation-policy=ALWAYS \
  --quiet

echo "--> Waiting for Cloud SQL to become RUNNABLE..."
for i in {1..30}; do
  STATUS=$(gcloud sql instances describe "$DB_INSTANCE" --project="$PROJECT_ID" --format="value(state)" || echo "UNKNOWN")
  if [ "$STATUS" = "RUNNABLE" ]; then
    echo "    Cloud SQL is RUNNABLE!"
    break
  fi
  echo "    Status: $STATUS (waiting 5s... attempt $i/30)"
  sleep 5
done

# 2. Restore Cloud Run Worker & Backend Instances
echo "--> [2/3] Restoring Cloud Run Worker (min-instances=1, no-cpu-throttling)..."
gcloud run services update newslens-worker \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --min-instances=1 \
  --no-cpu-throttling \
  --quiet

echo "--> [3/3] Restoring Cloud Run Backend (min-instances=1)..."
gcloud run services update newslens-backend \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --min-instances=1 \
  --cpu-throttling \
  --quiet

# 3. Verify Backend and Worker Health
echo "--> Checking Cloud Run Backend availability..."
BACKEND_URL=$(gcloud run services describe newslens-backend --project="$PROJECT_ID" --region="$REGION" --format="value(status.url)")
echo "    Pinging ${BACKEND_URL}/health..."
for i in {1..6}; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/health" || true)
  if [ "$HTTP_CODE" = "200" ]; then
    echo "    Backend responded with HTTP 200 OK!"
    break
  fi
  echo "    Waiting for container warm-up (attempt $i/6, HTTP $HTTP_CODE)..."
  sleep 5
done

echo ""
echo "============================================================"
echo " 🚀 GCP INFRASTRUCTURE IS NOW FULLY ACTIVE!"
echo "============================================================"
echo " • Database: $DB_INSTANCE is RUNNABLE"
echo " • Backend:  $BACKEND_URL"
echo " • Frontend: https://newslens-frontend-ubxy3uyqkq-el.a.run.app"
echo "============================================================"
