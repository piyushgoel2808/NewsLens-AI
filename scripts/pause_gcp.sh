#!/usr/bin/env bash
# =============================================================================
# NewsLens-AI: Pause GCP Production Infrastructure to Minimize Costs
# =============================================================================
# This script stops all billable compute resources on Google Cloud:
# 1. Cloud SQL (newslens-mysql) -> activation-policy=NEVER (stops compute, preserves data)
# 2. Cloud Run (newslens-worker) -> min-instances=0 (scales to 0 instances when idle)
# 3. Cloud Run (newslens-backend) -> min-instances=0 (scales to 0 instances when idle)
#
# When paused, your monthly bill drops from ~$100/mo to <$0.50/mo (disk storage only).
# =============================================================================

set -euo pipefail

PROJECT_ID="newslens-ai-prod"
REGION="asia-south1"
DB_INSTANCE="newslens-mysql"

echo "============================================================"
echo " Pausing NewsLens-AI GCP Infrastructure ($PROJECT_ID)"
echo "============================================================"

# 1. Scale Cloud Run Services to Zero Instances
echo "--> [1/3] Scaling Cloud Run Worker to 0 min-instances..."
gcloud run services update newslens-worker \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --min-instances=0 \
  --cpu-throttling \
  --quiet

echo "--> [2/3] Scaling Cloud Run Backend to 0 min-instances..."
gcloud run services update newslens-backend \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --min-instances=0 \
  --cpu-throttling \
  --quiet

# 2. Stop Cloud SQL Instance
echo "--> [3/3] Pausing Cloud SQL Database ($DB_INSTANCE)..."
gcloud sql instances patch "$DB_INSTANCE" \
  --project="$PROJECT_ID" \
  --activation-policy=NEVER \
  --quiet

echo ""
echo "============================================================"
echo " ✅ GCP INFRASTRUCTURE SUCCESSFULLY PAUSED!"
echo "============================================================"
echo " • Cloud Run: Scaled to 0 instances (Bill: \$0.00)"
echo " • Cloud SQL: Compute stopped (Bill: \$0.00 compute)"
echo " • Cloud Storage & Database Disks: Preserved safely (< \$0.50/month)"
echo ""
echo "To resume anytime, run: ./scripts/resume_gcp.sh"
echo "============================================================"
