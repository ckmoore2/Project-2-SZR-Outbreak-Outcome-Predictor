#!/usr/bin/env bash
# deploy.sh — build, push, and deploy the SZR Predictor to Cloud Run.
# Prerequisites: docker, gcloud CLI authenticated with the target project.
# Usage: bash deploy.sh
set -euo pipefail

PROJECT_ID=river-yew-486018-v3
IMAGE=gcr.io/$PROJECT_ID/szr-predictor

echo "Building image for linux/amd64..."
docker build --platform linux/amd64 -t "$IMAGE" .

echo "Pushing image to Google Container Registry..."
docker push "$IMAGE"

echo "Deploying to Cloud Run (us-east1)..."
gcloud run deploy szr-predictor \
  --image "$IMAGE" \
  --platform managed \
  --region us-east1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --project "$PROJECT_ID"

echo ""
echo "Deployment complete. Retrieve the live URL with:"
echo "  gcloud run services describe szr-predictor --region us-east1 --project $PROJECT_ID --format 'value(status.url)'"
