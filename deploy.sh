#!/usr/bin/env bash
# deploy.sh — build, push, and deploy the SZR Predictor to Cloud Run.
# Prerequisites: docker, gcloud CLI authenticated with the target project.
#
# Usage:
#   bash deploy.sh            # build for Cloud Run (linux/amd64) and deploy
#   bash deploy.sh --local    # build for local testing using native architecture
set -euo pipefail

PROJECT_ID=szr-outbreak-outcome-predictor
IMAGE=gcr.io/$PROJECT_ID/szr-predictor

# ── Detect host architecture ──────────────────────────────────────────────────
HOST_ARCH=$(uname -m)
case "$HOST_ARCH" in
  arm64|aarch64) HOST_PLATFORM="linux/arm64" ;;
  x86_64|amd64)  HOST_PLATFORM="linux/amd64" ;;
  *)             HOST_PLATFORM="linux/$HOST_ARCH" ;;
esac

# ── Parse arguments ───────────────────────────────────────────────────────────
MODE="deploy"
if [ "${1:-}" = "--local" ]; then
  MODE="local"
fi

if [ "$MODE" = "local" ]; then
  # ── Local build: use native architecture for fastest build + no emulation ──
  echo "Host architecture: $HOST_ARCH → building for $HOST_PLATFORM (native)"
  docker build --platform "$HOST_PLATFORM" -t szr-predictor-local .
  echo ""
  echo "Local image built: szr-predictor-local"
  echo "Run with:  docker run -p 8501:8080 szr-predictor-local"
  exit 0
fi

# ── Deployment build: always target linux/amd64 for Cloud Run ─────────────────
DEPLOY_PLATFORM="linux/amd64"

if [ "$HOST_PLATFORM" != "$DEPLOY_PLATFORM" ]; then
  echo "Host: $HOST_ARCH ($HOST_PLATFORM) → cross-compiling to $DEPLOY_PLATFORM for Cloud Run"
  echo "(emulation via Docker buildx — this will be slower than a native build)"
else
  echo "Host: $HOST_ARCH → building natively for $DEPLOY_PLATFORM"
fi

echo ""
echo "Building image for $DEPLOY_PLATFORM..."
docker build --platform "$DEPLOY_PLATFORM" -t "$IMAGE" .

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
