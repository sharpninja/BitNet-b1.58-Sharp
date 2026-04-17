#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  install-bitnet-worker.sh — generic one-shot worker launcher.
#
#  Pulls ghcr.io/sharpninja/bitnetsharp-worker:latest and runs it with
#  the hardened `docker run` flags used in production. Reads all
#  configuration from environment variables.
#
#  Required environment variables:
#    BITNET_COORDINATOR_URL   Coordinator base URL (e.g. https://host:5001/)
#    BITNET_WORKER_API_KEY    Shared API key set by the operator on the
#                             coordinator via Coordinator__WorkerApiKey
#
#  Optional:
#    BITNET_WORKER_ID         Stable worker identity  (default: $(hostname))
#    BITNET_WORKER_NAME       Human-friendly display  (default: $(hostname))
#    BITNET_CPU_THREADS       Hard cap on threads     (default: all logical CPUs)
#    BITNET_HEARTBEAT_SECONDS Heartbeat cadence       (default: 30)
#    BITNET_LOG_LEVEL         Serilog minimum level   (default: info)
#    BITNET_IMAGE             image:tag               (default: ghcr.io/sharpninja/bitnetsharp-worker:latest)
#    BITNET_CONTAINER_NAME    container name          (default: bitnet-worker)
#
#  Example:
#    export BITNET_COORDINATOR_URL='https://bitnet.example.com:5001/'
#    export BITNET_WORKER_API_KEY='the-secret-the-operator-set'
#    ./install-bitnet-worker.sh
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

fail() {
    echo "[installer] ERROR: $*" >&2
    exit 2
}

: "${BITNET_COORDINATOR_URL:?BITNET_COORDINATOR_URL is not set. Export it before running.}"
: "${BITNET_WORKER_API_KEY:?BITNET_WORKER_API_KEY is not set. Ask the operator for the value set on Coordinator__WorkerApiKey.}"

WORKER_ID="${BITNET_WORKER_ID:-$(hostname)}"
WORKER_NAME="${BITNET_WORKER_NAME:-$(hostname)}"
HEARTBEAT="${BITNET_HEARTBEAT_SECONDS:-30}"
LOG_LEVEL="${BITNET_LOG_LEVEL:-info}"
IMAGE="${BITNET_IMAGE:-ghcr.io/sharpninja/bitnetsharp-worker:latest}"
CONTAINER_NAME="${BITNET_CONTAINER_NAME:-bitnet-worker}"

echo "[installer] Using Docker worker image."
echo "[installer] coordinator: $BITNET_COORDINATOR_URL"
echo "[installer] worker id  : $WORKER_ID"
echo "[installer] image      : $IMAGE"

# Remove any pre-existing container with the same name so the run
# picks up the freshly-pulled image.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

env_args=(
    -e "BITNET_COORDINATOR_URL=$BITNET_COORDINATOR_URL"
    -e "BITNET_WORKER_API_KEY=$BITNET_WORKER_API_KEY"
    -e "BITNET_WORKER_ID=$WORKER_ID"
    -e "BITNET_WORKER_NAME=$WORKER_NAME"
    -e "BITNET_HEARTBEAT_SECONDS=$HEARTBEAT"
    -e "BITNET_LOG_LEVEL=$LOG_LEVEL"
)
if [[ -n "${BITNET_CPU_THREADS:-}" ]]; then
    env_args+=(-e "BITNET_CPU_THREADS=$BITNET_CPU_THREADS")
fi

docker run \
    -d \
    --pull=always \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --read-only \
    --tmpfs /tmp:size=64m,mode=1777 \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    "${env_args[@]}" \
    "$IMAGE" \
    || fail "docker run failed"

echo "[installer] Worker container '$CONTAINER_NAME' started. Tail logs with:"
echo "             docker logs -f $CONTAINER_NAME"
