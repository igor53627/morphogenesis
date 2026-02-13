#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/ops/datadog/docker-compose.yml"

if [[ -z "${DD_API_KEY:-}" ]]; then
    echo "DD_API_KEY is required"
    exit 1
fi

KEEP_AGENT="${KEEP_DATADOG_AGENT:-0}"
WAIT_SECS="${DATADOG_READY_TIMEOUT_SEC:-90}"
DD_HOSTNAME="${DD_HOSTNAME:-morphogenesis-e2e}"
export DD_HOSTNAME

cleanup() {
    if [[ "${KEEP_AGENT}" != "1" ]]; then
        docker compose -f "${COMPOSE_FILE}" down --remove-orphans >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "[datadog] starting agent"
docker compose -f "${COMPOSE_FILE}" up -d datadog-agent

echo "[datadog] waiting for APM intake on http://127.0.0.1:8126/info"
READY=0
for ((i = 1; i <= WAIT_SECS; i++)); do
    if curl -fsS "http://127.0.0.1:8126/info" >/dev/null; then
        READY=1
        break
    fi
    if ! docker compose -f "${COMPOSE_FILE}" ps --status running datadog-agent | rg -q "datadog-agent"; then
        echo "[datadog] agent container is not running"
        docker compose -f "${COMPOSE_FILE}" logs --tail=80 datadog-agent || true
        exit 1
    fi
    sleep 1
done

if [[ "${READY}" != "1" ]]; then
    echo "[datadog] agent did not become ready"
    docker compose -f "${COMPOSE_FILE}" logs --tail=80 datadog-agent || true
    exit 1
fi

echo "[datadog] running RPC E2E with OTel export"
ENABLE_DATADOG=1 \
OTEL_ENDPOINT="${OTEL_ENDPOINT:-http://127.0.0.1:4317}" \
OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-morphogen-rpc-adapter}" \
OTEL_ENV="${OTEL_ENV:-e2e}" \
OTEL_VERSION="${OTEL_VERSION:-local}" \
"${ROOT_DIR}/scripts/test_rpc_e2e.sh"

echo "[datadog] done"
