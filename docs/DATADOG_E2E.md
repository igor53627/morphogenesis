# Datadog APM Baseline for RPC E2E

This runbook configures a minimal Datadog setup for local/temporary E2E runs on real hardware.

## What is covered

- Datadog Agent via Docker Compose (`ops/datadog/docker-compose.yml`)
- OTLP trace export from `morphogen-rpc-adapter`
- OTLP trace export from `morphogen-e2e-client` (instrumented E2E caller)
- Existing E2E script with optional telemetry mode (`scripts/test_rpc_e2e.sh`)
- One-command wrapper that starts Agent + runs E2E (`scripts/test_rpc_e2e_datadog.sh`)

## 1) Export Datadog credentials

```bash
export DD_API_KEY="<your_api_key>"
export DD_SITE="datadoghq.com"      # or datadoghq.eu
export DD_ENV="e2e"
export DD_HOSTNAME="morphogenesis-e2e"  # optional, defaults to this value
```

## 2) Run E2E with Datadog tracing

```bash
scripts/test_rpc_e2e_datadog.sh
```

Optional overrides:

```bash
OTEL_ENDPOINT=http://127.0.0.1:4317 \
OTEL_SERVICE_NAME=morphogen-rpc-adapter \
OTEL_CLIENT_SERVICE_NAME=morphogen-e2e-client \
OTEL_ENV=e2e \
OTEL_VERSION=local \
scripts/test_rpc_e2e_datadog.sh
```

Keep Agent running after script exit:

```bash
KEEP_DATADOG_AGENT=1 scripts/test_rpc_e2e_datadog.sh
```

## 3) Validate intake locally

```bash
curl -sS http://127.0.0.1:8126/info | jq '.version,.endpoints[]? | select(.|type=="string")'
```

## 4) Validate in Datadog UI

- APM -> Services should show both:
  - `morphogen-e2e-client`
  - `morphogen-rpc-adapter`
- Service map should include dependency edge:
  - `morphogen-e2e-client -> morphogen-rpc-adapter`
- Adapter service should have traces for RPC handlers such as:
  - `eth_getBalance`
  - `eth_getTransactionCount`
  - `eth_getCode`

## Adapter flags

`morphogen-rpc-adapter` now supports:

- `--otel-traces`
- `--otel-endpoint`
- `--otel-service-name`
- `--otel-env`
- `--otel-version`

Use directly (without wrapper) if your Agent is already running:

```bash
cargo run -p morphogen-rpc-adapter -- \
  --otel-traces \
  --otel-endpoint http://127.0.0.1:4317 \
  --otel-service-name morphogen-rpc-adapter \
  --otel-env e2e \
  --otel-version local
```
