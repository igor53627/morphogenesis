# Production Ops Bundle

This directory provides canonical deployment artifacts and runbooks for:

- `morphogen-server` (two instances: A/B)
- `morphogen-rpc-adapter`
- supporting dependency for code artifacts (`code-data` static file service)

## Artifacts

- Compose orchestration: `ops/compose/docker-compose.prod.yml`
- Local code-data startup-order override: `ops/compose/docker-compose.prod.local-code-data.yml`
- Server container image spec: `ops/docker/Dockerfile.server`
- Adapter container image spec: `ops/docker/Dockerfile.rpc-adapter`
- Environment templates:
  - `ops/env/morphogen-prod.env.example`
  - `ops/env/morphogen-server-a.env.example`
  - `ops/env/morphogen-server-b.env.example`
  - `ops/env/morphogen-rpc-adapter.env.example`
- Runbooks:
  - `ops/runbooks/deploy.md`
  - `ops/runbooks/rollback.md`
  - `ops/runbooks/incident-response.md`
  - `ops/runbooks/maintenance.md`

## Startup Ordering

Recommended startup ordering:

1. Ensure code artifacts are reachable (`code-data` for dictionary/CAS or external equivalents).
2. Start `morphogen-server-a` and `morphogen-server-b`.
3. Verify both server instances are healthy (`GET /health` and `GET /epoch`).
4. Start `morphogen-rpc-adapter` and verify JSON-RPC readiness.
5. Cut client traffic to adapter only after all checks pass.

The base compose file encodes startup ordering for server dependencies with `depends_on` health conditions. For local dictionary/CAS mode, use `ops/compose/docker-compose.prod.local-code-data.yml` with `--profile local-code-data` to gate adapter startup on `code-data` health.

## Environment Variables and Secret Handling

### Server (`morphogen-server`)

Required for prod:

- Server A:
  - `MORPHOGEN_SERVER_A_ENV=prod`
  - `MORPHOGEN_SERVER_A_BIND_ADDR=0.0.0.0:3000`
  - `MORPHOGEN_SERVER_A_MATRIX_FILE=/var/lib/morphogen/matrix.bin`
  - `MORPHOGEN_SERVER_A_PAGE_PRG_KEY_0` and `MORPHOGEN_SERVER_A_PAGE_PRG_KEY_1` (non-zero 16-byte hex each)
- Server B:
  - `MORPHOGEN_SERVER_B_ENV=prod`
  - `MORPHOGEN_SERVER_B_BIND_ADDR=0.0.0.0:3000`
  - `MORPHOGEN_SERVER_B_MATRIX_FILE=/var/lib/morphogen/matrix.bin`
  - `MORPHOGEN_SERVER_B_PAGE_PRG_KEY_0` and `MORPHOGEN_SERVER_B_PAGE_PRG_KEY_1` (non-zero 16-byte hex each; distinct from A)

Optional tuning:

- `MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS` / `MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS`
- `MORPHOGEN_SERVER_A_MERGE_INTERVAL_MS` / `MORPHOGEN_SERVER_B_MERGE_INTERVAL_MS`

### Adapter (`morphogen-rpc-adapter`)

Required:

- `MORPHOGEN_SERVER_IMAGE` (immutable tag/digest)
- `MORPHOGEN_RPC_ADAPTER_IMAGE` (immutable tag/digest)
- `UPSTREAM_RPC_URL`
- `DICT_URL`
- `CAS_URL`

Optional:

- `RPC_BIND_HOST` (default `127.0.0.1`; set explicitly if publishing beyond loopback)
- OpenTelemetry flags/endpoint values

Secret handling guidance:

- Do not commit real secrets in `ops/env/*.env`.
- Keep committed files as `*.env.example` only.
- Inject real secrets from your platform secret manager (Kubernetes Secret, Vault, SSM, etc.) at deploy time.
- Rotate PRG keys and admin/API secrets on a fixed cadence and on incident response triggers.

## Health Checks

Server health:

- `GET /health` should return `{"status":"ok", ...}`
- `GET /epoch` should return current epoch metadata

Adapter health:

- JSON-RPC POST on `/`:
  - Method: `web3_clientVersion`
  - Expect a JSON-RPC `result` field

Metrics:

- Server exports `GET /metrics` when built with metrics support.

## Release Hygiene

- Keep image tags immutable per release.
- Prefer digest-pinned image refs in `MORPHOGEN_SERVER_IMAGE` / `MORPHOGEN_RPC_ADAPTER_IMAGE`.
- Store compose/env changes in change control.
- Validate runbooks after every operational change.
