# Deploy Runbook

## Scope

Deploy `morphogen-server` (A/B), `morphogen-rpc-adapter`, and optional local `code-data` using `ops/compose/docker-compose.prod.yml`.

## Preconditions

1. Operator has production access and change window approval.
2. Real env files are created from `ops/env/*.env.example` in a secure path (not committed) and loaded through `--env-file`.
3. Matrix snapshots for server A/B are present at:
   - `ops/compose/state/server-a/matrix.bin`
   - `ops/compose/state/server-b/matrix.bin`
4. Dictionary/CAS input is available through one of:
   - External endpoints via `DICT_URL` and `CAS_URL`
   - Local files under `ops/compose/data/code/` with compose profile `local-code-data`
5. Runtime volume directories are writable by container uid `10001`:
   - `ops/compose/state/server-a/`
   - `ops/compose/state/server-b/`

## Deploy Steps

1. Set required env values (in shell exports or `/secure/path/morphogen-prod.env`):
   - `MORPHOGEN_SERVER_IMAGE` (immutable tag/digest)
   - `MORPHOGEN_RPC_ADAPTER_IMAGE` (immutable tag/digest)
   - `UPSTREAM_RPC_URL`
   - `DICT_URL`
   - `CAS_URL`
   - `MORPHOGEN_SERVER_A_PAGE_PRG_KEY_0`
   - `MORPHOGEN_SERVER_A_PAGE_PRG_KEY_1`
   - `MORPHOGEN_SERVER_B_PAGE_PRG_KEY_0`
   - `MORPHOGEN_SERVER_B_PAGE_PRG_KEY_1`
   - Optional bind override: `RPC_BIND_HOST` (defaults to `127.0.0.1`)
   - For local dictionary/CAS mode set:
     - `DICT_URL=http://code-data/mainnet_compact.dict`
     - `CAS_URL=http://code-data/cas`
2. Pull and start services:
   - External dictionary/CAS mode:
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml pull`
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml up -d --no-build`
   - Local dictionary/CAS mode:
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml -f ops/compose/docker-compose.prod.local-code-data.yml --profile local-code-data pull`
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml -f ops/compose/docker-compose.prod.local-code-data.yml --profile local-code-data up -d --no-build`
3. Verify service health:
   - External dictionary/CAS mode:
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml exec -T morphogen-server-a curl -fsS http://127.0.0.1:3000/health`
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml exec -T morphogen-server-b curl -fsS http://127.0.0.1:3000/health`
   - Local dictionary/CAS mode:
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml -f ops/compose/docker-compose.prod.local-code-data.yml --profile local-code-data exec -T morphogen-server-a curl -fsS http://127.0.0.1:3000/health`
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml -f ops/compose/docker-compose.prod.local-code-data.yml --profile local-code-data exec -T morphogen-server-b curl -fsS http://127.0.0.1:3000/health`
   - `curl -fsS -H 'content-type: application/json' --data '{"jsonrpc":"2.0","id":1,"method":"web3_clientVersion","params":[]}' http://127.0.0.1:8545`
4. Verify privacy-path readiness:
   - `curl -fsS -H 'content-type: application/json' --data '{"jsonrpc":"2.0","id":1,"method":"eth_getBalance","params":["0x0000000000000000000000000000000000000000","latest"]}' http://127.0.0.1:8545`
5. Shift external traffic to adapter endpoint after checks are green.

## Post-Deploy Validation

1. Review adapter logs for fail-closed/fallback warnings.
2. Confirm server `/epoch` values are stable and non-empty.
3. Confirm monitoring receives health/latency/error telemetry.
