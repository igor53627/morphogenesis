# Rollback Runbook

## Rollback Triggers

1. New release fails health checks.
2. Error rate or latency regression breaches SLO.
3. Data/epoch mismatch after deploy.
4. Privacy policy behavior changes unexpectedly.

## Rollback Strategy

Use immutable image tags and keep last-known-good compose/env bundle available.

## Rollback Steps

1. Freeze new rollouts and announce rollback start in ops channel.
2. Restore previous release artifact versions:
   - previous immutable image refs for server/adapter (`MORPHOGEN_SERVER_IMAGE`, `MORPHOGEN_RPC_ADAPTER_IMAGE`)
   - previous env bundle (keys/endpoints/settings)
3. Restart previous version:
   - External dictionary/CAS mode:
     - Ensure `DICT_URL` and `CAS_URL` are set to previous external endpoints.
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml pull`
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml up -d --no-build`
   - Local dictionary/CAS mode:
      - `export DICT_URL=http://code-data/mainnet_compact.dict`
      - `export CAS_URL=http://code-data/cas`
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml -f ops/compose/docker-compose.prod.local-code-data.yml --profile local-code-data pull`
     - `docker compose --env-file /secure/path/morphogen-prod.env -f ops/compose/docker-compose.prod.yml -f ops/compose/docker-compose.prod.local-code-data.yml --profile local-code-data up -d --no-build`
4. Re-run health checks:
   - server `GET /health`
   - adapter `web3_clientVersion` JSON-RPC
5. Confirm error rates and user traffic recover to baseline.

## Rollback Verification

1. Check adapter logs for degraded fallback warnings.
2. Check server `/epoch` metadata consistency.
3. Validate at least one private read request through adapter.

## Exit Criteria

Rollback is complete when health checks, baseline metrics, and private-path checks are stable for the agreed observation window.
