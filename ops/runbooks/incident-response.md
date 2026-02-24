# Incident Response Runbook

## Incident Classification

Use severity based on user impact and privacy risk:

1. `SEV-1`: complete outage, security/privacy breach, or prolonged data unavailability.
2. `SEV-2`: major performance degradation or partial outage.
3. `SEV-3`: localized issue with workaround.

## Immediate Actions

1. Open incident channel and assign incident commander.
2. Capture timestamps, affected endpoints, and current release version.
3. Run quick health triage:
   - server A/B `GET /health`
   - adapter `web3_clientVersion`
   - adapter private read smoke call (`eth_getBalance`)
4. Decide: mitigate in place vs rollback.

## Common Failure Modes

### Server unhealthy

1. Check container logs for startup/config errors.
2. Validate matrix file mount and permissions.
3. Validate page PRG keys are present/non-zero in prod.
4. Restart affected server instance and re-check `/health` and `/epoch`.

### Adapter unhealthy

1. Validate upstream connectivity and DNS.
2. Validate PIR server A/B reachability from adapter network.
3. Check for fail-closed behavior (expected in prod by default).
4. If incident requires temporary degraded mode, use approved change path and explicit privacy acknowledgement flags.

### Elevated error rate / latency

1. Check CPU/memory saturation and request concurrency.
2. Check upstream timeout behavior and dependency latency.
3. Reduce traffic or enable controlled throttling while investigating.

## Communications

1. Update incident channel every 15 minutes (or agreed cadence).
2. Record customer-facing status updates.
3. Track decisions, mitigations, and timeline for postmortem.

## Closure

1. Confirm health and metrics stabilize.
2. Remove temporary mitigations.
3. Open postmortem with root cause and action items.
