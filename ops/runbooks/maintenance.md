# Maintenance Runbook

## Routine Maintenance Tasks

1. Rotate PRG keys and privileged credentials on schedule.
2. Refresh matrix snapshots and verify checksum/integrity before rollout.
3. Rebuild images with security patches (base image and Rust deps).
4. Validate compose/env drift against repository artifacts.
5. Verify backup and restore process for snapshot/state data.

## Patch Procedure

1. Build updated images in staging.
2. Run smoke checks for server and adapter.
3. Promote to production during approved maintenance window.
4. Observe error/latency metrics and health checks.

## Capacity and Performance Checks

1. Monitor scan concurrency and queueing pressure.
2. Tune `MORPHOGEN_SERVER_A_MAX_CONCURRENT_SCANS` and `MORPHOGEN_SERVER_B_MAX_CONCURRENT_SCANS` conservatively.
3. Legacy fallback remains available via `MORPHOGEN_SERVER_MAX_CONCURRENT_SCANS` if A/B-prefixed values are not set.
4. Validate adapter upstream timeout settings and retry pressure.

## Backup and Restore Drill

1. Take periodic snapshot backups for server A/B matrix data.
2. Perform restore drill at least quarterly:
   - restore snapshot to standby path
   - start service with restored data
   - verify `/health` and `/epoch`
3. Record drill duration and recovery observations.

## Maintenance Exit Criteria

1. All health checks green.
2. Private-path smoke tests pass.
3. No sustained regression in error rate or latency.
