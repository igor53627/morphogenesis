---
id: TASK-17
title: "EPIC: Deterministic E2E test suite with subset data"
status: Done
assignee: []
created_date: '2026-02-11 20:38'
updated_date: '2026-02-11 23:22'
labels:
  - rpc
  - testing
  - integration
  - e2e
  - epic
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build a reliable end-to-end test suite for the RPC adapter using a deterministic subset dataset instead of full mainnet state.

### Why subset data

Subset data is the right approach for E2E:
- fast enough for CI and local iteration
- deterministic outputs (no chain drift, no flaky assertions)
- lower infra cost and simpler debugging

Use live-mainnet checks only as optional smoke tests, not as the primary gating signal.

### Scope

1. Create a reproducible fixture dataset (accounts, storage slots, code IDs/hashes/bytecode, and minimal block/receipt/log fixtures) for a small address set.
2. Stand up a full local E2E harness:
   - PIR Server A + PIR Server B
   - RPC adapter
   - dictionary/CAS fixture service
3. Add deterministic assertions covering critical RPC paths:
   - `eth_getBalance`
   - `eth_getTransactionCount`
   - `eth_getStorageAt`
   - `eth_getCode`
   - `eth_call`
   - `eth_estimateGas`
   - `eth_getLogs` / filter lifecycle over cached blocks
4. Integrate this deterministic E2E suite into CI.
5. Keep existing upstream-fallback wallet tests as non-gating smoke checks.

### Subtasks

- TASK-17.1: Build subset fixtures for deterministic E2E.
- TASK-17.2: Harden E2E launcher and readiness checks.
- TASK-17.3: Add deterministic private-path RPC E2E assertions.
- TASK-17.4: Add CI job for deterministic E2E.
- TASK-17.5: Add nightly live-upstream smoke tests.

### Acceptance criteria

- Deterministic E2E suite runs locally with one command and passes consistently.
- CI runs deterministic E2E on every PR.
- Tests validate private-path behavior (not only `--fallback-to-upstream`).
- Failure output is actionable (clear method + expected vs actual).
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed deterministic E2E epic:
- subset fixtures checked in under `fixtures/e2e/`
- deterministic local E2E harness in `scripts/test_rpc_e2e.sh`
- deterministic private-path assertions including call/estimateGas/log/filter flows
- PR-gating deterministic E2E CI job in `.github/workflows/ci.yml`
- nightly live-upstream smoke workflow in `.github/workflows/nightly-smoke.yml`
<!-- SECTION:FINAL_SUMMARY:END -->
