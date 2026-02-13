---
id: TASK-17.3
title: Add deterministic private-path RPC E2E assertions
status: Done
assignee: []
created_date: '2026-02-11 20:42'
updated_date: '2026-02-11 23:12'
labels:
  - rpc
  - testing
  - integration
  - e2e
dependencies:
  - TASK-17.1
  - TASK-17.2
parent_task_id: TASK-17
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement deterministic E2E assertions for private-path methods:
- `eth_getBalance`
- `eth_getTransactionCount`
- `eth_getStorageAt`
- `eth_getCode`
- `eth_call`
- `eth_estimateGas`
- `eth_getLogs` and filter lifecycle over cached fixture blocks

Focus on deterministic expected values from subset fixtures, not upstream-live responses.

Acceptance:
- one command runs the suite end-to-end locally
- failures print method, input, expected, and actual values
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded deterministic E2E assertions in `scripts/test_rpc_e2e.sh` to cover:
- `eth_call`
- `eth_estimateGas`
- `eth_getLogs` on cached deterministic block
- `eth_newFilter` + `eth_getFilterLogs` + `eth_getFilterChanges`
- `eth_newBlockFilter` + mined-block change detection

Added `mock_upstream` binary (`crates/morphogen-rpc-adapter/src/bin/mock_upstream.rs`) to provide deterministic upstream block/receipt/log data and `evm_mine` for block-filter progression.
<!-- SECTION:FINAL_SUMMARY:END -->
