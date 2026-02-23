---
id: TASK-36
title: Close remaining JSON-RPC compatibility gaps in adapter
status: Done
assignee: []
created_date: '2026-02-22 10:42'
updated_date: '2026-02-23 05:59'
labels:
  - rpc-adapter
  - compatibility
  - production
dependencies: []
references:
  - crates/morphogen-rpc-adapter/src/evm.rs
  - crates/morphogen-rpc-adapter/src/block_cache.rs
  - crates/morphogen-rpc-adapter/src/main.rs
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A few method-parameter semantics are intentionally unsupported today and can break provider/wallet parity in production environments.

Close or explicitly gate these gaps with predictable behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Support EIP-1898 block object form (or provide feature-gated behavior with explicit, documented fallback strategy)
- [x] #2 Support safe/finalized block tags in log filtering paths
- [x] #3 Handle state-overrides for estimate/access-list with deterministic behavior and privacy policy integration
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started TASK-36 implementation. Scoping remaining JSON-RPC compatibility gaps and preparing TDD updates in morphogen-rpc-adapter.

Implemented TDD updates and compatibility fixes in morphogen-rpc-adapter: (1) EIP-1898 block object form now supported in local EVM block spec parsing, including blockNumber and blockHash with requireCanonical handling; (2) safe/finalized tags are accepted in log filter parsing paths and resolve deterministically to the current local latest cached block; (3) state-overrides handling for eth_estimateGas/eth_createAccessList is now deterministic and privacy-policy integrated via strict override-shape validation and fail-closed gating before any upstream proxy. Added/updated unit tests in evm.rs, block_cache.rs, and main.rs; cargo test/clippy/fmt all pass for morphogen-rpc-adapter.

Post-completion hardening pass: addressed roborev findings with follow-up commits for finality-tag semantics, stale-cache normalization, and additional handler-path tests. Opened PR #14: https://github.com/igor53627/morphogenesis/pull/14
<!-- SECTION:NOTES:END -->
