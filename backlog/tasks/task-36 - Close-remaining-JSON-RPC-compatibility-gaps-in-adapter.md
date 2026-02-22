---
id: TASK-36
title: Close remaining JSON-RPC compatibility gaps in adapter
status: To Do
assignee: []
created_date: '2026-02-22 10:42'
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
- [ ] #1 Support EIP-1898 block object form (or provide feature-gated behavior with explicit, documented fallback strategy)
- [ ] #2 Support safe/finalized block tags in log filtering paths
- [ ] #3 Handle state-overrides for estimate/access-list with deterministic behavior and privacy policy integration
<!-- AC:END -->
