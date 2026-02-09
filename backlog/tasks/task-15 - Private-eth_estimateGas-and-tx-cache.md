---
id: TASK-15
title: Private eth_estimateGas and local transaction cache
status: To Do
assignee: []
created_date: '2026-02-09 06:45'
labels: []
dependencies:
  - TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close two privacy gaps identified from real wallet/dApp RPC call distribution analysis:

1. **Private eth_estimateGas**: Currently passthrough — leaks full tx calldata to upstream
   before the user even sends it. Fix: reuse local EVM execution (same as eth_call),
   extract gas_used + 20% safety margin.

2. **Local transaction cache**: eth_getTransactionByHash and eth_getTransactionReceipt are
   passthrough — leak which specific txs the user cares about (88 combined triggers across
   Aave, Helios, Rotki, Ambire, Zerion). Fix: background poller subscribes to new blocks
   (public, everyone does this), caches tx objects and receipts. Serve recent txs from cache
   privately; fall through to upstream for historical txs only.

### Implementation

**Part 1 — eth_estimateGas (evm.rs + main.rs):**
- Refactor: extract `run_evm()` helper from `execute_eth_call()` returning raw ExecutionResult
- New `execute_eth_estimate_gas()` calls run_evm, returns gas_used * 120 / 100
- Fix basefee=0 in block env for simulation (prevents GasPriceLessThanBasefee error)
- Remove eth_estimateGas from PASSTHROUGH_METHODS, add custom handler

**Part 2 — Transaction cache (block_cache.rs + main.rs):**
- New BlockCache struct: HashMap<B256, Value> for txs and receipts, VecDeque for eviction
- Background poller (tokio::spawn): polls eth_blockNumber every 2s, fetches full blocks +
  receipts via eth_getBlockReceipts, caches last 64 blocks
- Custom handlers for eth_getTransactionByHash and eth_getTransactionReceipt: check cache
  first, passthrough on miss
<!-- SECTION:DESCRIPTION:END -->
