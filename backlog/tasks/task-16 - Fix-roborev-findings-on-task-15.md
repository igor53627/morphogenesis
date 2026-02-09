---
id: TASK-16
title: Fix roborev findings on TASK-15 (estimateGas + tx cache)
status: In Progress
assignee: []
created_date: '2026-02-09 07:00'
labels: []
dependencies:
  - TASK-15
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address roborev code review findings on the TASK-15 commit (bdfd5f1).

### High severity
1. **eth_estimateGas rejects 3-param calls**: Some clients send state overrides as 3rd param.
   Accept up to 3 params (ignore overrides for now, or fallback to upstream if present).
2. **DoS via unbounded hex::decode**: tx hash handlers decode user input before checking length.
   Pre-check string length before decoding.

### Medium severity
3. **Pre-existing DoS in eth_getStorageAt**: Same unbounded decode issue for slot hex.
4. **Orphaned receipts on eviction**: Eviction only tracks tx hashes, not receipt hashes.
   Track all hashes per block for eviction.
5. **No reorg handling**: Block poller assumes monotonic block numbers. Detect and handle reorgs.

### Low severity
6. **basefee=0 affects BASEFEE opcode**: Acceptable per plan, matches Geth eth_call behavior.
7. **Duplicate tx hash parsing**: Share parse_tx_hash between main.rs and block_cache.rs.
<!-- SECTION:DESCRIPTION:END -->
