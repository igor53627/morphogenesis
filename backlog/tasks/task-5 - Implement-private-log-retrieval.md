---
id: TASK-5
title: Implement private log retrieval
status: Done
assignee: []
created_date: '2026-02-03 14:30'
updated_date: '2026-02-22 03:50'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make eth_getLogs private. Logs are fundamentally different from accounts/storage — they are
variable-size, range-queried, and unbounded. Our DPF-based PIR (point queries over uniform
cuckoo-hashed rows) doesn't fit.

### Approach: local cache + client-side filtering (no new PIR scheme)

Key insight: the block cache poller already fetches ALL recent blocks' receipts, which
contain logs. We can extract logs from cached receipts and filter locally — the upstream
never sees the address/topic filter. This gives full privacy for recent blocks (last 64)
with zero additional infrastructure.

For historical logs, fall through to upstream (same pattern as tx/receipt cache).

### Why not a new PIR scheme?

Logs violate every assumption of our PIR:
- Not uniform size (variable-length ABI-encoded data)
- Not point-addressable (eth_getLogs is a range/filter query)
- Not pre-indexable (can't cuckoo-hash a filter predicate)
- Unbounded growth (new logs every block)

Alternative schemes considered:
- **Bloom filter PIR matrix**: Requires a separate matrix, two-phase query, false positives
- **Keyword PIR**: Academic, different server infrastructure needed
- **CAS/IPFS/DHT**: Querying by content hash from CDN is private only if requests blend
  into the anonymity set (everyone fetches same data). For selective historical queries,
  still leaks which blocks you care about.

The local cache approach is simpler, requires no new infrastructure, and covers the common
case (wallets query recent logs, not historical).

### Subtasks

- **TASK-5.1**: Private eth_getLogs via local receipt cache (recent blocks)
- **TASK-5.2** (future): Private historical log queries if needed (would require new scheme)
<!-- SECTION:DESCRIPTION:END -->
