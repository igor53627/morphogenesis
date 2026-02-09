---
id: TASK-5.1
title: Private eth_getLogs via local receipt cache
status: Done
assignee: []
created_date: '2026-02-09 08:00'
labels: []
dependencies:
  - TASK-15
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make eth_getLogs private for recent blocks by extracting logs from the locally-cached
receipts and filtering client-side. No new PIR scheme needed — the block cache poller
already fetches all recent receipts (which contain logs). Historical log queries fall
through to upstream (acceptable degradation, same as tx/receipt cache pattern).

### Privacy model

- **Recent blocks (last 64)**: Fully private. The poller fetches ALL blocks and receipts
  indistinguishably from any syncing node. Logs are extracted from cached receipts and
  filtered locally. Upstream never sees the address/topic filter.
- **Historical blocks**: Falls through to upstream (privacy leak, same as today). Users
  querying historical logs accept this tradeoff.
- **Anonymity set**: The poller's traffic pattern (fetch every block's receipts) is
  identical to any light client or block explorer. No selective fetching.

### Design

**Block cache changes (`block_cache.rs`):**

1. Add `logs: HashMap<u64, Vec<Value>>` to BlockCache — maps block_number to that block's
   log entries (extracted from receipts at insert time)
2. In `insert_block()`, extract logs from each receipt's `logs` array and store per-block
3. Eviction: remove logs for evicted blocks (same as tx/receipt eviction)
4. New method: `get_logs(&self, filter) -> Vec<Value>` that:
   - Iterates cached blocks in the filter's `fromBlock..=toBlock` range
   - For each block's logs, applies the filter:
     - `address` match (single address or list)
     - `topics[0..3]` match (with null = wildcard, per EIP-1474)
   - Returns matching logs

**Handler changes (`main.rs`):**

5. Remove `"eth_getLogs"` from `PASSTHROUGH_METHODS`
6. New handler parses the filter object:
   - `fromBlock` / `toBlock` (hex or "latest") — resolve to block numbers
   - `address` (single or array)
   - `topics` (array of [topic | null | [topic, ...]])
7. Check if the entire requested range is within the cache window:
   - **Yes**: Filter locally, return results (fully private)
   - **No**: Proxy to upstream (log with privacy warning)
   - **Partial overlap**: Proxy entire range to upstream (simpler than merging, avoids
     correctness issues with partial results)

**Filter matching logic:**

```
fn log_matches_filter(log: &Value, filter: &LogFilter) -> bool {
    // Address check
    if let Some(addrs) = &filter.addresses {
        let log_addr = log["address"].as_str();
        if !addrs.iter().any(|a| a == log_addr) { return false; }
    }
    // Topic checks (position-sensitive, null = wildcard)
    for (i, topic_filter) in filter.topics.iter().enumerate() {
        if topic_filter.is_null() { continue; } // wildcard
        let log_topic = log["topics"][i].as_str();
        match topic_filter {
            Single(t) => if log_topic != Some(t) { return false; },
            Array(ts) => if !ts.iter().any(|t| log_topic == Some(t)) { return false; },
        }
    }
    true
}
```

### Bandwidth estimate

- ~200 txs/block x ~2-5 logs/tx x ~256 bytes/log = ~150-250 KB/block
- 64 cached blocks = ~10-16 MB total log cache
- Already fetching receipts (which contain logs) — zero additional RPC calls

### What this does NOT cover

- Private historical log queries (requires separate PIR scheme — TASK-5.2 if needed)
- Filter APIs (eth_newFilter, eth_getFilterChanges) — these could be layered on top
  by running the same local filter on each new block, but left for a separate task
- `blockHash` filter parameter (single block by hash — easy to add but omitted for scope)

### Test plan

- Unit tests for log filter matching (address, topics, wildcards, arrays)
- Unit tests for get_logs range queries against cached blocks
- Update `test_wallet_compat.sh` to verify eth_getLogs returns results for recent blocks
- Verify passthrough still works for historical block ranges
<!-- SECTION:DESCRIPTION:END -->
