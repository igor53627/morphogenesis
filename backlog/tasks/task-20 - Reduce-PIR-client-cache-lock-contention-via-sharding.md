---
id: TASK-20
title: Reduce PIR client cache lock contention via sharding
status: Done
assignee: []
created_date: '2026-02-18 19:52'
updated_date: '2026-02-21 13:03'
labels:
  - performance
  - rpc
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PirClient currently uses a single async Mutex over the whole cache, which can serialize concurrent query paths under load.

Scope:
- Replace single-lock cache design with sharded locking or equivalent low-contention approach.
- Preserve epoch-aware cache invalidation semantics and eviction behavior.
- Validate correctness under concurrent single and batch query flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PirClient cache no longer relies on a single global Mutex for all cache operations
- [x] #2 Epoch-change invalidation and eviction semantics are preserved
- [x] #3 Concurrency-focused tests added/updated for single and batch query paths
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-21: Replaced PirClient cache single global tokio::Mutex<PirCache> with an internally sharded cache (32 shards by default) using per-shard mutexes and atomic epoch tracking. This removes the single lock bottleneck on cache hit/miss paths in both execute_pir_query_with_retry (single) and execute_batch_pir_query_with_retry (batch).

Epoch semantics are preserved via shard-epoch synchronization: on epoch transition, touched shards clear stale entries before serving/inserting, and stale older-epoch inserts are ignored. Eviction semantics remain clear-on-pressure, now at shard scope (capacity_per_shard = ceil(total_capacity / shard_count)).

Validation: cargo test --package morphogen-client (47 passed). Added concurrency-focused tests for both single and batch flows: single_query_uses_cache_concurrently and batch_query_uses_cache_concurrently, plus direct sharded cache stress test sharded_cache_supports_concurrent_put_get.

Observed impact: correctness holds under concurrent cache access without serialized global lock sections; cache operations now contend only within the computed shard for each key.

2026-02-21 benchmark (local microbenchmark, ignored test):
Command: cargo test --package morphogen-client benchmark_cache_contention_sharded_vs_single_mutex -- --ignored --nocapture
Read-heavy (256,000 ops, 64 workers): single-mutex 522.30 ms (490k ops/s) vs sharded 61.66 ms (4.15M ops/s), 8.47x throughput improvement.
Mixed (87.5% get / 12.5% put, 256,000 ops, 64 workers): single-mutex 530.20 ms (483k ops/s) vs sharded 70.76 ms (3.62M ops/s), 7.49x throughput improvement.

Scope note: this is an in-process cache microbenchmark (no network I/O); it isolates cache lock contention impact only.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 morphogen-client tests pass
- [x] #2 Task notes include contention rationale and measured/observed impact
<!-- DOD:END -->
