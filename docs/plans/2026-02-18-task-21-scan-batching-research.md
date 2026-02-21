# TASK-21 Research: Server-Side Multi-Query Scan Batching ROI

## Question
Should we implement true scan-level batching in `morphogen-server` (single matrix pass serving multiple PIR queries), instead of current per-query scan execution inside `/query/batch`?

## Current State

### 1) Network-level batching exists
- Client batches misses up to 32 (`MAX_BATCH_SIZE`) and calls `/query/batch`.
- Server exposes `/query/batch` and validates `1..=32` queries.

References:
- `crates/morphogen-client/src/network.rs` (`MAX_BATCH_SIZE`, `execute_batch_pir_query_with_retry`)
- `crates/morphogen-server/src/network/api.rs:112`
- `crates/morphogen-server/src/network/api.rs:339`

### 2) Scan-level batching does not exist
Inside `batch_query_handler`, server still executes scan per query:
- loops queries (`for keys in &all_keys`)
- calls `scan_main_matrix(...)` per query
- applies delta per query

References:
- `crates/morphogen-server/src/network/api.rs:408`
- `crates/morphogen-server/src/network/api.rs:410`
- `crates/morphogen-server/src/network/api.rs:412`

### 3) Internal dead batching path was removed in TASK-19
- Old `batch_size` plumbing in scan API is now compatibility-only and ignored.
- Preferred path is no-batch scan API.

References:
- `crates/morphogen-server/src/scan.rs:395`
- `crates/morphogen-server/src/scan.rs:412`

## Baseline Measurement

Command:

```bash
cargo run -p morphogen-server --features parallel --bin bench_scan -- \
  --rows 262144 --iterations 20 --warmup-iterations 3 --scan-only --parallel
```

Observed:
- `scan_ms=2148` for 20 iterations
- ~`107 ms/query`
- ~`0.58 GB/s`

Interpretation:
- Current CPU scan path is linear per query; request-level batching saves network round trips but not matrix passes.

## Batching Design Options

### Option A: True fused multi-query scan (recommended to prototype)
Implement `scan_main_matrix_parallel_multi`:
- input: `&[[K; 3]]`
- single pass over matrix chunk
- evaluate masks for all queries per row
- update all query accumulators in one pass

Expected upside:
- fewer matrix rereads when batch size > 1
- improved throughput for `/query/batch`

Main risks:
- CPU compute grows with query count (`Q * 3` key evals per row)
- accumulator memory footprint (`Q * 3 * row_size_bytes`)
- implementation complexity in AVX-512 path

### Option B: Keep current scan path (status quo)
- no complexity increase
- predictable behavior
- no scan throughput gain from batching

### Option C: Hybrid small-batch fusion
- only fuse up to small `Q` (e.g., 4 or 8), fallback to per-query loop for larger batches
- lower risk than full dynamic batching
- still captures most practical wallet burst benefits

## Recommended Research Implementation
1. Build CPU portable prototype first (`scan_main_matrix_parallel_multi_portable`).
2. Wire only `/query/batch` to prototype behind feature flag (or runtime guard).
3. Keep current path as fallback.
4. Add correctness tests:
   - fused result == N independent `scan_main_matrix` results
   - epoch consistency and delta application parity
5. Benchmark by batch size `Q = 1, 2, 4, 8, 16`.

## Go / No-Go Criteria
Go if all are true on local/CI benchmark fixture:
- Throughput gain >= 1.5x at `Q=8` vs current per-query loop
- No p95 latency regression for `Q=1`
- No correctness mismatches vs existing path
- Code complexity remains maintainable (portable path first)

No-Go if:
- Gain < 20% at `Q>=4`, or
- `Q=1` regresses materially, or
- complexity/risk too high for near-term priorities.

## Recommendation
Proceed with a prototype (Option A/Hybrid) as an experiment task, not immediate production switch. Current architecture leaves meaningful throughput on the table for batched requests, but we need data to confirm memory-vs-compute tradeoffs.
