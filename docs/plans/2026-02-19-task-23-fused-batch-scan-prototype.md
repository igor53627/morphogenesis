# TASK-23: Fused Multi-Query Row Scan Prototype (`/query/batch`)

## Summary
Implemented a guarded prototype of true scan-level batching for row PIR:
- New fused matrix primitive in `scan.rs`:
  - `scan_main_matrix_multi(...)`
- `/query/batch` now has a selectable fused path behind compile-time feature flag:
  - `--features fused-batch-scan` enables fused scan for multi-query batches
  - default builds keep fused path disabled to preserve stable behavior

## Correctness

Added parity and consistency tests:
- `scan::tests::multi_scan_matches_independent_scans`
- `scan::tests::multi_scan_empty_queries_returns_empty_results`
- `network::api::tests::batch_snapshot_scan_fused_matches_legacy_with_pending_delta`
- `network::api::tests::batch_query_consistency_fallback_returns_service_unavailable_on_persistent_mismatch`

Result: fused snapshot path matches legacy outputs (including pending delta), and consistency fallback behavior is unchanged.

## Benchmark Method

Command (legacy/default):

```bash
cargo run --package morphogen-server --release --features network \
  --bin bench_batch_query -- \
  --rows 262144 --iterations 30 --warmup-iterations 3 --batch-sizes 1,2,4,8,16,32 --with-delta
```

Command (fused prototype):

```bash
cargo run --package morphogen-server --release --features network,fused-batch-scan \
  --bin bench_batch_query -- \
  --rows 262144 --iterations 30 --warmup-iterations 3 --batch-sizes 1,2,4,8,16,32 --with-delta
```

Raw outputs:
- `docs/benchmarks/2026-02-19-task-23-batch-query-legacy-iter30.csv`
- `docs/benchmarks/2026-02-19-task-23-batch-query-fused-iter30.csv`

## Results

| Q | Legacy ms/query | Fused ms/query | Relative |
|---|----------------:|---------------:|---------:|
| 1  | 5.26 | 5.28 | 0.99x |
| 2  | 5.28 | 7.31 | 0.72x |
| 4  | 5.29 | 7.17 | 0.74x |
| 8  | 5.28 | 7.05 | 0.75x |
| 16 | 5.28 | 7.10 | 0.74x |
| 32 | 5.29 | 7.05 | 0.75x |

At `Q=8`:
- Legacy throughput: ~`189.26 qps`
- Fused prototype throughput: ~`141.78 qps`
- Throughput ratio: ~`0.75x` (regression)

## Go/No-Go Decision

No-Go for immediate rollout of this prototype implementation.

Reason:
- Fails target criterion (`>=1.5x` throughput gain at `Q=8`)
- Shows regression for `Q>=2`

Mitigation applied:
- Keep fused path behind compile-time feature flag; legacy per-query scan remains default execution path.

## Next Optimization Directions

1. Reduce fused-path key-eval overhead by using block/range mask generation where possible.
2. Improve accumulator/layout strategy to reduce per-row per-query write amplification.
3. Revisit fused-path vectorization and parallel decomposition before re-evaluating rollout.
