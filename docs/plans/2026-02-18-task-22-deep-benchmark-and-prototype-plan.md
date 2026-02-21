# TASK-22: Deep Benchmark and Prototype Plan for True Scan-Level Batching

## Goal
Decide whether to prioritize true server-side scan batching (single matrix pass for multiple PIR queries) beyond current request-level `/query/batch`.

## Benchmark Method

### Harness
- Added in-process benchmark binary: `crates/morphogen-server/src/bin/bench_batch_query.rs`
- Measures `batch_query_handler` directly (avoids HTTP/network noise while preserving API-layer logic and scan path).

### Commands

```bash
cargo run --package morphogen-server --release --features network --bin bench_batch_query -- \
  --rows 262144 --iterations 30 --warmup-iterations 3 --batch-sizes 1,2,4,8,16,32 --with-delta
```

```bash
cargo run --package morphogen-server --release --bin bench_scan -- \
  --rows 262144 --iterations 30 --warmup-iterations 3 --scan-only
```

### Raw Outputs
- `docs/benchmarks/2026-02-18-task-22-batch-query-bench-iter30.csv`
- `docs/benchmarks/2026-02-18-task-22-batch-query-bench.csv`
- `docs/benchmarks/2026-02-18-task-22-bench-scan-iter30.txt`

## Benchmark Results (`rows=262144`, `row_size=256`, release build)

| Q | ms/batch | ms/query | p95 batch ms | queries/sec | effective GB/s |
|---|---------:|---------:|-------------:|------------:|---------------:|
| 1  | 5.23  | 5.23 | 5.36   | 191.10 | 11.94 |
| 2  | 10.50 | 5.25 | 10.61  | 190.53 | 11.91 |
| 4  | 21.09 | 5.27 | 21.29  | 189.65 | 11.85 |
| 8  | 42.13 | 5.27 | 42.59  | 189.89 | 11.87 |
| 16 | 84.09 | 5.26 | 84.50  | 190.27 | 11.89 |
| 32 | 168.47 | 5.26 | 169.22 | 189.94 | 11.87 |

`bench_scan` corroboration for single-query scan:
- `scan_ms=158` over 30 iterations => ~`5.27 ms/query`, `11.86 GB/s`

## Interpretation

1. `ms/query` is flat (~5.23-5.27 ms) as Q increases.
2. `ms/batch` is almost perfectly linear in Q.
3. `queries/sec` and effective scan bandwidth are nearly constant.

Conclusion: current `/query/batch` performs per-query full matrix scans and does not reuse matrix passes across batched queries.

This matches code:
- `batch_query_handler` loops queries and calls `scan_main_matrix` per query:
  - `crates/morphogen-server/src/network/api.rs:408`
  - `crates/morphogen-server/src/network/api.rs:410`
- compatibility `batch_size` parameter in scan API is ignored:
  - `crates/morphogen-server/src/scan.rs:395`
  - `crates/morphogen-server/src/scan.rs:406`

## Prototype Design (Concrete Path)

### 1) New fused scan primitive (portable first)
Add a new function in `scan.rs`:
- `scan_main_matrix_multi_portable<K: DpfKey>(matrix: &ChunkedMatrix, all_keys: &[[K;3]], row_size: usize) -> Vec<[Vec<u8>;3]>`

Behavior:
- Single pass over matrix rows.
- For each row, evaluate all query masks (`Q*3` evals).
- XOR row bytes into per-query accumulators.

### 2) API integration
In `batch_query_handler` (`crates/morphogen-server/src/network/api.rs`):
- Keep current epoch snapshot logic.
- Replace per-query scan loop with fused call when `Q >= 2` and feature flag/runtime guard enabled.
- Keep current path as fallback.

### 3) Delta application parity
Current delta logic loops `entries x queries x 3`; keep same semantics, but apply into fused results buffer.

### 4) Parallel extension (phase 2)
After portable correctness:
- Add `scan_main_matrix_parallel_multi` modeled on `scan_main_matrix_parallel` fold/reduce:
  - `crates/morphogen-server/src/scan.rs:505`

## Complexity and Tradeoffs

Current per batch:
- Matrix reads: `Q * matrix_bytes`
- Key evals: `Q * 3 * rows`
- Accumulator memory: `Q * 3 * row_size`

Fused scan expected:
- Matrix reads: `1 * matrix_bytes` (major reduction)
- Key evals: still `Q * 3 * rows` (same order)
- Accumulator memory: same order, but hotter working set if laid out contiguously

Risk areas:
- More key-eval compute per row in a tight loop can become CPU-bound at higher Q.
- AVX-512 path complexity rises significantly; start portable to control risk.
- Correctness risk around epoch consistency + delta parity needs explicit equivalence tests.

## Go / No-Go Criteria

Go if all pass:
1. Throughput gain >= 1.5x at `Q=8` versus current `/query/batch` loop.
2. No `Q=1` regression larger than 5% in p95 latency.
3. Bit-exact parity with existing results (including pending delta cases).
4. Maintainable implementation (portable path + tests) before AVX-512 specialization.

No-Go if any:
1. Gain < 20% for `Q>=4`.
2. Q=1 regression >5%.
3. Frequent complexity/correctness issues during implementation.

## Recommendation

Proceed with a guarded prototype.

Evidence from this benchmark is strong: request batching currently saves round trips only; server scan work scales linearly with Q. A fused scan experiment is likely worth it, but should be staged:
1. Portable fused path + parity tests.
2. Benchmark gate against the criteria above.
3. Only then decide on defaulting/rolling out.
