# TASK-26: Adaptive GPU Micro-Batching (Cap at 2)

## Summary
Implemented adaptive micro-batching for `POST /query/page/gpu/batch` on the CUDA path:
- effective GPU launch batch size is capped at `2`,
- larger request batches are split into ordered micro-batches (`[0..2]`, `[2..4]`, ...),
- response order is preserved.

The CPU fallback path is unchanged.

## Code Changes

- Added micro-batch size constant and planner:
  - `GPU_MICRO_BATCH_SIZE = 2`
  - `gpu_micro_batch_ranges(total_queries) -> Vec<(start, end)>`
- Updated CUDA execution in `page_query_gpu_batch_handler` to:
  - iterate through micro-batch ranges,
  - call `scan_batch_optimized` per chunk,
  - append chunk results in request order.

Code:
- `crates/morphogen-server/src/network/api.rs`

## TDD Coverage

Added unit tests first (red->green) for planner behavior:
- `gpu_micro_batch_ranges_caps_each_batch_to_two`
- `gpu_micro_batch_ranges_preserves_order_and_full_coverage`

Code:
- `crates/morphogen-server/src/network/api.rs`

Validation:
- `cargo test --package morphogen-server --features network`

## H100 Benchmark

Task-26 artifact (CUDA mode):
- `docs/benchmarks/2026-02-19-task-26-page-gpu-batch-h100.csv`

Baseline artifact from TASK-24:
- `docs/benchmarks/2026-02-19-task-24-page-gpu-batch-h100.csv`

Same benchmark shape (`pages=65536`, `domain_bits=16`, `iterations=30`, `warmup=5`):

| Q | TASK-24 Speedup | TASK-26 Speedup | TASK-24 Batch ms | TASK-26 Batch ms |
|---|-----------------|-----------------|------------------|------------------|
| 2 | 1.16x | 1.15x | 0.87 | 0.88 |
| 4 | 0.31x | 1.15x | 6.44 | 1.76 |
| 8 | 0.18x | 1.16x | 22.72 | 3.51 |

## Result

Adaptive micro-batching removes the severe large-batch regression seen in TASK-24 for tested sizes and keeps small-batch behavior neutral-to-positive.
