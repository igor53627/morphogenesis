# TASK-31: Per-Workspace-Pool Prototype for Concurrent GPU Page PIR

## Summary

Implemented a bounded pooled workspace strategy in `GpuScanner` to remove the single global
`batch_workspace` mutex hot path.

Key change:
- Replace one global `Mutex<Option<BatchWorkspace>>` with a workspace pool:
  - `batch_workspace_pool: Vec<Mutex<Option<BatchWorkspace>>>`
  - round-robin slot selection via `next_workspace_slot: AtomicUsize`
  - configurable pool size via `MORPHOGEN_GPU_WORKSPACE_POOL_SIZE` (default `4`, max `8`)

This preserves bounded memory behavior (lazy, per-slot allocation with capped slot count) and
keeps correctness safeguards unchanged.

## Validation

Local (non-CUDA) tests added and passed:
- `workspace_pool_size_defaults_and_clamps`
- `workspace_pool_slot_wraps_with_modulo`

Command:
- `cargo test --package morphogen-gpu-dpf workspace_pool`

## H100 Benchmark

Artifact:
- `docs/benchmarks/2026-02-21-task-31-page-gpu-workspace-pool-h100.csv`

Environment:
- Modal app: `ap-IOkvgjXKnjz1hA5UyzhMN6`
- GPU: `NVIDIA H100 80GB HBM3`
- Matrix: `/data/mainnet_compact.bin`
- Query mix: `Q=8`
- Iterations: `5`, warmup: `1`
- `MORPHOGEN_GPU_STREAMS=1`, `MORPHOGEN_GPU_BATCH_POLICY=latency`
- Compared:
  - baseline: `MORPHOGEN_GPU_WORKSPACE_POOL_SIZE=1`
  - pooled: `MORPHOGEN_GPU_WORKSPACE_POOL_SIZE=4`
- Concurrency points: `1`, `8`, `16`

Results (batch path):

| Variant | Concurrency | batch_ms_per_batch | batch_qps |
|---|---:|---:|---:|
| baseline (pool=1) | 1 | 2963.38 | 2.70 |
| pooled (pool=4) | 1 | 2965.19 | 2.70 |
| baseline (pool=1) | 8 | 23720.01 | 2.70 |
| pooled (pool=4) | 8 | 23713.15 | 2.70 |
| baseline (pool=1) | 16 | 47428.25 | 2.70 |
| pooled (pool=4) | 16 | 47442.71 | 2.70 |

Correctness:
- `checksum_match=1` for all rows.

## Interpretation

The pooled workspace prototype successfully removes the single global buffer lock from the code
path, but on this workload there is no measurable throughput/latency improvement. This indicates
kernel/stream execution serialization dominates runtime at `streams=1` and `Q=8` for mainnet-sized
matrix scans, so lock removal alone is not the limiting factor.

Potential follow-up:
- Pair workspace pooling with per-slot stream execution (not only per-slot buffers) to reduce
  default-stream serialization under concurrent request load.
