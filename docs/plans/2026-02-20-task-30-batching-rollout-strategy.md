# TASK-30: H100 Batching Prototype Comparison and Rollout Decision

## Summary

Ran a controlled H100 benchmark matrix across five GPU batching strategies with identical inputs:

- dataset: `/data/mainnet_compact.bin` (real mainnet volume)
- query mix: `Q=1,2,4,8,16,32`
- benchmark settings: `iterations=2`, `warmup=1`, `concurrency=1`
- binary: `bench_page_gpu_batch` (`--features network,cuda`)

Raw artifact:

- `docs/benchmarks/2026-02-20-task-30-page-gpu-batch-h100-matrix.csv`

Run metadata:

- Modal app: `ap-uWcW0hl1gC1o2kC3Yb0blq`
- GPU: `NVIDIA H100 80GB HBM3`

## Strategies Compared

1. `baseline_full_batch`: `policy=latency`, `streams=1`, `cuda_graph=false`, `tile=16`
2. `adaptive_micro_batch`: `policy=adaptive`, `threshold=4`, `streams=1`, `cuda_graph=false`, `tile=16`
3. `multi_stream_4`: `policy=latency`, `streams=4`, `cuda_graph=false`, `tile=16`
4. `cuda_graph_replay`: `policy=latency`, `streams=1`, `cuda_graph=true`, `tile=16`
5. `tiled_tile4`: `policy=latency`, `streams=1`, `cuda_graph=false`, `tile=4`

## Results

### Batch-vs-single speedup (`batch_vs_single_speedup`)

| Q | baseline | adaptive | multi-stream-4 | cuda-graph | tiled-4 |
|---|---:|---:|---:|---:|---:|
| 1  | 1.00x | 1.00x | 1.00x | 1.00x | 1.01x |
| 2  | 1.02x | 1.02x | 1.01x | 1.02x | 1.02x |
| 4  | 0.22x | 0.23x | 1.02x | 0.22x | 0.22x |
| 8  | 0.08x | 1.02x | 1.02x | 0.08x | 0.22x |
| 16 | 0.06x | 1.02x | 1.02x | 0.06x | 0.22x |
| 32 | 0.06x | 1.02x | 1.02x | 0.06x | 0.22x |

### Batch latency (`batch_ms_per_batch`)

| Q | baseline | adaptive | multi-stream-4 | cuda-graph | tiled-4 |
|---|---:|---:|---:|---:|---:|
| 1  | 30.14 | 30.23 | 30.11 | 30.12 | 30.24 |
| 2  | 59.53 | 59.65 | 59.76 | 59.46 | 59.76 |
| 4  | 540.10 | 539.84 | 119.06 | 543.10 | 539.06 |
| 8  | 2965.05 | 238.18 | 237.46 | 2966.01 | 1080.83 |
| 16 | 7927.78 | 475.88 | 474.65 | 7932.62 | 2166.32 |
| 32 | 15877.60 | 951.59 | 948.21 | 15863.46 | 4310.66 |

### Batch throughput (`batch_qps`)

| Q | baseline | adaptive | multi-stream-4 | cuda-graph | tiled-4 |
|---|---:|---:|---:|---:|---:|
| 1  | 33.17 | 33.08 | 33.21 | 33.20 | 33.07 |
| 2  | 33.60 | 33.53 | 33.47 | 33.64 | 33.47 |
| 4  | 7.41  | 7.41  | 33.60 | 7.37  | 7.42  |
| 8  | 2.70  | 33.59 | 33.69 | 2.70  | 7.40  |
| 16 | 2.02  | 33.62 | 33.71 | 2.02  | 7.39  |
| 32 | 2.02  | 33.63 | 33.75 | 2.02  | 7.42  |

### Correctness parity

- `checksum_match=1` for all 30 rows (all strategies and all Q values).
- No checksum mismatches observed.

## Decision

### Recommendation

- **Go** with `multi_stream_4` as the production batching direction.
  - It is the only strategy that avoids high-Q collapse while also fixing the Q=4 regression.
  - It is consistently ~`1.02x` batch-vs-single across `Q=4..32`.

### No-go paths

- **No-go** for `baseline_full_batch` and `cuda_graph_replay` at current kernel shapes (`Q>=4` severe regression).
- **No-go** for `tiled_tile4` as standalone rollout strategy (improves over full-batch collapse but remains materially below single-loop throughput).

### Fallback strategy

- Fallback to `adaptive_micro_batch` (single-stream adaptive policy) if multi-stream causes operational issues.
- For an immediate conservative fallback with guaranteed micro-batching behavior, use `MORPHOGEN_GPU_BATCH_POLICY=throughput`.

## Rollout Guard (Explicit Config)

Selected strategy is guarded entirely by runtime config:

- `MORPHOGEN_GPU_STREAMS=4`
- `MORPHOGEN_GPU_BATCH_POLICY=latency`
- `MORPHOGEN_GPU_CUDA_GRAPH=false`
- `MORPHOGEN_GPU_BATCH_TILE_SIZE=16`

Fallback toggles:

- `MORPHOGEN_GPU_STREAMS=1`
- `MORPHOGEN_GPU_BATCH_POLICY=adaptive` (or `throughput` for strict micro-batch mode)
