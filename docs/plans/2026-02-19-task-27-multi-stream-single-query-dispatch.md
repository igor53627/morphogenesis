# TASK-27: Multi-Stream Single-Query GPU Dispatch

## Summary
Implemented host-side multi-stream single-query dispatch for the CUDA path of:
- `POST /query/page/gpu/batch`

Design:
- keep each kernel launch at single-query shape (`batch_size=1`),
- distribute query launches across multiple CUDA streams,
- preserve response ordering,
- keep existing micro-batch behavior for single-stream mode.

Configuration:
- `MORPHOGEN_GPU_STREAMS` env var controls stream count.
- parser clamps to `1..=8`.
- stream count `1` uses existing path; `>1` enables multi-stream single-query dispatch.

## Code Changes

### Server API routing/selection
- Added stream-count parsing and runtime selection in GPU batch handler.
- New parser helpers:
  - `parse_gpu_stream_count()`
  - `configured_gpu_stream_count()`
- If `MORPHOGEN_GPU_STREAMS > 1`, server calls new scanner multi-stream path.

Code:
- `crates/morphogen-server/src/network/api.rs`

### GPU scanner implementation
- Added:
  - `GpuScanner::scan_batch_single_query_multistream_optimized(...)`
- Behavior:
  - pre-allocates per-query key/output buffers,
  - forks non-default streams via `fork_default_stream()`,
  - launches kernels with `launch_on_stream(...)`,
  - synchronizes back to default stream, then reclaims outputs in-order.

Code:
- `crates/morphogen-gpu-dpf/src/kernel.rs`

### Benchmark harness
- Added benchmark arg:
  - `--gpu-streams <N>`
- Benchmark sets `MORPHOGEN_GPU_STREAMS` for in-process API calls.

Code:
- `crates/morphogen-server/src/bin/bench_page_gpu_batch.rs`

## TDD / Tests

Added parser tests first (red -> green):
- `parse_gpu_stream_count_supports_required_values`
- `parse_gpu_stream_count_defaults_and_clamps`

Also added benchmark config parse test:
- `parse_config_reads_gpu_streams`

Validation:
- `cargo test --package morphogen-server --features network`

## H100 Benchmark

Artifact:
- `docs/benchmarks/2026-02-19-task-27-page-gpu-streams-h100.csv`

Setup:
- GPU: NVIDIA H100 80GB
- `pages=65536`, `domain_bits=16`, `iterations=30`, `warmup=5`
- batch sizes tested: `Q=2,4,8`
- stream counts tested: `1,2,4,8`

Observed speedup (`batch_vs_single_speedup`):

| Streams | Q=2 | Q=4 | Q=8 |
|---------|-----|-----|-----|
| 1 | 1.16x | 1.16x | 1.16x |
| 2 | 1.58x | 1.70x | 1.75x |
| 4 | 1.57x | 2.67x | 2.85x |
| 8 | 1.58x | 2.56x | 2.69x |

Comparison vs TASK-26 (`streams=1` micro-batching):
- stream=1 matches prior behavior (~1.15-1.16x at Q=2,4,8).
- stream>=2 substantially improves throughput; stream=4 is best on tested points.

## Correctness

No correctness regressions observed in benchmark output:
- for every row, `checksum_single == checksum_batch`.
- response order is preserved by in-order result assembly.
