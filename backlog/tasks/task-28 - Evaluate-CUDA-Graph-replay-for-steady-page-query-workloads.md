---
id: TASK-28
title: Evaluate CUDA Graph replay for steady page query workloads
status: Done
assignee: []
created_date: '2026-02-19 12:04'
updated_date: '2026-02-20 12:28'
labels:
  - performance
  - cuda
  - batching
  - research
dependencies: []
documentation:
  - docs/plans/2026-02-19-task-24-gpu-page-batch-endpoint.md
  - docs/benchmarks/2026-02-20-task-28-page-gpu-batch-h100-cuda-graph.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a CUDA Graph prototype for repeated page-query execution paths and compare launch overhead against non-graph stream dispatch.

Goal: reduce CPU-side launch overhead for stable request shapes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Graph capture/replay can be toggled by feature flag or runtime config
- [x] #2 Results match non-graph execution for identical inputs
- [x] #3 H100 benchmark quantifies launch-overhead impact
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-20: Started implementation after TASK-32 closure. Next step: add graph toggle plumbing + correctness tests before benchmarking.

2026-02-20: Added runtime CUDA graph toggle `MORPHOGEN_GPU_CUDA_GRAPH` (plus bench flag `--gpu-cuda-graph`) and wired GPU batch handler to pass graph mode into scanner dispatch paths.

Implemented CUDA graph prototype in `morphogen-gpu-dpf::GpuScanner` for eligible single-launch optimized batch shapes (`1|2|4|8|16` queries), with signature-based cache and replay. For unsupported capture/launch cases, code falls back to direct kernel launch to avoid correctness regressions.

H100 benchmark on real mainnet volume (`/data/mainnet_compact.bin`) with throughput policy, stream=1, concurrency=1, batch sizes 2/4/8, iterations=5, warmup=1 is recorded in `docs/benchmarks/2026-02-20-task-28-page-gpu-batch-h100-cuda-graph.csv`.

Result: graph mode is functionally correct (checksum parity between single and batch paths remains intact) but shows no material latency/QPS gain vs non-graph in this workload; keep default `MORPHOGEN_GPU_CUDA_GRAPH=false` and treat graph path as optional/experimental for now.

Measured graph-mode deltas vs non-graph on this run: batch latency +0.25% (q=2), +0.57% (q=4), +0.13% (q=8); batch QPS deltas were within -0.06%. This indicates no launch-overhead win at current request shape.
<!-- SECTION:NOTES:END -->
