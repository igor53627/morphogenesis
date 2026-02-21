---
id: TASK-29
title: Prototype tiled batch kernel to limit shared-memory pressure
status: Done
assignee: []
created_date: '2026-02-19 12:04'
updated_date: '2026-02-20 13:23'
labels:
  - performance
  - cuda
  - batching
  - research
dependencies: []
documentation:
  - docs/plans/2026-02-19-task-24-gpu-page-batch-endpoint.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement tiled query batching inside GPU execution (for example tile=2 or tile=4), processing larger external batches as multiple tiles and combining outputs.

Goal: retain batching benefits while preventing occupancy collapse at high Q.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tile size is explicit and configurable
- [x] #2 Kernel/output correctness matches baseline for all tested Q
- [x] #3 H100 benchmark covers Q=4,8,16,32 and reports speedup/throughput
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-20: Prioritize this task for mainnet-scale optimization. TASK-25 real-volume H100 benchmark (docs/benchmarks/2026-02-19-task-25-page-gpu-batch-h100-mainnet-volume.csv) shows only ~1.03-1.04x batch speedup at Q=2/4/8, indicating kernel-level work likely needed beyond runtime-path optimizations.

2026-02-20: Started after TASK-28/32 and review cleanup; proceeding with test-first tiled-batch kernel prototype.

2026-02-20: Implemented explicit tiled GPU batch launch cap via runtime env `MORPHOGEN_GPU_BATCH_TILE_SIZE` (server parser clamp 1..16), scanner API `scan_batch_optimized_tiled[_with_graph]`, and tiled launch planner in `morphogen-gpu-dpf::kernel` so larger external batches are executed as multiple internal tiles.

Added test coverage: server parse test for tile knob, bench CLI parse test (`--gpu-batch-tile-size`), tiled launch planner unit tests, and CUDA-gated correctness test comparing tiled vs baseline optimized outputs for Q=1,2,4,8,16,32.

Bench harness now sets and reports tile-size knob.

2026-02-20: Ran Modal H100 benchmark (app run `ap-arl9xv3nZMl51jMHIIXLop`) with real volume matrix `/data/mainnet_compact.bin`, policy=`latency` (full-batch dispatch), tile_size=4, iterations=2, warmup=1. Artifact: `docs/benchmarks/2026-02-20-task-29-page-gpu-batch-h100-tiled.csv`.

Results (batch_vs_single_speedup): Q=4 0.22x, Q=8 0.22x, Q=16 0.22x, Q=32 0.22x. Throughput dropped from ~32.95 QPS (single loop) to ~7.35-7.37 QPS (batch) under this tiled full-batch prototype.

Conclusion: explicit tiling knob and correctness wiring are complete, but tile_size=4 under full-batch latency policy regresses throughput materially on this workload; keep as configurable prototype and defer rollout decision to TASK-30 comparison matrix.
<!-- SECTION:NOTES:END -->
