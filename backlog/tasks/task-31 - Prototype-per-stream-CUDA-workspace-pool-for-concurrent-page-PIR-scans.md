---
id: TASK-31
title: Prototype per-stream CUDA workspace pool for concurrent page-PIR scans
status: Done
assignee: []
created_date: '2026-02-20 11:30'
updated_date: '2026-02-21 15:09'
labels:
  - performance
  - cuda
  - server
dependencies: []
documentation:
  - docs/plans/2026-02-21-task-31-workspace-pool-prototype.md
  - docs/benchmarks/2026-02-21-task-31-page-gpu-workspace-pool-h100.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current GPU scan path reuses a single shared batch workspace, which serializes concurrent requests and likely limits throughput under load.

Implement a per-stream (or pooled) CUDA workspace strategy so concurrent requests avoid lock contention on temporary device/host buffers while preserving correctness and bounded memory use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace allocation strategy supports concurrent GPU requests without a single global buffer lock hot path
- [x] #2 Correctness parity holds vs current path (matching checksums/results across concurrency tests)
- [x] #3 H100 benchmark reports throughput/latency at concurrency 1,8,16 and compares against current baseline
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-21: Implemented pooled GPU batch workspace in morphogen-gpu-dpf::GpuScanner to remove single global workspace lock. Added bounded workspace pool (`MORPHOGEN_GPU_WORKSPACE_POOL_SIZE`, default=4, max=8), round-robin slot selection, and per-slot lazy allocation/reuse.

TDD: added kernel unit tests for pool sizing and slot wrap behavior (`workspace_pool_size_defaults_and_clamps`, `workspace_pool_slot_wraps_with_modulo`), then implemented parser/selection + pool wiring.

H100 benchmark (Modal app ap-IOkvgjXKnjz1hA5UyzhMN6) completed at concurrency 1/8/16 comparing baseline pool=1 vs pooled pool=4 on `/data/mainnet_compact.bin` (Q=8, iterations=5, warmup=1, streams=1, policy=latency). Artifact: docs/benchmarks/2026-02-21-task-31-page-gpu-workspace-pool-h100.csv.

Observed: correctness parity held (checksum_match=1 for all rows), but throughput/latency stayed effectively unchanged (batch_qps ~2.70 across both variants). Interpretation: default-stream/kernel serialization dominates this workload; lock removal alone is not the primary bottleneck.
<!-- SECTION:NOTES:END -->
