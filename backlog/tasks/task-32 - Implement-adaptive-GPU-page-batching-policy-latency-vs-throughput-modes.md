---
id: TASK-32
title: Implement adaptive GPU page batching policy (latency vs throughput modes)
status: Done
assignee: []
created_date: '2026-02-20 11:30'
updated_date: '2026-02-20 11:48'
labels:
  - performance
  - cuda
  - batching
  - server
dependencies: []
documentation:
  - docs/benchmarks/2026-02-19-task-25-page-gpu-batch-h100-mainnet-volume.csv
  - docs/benchmarks/2026-02-20-task-32-page-gpu-batch-h100-mainnet-policies.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current micro-batching is static. Real mainnet-volume benchmark shows only modest gains, suggesting batching policy should adapt to runtime load and latency targets.

Introduce an adaptive policy that can select effective batching strategy (for example batch=1 vs batch=2) based on queue depth / concurrency / latency SLO and expose config/metrics for tuning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Batching policy can switch behavior based on runtime load signals without correctness regressions
- [x] #2 Policy is configurable (safe defaults + explicit override) and observable via metrics/logging
- [x] #3 H100 benchmark on real volume compares latency/QPS across policy modes and includes recommendation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-20: Started implementation. Plan: add test coverage for adaptive policy selection and then wire policy-based dispatch into GPU batch path with safe defaults.

2026-02-20: Completed adaptive GPU batch policy wiring in `network/api.rs` with policy modes `adaptive|throughput|latency`, threshold env `MORPHOGEN_GPU_BATCH_ADAPTIVE_THRESHOLD` (default 4, clamped 1..=32), and dispatch metric `gpu_batch_dispatch_mode_total{mode=...}` documented in `docs/OBSERVABILITY.md`.

H100 real-volume policy benchmark (Modal, matrix `/data/mainnet_compact.bin`, batch sizes 2/4/8, gpu_streams=1, concurrency=1, iterations=20, warmup=3, adaptive_threshold=4) saved at `docs/benchmarks/2026-02-20-task-32-page-gpu-batch-h100-mainnet-policies.csv`.

Recommendation: use `throughput` policy for now on current mainnet-volume shape; `adaptive` at threshold 4 regresses at Q=4 because it switches to full-batch there, and `latency` regresses heavily for Q>=4.
<!-- SECTION:NOTES:END -->
