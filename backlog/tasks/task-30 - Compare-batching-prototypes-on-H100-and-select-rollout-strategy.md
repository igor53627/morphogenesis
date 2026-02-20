---
id: TASK-30
title: Compare batching prototypes on H100 and select rollout strategy
status: Done
assignee: []
created_date: '2026-02-19 12:05'
updated_date: '2026-02-20 20:50'
labels:
  - performance
  - cuda
  - batching
  - research
dependencies:
  - TASK-26
  - TASK-27
  - TASK-28
  - TASK-29
documentation:
  - docs/plans/2026-02-19-task-24-gpu-page-batch-endpoint.md
  - docs/plans/2026-02-20-task-30-batching-rollout-strategy.md
  - docs/benchmarks/2026-02-20-task-30-page-gpu-batch-h100-matrix.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a controlled benchmark matrix across batching prototypes and decide production direction.

Compare baseline, adaptive micro-batch, multi-stream dispatch, CUDA Graph replay, and tiled kernel on H100 with identical datasets and query mixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decision report includes latency, throughput, and correctness parity for Q=1,2,4,8,16,32
- [x] #2 Recommendation includes go/no-go and fallback strategy
- [x] #3 Selected strategy has explicit rollout guard (feature flag or config)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-20: Started TASK-30. Running controlled H100 benchmark matrix across baseline, adaptive micro-batch, multi-stream, CUDA graph, and tiled kernel with identical dataset/query mix before writing rollout recommendation.

2026-02-20: Completed controlled H100 comparison across five strategies (baseline full-batch, adaptive micro-batch, multi-stream-4, CUDA graph replay, tiled tile=4) on identical real-volume dataset/query mix (`Q=1,2,4,8,16,32`, iterations=2, warmup=1).

Artifact: `docs/benchmarks/2026-02-20-task-30-page-gpu-batch-h100-matrix.csv` (Modal app `ap-uWcW0hl1gC1o2kC3Yb0blq`). Correctness parity held for all rows (`checksum_match=1`).

Decision report: `docs/plans/2026-02-20-task-30-batching-rollout-strategy.md`. Recommendation is GO for `multi_stream_4` rollout (`MORPHOGEN_GPU_STREAMS=4` + latency policy), NO-GO for full-batch/cuda-graph/tiled as standalone rollout, with adaptive/throughput policy as explicit fallback.
<!-- SECTION:NOTES:END -->
