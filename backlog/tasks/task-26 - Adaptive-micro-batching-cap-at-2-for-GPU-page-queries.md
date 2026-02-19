---
id: TASK-26
title: Adaptive micro-batching cap at 2 for GPU page queries
status: Done
assignee: []
created_date: '2026-02-19 12:04'
updated_date: '2026-02-19 12:13'
labels:
  - performance
  - cuda
  - batching
  - research
dependencies: []
documentation:
  - docs/plans/2026-02-19-task-24-gpu-page-batch-endpoint.md
  - docs/plans/2026-02-19-task-26-adaptive-micro-batching.md
  - docs/benchmarks/2026-02-19-task-26-page-gpu-batch-h100.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement adaptive micro-batching for GPU page queries where effective batch size is capped at 2, splitting larger request groups into Q=2 chunks.

Goal: preserve the only currently positive region observed on H100 while avoiding large-batch regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Batch scheduler enforces max effective micro-batch of 2 on GPU path
- [x] #2 Correctness parity holds vs existing single-query loop (checksum or byte-equality)
- [x] #3 H100 benchmark includes Q=2,4,8 and reports speedup vs baseline
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-19: Started implementation. Following TDD: add failing coverage for micro-batch cap at 2, then implement GPU-path scheduling split.

2026-02-19: Implemented GPU adaptive micro-batching (cap=2) for /query/page/gpu/batch by splitting GPU launches into range chunks of size <=2 while preserving output order.
Added unit tests for planner behavior and coverage in crates/morphogen-server/src/network/api.rs.
Validation: cargo test --package morphogen-server --features network (pass).
H100 benchmark artifact: docs/benchmarks/2026-02-19-task-26-page-gpu-batch-h100.csv (mode=cuda).
Compared to task-24 H100 baseline (docs/benchmarks/2026-02-19-task-24-page-gpu-batch-h100.csv):
- Q=2 speedup 1.16x -> 1.15x (effectively neutral)
- Q=4 speedup 0.31x -> 1.15x
- Q=8 speedup 0.18x -> 1.16x
This removes the prior large-batch regression for tested sizes.
<!-- SECTION:NOTES:END -->
