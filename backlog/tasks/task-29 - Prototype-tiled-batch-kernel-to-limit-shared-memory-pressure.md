---
id: TASK-29
title: Prototype tiled batch kernel to limit shared-memory pressure
status: To Do
assignee: []
created_date: '2026-02-19 12:04'
updated_date: '2026-02-20 11:31'
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
- [ ] #1 Tile size is explicit and configurable
- [ ] #2 Kernel/output correctness matches baseline for all tested Q
- [ ] #3 H100 benchmark covers Q=4,8,16,32 and reports speedup/throughput
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-20: Prioritize this task for mainnet-scale optimization. TASK-25 real-volume H100 benchmark (docs/benchmarks/2026-02-19-task-25-page-gpu-batch-h100-mainnet-volume.csv) shows only ~1.03-1.04x batch speedup at Q=2/4/8, indicating kernel-level work likely needed beyond runtime-path optimizations.
<!-- SECTION:NOTES:END -->
