---
id: TASK-22
title: Deep benchmark and prototype plan for true scan-level batching
status: Done
assignee: []
created_date: '2026-02-18 21:49'
updated_date: '2026-02-18 22:06'
labels:
  - performance
  - research
  - server
dependencies: []
documentation:
  - docs/plans/2026-02-18-task-22-deep-benchmark-and-prototype-plan.md
  - docs/benchmarks/2026-02-18-task-22-batch-query-bench-iter30.csv
  - docs/benchmarks/2026-02-18-task-22-bench-scan-iter30.txt
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up research for server-side scan-level batching (single matrix pass for multiple PIR queries) to decide implementation priority.

Scope:
- Benchmark current /query/batch behavior across batch sizes Q={1,2,4,8,16,32}.
- Prototype-level design and complexity analysis for fused multi-query scan path.
- Capture expected CPU/memory tradeoffs and integration points in scan and API layers.
- Produce clear go/no-go recommendation with success thresholds and rollout strategy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Benchmark table recorded for Q={1,2,4,8,16,32} with comparable setup
- [x] #2 Design note documents at least one concrete fused-scan implementation path and risks
- [x] #3 Recommendation includes explicit go/no-go thresholds and next action
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented dedicated in-process benchmark binary: crates/morphogen-server/src/bin/bench_batch_query.rs

Release benchmark (rows=262144, iter=30, warmup=3, with_delta): ms/query stayed flat at ~5.23-5.27 from Q=1..32; ms/batch scaled linearly with Q; throughput ~190 queries/sec constant.

Conclusion: /query/batch currently provides request batching only and does not fuse matrix scans. Recommendation: prototype guarded fused multi-query scan (portable first) and gate rollout on >=1.5x throughput gain at Q=8 with <=5% Q=1 p95 regression.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog notes link to research doc and benchmark outputs
<!-- DOD:END -->
