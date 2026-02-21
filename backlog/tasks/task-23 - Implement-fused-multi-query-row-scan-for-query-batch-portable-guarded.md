---
id: TASK-23
title: Implement fused multi-query row scan for /query/batch (portable + guarded)
status: Done
assignee: []
created_date: '2026-02-19 05:19'
updated_date: '2026-02-19 08:36'
labels:
  - performance
  - server
  - batching
dependencies:
  - TASK-22
documentation:
  - docs/plans/2026-02-19-task-23-fused-batch-scan-prototype.md
  - docs/benchmarks/2026-02-19-task-23-batch-query-legacy-iter30.csv
  - docs/benchmarks/2026-02-19-task-23-batch-query-fused-iter30.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement true scan-level batching for row PIR so /query/batch can serve multiple queries in a single matrix pass.

Scope:
- Add a portable fused scan primitive in scan.rs that accepts N queries (each with 3 keys) and accumulates all outputs in one matrix traversal.
- Integrate into batch_query_handler behind a runtime/feature guard with fallback to current per-query scan loop.
- Preserve epoch-consistency and delta-application semantics.
- Follow TDD: add failing parity tests before implementation, then benchmark and refine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parity tests cover fused vs per-query outputs (including pending delta and consistency path)
- [x] #2 batch_query_handler can select fused path with safe fallback to existing loop
- [x] #3 Benchmark evidence recorded for Q={1,2,4,8,16,32} and go/no-go decision documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented fused multi-query row scan primitive: scan_main_matrix_multi in scan.rs and integrated /query/batch snapshot path via guarded selector.

Guard: MORPHOGEN_FUSED_BATCH_SCAN (default disabled for safety); legacy path remains fallback/default.

Added parity/consistency tests for fused-vs-legacy snapshot behavior and persistent epoch mismatch fallback.

Benchmark comparison (rows=262144, iter=30): fused prototype regressed throughput (~0.75x at Q=8 vs legacy), so rollout decision is No-Go for now. Retained as guarded experimental path only.

2026-02-19 follow-up: fused scan is now compile-time gated behind cargo feature 'fused-batch-scan' (default off) instead of env toggle.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Server tests pass with network/parallel feature combinations
- [x] #2 Task notes link benchmark artifacts and final decision
<!-- DOD:END -->
