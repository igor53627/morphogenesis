---
id: TASK-27
title: Prototype multi-stream single-query dispatch for GPU page PIR
status: Done
assignee: []
created_date: '2026-02-19 12:04'
updated_date: '2026-02-19 16:17'
labels:
  - performance
  - cuda
  - batching
  - research
dependencies: []
documentation:
  - docs/plans/2026-02-19-task-27-multi-stream-single-query-dispatch.md
  - docs/benchmarks/2026-02-19-task-27-page-gpu-streams-h100.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prototype host-side dispatch that runs multiple single-query GPU scans concurrently using CUDA streams (no large in-kernel batch).

Goal: improve throughput by overlapping kernels/memory activity without shared-memory pressure from large batches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configurable stream count (at least 1,2,4,8)
- [x] #2 No correctness regression vs baseline responses
- [x] #3 H100 benchmark compares throughput/latency against baseline and micro-batching
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-02-19: Started TASK-27. Plan: add test-first coverage for configurable stream dispatch, then implement CUDA multi-stream single-query prototype and benchmark on H100.

2026-02-19: Implemented configurable multi-stream single-query CUDA dispatch for /query/page/gpu/batch.
Config: MORPHOGEN_GPU_STREAMS (clamped to 1..=8).
TDD: added stream-count parser tests and benchmark parser test before implementation.
Validation: cargo test --package morphogen-server --features network (pass).
H100 benchmark matrix artifact: docs/benchmarks/2026-02-19-task-27-page-gpu-streams-h100.csv.
Observed speedups (batch_vs_single_speedup):
- streams=1: Q2=1.16x, Q4=1.16x, Q8=1.16x
- streams=2: Q2=1.58x, Q4=1.70x, Q8=1.75x
- streams=4: Q2=1.57x, Q4=2.67x, Q8=2.85x
- streams=8: Q2=1.58x, Q4=2.56x, Q8=2.69x
On tested workload, stream=4 was best overall for Q>=4.
Correctness: checksum_single == checksum_batch on all benchmark rows.

2026-02-19: Addressed RoboRev findings post-commit adf33cd7. Fixes: (1) CUDA multistream kernel selection now uses a single `using_optimized_batch1` guard for both kernel function and shared-memory mask sizing; (2) /query/page/gpu/batch now enforces strict GPU result count invariants (including multistream path) and adds unit tests for mismatch handling; (3) bench_page_gpu_batch state initialization no longer mutates immutable slices via pointer cast and now uses safe `ChunkedMatrix::write_row` filling.

2026-02-19: Added branch-boundary invariant helper validated_gpu_results_with_keys and tests for matching/mismatched lengths to prevent zip-truncation regressions in page_query_gpu_batch_handler.
<!-- SECTION:NOTES:END -->
