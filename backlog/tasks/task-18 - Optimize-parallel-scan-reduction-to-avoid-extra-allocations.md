---
id: TASK-18
title: Optimize parallel scan reduction to avoid extra allocations
status: Done
assignee: []
created_date: '2026-02-18 19:51'
updated_date: '2026-02-18 20:18'
labels:
  - performance
  - server
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce hot-path allocation and memory traffic in parallel matrix scan by replacing collect-then-reduce with thread-local fold/reduce accumulation.

Scope:
- Update scan_main_matrix_parallel in crates/morphogen-server/src/scan.rs to avoid building Vec<[Vec<u8>;3]> for all chunks.
- Use Rayon fold/reduce (or equivalent) so each worker accumulates directly into local result buffers, then combines once.
- Preserve current behavior and output correctness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parallel scan no longer collects per-chunk results into Vec<[Vec<u8>;3]>
- [x] #2 Behavioral parity proven by existing tests plus at least one new targeted test for result equivalence
- [x] #3 Benchmark/profiling note added comparing before/after (latency and/or allocation impact)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented fold/reduce accumulation in scan_main_matrix_parallel to remove per-chunk Vec<[Vec<u8>;3]> collection.
Added regression test scan::tests::parallel_scan_matches_portable_across_chunks (feature=parallel) comparing parallel output against portable path across many chunks with tail handling.

Benchmark (dev profile, same command before/after):
  cargo run -p morphogen-server --features parallel --bin bench_scan -- --rows 262144 --iterations 30 --warmup-iterations 3 --scan-only --parallel
  Before: scan_ms=3218, gb_per_sec=0.58
  After:  scan_ms=3240, gb_per_sec=0.58
Observed latency/throughput difference is within noise; structural allocation reduction landed as intended.

2026-02-19: Addressed roborev job 776 low finding by forcing parallel equivalence test to run inside a dedicated Rayon thread pool (num_threads=2), ensuring fold/reduce combiner path coverage in CI.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Code compiles and tests pass for morphogen-server with parallel feature
- [x] #2 Change is documented in task notes with measured impact
<!-- DOD:END -->
