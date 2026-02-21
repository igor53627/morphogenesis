---
id: TASK-24
title: Add GPU page batch endpoint and wire batched CUDA scan path
status: Done
assignee: []
created_date: '2026-02-19 05:19'
updated_date: '2026-02-19 09:09'
labels:
  - performance
  - cuda
  - server
  - network
dependencies:
  - TASK-22
documentation:
  - docs/morphogenesis_protocol.md
  - docs/plans/2026-02-19-task-24-gpu-page-batch-endpoint.md
  - docs/benchmarks/2026-02-19-task-24-page-gpu-batch-cpu-fallback.csv
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose and integrate a batched GPU page query path so multiple page queries can be executed in one CUDA batch launch path.

Scope:
- Add API endpoint (or equivalent transport path) for batched GPU page queries.
- Use GpuScanner::scan_batch* APIs instead of repeated single-query scan calls.
- Preserve epoch and delta consistency semantics with clear fallback behavior.
- Follow TDD: add failing API and parity tests before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Batched GPU endpoint validates request size/shape and returns deterministic ordering
- [x] #2 Implementation uses scan_batch (optimized variant when available) and preserves correctness vs repeated single-query results
- [x] #3 Benchmark output captures latency/throughput for batch sizes and compares against single-query loop
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented /query/page/gpu/batch with request validation, deterministic ordering, CUDA batched scan path, and CPU fallback.

CUDA path uses GpuScanner::scan_batch_optimized plus epoch consistency checks and per-query delta merge.

Added integration tests for ordering and validation plus bench_page_gpu_batch benchmark artifact for batch vs single loop comparison.

2026-02-19: Validated benchmark on Modal H100 via `bench_page_gpu_batch` with `--features network,cuda`.
Artifact: docs/benchmarks/2026-02-19-task-24-page-gpu-batch-h100.csv.
Run metadata: gpu_info=NVIDIA H100 80GB HBM3, 81559 MiB, mode=cuda, pages=65536, domain_bits=16, iterations=30, warmup=5.
Observed speedup: batch size 2 improved vs single-loop (~1.16x), but >=4 regressed; this likely reflects current kernel occupancy/shared-memory behavior for larger batches.
Also fixed CUDA-only compile regressions discovered during Modal run: moved-key borrow in `network/api.rs` and `Arc` ownership in `bench_page_gpu_batch.rs`.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Network API tests (including GPU fallback cases) pass
- [x] #2 Protocol/API docs updated for new batched GPU query path
<!-- DOD:END -->
