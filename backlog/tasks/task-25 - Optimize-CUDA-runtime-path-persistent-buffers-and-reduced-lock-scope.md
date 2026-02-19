---
id: TASK-25
title: 'Optimize CUDA runtime path: persistent buffers and reduced lock scope'
status: Done
assignee: []
created_date: '2026-02-19 05:19'
updated_date: '2026-02-19 19:48'
labels:
  - performance
  - cuda
  - server
dependencies:
  - TASK-22
documentation:
  - docs/OBSERVABILITY.md
  - docs/plans/2026-02-18-task-22-deep-benchmark-and-prototype-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve GPU query runtime efficiency by reducing per-request memory overhead and lock contention in the server CUDA path.

Scope:
- Reduce/avoid per-request device allocation and host<->device staging overhead via reusable buffers or pooling strategy.
- Narrow lock scope around gpu_matrix access in page_query_gpu_handler to avoid blocking unrelated work.
- Add lightweight instrumentation to separate transfer, kernel, and merge phases.
- Follow TDD: add regression/perf-guard tests where feasible before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server GPU path avoids repeated allocation/copy hot spots for steady-state requests
- [x] #2 gpu_matrix lock hold time is reduced and correctness under epoch transitions is preserved
- [x] #3 Before/after benchmark or profiling artifacts show measurable latency/QPS improvement
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- 2026-02-19: Implemented server-side GPU lock-scope reduction by scanning under `gpu_matrix` lock and moving delta-merge/proof work outside lock in both `/query/page/gpu` and `/query/page/gpu/batch`.
- 2026-02-19: Added reusable CUDA batch workspace in `GpuScanner` (`out_accumulators`, `out_verifiers`, `key_buffer`, host buffers) to reduce per-request device allocation/copy churn in `scan_batch_opts`.
- 2026-02-19: Added phase instrumentation plumbing (`transfer_h2d`, `kernel`, `transfer_d2h`, `merge`) and updated docs metric table.
- 2026-02-19: Added tests for timing aggregation and lock-scope helper; full `cargo test --package morphogen-server --features network` passes.

- 2026-02-19: Modal H100 benchmark artifact captured at `docs/benchmarks/2026-02-19-task-25-page-gpu-batch-h100.csv` (mode=cuda, pages=65536, domain_bits=16, iterations=30, warmup=5, gpu_streams=1).
- 2026-02-19: Benchmark rerun exposed checksum mismatch after introducing persistent buffers; fixed by zeroing reused output/verifier regions via `CudaDevice::memset_zeros` before each scan request. Added checksum assertion in `bench_page_gpu_batch` to guard future regressions.
- 2026-02-19: Compared with TASK-26 H100 baseline (`docs/benchmarks/2026-02-19-task-26-page-gpu-batch-h100.csv`): batch latency/QPS is roughly flat (within ~1-2%), while batch-vs-single speedup increased because single-path latency rose in this run. Lock-scope benefit likely requires concurrent-load benchmark to measure directly.

- 2026-02-19: Added concurrent-load H100 benchmark matrix at `docs/benchmarks/2026-02-19-task-25-page-gpu-batch-h100-concurrency.csv` using `bench_page_gpu_batch --concurrency={1,8,16}` (Q=2,4,8). All runs had matching checksums (single == batch).

- 2026-02-19: Versus TASK-26 baseline at concurrency=1, batch latency improved modestly (~0.6-1.1%) and batch QPS improved (~0.8-1.5%) across Q=2,4,8. Under higher concurrency, aggregate batch QPS remained stable around ~2.16k-2.24k and stayed consistently above single-loop path by ~26-30%.

- 2026-02-19: Added real-snapshot support to `bench_page_gpu_batch` via `--matrix-file` and `--chunk-size-bytes`, with streamed `ChunkedMatrix` loading and auto domain-bits detection from page count (guarded by `MAX_DOMAIN_BITS`).

- 2026-02-19: Ran H100 benchmark against Modal volume file `/data/mainnet_compact.bin` (64.2 GiB) and saved artifact `docs/benchmarks/2026-02-19-task-25-page-gpu-batch-h100-mainnet-volume.csv`. Results: Q=2 single 61.26ms vs batch 59.08ms (1.04x), Q=4 122.57ms vs 117.82ms (1.04x), Q=8 245.08ms vs 236.91ms (1.03x).
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Task notes include profiling traces/metrics and chosen design tradeoffs
- [x] #2 Observability docs updated for any new GPU timing metrics
<!-- DOD:END -->
