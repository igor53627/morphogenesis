# TASK-24: GPU Page Batch Endpoint and Batched Scan Wiring

## Summary
Implemented batched GPU page query support on the server API:
- New endpoint: `POST /query/page/gpu/batch`
- Request shape: `{ "queries": [ { "keys": ["0x..","0x..","0x.."] }, ... ] }`
- Response shape: `{ "epoch_id": <u64>, "results": [ { "pages": [...] }, ... ] }`

The endpoint:
- validates `1..=MAX_BATCH_SIZE` query count,
- validates each query contains exactly 3 ChaCha keys,
- preserves response ordering,
- supports CUDA path (`scan_batch_optimized`) and CPU fallback.

## Implementation Notes

### API additions
- `BatchGpuPageQueryRequest`
- `BatchGpuPageQueryResponse`
- `BatchGpuPageQueryResult`

Code:
- `crates/morphogen-server/src/network/api.rs`

### Router wiring
- Added route:
  - `/query/page/gpu/batch`

Code:
- `crates/morphogen-server/src/network/api.rs`

### CUDA path behavior
- Uses `GpuScanner::scan_batch_optimized(...)`.
- Applies pending delta per query via `scan_delta_for_gpu`.
- Performs the same epoch consistency checks used in single-query GPU path.
- Returns `503` on repeated consistency failures.

### CPU fallback behavior
- Reuses page collection + CPU fused evaluator (`eval_fused_3dpf_cpu`) per query.
- Preserves deterministic request order in response.

## Tests

Added integration tests:
- `gpu_batch_query_returns_results_in_request_order_cpu_fallback`
- `gpu_batch_query_rejects_empty_batch`
- `gpu_batch_query_rejects_wrong_key_count`

Code:
- `crates/morphogen-server/tests/network_api.rs`

## Benchmark Evidence

Environment in this workspace does not provide CUDA runtime for endpoint benchmarking,
so measurements below are CPU fallback endpoint comparisons.

Benchmark binary:
- `crates/morphogen-server/src/bin/bench_page_gpu_batch.rs`

Command:

```bash
cargo run --package morphogen-server --release --features network --bin bench_page_gpu_batch -- \
  --num-pages 256 --domain-bits 8 --iterations 30 --warmup-iterations 3 --batch-sizes 1,2,4,8,16,32
```

Raw output:
- `docs/benchmarks/2026-02-19-task-24-page-gpu-batch-cpu-fallback.csv`

Observed (`batch_vs_single_speedup`):
- Q=1: `1.09x`
- Q=2: `1.10x`
- Q=4: `0.98x`
- Q=8: `0.99x`
- Q=16: `1.00x`
- Q=32: `1.00x`

Interpretation:
- CPU fallback endpoint overhead for batch vs single loop is roughly neutral at larger Q.
- Main expected gains for this feature are on CUDA path by reducing host/API overhead and using batched kernel launch.

### Modal H100 CUDA validation (2026-02-19)

Ran the same benchmark on Modal with `H100!` and `--features network,cuda`.
During this validation, two CUDA-only compile issues were fixed:
- `crates/morphogen-server/src/network/api.rs`: avoid moving `ChaChaKey` values when calling `scan_delta_for_gpu`.
- `crates/morphogen-server/src/bin/bench_page_gpu_batch.rs`: keep `Arc<ChunkedMatrix>` available for GPU upload path.

Artifact:
- `docs/benchmarks/2026-02-19-task-24-page-gpu-batch-h100.csv`

Observed (`batch_vs_single_speedup`):
- Q=1: `1.01x`
- Q=2: `1.16x`
- Q=4: `0.31x`
- Q=8: `0.18x`
- Q=16: `0.10x`
- Q=32: `0.10x`

Interpretation:
- CUDA path is validated (`mode=cuda`) on NVIDIA H100 80GB.
- Current batched kernel path only improves at very small batch (Q=2) in this setup; larger batches regress, consistent with known occupancy/shared-memory pressure behavior.
