# Observability and Tracing

Morphogenesis PIR provides built-in support for structured logging and real-time metrics to ensure reliable operation in production environments.

## Structured Logging (`tracing`)

The server uses the `tracing` crate for asynchronous, structured logging. This allows for better context in logs (e.g., request IDs, spans) and easy ingestion into log management systems.

### Configuration

Log levels are controlled via the `RUST_LOG` environment variable.

- **Production:** `RUST_LOG=info` (default)
- **Debugging:** `RUST_LOG=morphogen_server=debug,tower_http=debug`
- **Tracing:** `RUST_LOG=morphogen_server=trace`

### Output Format

By default, the server outputs logs in a human-readable text format. In production environments, it can be configured to output **JSON** for easier parsing by ELK, Datadog, or CloudWatch.

### Key Instrumentation

The following operations are instrumented with spans:

- `query_handler`: Tracks the lifecycle of a row-level PIR query.
- `page_query_handler`: Tracks a CPU-based page PIR query.
- `page_query_gpu_handler`: Tracks a GPU-accelerated PIR query.
- `EpochManager::try_advance`: Logs epoch transition details and merge performance.

## Real-time Metrics (Prometheus)

The server exposes a `/metrics` endpoint in Prometheus format.

### Available Metrics

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `pir_query_count_total` | Counter | Total number of PIR queries received (labeled by `type`). |
| `pir_query_duration_seconds` | Histogram | Latency distribution of PIR queries. |
| `pir_epoch_id` | Gauge | The current active epoch ID. |
| `gpu_scan_duration_seconds` | Histogram | Latency of the GPU-only scan kernel execution. |
| `gpu_query_phase_duration_seconds` | Histogram | GPU request phase latency with labels `endpoint` (`gpu`/`gpu_batch`) and `phase` (`transfer_h2d`, `kernel`, `transfer_d2h`, `merge`). |
| `gpu_batch_dispatch_mode_total` | Counter | GPU batch dispatch mode selections labeled by `mode` (`multistream`, `full_batch`, `micro_batch2`). |
| `gpu_vram_used_bytes` | Gauge | Current VRAM utilization (if CUDA enabled). |

### Monitoring Endpoint

Access metrics locally:
```bash
curl http://localhost:3000/metrics
```

### Prometheus Scrape Configuration

Add the following to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'morphogen-server'
    static_configs:
      - targets: ['localhost:3000']
```

### GPU Batch Dispatch Knobs

For `/query/page/gpu/batch`, runtime dispatch can be tuned with:

- `MORPHOGEN_GPU_STREAMS`: stream count (clamped to `1..=8`).
- `MORPHOGEN_GPU_BATCH_POLICY`: `adaptive` (default), `throughput`, or `latency`.
- `MORPHOGEN_GPU_BATCH_ADAPTIVE_THRESHOLD`: query-count threshold used by `adaptive` mode (clamped to `1..=32`).
- `MORPHOGEN_GPU_BATCH_TILE_SIZE`: per-launch query tile cap for optimized batch scans (clamped to `1..=16`).
- `MORPHOGEN_GPU_CUDA_GRAPH`: `true/false` toggle for CUDA graph capture+replay on eligible single-launch batch shapes (`1|2|4|8|16` queries).

## Performance Impact

The observability stack is designed for high-performance PIR:
- **Tracing:** Low-overhead macros; disabled at compile-time if the `tracing` feature is off.
- **Metrics:** Atomic counters and thread-local histograms to minimize lock contention.
- **Micro-benchmarking:** For sub-millisecond precision on the GPU path, use the dedicated `profile_gpu_dpf` binary which uses high-resolution CUDA events.

## OpenTelemetry Tracing

For distributed tracing setup with various backends (SigNoz, SkyWalking, Jaeger, etc.),
see `docs/OPENTELEMETRY.md`.
