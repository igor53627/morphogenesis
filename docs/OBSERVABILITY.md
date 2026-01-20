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

## Performance Impact

The observability stack is designed for high-performance PIR:
- **Tracing:** Low-overhead macros; disabled at compile-time if the `tracing` feature is off.
- **Metrics:** Atomic counters and thread-local histograms to minimize lock contention.
- **Micro-benchmarking:** For sub-millisecond precision on the GPU path, use the dedicated `profile_gpu_dpf` binary which uses high-resolution CUDA events.
