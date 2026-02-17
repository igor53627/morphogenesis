# OpenTelemetry Observability Setup

Morphogenesis supports distributed tracing via OpenTelemetry (OTLP). This is vendor-neutral and works with any OTLP-compatible backend.

## Supported Backends

The following open-source APM tools have been tested or are expected to work:

| Tool | Origin | Service Map | Setup Difficulty |
|------|--------|-------------|------------------|
| [SigNoz](https://signoz.io) | India | ⭐⭐⭐⭐ | Easy |
| [Apache SkyWalking](https://skywalking.apache.org) | China | ⭐⭐⭐⭐⭐ | Medium |
| [Pinpoint](https://pinpoint-apm.gitbook.io/pinpoint) | Korea | ⭐⭐⭐⭐⭐ | Medium |
| [Grafana Tempo](https://grafana.com/oss/tempo/) | EU/US | ⭐⭐⭐⭐ | Medium |
| [Jaeger](https://www.jaegertracing.io) | US | ⭐⭐⭐ | Hard |

## Quick Start with SigNoz (Recommended)

```bash
# 1. Clone and start SigNoz
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy/
docker-compose up -d

# 2. Start morphogen-rpc-adapter with tracing
cargo run -p morphogen-rpc-adapter -- \
  --otel-traces \
  --otel-endpoint http://127.0.0.1:4317 \
  --otel-service-name morphogen-rpc-adapter

# 3. Run E2E client with tracing
cargo run -p morphogen-rpc-adapter --bin morphogen-e2e-client -- \
  --otel-traces \
  --otel-endpoint http://127.0.0.1:4317

# 4. Open SigNoz UI at http://localhost:3301
```

## Quick Start with Apache SkyWalking

```bash
# 1. Start SkyWalking with Docker Compose
wget https://raw.githubusercontent.com/apache/skywalking/master/docker/docker-compose.yml
docker-compose up -d

# 2. Configure adapter to send to SkyWalking OTLP endpoint
cargo run -p morphogen-rpc-adapter -- \
  --otel-traces \
  --otel-endpoint http://127.0.0.1:11800 \
  --otel-service-name morphogen-rpc-adapter

# 3. Open SkyWalking UI at http://localhost:8080
```

## Configuration Options

Both the RPC adapter and E2E client support these OpenTelemetry flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--otel-traces` | `false` | Enable OTLP trace export |
| `--otel-endpoint` | `http://127.0.0.1:4317` | OTLP collector endpoint |
| `--otel-service-name` | `morphogen-rpc-adapter` | Service name in traces |
| `--otel-env` | `e2e` | Deployment environment tag |
| `--otel-version` | `local` | Service version tag |

## Environment Variables

You can also configure via environment:

```bash
export OTEL_ENDPOINT=http://127.0.0.1:4317
export OTEL_SERVICE_NAME=morphogen-rpc-adapter
export OTEL_ENV=production
export OTEL_VERSION=1.0.0
export RUST_LOG=morphogen_rpc_adapter=info,morphogen_e2e_client=info
```

## Trace Context Propagation

The adapter automatically extracts trace context from incoming HTTP requests:
- `traceparent` header (W3C standard)
- `tracestate` header
- `baggage` header

This enables distributed tracing across service boundaries.

## Viewing Traces

After setup, you should see:

1. **Service List**: Both `morphogen-rpc-adapter` and `morphogen-e2e-client`
2. **Service Map**: Visual topology showing the dependency edge
3. **Traces**: Individual RPC requests with timing and attributes

### Example Traces

- `eth_getBalance` - Private balance query via PIR
- `eth_getCode` - Contract code fetch via CAS
- `eth_call` - Local EVM execution
- `eth_estimateGas` - Gas estimation

## Troubleshooting

### No traces appearing

1. Check OTLP endpoint is reachable: `curl http://localhost:4317`
2. Verify `--otel-traces` flag is enabled
3. Check `RUST_LOG` includes your service name
4. Ensure backend collector is running

### Missing service map edges

- Both services must use the same OTLP endpoint
- Trace context must propagate between client and server
- Check that `traceparent` header is being sent/received

## Architecture

```
┌─────────────────────┐
│  morphogen-e2e-client │──┐
└─────────────────────┘  │
                         │ OTLP/gRPC
                         ▼
              ┌─────────────────────┐
              │   OTLP Collector    │
              │  (SigNoz/SkyWalking/│
              │   Jaeger/Tempo)     │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  morphogen-rpc-adapter│
              │  (HTTP + OTLP export) │
              └─────────────────────┘
```

## References

- [OpenTelemetry](https://opentelemetry.io/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [SigNoz Documentation](https://signoz.io/docs/)
- [SkyWalking Documentation](https://skywalking.apache.org/docs/)
