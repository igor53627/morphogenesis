# Morphogenesis PIR

2-server DPF-based Private Information Retrieval for Ethereum state.

## Current Status

- Full mainnet matrix PIR on H100: **32.1 ms** latency (subtree kernel), **2.14 TB/s** throughput
- B200 projection: **27.4 ms** latency, **2.51 TB/s** throughput
- 2.15 billion rows (full Mainnet accounts + storage) in GPU VRAM
- RPC adapter: drop-in JSON-RPC proxy with 20+ private methods
- Information-Theoretic privacy under 2-server semi-honest model

## Performance

| Hardware | VRAM | Throughput | Latency | Concurrent Clients (< 600 ms) | Source |
|----------|------|------------|---------|-------------------------------|--------|
| NVIDIA B200 | 192 GB | 2,510 GB/s | 27.4 ms | ~21 | Projected |
| NVIDIA H200 | 141 GB | 2,235 GB/s | 30.8 ms | ~19 | Projected |
| NVIDIA H100 | 80 GB | 2,143 GB/s | 32.1 ms | ~18 | Measured |

Assumes full mainnet matrix in GPU VRAM, one in-flight query per client, and excludes network RTT/application overhead. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for benchmark methodology.

## Quick Start

```bash
# Build with AVX-512 and parallel support
cargo build --release --features avx512,parallel

# Run benchmark (75GB matrix, 3 iterations)
./target/release/bench_scan --rows 78643200 --iterations 3 --warmup-iterations 1 --scan-only --parallel

# Keep provider key out of inline commands/snippets
export UPSTREAM_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/<KEY>"

# Run RPC adapter (connects to PIR servers + upstream RPC)
cargo run -p morphogen-rpc-adapter -- \
  --pir-server-a http://pir1:8080 \
  --pir-server-b http://pir2:8080 \
  --upstream "$UPSTREAM_RPC_URL"
```

## Development Workflow

Required tooling/hooks are documented in [.mandatory-tooling](.mandatory-tooling). Task tracking lives in [backlog/](backlog/) and reviews are managed with `roborev`.
Every code or docs change must update the related backlog task status/notes before merge.

## Deterministic E2E

Run the full local deterministic RPC adapter E2E suite (subset fixtures, no live-mainnet dependency):

```bash
bash scripts/test_rpc_e2e.sh
```

## Architecture

```
crates/
├── morphogen-core/        # Core types (DeltaBuffer, EpochSnapshot, GlobalState, Cuckoo hashing)
├── morphogen-dpf/         # DPF key trait and implementations (AES-based, fss-rs)
├── morphogen-gpu-dpf/     # GPU-accelerated DPF using ChaCha8 PRG (CUDA)
├── morphogen-storage/     # AlignedMatrix, ChunkedMatrix storage primitives
├── morphogen-server/      # Scan kernel, HTTP/WebSocket server, benchmarks
├── morphogen-client/      # PIR client with network layer, caching, batch aggregation
├── morphogen-wasm-gateway/ # Browser EIP-1193 facade (WASM, private reads + passthrough)
├── morphogen-rpc-adapter/ # JSON-RPC proxy: private methods via PIR, passthrough to upstream
└── reth-adapter/          # Reth integration for mainnet snapshot ETL
```

## RPC Adapter

The adapter runs on `:8545` as a drop-in replacement for standard Ethereum RPC providers. Compatible with MetaMask, Rabby, Frame, and any EIP-1193 wallet.

| Category | Methods | Mechanism |
|----------|---------|-----------|
| Private (PIR) | `eth_getBalance`, `eth_getTransactionCount`, `eth_getCode`, `eth_getStorageAt` | DPF query to PIR servers |
| Private (EVM) | `eth_call`, `eth_estimateGas` | Local revm with PIR-backed state |
| Private (Cache) | `eth_getLogs`, `eth_getTransactionByHash`, `eth_getTransactionReceipt` | Block cache (64 blocks) |
| Private (Filters) | `eth_newFilter`, `eth_newBlockFilter`, `eth_getFilterChanges`, `eth_getFilterLogs` | Local filter state |
| Relay | `eth_sendRawTransaction` | Flashbots Protect |
| Passthrough | `eth_blockNumber`, `eth_chainId`, `eth_gasPrice`, `eth_getBlockByNumber`, etc. | Forwarded to upstream |

## Browser WASM Gateway

For browser-first EIP-1193 integration (without a localhost proxy), see:

- [WASM Gateway Guide](docs/WASM_GATEWAY.md)
- [WASM Demo Integration](examples/wasm-gateway/README.md)

## Features

- `avx512` - Enable AVX-512 SIMD optimizations
- `parallel` - Enable multi-threaded chunk processing (rayon)
- `profiling` - Enable detailed timing instrumentation
- `verifiable-pir` - Enable sumcheck/binius proof plumbing

## Documentation

- [Protocol & Architecture](docs/morphogenesis_protocol.md) - v5.0
- [Paper (LaTeX)](docs/paper/morphogenesis.pdf) - Academic paper
- [Performance](docs/PERFORMANCE.md) - Benchmark results and analysis
- [WASM Gateway](docs/WASM_GATEWAY.md) - Browser setup, method matrix, CORS, and MVP limits
- [Profiling Guide](docs/PROFILING_GUIDE.md) - How to profile scan engines
- [Cryptography](docs/CRYPTOGRAPHY.md) - Fused kernel and DPF internals
- [Crypto Analysis](docs/CRYPTO_ANALYSIS.md) - ChaCha8 vs AES on GPUs
- [Backlog](backlog/) - Active task management (use `backlog` CLI)

## Key Concepts

- **Epoch-Based Delta-PIR**: Wait-free snapshot isolation for live updates
- **Parallel Cuckoo Retrieval**: 3 simultaneous DPF queries per request
- **Copy-on-Write Merge**: Striped CoW for O(delta) epoch transitions
- **Access-List Prefetch**: EIP-2930 batch PIR for O(1) round-trip `eth_call`
- **Block Cache**: 64-block FIFO with reorg detection for private log/tx queries

## License

MIT
