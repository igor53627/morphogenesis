# Morphogenesis PIR

2-party DPF-based Private Information Retrieval for Ethereum state.

## Current Status

- Full mainnet matrix PIR on H100: **32.1 ms** latency (subtree kernel), **2.14 TB/s** throughput
- B200 projection: **27.4 ms** latency, **2.51 TB/s** throughput
- 2.15 billion rows (full Mainnet accounts + storage) in GPU VRAM
- RPC adapter: drop-in JSON-RPC proxy with 20+ private methods
- Information-Theoretic privacy under 2-server semi-honest model

## Performance

| Hardware | VRAM | Throughput | Latency | Concurrent Clients (< 600 ms) |
|----------|------|------------|---------|-------------------------------|
| NVIDIA B200 | 192 GB | 2,510 GB/s | 27.4 ms | ~21 |
| NVIDIA H200 | 141 GB | 2,235 GB/s | 30.8 ms | ~19 |
| NVIDIA H100 | 80 GB | 2,143 GB/s | 32.1 ms | ~18 |

## Quick Start

```bash
# Build with AVX-512 and parallel support
cargo build --release --features avx512,parallel

# Run benchmark (75GB matrix, 3 iterations)
./target/release/bench_scan --rows 78643200 --iterations 3 --warmup-iterations 1 --scan-only --parallel

# Run RPC adapter (connects to PIR servers + upstream RPC)
cargo run -p morphogen-rpc-adapter -- \
  --server1 http://pir1:8080 \
  --server2 http://pir2:8080 \
  --upstream https://eth-mainnet.g.alchemy.com/v2/KEY
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
├── morphogen-rpc-adapter/ # JSON-RPC proxy: private methods via PIR, passthrough to upstream
└── reth-adapter/          # Reth integration for mainnet snapshot ETL
```

## RPC Adapter

The adapter runs on `:8545` as a drop-in replacement for standard Ethereum RPC providers. Compatible with MetaMask, Rabby, Frame, and any EIP-1193 wallet.

| Category | Methods | Mechanism |
|----------|---------|-----------|
| Private (PIR) | `getBalance`, `getTransactionCount`, `getCode`, `getStorageAt` | DPF query to PIR servers |
| Private (EVM) | `eth_call`, `eth_estimateGas` | Local revm with PIR-backed state |
| Private (Cache) | `eth_getLogs`, `getTransactionByHash`, `getTransactionReceipt` | Block cache (64 blocks) |
| Private (Filters) | `newFilter`, `newBlockFilter`, `getFilterChanges`, `getFilterLogs` | Local filter state |
| Relay | `eth_sendRawTransaction` | Flashbots Protect |
| Passthrough | `blockNumber`, `chainId`, `gasPrice`, `getBlockByNumber`, etc. | Forwarded to upstream |

## Features

- `avx512` - Enable AVX-512 SIMD optimizations
- `parallel` - Enable multi-threaded chunk processing (rayon)
- `profiling` - Enable detailed timing instrumentation
- `verifiable-pir` - Enable sumcheck/binius proof plumbing

## Documentation

- [Protocol & Architecture](docs/morphogenesis_protocol.md) - v5.0
- [Paper (LaTeX)](docs/paper/morphogenesis.pdf) - Academic paper
- [Performance](docs/PERFORMANCE.md) - Benchmark results and analysis
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
