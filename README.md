# Morphogenesis PIR

High-throughput Private Information Retrieval for Ethereum light clients.

## Performance

| Metric | Value |
|--------|-------|
| Scan throughput | **383 GB/s** |
| Scan latency (75GB) | **~196ms** |
| Target | 140 GB/s (exceeded 2.7x) |

## Quick Start

```bash
# Build with AVX-512 and parallel support
cargo build --release --features avx512,parallel

# Run benchmark (75GB matrix, 3 iterations)
./target/release/bench_scan --rows 78643200 --iterations 3 --warmup-iterations 1 --scan-only --parallel
```

## Development Workflow (MANDATORY)

This project uses mandatory tooling for task tracking and code quality:

### backlog.md - Task Management
```bash
# View all tasks
backlog task list

# Work on a task
backlog task edit TASK-X -s "In Progress"

# Commit with task reference (auto-linked via git hook)
git commit -m "feat: implement feature

Addresses TASK-X"
```

### roborev - Continuous Code Review
```bash
# View latest reviews
roborev list

# Check specific commit
roborev show COMMIT_SHA

# Address findings
roborev address JOB_ID
```

### Git Hook Setup
The `.git/hooks/post-commit` hook automatically:
1. Triggers roborev review for every commit
2. Links commits to tasks when message includes "Addresses TASK-X"
3. Updates zoekt search index

**All commits must reference a task ID** and will be automatically reviewed.

## Architecture

```
crates/
├── morphogen-core/     # Core types (DeltaBuffer, EpochSnapshot, GlobalState)
├── morphogen-dpf/      # DPF key trait and implementations
├── morphogen-storage/  # AlignedMatrix, ChunkedMatrix
└── morphogen-server/   # Scan kernel, server, benchmarks
```

## Features

- `avx512` - Enable AVX-512 SIMD optimizations
- `parallel` - Enable multi-threaded chunk processing (rayon)
- `profiling` - Enable detailed timing instrumentation

## Documentation

- [Kanban](docs/KANBAN.md) - Project status and tasks (historic)
- [Backlog](backlog/) - Active task management (use `backlog` CLI)
- [Protocol Spec](docs/morphogenesis_protocol.md) - PRD v3.2
- [Paper](docs/morphogenesis_paper.md) - Epoch-Based Delta-PIR
- [EDD](docs/morphogenesis_EDD.md) - Engineering design
- [Performance](docs/PERFORMANCE.md) - Optimization findings
- [Profiling Guide](docs/PROFILING_GUIDE.md) - How to profile
- [Cryptography & Core Mechanics](docs/CRYPTOGRAPHY.md) - Explaining the fused kernel and DPF logic
- [Crypto Analysis](docs/CRYPTO_ANALYSIS.md) - Why we use ChaCha8 over AES on GPUs

## Key Concepts

- **Epoch-Based Delta-PIR**: Wait-free snapshot isolation for live updates
- **Parallel Cuckoo Retrieval**: 3 simultaneous DPF queries per request
- **Copy-on-Write Merge**: Striped CoW for O(delta) epoch transitions
- **AVX-512 Scan Kernel**: 64-byte SIMD with 8-row unrolling

## License

MIT
