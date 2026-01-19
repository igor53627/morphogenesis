# Profiling Guide for Morphogen Server

## Built-in Profiling

The server includes a `profiling` feature that adds detailed timing instrumentation.

## Benchmark Flags

Use these flags to separate setup, reduce page faults, and focus on scan-only timing:

- `--no-fill`: skip filling the matrix with a pattern (faster setup).
- `--warmup-iterations N`: pre-touch the matrix to reduce page faults.
- `--scan-only`: print scan timing only (omit setup timing).

Example:
```bash
cargo run -p morphogen-server --bin bench_scan --release --features profiling,avx512 -- \
  --rows 78643200 --iterations 1 --warmup-iterations 1 --scan-only
```

### Enable Profiling

```bash
cargo run -p morphogen-server --bin bench_scan --release --features profiling,avx512 -- --rows 78643200 --iterations 1
```

This will output timing breakdowns to stderr showing:
- Time spent in `scan_main_matrix` vs `scan_delta`
- Time spent in AVX-512 kernel initialization
- Time spent in mask evaluation
- Time spent in data processing loops

### Example Output

```
scan_main_matrix breakdown:
  scan_main_matrix_init: 0.01ms
  scan_main_matrix_avx512_path: 0.02ms
  scan_main_matrix_complete: 19120.50ms

avx512 kernel breakdown:
  avx512_init: 0.01ms
  avx512_setup: 0.02ms
  avx512_first_chunk: 0.05ms
  avx512_first_mask_eval: 0.10ms
  avx512_first_mask_compute: 0.12ms
  avx512_first_row_loop_start: 0.15ms
  avx512_complete: 19120.50ms

Profiling breakdown:
  scan_start: 0.00ms
  scan_main_matrix: 19120.50ms
  scan_delta: 0.01ms
```

## External Profiling Tools

### 1. perf (Linux)

**Record profiling data:**
```bash
perf record -g --call-graph dwarf \
  cargo run -p morphogen-server --bin bench_scan --release --features avx512 -- \
  --rows 78643200 --iterations 1
```

**View report:**
```bash
perf report
```

**View annotated source:**
```bash
perf annotate
```

**Top functions:**
```bash
perf top -p $(pgrep bench_scan)
```

### 2. cargo-flamegraph

**Install:**
```bash
cargo install flamegraph
```

**Generate flamegraph:**
```bash
cargo flamegraph --bin bench_scan --release --features avx512 -- \
  --rows 78643200 --iterations 1
```

This creates `flamegraph.svg` showing the call stack and time spent in each function.

### 3. valgrind (callgrind)

**Profile with callgrind:**
```bash
valgrind --tool=callgrind \
  cargo run -p morphogen-server --bin bench_scan --release --features avx512 -- \
  --rows 78643200 --iterations 1
```

**View with kcachegrind:**
```bash
kcachegrind callgrind.out.*
```

### 4. Intel VTune (if available)

```bash
vtune -collect hotspots -result-dir ./vtune_results \
  cargo run -p morphogen-server --bin bench_scan --release --features avx512 -- \
  --rows 78643200 --iterations 1
```

## What to Look For

### High-Level Bottlenecks
1. **DPF evaluation time** - If `avx512_first_mask_eval` takes significant time, DPF calls are the bottleneck
2. **Memory access time** - If loop time >> computation time, memory bandwidth is the limit
3. **Cache misses** - Use `perf stat -e cache-misses` to check

### Low-Level Analysis
1. **Function hotspots** - Use `perf report` to see which functions take most time
2. **Instruction-level** - Use `perf annotate` to see which instructions are slow
3. **Memory stalls** - Check for high `cycles:u` vs `instructions:u` ratio

### Common Issues

**If DPF evaluation is slow:**
- Per-row `eval_bit()` calls are the bottleneck
- Solution: Implement vectorized keystream generation

**If memory access is slow:**
- Not saturating memory bandwidth
- Solution: Better prefetching, larger unrolls

**If computation is slow:**
- AVX-512 instructions not being used efficiently
- Solution: Better instruction scheduling, more parallelism

## Quick Profiling Commands

```bash
# Built-in profiling
cargo run -p morphogen-server --bin bench_scan --release --features profiling,avx512 -- --rows 78643200 --iterations 1

# perf top (real-time)
perf top -p $(pgrep -f bench_scan)

# perf stat (summary)
perf stat cargo run -p morphogen-server --bin bench_scan --release --features avx512 -- --rows 78643200 --iterations 1

# Cache statistics
perf stat -e cache-references,cache-misses,LLC-loads,LLC-load-misses \
  cargo run -p morphogen-server --bin bench_scan --release --features avx512 -- \
  --rows 78643200 --iterations 1
```

## Page-Level PIR Profiling (profile_dpf)

The `profile_dpf` binary provides DPF-specific profiling for page-level PIR.

### Build

```bash
cargo build --release --package morphogen-dpf --features fss --bin profile_dpf
```

### Usage

```bash
# Run all benchmarks (contiguous, fragmented, chunk sweep)
./target/release/profile_dpf <domain_bits>

# Specific modes
./target/release/profile_dpf <domain_bits> contiguous [chunk_size]
./target/release/profile_dpf <domain_bits> fragmented [chunk_size]
./target/release/profile_dpf <domain_bits> sweep
./target/release/profile_dpf <domain_bits> optimal    # Uses chunk=131072
./target/release/profile_dpf <domain_bits> masks      # Analyze mask distribution
```

### Examples

```bash
# 16-bit domain (64K pages, 256MB) - quick test
./target/release/profile_dpf 16

# 18-bit domain (256K pages, 1GB) - chunk sweep
./target/release/profile_dpf 18 sweep

# 20-bit domain (1M pages, 4GB) - optimal settings
./target/release/profile_dpf 20 optimal

# Profile with flamegraph
cargo flamegraph --bin profile_dpf --package morphogen-dpf --features fss -- 18
```

### Example Output (chunk sweep)

```
=== DPF Profile: 18-bit domain ===
Pages: 262144, Data: 1024 MB
Using contiguous allocation for chunk sweep...

   Chunk    DPF(ms)    XOR(ms)  Total(ms)       DPF%
----------------------------------------------------
    4096      19.71      17.19      36.90      53.4%
    8192      13.69      16.95      30.65      44.7%
   16384      11.24      17.02      28.27      39.8%
   32768       9.42      16.85      26.27      35.9%
   65536       8.69      16.84      25.53      34.0%
  131072       8.27      16.83      25.10      33.0%
```

### Key Insights

1. **Optimal chunk size**: 65536 (1MB buffer) provides best performance
2. **DPF vs XOR split**: At optimal chunk, DPF is ~33%, XOR is ~67%
3. **XOR is memory-bound**: Constant ~17ms regardless of chunk size
4. **Contiguous vs fragmented**: No difference when data fits in RAM
5. **Thread scaling**: 3.4x speedup with 8 threads (rayon)
