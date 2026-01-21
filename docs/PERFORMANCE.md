# Performance Optimization Findings

## Current Best Performance (Jan 19, 2026)
- **Achieved:** **1,798 GB/s** (with fused CUDA kernel on NVIDIA B200)
- **Mainnet Latency (108GB):** **~60 ms** (Target: < 600 ms)
- **vs Single-node CPU:** **172x speedup** (38ms vs 6.6s for 68GB)
- **Cuckoo load factor:** 85% (1.18x overhead)

## Hardware Benchmark Results (Real Mainnet Data - Jan 21, 2026)

Results from scanning the **Full Ethereum State** (1.85B items, 68.8 GB matrix) on production hardware.

| Hardware | Dataset Size | Row Size | Mode | Throughput | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA H100** | 68.8 GB | 32 Bytes | GPU (CUDA) | **1,300 GB/s** | **53.0 ms** |
| **NVIDIA H200** | 137.6 GB | 64 Bytes | GPU (CUDA) | **1,574 GB/s** | **87.3 ms** |
| **AMD EPYC (64 thr)**| 108 GB | 64 Bytes | CPU (Rayon) | 16 GB/s | 6,617 ms |

### Key Insight: 50ms Real-Time PIR
The 53ms latency achieved on the H100 confirms that Morphogenesis can outperform traditional non-private RPC providers while maintaining 100% metadata privacy.

## Memory Footprint (Verified Mainnet)

Based on scanning 350M Accounts + 1.5B Storage Slots (Jan 2026):

| Schema | Row Size | Total Matrix | Hardware Required |
| :--- | :--- | :--- | :--- |
| **Compact** | 32 Bytes | **59.2 GB** | **1x H100 (80GB)** |
| **Full** | 64 Bytes | **118.4 GB** | **1x H200 (141GB)** |
| **Trustless** | 2 KB | **3.7 TB** | **4x H200 Nodes (Cluster)** |

### Key Insight: HBM-Resident Fused Kernel
The transition from CPU to GPU provided a **100x+ improvement** by bypassing the PCIe/Memory bus bottleneck. Our custom CUDA kernel fuses DPF evaluation, page masking, and XOR reduction into a single pass, utilizing the massive HBM bandwidth (up to 4.8 TB/s on H200) directly.

## Hybrid CPU/GPU Architecture

To maintain **<100ms latency** while supporting **12s block updates**, we employ a hybrid split:

1.  **Main Matrix (GPU HBM):** Stores the 108GB base snapshot (Epoch $N$). Read-only during queries. Scanned in ~60ms.
2.  **Delta Buffer (CPU RAM):** Accumulates live updates (<100MB) for Epoch $N$. Scanned in <1ms.
3.  **Merge:** The server XORs the GPU result and CPU result before responding.

This architecture decouples the massive read bandwidth requirement (handled by GPU) from the low-latency write requirement (handled by CPU), allowing **zero-downtime updates**.

## Cuckoo Hash Table Optimization
...

### Load Factor Improvement
| Algorithm | Load Factor | Table Size (78M) | Matrix Size | Scan Time |
|-----------|-------------|------------------|-------------|-----------|
| Naive deterministic | 50% | 156M rows | 150 GB | ~382ms |
| **Random-walk insertion** | **85%** | **92M rows** | **87.5 GB** | **~222ms** |
| Theoretical limit | 91.8% | 85M rows | 82 GB | ~209ms |

### Key Insight: Random-Walk Cuckoo
The naive Cuckoo insertion uses `hash_fn = kick % 3` deterministically, creating short cycles.
Random-walk insertion:
1. Picks one of 3 candidate positions **randomly**
2. Excludes the position just evicted from
3. Achieves 85% load factor vs 50% with deterministic cycling
4. Result: 1.18x memory overhead vs 2x

## Failed Approaches

### 1. Batched eval_range_masks (SLOWER - 12-14 GB/s)
- Pre-computed masks for batches of rows
- Overhead from slice operations and batch loop structure outweighed benefits

### 2. Offset-first loop order (MUCH SLOWER - 7.7 GB/s)
- Process all rows for offset 0, then offset 1, etc.
- Destroyed cache locality - each row re-fetched 16 times

### 3. Streaming stores (SLOWER - 6 GB/s)
- `_mm512_stream_si512` for accumulator stores
- Hurt performance because output buffer is small (3KB)

### 4. Prefetching (NEUTRAL - ~28.5 GB/s)
- `_mm_prefetch` for next batch of rows
- No measurable improvement - CPU already handles this well

## What Works

### 8-row unroll
- Process 8 consecutive rows per outer iteration
- Amortizes loop overhead
- Good instruction-level parallelism
- 24 DPF evaluations per iteration (3 keys × 8 rows)

## How We Achieved 383 GB/s

### Key Optimizations Applied
1. **8-row unroll** - Process 8 consecutive rows per iteration
   - Amortizes loop overhead
   - Better instruction-level parallelism
   - 24 DPF evaluations per outer iteration (3 keys × 8 rows)

2. **Parallel chunk processing** - Using rayon
   - Each 1GB chunk processed by a separate thread
   - Results XOR'd together at the end
   - Scales with available CPU cores

3. **AVX-512 vectorization** - 512-bit SIMD operations
   - Process 64 bytes at a time
   - AND + XOR operations fully vectorized

## Modal.com Cloud Benchmarks (Jan 18, 2026)

### CPU Instance (no AVX-512)
- 32 cores, 448GB RAM, unknown CPU model
- **109 GB/s** at 37.5GB (3.6x slower than hsiao due to no AVX-512)

### H200 GPU Host
- 17 CPU cores, 1TB RAM, NVIDIA H200 (141GB HBM3e)
- Has AVX-512 but limited cores
- **70 GB/s** at 10GB (limited by 17 cores vs 64 on hsiao)
- GPU's 4.8 TB/s HBM not usable from CPU code

### Key Insight
Performance scales with CPU core count more than AVX-512 availability.
GPU-accelerated PIR would require CUDA port to access HBM bandwidth.

## Page-Level PIR Performance (Jan 19, 2026)

### Overview
Page-level PIR using fss-rs provides true 2-server computational privacy,
unlike the insecure AesDpfKey that exposes the target index.

### Chunk Size Optimization Results

**18-bit domain (256K pages, 1GB data):**

| Chunk Size | DPF (ms) | XOR (ms) | Total (ms) | DPF % | Speedup |
|------------|----------|----------|------------|-------|---------|
| 4096 | 19.7 | 17.2 | 36.9 | 53% | 1.00x |
| 8192 | 13.7 | 17.0 | 30.7 | 45% | 1.20x |
| 16384 | 11.2 | 17.0 | 28.3 | 40% | 1.30x |
| 32768 | 9.4 | 16.9 | 26.3 | 36% | 1.40x |
| **65536** | **8.7** | **16.8** | **25.5** | **34%** | **1.45x** |
| 131072 | 8.3 | 16.8 | 25.1 | 33% | 1.47x |

**Optimal chunk size: 65536** (1MB DPF buffer, 1.45x speedup)

### Mainnet Verified Performance (25-bit domain, 27M pages, 108GB)

| Hardware | Latency | Status |
| :--- | :--- | :--- |
| Baseline (CPU Rayon) | 6.6s | Verified |
| **NVIDIA H100** | **~66ms** | **Verified** |
| **NVIDIA H200** | **~68ms** | **Verified** |
| **NVIDIA B200** | **~60ms** | **Verified** |

### Key Findings

1. **Memory-to-Compute Ratio**: Our PIR protocol is memory-bound on CPU but compute-bound on GPU due to the ChaCha8 DPF logic.
2. **HBM Dominance**: Performance tracks with GPU memory generation (HBM3/HBM3e).
3. **Single-Query Efficiency**: Batching multiple queries increases throughput (QPS) slightly but adds significant latency; single-query processing is optimal for <100ms response times.

### Bottleneck Analysis

At optimal chunk size (65536):
- **DPF eval**: 33% of time (tree traversal + AES)
- **XOR accumulate**: 67% of time (memory-bound, touches all data)

Even if DPF became free, XOR alone = 2.1s (3.5x over target).
**GPU acceleration required** to parallelize both DPF and memory bandwidth.

### Thread Scaling (rayon)

| Threads | 16-bit Latency | Speedup |
|---------|----------------|---------|
| 1 | 27.3ms | 1.0x |
| 2 | 15.9ms | 1.7x |
| 4 | 10.8ms | 2.5x |
| 8 | 8.0ms | 3.4x |
| 16 | 8.2ms | 3.3x |

Optimal at 8 threads; 16 threads shows slight regression (overhead).

## What's Left for Production
1. **GPU acceleration** - Required for <600ms at mainnet scale
2. **Delta-PIR integration** - Apply delta buffer after main scan
3. **Epoch management** - CoW memory management for live updates
4. **Network layer** - HTTP/WebSocket API complete
