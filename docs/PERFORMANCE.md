# Performance Optimization Findings

## Current Best Performance (Jan 18, 2026)
- **Achieved:** **393 GB/s** (with real AES-DPF + 8-row unroll + rayon parallel)
- **Target:** 140 GB/s [PASS] **EXCEEDED by 2.8x**
- **vs Single-threaded:** 13.8x improvement
- **Cuckoo overhead:** 1.18x (vs 2x with naive 50% load factor)

## Query Mode Performance (hsiao benchmarks)

| Mode | Row Size | Matrix (78M @ 85%) | Scan Time | Concurrent Clients (<600ms) |
|------|----------|-------------------|-----------|----------------------------|
| **Privacy-Only** (default) | 256 bytes | 22 GB | **~66ms** | **~9** |
| Trustless | 2 KB | 175 GB | ~439ms | 1 |

## Server Specs (hsiao)
- **CPU:** AMD EPYC 9375F 32-Core (64 threads)
- **RAM:** 1.1 TB
- **Single-thread read bandwidth:** ~41 GB/s

## Optimization History

| Version | Performance | Change |
|---------|-------------|--------|
| Baseline (2-row unroll) | 20 GB/s | - |
| 4-row unroll | 24.85 GB/s | +24% |
| 8-row unroll | 28.57 GB/s | +43% |
| 8-row unroll + parallel (DummyDpfKey) | 383 GB/s | +19x |
| 8-row unroll + parallel (AesDpfKey, 75GB) | 393 GB/s | +20x |
| **8-row unroll + parallel (AesDpfKey, 87.5GB 85% Cuckoo)** | **393 GB/s** | **+20x** |

## Cuckoo Hash Table Optimization

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

## What's Left for Production
1. **Delta-PIR integration** - Apply delta buffer after main scan
2. **Epoch management** - CoW memory management for live updates
3. **Network layer** - gRPC/HTTP API for client queries
