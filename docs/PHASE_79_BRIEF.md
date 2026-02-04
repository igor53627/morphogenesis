# Phase 79: Kernel Optimization Brief

**Goal:** Increase GPU PIR throughput from **1.3 TB/s** to **> 3 TB/s** on H100.

## Optimization Journey

We tested multiple strategies to break the 1.3 TB/s bottleneck:

1.  **Baseline (53ms):** Thread-per-Page. Uncoalesced memory access (4KB stride) but full compute parallelism.
2.  **Plan A (1500ms):** Warp-per-Page. Perfect coalescing but serialized PRG (Lane 0 only). Proved PRG-bound.
3.  **Plan C (48ms):** Transposed Layout. Fixed memory coalescing for Thread-per-Page. Proved that even with perfect memory, we are bound by DPF compute.
4.  **Plan I (52ms):** Parallel DPF + Warp-per-Page. Restored compute parallelism. Matched baseline, proving memory coalescing gain was negligible while compute bound.
5.  **Plan K (27.4ms):** **Subtree Caching (Winner)**.
    - Evaluates top 15 levels of DPF tree once per tile.
    - Reduces per-page compute by 60%.
    - Combined with Warp-per-Page coalescing, this breaks both barriers.

## Final Status (Jan 21, 2026)
- **Goal:** Optimize GPU Kernel for >3 TB/s.
- **Result:** **27.4ms** latency on B200 (2.51 TB/s).
- **Architecture (Plan K):**
  - **Memory:** Warp-per-Page (Coalesced).
  - **Compute:** Subtree DPF Caching (Tile Size: 2048 pages).
  - **Parallelism:** Full 32-lane DPF evaluation.
- **Improvement:** 1.93x speedup over H100 baseline.
- **Conclusion:** We are within 6ms of the memory bandwidth limit (21ms).

## Optimization Strategy: Warp-per-Page

We will refactor the kernel to assign a **Warp (32 threads)** to a single Page.

### 1. Memory Access (Coalesced)
- Threads 0-31 read contiguous `uint4` (16-byte) blocks.
- Thread 0 reads `Page[0]`.
- Thread 1 reads `Page[16]`.
- ...
- Thread 31 reads `Page[496]`.
- **Result:** Perfect coalescing. We load 512 bytes per instruction per warp.

### 2. DPF Evaluation (Broadcast)
- Since all 32 threads in a warp process the *same* page index, they need the *same* DPF mask.
- **Optimization:** Only **Lane 0** evaluates the DPF PRG (ChaCha8).
- **Broadcast:** Lane 0 broadcasts the 16-byte mask to Lanes 1-31 using `__shfl_sync` (warp shuffle).
- **Savings:** Reduces compute load by 96% (1 PRG eval per 32 threads instead of 32).

## Implementation Steps
1.  Modify `crates/morphogen-gpu-dpf/cuda/fused_kernel.cu`.
2.  Update `crates/morphogen-gpu-dpf/src/kernel.rs` launch params (Grid/Block dims).
    - Block Dim: 256 threads (8 warps).
    - Each Block processes 8 pages in parallel (instead of 256).
    - Grid Dim needs to increase by 32x to cover the same number of pages.

## expected Outcome
- Throughput should scale linearly with memory bandwidth.
- Target: **3 TB/s** (approx 70-80% efficiency).
- QPS capacity per H100 should triple (~60 QPS).
