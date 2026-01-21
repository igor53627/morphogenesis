# Phase 79: Kernel Optimization Brief

**Goal:** Increase GPU PIR throughput from **1.3 TB/s** to **> 3 TB/s** on H100.

## Current Status (Jan 21, 2026)
- **Baseline:** 53ms latency on H100 (68GB Matrix).
- **Bottleneck:** Memory Bandwidth Efficiency (~27% of peak).
- **Root Cause:** The `fused_kernel.cu` uses a **Thread-per-Page** mapping.
    - Thread 0 reads Page 0 (Addr 0).
    - Thread 1 reads Page 1 (Addr 4096).
    - This stride (4KB) breaks memory coalescing. The GPU memory controller fetches 32-byte cache lines, but we only use 16 bytes from each, wasting bandwidth.

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
