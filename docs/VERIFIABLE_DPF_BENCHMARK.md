# Benchmark Report: Verifiable DPF (Binius/SumCheck)

**Date:** Jan 23, 2026
**Device:** Apple M1/M2/M3 (Simulated High-End CPU)
**Target:** Ethereum Mainnet State (~270M rows, $2^{28}$)

## 1. Executive Summary
We implemented a multi-threaded CPU benchmark for the "Verifiable DPF" protocol using Binius-compatible binary fields ($GF(2^{128})$). This protocol generates a wide execution trace of the PRG expansion, compresses it via Random Linear Combinations (RLC), and proves integrity via SumCheck.

**Key Findings:**
1.  **CPU is too slow for Per-Query Proofs:** Extrapolated latency is **98 seconds**. Target is **<50ms**.
2.  **CPU is fast enough for Per-Epoch Proofs:** 98 seconds is well within the 10-minute epoch window.
3.  **GPU Acceleration is Mandatory:** To achieve per-query verifiability, we need a **2000x speedup**, which is only possible by fusing the verification into the GPU PIR kernel (memory-bound).

## 2. Benchmark Results (CPU)
Run parameters: $N=2^{20}$ (1M rows), Trace Width = 16 columns. Parallelized with `rayon`.

| Step | Time (1M Rows) | Throughput | Projected (270M) |
| :--- | :--- | :--- | :--- |
| **Trace Gen** | 0.0203 s | 51.71 M/s | 5.19 s |
| **RLC Batching** | 0.2108 s | 4.97 M/s | 53.96 s |
| **SumCheck** | 0.1513 s | 6.93 M/s | 38.73 s |
| **TOTAL** | **0.3823 s** | **-** | **97.88 s** |

## 3. Analysis

### 3.1 Trace Complexity
The benchmark assumes a trace width of 16 columns (128-bit each) to represent the PRG state.
*   **Generation:** Fast (51 M/s). Limited by memory write speed.
*   **Compression (RLC):** The most expensive step (54% of total time). It requires reading 16 streams and writing 1 stream.
*   **SumCheck:** Surprisingly fast (6.9 M/s) due to efficient binary field arithmetic and parallelization.

### 3.2 The "Per-Query" Gap
*   **Requirement:** The server must attach a proof to *every* query response.
*   **Latency Budget:** < 50ms (ideally 0ms overhead over scan).
*   **Current Reality:** ~98,000 ms.
*   **The Fix:**
    1.  **Fusion:** Don't write the trace to memory. Generate $Q[i]$ and immediately fold it into the SumCheck accumulator in registers.
    2.  **GPU Bandwidth:** H100 (3 TB/s) vs CPU RAM (~100 GB/s).
    3.  **Zero Overhead:** The PIR scan already reads the DB. If we perform the proof *arithmetic* faster than the memory fetch, the proof becomes "free" (latency hidden by memory).

## 4. Next Steps
1.  **Verify GPU Arithmetic Intensity:** Confirm that $GF(2^{128})$ operations on B200/H100 are fast enough to keep up with 3 TB/s memory bandwidth.
2.  **Implement Fused Kernel:** Port the logic from `SumCheckProver` directly into the CUDA kernel (`fused_kernel.cu`).
3.  **Use CPU for Layer 2:** Proceed with the current CPU implementation for the "Database Integrity" (Phase 2) proofs.
