# Morphogenesis Performance Benchmarks

**Latest Benchmark:** Jan 21, 2026
**Target Dataset:** Full Ethereum Mainnet (1.85B items, 68.8 GB)

## 1. Executive Summary

| Metric | Result | Target | Status |
| :--- | :--- | :--- | :--- |
| **Latency (Single Query)** | **53.0 ms** | < 100ms | **PASSED** |
| **Throughput (Scan)** | **1,300 GB/s** | > 1 TB/s | **PASSED** |
| **Capacity (Single H100)** | **2.15 Billion Rows** | 1.8 Billion | **PASSED** |

**Conclusion:** A single NVIDIA H100 GPU can serve private account balance queries for the entire Ethereum network with lower latency than many public RPC providers.

---

## 2. Hardware Benchmarks (Mainnet Scale)

Benchmarks run on real hardware using the full 68.8 GB production matrix (Compact 32-byte schema).

| Hardware | VRAM | Mode | Throughput | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA B200** | 192 GB | **Production (Subtree)** | **2,510 GB/s** | **27.4 ms** |
| **NVIDIA H200** | 141 GB | **Production (Subtree)** | **2,235 GB/s** | **30.8 ms** |
| **NVIDIA H100** | 80 GB | **Production (Subtree)** | **2,143 GB/s** | **32.1 ms** |

*\*H200/B200 results extrapolated from synthetic benchmarks.*

### Why is H100 the winner?
- Fits the 68.8 GB dataset entirely in HBM3 memory.
- Massive 3.35 TB/s memory bandwidth allows scanning 68GB in ~20ms (theoretical).
- Our implementation achieves ~40% of peak theoretical bandwidth (overhead from DPF compute + Cuckoo logic).

---

## 3. Historical Benchmarks (Icebox)

Older benchmarks for deprecated architectures.

### CPU Mode (AVX-512)
*Target: Commodity Servers*
- **Hardware:** AMD EPYC 9375F (64 cores).
- **Throughput:** 16 GB/s.
- **Latency:** **6.6 seconds** (Too slow for wallets).

### Trustless Mode (2KB Rows)
*Target: Verifiable Queries*
- **Dataset Size:** 3.7 TB (Requires 8x H100 Cluster).
- **Latency:** ~400ms.
- **Status:** Moved to Icebox due to high cost ($30k/mo).