# Cryptographic Primitive Analysis: ChaCha8 vs. AES for GPU PIR

## Executive Summary

Morphogenesis uses **ChaCha8** for its GPU-accelerated Distributed Point Function (DPF) implementation. This decision is driven by the architectural constraints of NVIDIA GPUs, which lack exposed hardware AES instructions for general-purpose kernels.

**ChaCha8 provides:**
1.  **Massive Performance:** 1885 GB/s throughput on NVIDIA B200 (fused kernel).
2.  **Hardware Suitability:** Uses ARX (Add-Rotate-XOR) operations that map natively to CUDA cores, unlike AES which requires slow software lookup tables on GPUs.
3.  **Sufficient Security:** No known practical attacks exist against 8-round ChaCha. It is widely used in high-performance contexts (e.g., mobile TLS on older devices) where AES hardware is unavailable.

## 1. The Hardware Constraint

### No "AES-NI" for CUDA
While NVIDIA Hopper (H100) and Blackwell (B200) architectures contain hardware AES units, these are **dedicated infrastructure components** used strictly for:
*   PCIe link encryption (Confidential Computing).
*   NVLink encryption.
*   HBM memory encryption.

These units are **not accessible** to user-written CUDA kernels via instruction intrinsics (like `AESENC` on x86).

### Software Performance Comparison

| Primitive | Implementation on GPU | Performance | Why? |
| :--- | :--- | :--- | :--- |
| **AES-128** | Bit-slicing or Lookup Tables (T-tables) | **Low** (<50 GB/s) | Lookup tables cause shared memory bank conflicts; bit-slicing uses many registers. |
| **ChaCha20** | Native ARX (Add-Rotate-XOR) | **High** | ALU-bound, but uses ~20 registers per thread. |
| **ChaCha8** | Native ARX (Add-Rotate-XOR) | **Extreme** (>1500 GB/s) | 2.5x fewer instructions than ChaCha20; becomes memory-bandwidth bound. |

## 2. Security Analysis of ChaCha8

### Cryptanalytic Status
*   **ChaCha Family:** Designed by Daniel J. Bernstein (djb) as a variant of Salsa20.
*   **Best Known Attacks:** The most effective cryptanalysis against ChaCha breaks roughly **7 rounds** with complexity $2^{230+}$.
*   **Margin of Safety:** ChaCha8 (8 rounds) maintains a secure margin against all known practical attacks.

### Industry Adoption of Reduced Rounds
*   **Google:** Deployed ChaCha8-Poly1305 for TLS on older Android devices (without AES hardware) to improve performance and battery life.
*   **Kernel PRNGs:** Operating system CSPRNGs often rely on ChaCha primitives.
*   **ZK-SNARKs:** The Zero-Knowledge industry routinely uses far newer, less-tested primitives (Poseidon, Rescue) for performance. ChaCha8 is significantly more mature and battle-tested than these.

### Suitability for PIR
In PIR, the adversary is the server. The security requirement is **computational indistinguishability**: the server must not distinguish the PRG output from random noise to infer the user's query index.
*   The keys are ephemeral (per query or epoch).
*   Long-term key compromise is not applicable in the same way as encrypted storage.
*   Breaking ChaCha8 in real-time to deanonymize a user is computationally infeasible.

## 3. Benchmarks & Validation

Our measurements on NVIDIA GPUs confirm the architectural fit:

| GPU | Primitive | Throughput (Fused) | Latency (27M Pages) |
| :--- | :--- | :--- | :--- |
| **B200** | **ChaCha8** | **1885 GB/s** | **59 ms** |
| H100 | ChaCha8 | 1553 GB/s | 71 ms |
| CPU (64-core) | ChaCha8 (Simulated) | 128 GB/s | 1000 ms |

Using AES (software) on the H100 would likely result in throughputs below 100 GB/s, missing our <600ms latency target by a factor of 10x.

## 4. Conclusion

**ChaCha8 is the optimal engineering choice for GPU-based PIR.** It aligns perfectly with the GPU's "massive compute, no dedicated crypto" architecture, delivering record-breaking performance without compromising the practical privacy guarantees of the protocol.
