# Cryptography & Core Mechanics of Morphogenesis PIR

Morphogenesis implements a highly optimized **2-Party Private Information Retrieval (PIR)** system designed to scale to the full Ethereum state (108GB+). It combines theoretical cryptographic primitives with GPU-friendly engineering choices to achieve sub-100ms latency.

## 1. The "Full Scan" Explained

A "Full Scan" is the core operation performed by the server for every query. It ensures privacy by touching **every single byte** of the database, making it impossible for the server to know which specific record was accessed via memory access patterns.

On an NVIDIA B200 GPU, this scan takes **~66 milliseconds** for 108GB.

### The Pipeline (Fused Kernel)
The GPU executes four fused steps in a single pass over memory:

1.  **3-Key DPF Evaluation (Compute):**
    For every one of the 27,000,000 pages, the GPU evaluates 3 independent Distributed Point Function (DPF) keys. This involves traversing a GGM tree 25 levels deep using the **ChaCha8 PRG** to determine if the current page index matches one of the user's queries.

2.  **Global Memory Streaming (I/O):**
    The GPU pulls the entire 108GB database from HBM3e memory into its CUDA cores using vectorized 128-bit loads (`uint4`), saturating the memory bus at **1.66 TB/s**.

3.  **Fused Masking (Filter):**
    Each 4KB page is "masked" (bitwise AND) by the 16-byte DPF outputs. If the DPF indicates the page is not part of the query, it is effectively zeroed out.

4.  **Shared Memory & Reduction (Aggregation):**
    Results are accumulated in fast Shared Memory (L1 cache) and then atomically XORed into a global result buffer. The final output is just **12KB** (3 pages of 4KB), which is sent back to the client.

---

## 2. Cryptographic Primitives

### Distributed Point Functions (DPF)
Based on the **BGI (Boyle-Gilboa-Ishai) 2014** construction.
*   **Purpose:** Allows a client to split a query for index $x$ into two secret keys, $K_A$ and $K_B$.
*   **Property:** Each key expands to a pseudorandom stream. However, $Expand(K_A) \oplus Expand(K_B) = \delta_x$ (a vector that is 0 everywhere except at index $x$, where it is 1).
*   **Privacy:** Neither server can distinguish its key from a random string.

### ChaCha8 (Pseudorandom Generator)
The DPF requires a PRG to expand seed values. We use **ChaCha8**.
*   **Why Not AES?** NVIDIA GPUs (Hopper/Blackwell) do not expose hardware AES instructions to user kernels. Software AES is too slow.
*   **The Choice:** ChaCha8 uses ARX (Add-Rotate-XOR) operations which map natively to GPU ALUs. This design choice enables the system to be memory-bound (1.66 TB/s) rather than compute-bound.
*   **Security:** ChaCha8 is widely used in performance-critical applications (e.g., mobile TLS) and has no known practical attacks.

### Additive Secret Sharing (XOR)
The protocol relies on the homomorphic property of XOR ($\oplus$).
*   **Reconstruction:**
    $$Result = (DB \cdot Mask_A) \oplus (DB \cdot Mask_B)$$
    $$Result = DB \cdot (Mask_A \oplus Mask_B)$$
    $$Result = DB \cdot \delta_x = DB[x]$$
*   Because XOR is their own inverse, the "noise" introduced by the two servers cancels out perfectly, revealing only the requested data.

### Cuckoo Hashing (Addressing)
To map sparse Ethereum accounts (20-byte addresses) to a dense matrix:
*   **Mapping:** Accounts are inserted into a Cuckoo Hash Table with 3 hash functions.
*   **Querying:** A client searches for an account by querying all 3 possible locations simultaneously.
*   **Efficiency:** This allows us to achieve high storage density (~85-90% load factor) while guaranteeing constant-time lookups ($O(1)$) with only 3 parallel queries.
