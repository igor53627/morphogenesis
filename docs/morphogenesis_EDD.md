# Part 3: Engineering Design Document (EDD) v2.0

**Component:** `morphogen-server`
**Target Architecture:** NVIDIA H100 GPU (CUDA)
**Host:** x86_64 Linux

## 1. System Architecture

Morphogenesis employs a **Hybrid CPU/GPU Architecture** to solve the twin challenges of massive scale (1.8B items) and high-frequency updates (12s block time).

### 1.1 The "Privacy-Only" Model
We prioritize **Metadata Privacy** (hiding *who* you query) over Trustlessness.
- **Row Size:** 32 bytes (Compact Schema).
- **Verification:** Clients trust the server for data integrity but rely on PIR for privacy.
- **Data:** Accounts and Storage slots are mixed into a single address space.

*For details on Data Layouts, see [`DATA_STRUCTURES.md`](./DATA_STRUCTURES.md).*
*For archived Trustless/CPU modes, see [`ICEBOX.md`](./ICEBOX.md).*

### 1.2 The Hybrid Pipeline

1.  **Main Matrix (GPU HBM):**
    *   Holds the **Static Snapshot** (Epoch $N$).
    *   Size: ~68.8 GB (Full Mainnet).
    *   Read-only during queries.
    *   Resident in H100 VRAM (80GB).

2.  **Delta Buffer (CPU RAM):**
    *   Accumulates live updates (writes) for Epoch $N$.
    *   Size: Small (<100 MB).
    *   Reset on Epoch Rotation.

3.  **Query Execution Flow:**
    *   **Step A (GPU):** `GpuScanner` scans the 68GB Main Matrix. (~53ms).
    *   **Step B (CPU):** Server scans the `DeltaBuffer`. (<1ms).
    *   **Step C (Merge):** Server XORs the GPU results with the CPU Delta results.

## 2. Memory Management (GpuPageMatrix)

To maximize memory bandwidth (~3.35 TB/s on H100), we bypass standard array layouts.

### Paged Storage
The Matrix is stored as a collection of **4KB Pages**.
- **Alignment:** Data loaded from disk is padded to 4KB.
- **Access:** The fused CUDA kernel loads 128-bit vectors from these pages based on the DPF index.
- **Loading:** We use direct chunked loading (`from_chunked_matrix`) to stream data from disk to VRAM without double-buffering in RAM.

## 3. The Scan Engines

### 3.1 GPU Fused Kernel (CUDA)
The primary engine.
- **Logic:** Fuses DPF evaluation (AES-based PRG), masking, and XOR accumulation into a single kernel.
- **Performance:** **1,300 GB/s** effective throughput.
- **Latency:** **53ms** for 68GB.

### 3.2 Delta-PIR (CPU)
Scanning the small DeltaBuffer on the CPU.
- **Concurrency:** Uses `RwLock` and atomic epoch markers to ensure wait-free reads for the scanner.
- **Consistency:** The `GlobalState` ensures the CPU Delta and GPU Snapshot correspond to the same Epoch ID.

## 4. Epoch Management & Rotation

The system treats the Mainnet state as a sequence of **Static Snapshots** (Major Epochs).

1.  **Rotation:** A new snapshot (Matrix) is generated offline (via `reth-adapter`).
2.  **Upload:** Uploaded to Object Storage (R2).
3.  **Hot Swap:** The server downloads the new matrix, loads it into a *secondary* GPU buffer (if space permits) or performs a quick swap.
    *   *Note:* Single H100 cannot hold two 68GB matrices. Rotation requires a brief maintenance window or a Blue/Green deployment.

## 5. Network Layer

### 5.1 WebSocket Epoch Streaming
Clients subscribe to epoch updates via persistent WebSocket connection:
- Server pushes `EpochMetadata` (Seeds, Epoch ID).
- Client updates its Cuckoo addressing logic.

### 5.2 Query Protocol
Standard 2-Server PIR setup.
- Client sends `Query { keys_a }` to Server A and `Query { keys_b }` to Server B.
- Servers respond with `QueryResponse { payloads }`.
- Client XORs payloads to recover the 32-byte row.

## 6. Code Serving (Sidecar)

Since the PIR row (32 bytes) is too small for contract code, we use a **Sidecar Architecture**.
1.  **PIR Query:** Returns `Balance`, `Nonce`, and `CodeID`.
2.  **Lookaside:** Client resolves `CodeID` -> `CodeHash` via a dictionary.
3.  **Fetch:** Client fetches bytecode from a static CAS (CDN).

*See [`design/CODE_SERVING.md`](./design/CODE_SERVING.md) for details.*