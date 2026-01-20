# Part 3: Engineering Design Document (EDD) v1.1

**Component:** `morphogen-server`
**Target Architecture:** x86_64 (AVX-512 required)
**Rust Version:** `nightly` (for `std::simd` and intrinsics stability)

## 0. Query Modes

The system currently focuses on the **Privacy-Only** mode to maximize performance on available hardware.

```rust
pub enum QueryMode {
    PrivacyOnly,  // 256-byte or 32-byte rows
    // Trustless, // MOVED TO ICEBOX: Requires 2KB rows and multi-GPU cluster
}
```

| Mode | Row Size | Trust Model | Use Case |
|------|----------|-------------|----------|
| **Privacy-Only** | 32-64 bytes | Honest-but-curious | Fast queries, trusted operators |

### 0.1 Client Addressing Requirements

For clients to compute correct query positions, each epoch publishes:

```rust
pub struct EpochMetadata {
    pub epoch_id: u64,
    pub num_rows: usize,                        // Cuckoo table size
    pub seeds: [u64; 3],                        // Hash function seeds
    pub block_root: B256,                       // For consistency checking
}
```

**Stash handling:** At build-time, the server rehashes with new seeds until stash is empty (guaranteed at 85% load). Runtime additions (new accounts) go to Delta buffer at their first candidate position ($h_1$), which Delta-PIR scans automatically.

### 0.2 Icebox: Trustless Mode
**Trustless Mode** (2KB rows with Merkle Proofs) has been moved to the project Icebox due to the hardware requirement of an 8-GPU cluster to hold the ~850GB dataset in VRAM. It remains a valid future direction for fully adversarial environments.

## 1. Core Data Structures

### 1.1 The Aligned Matrix
To prevent AVX-512 faults, the main data buffer must be 64-byte aligned. Standard `Vec<u8>` provides no such guarantee. We will implement a custom `AlignedBuffer`.

```rust
use std::alloc::{alloc, dealloc, Layout};

pub struct AlignedMatrix {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
    layout: Layout,
}

impl AlignedMatrix {
    pub fn new(size_bytes: usize) -> Self {
        // Enforce 64-byte alignment for AVX-512 ZMM registers
        let layout = Layout::from_size_align(size_bytes, 64).unwrap();
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() { handle_alloc_error(layout); }
        
        // Zero-init (important for security)
        unsafe { std::ptr::write_bytes(ptr, 0, size_bytes) };
        
        Self { ptr, len: size_bytes, capacity: size_bytes, layout }
    }
    
    // Unsafe access for the JIT Kernel
    #[inline(always)]
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}
```

### 1.2 The Epoch Snapshot
We use `arc-swap` to manage the global pointer, allowing wait-free updates.

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;

pub struct GlobalState {
    // Atomic pointer to the current active epoch
    current_snapshot: ArcSwap<EpochSnapshot>,
}

pub struct EpochSnapshot {
    pub epoch_id: u64,
    pub matrix: Arc<ChunkedMatrix>, // The Frozen Matrix (Chunked for CoW)
    pub delta: DeltaBuffer,         // Append-only updates
}
```

### 1.3 Optimized Data Layouts (Code Indexing)

To fit the entire Ethereum State (Accounts + Storage, ~1.85 Billion items) into a single GPU's VRAM, we employ two optimized row schemas.

**Verification (Jan 20, 2026):** Scanned 350M Mainnet accounts. **Zero** accounts exceeded 128-bit balance. 16-byte balance storage is lossless.

#### Schema A: Compact (32 Bytes)
**Target:** NVIDIA H100 (80GB VRAM)
**Total Size:** ~60 GB.

| Offset | Size | Field | Notes |
| :--- | :--- | :--- | :--- |
| 0 | 16 | Balance | `uint128` (Safe for all ETH) |
| 16 | 8 | Nonce | `uint64` |
| 24 | 4 | CodeID | Dictionary Index (0=EOA) |
| 28 | 4 | Padding | Reserved (Flags/Version) |

#### Schema B: Full (64 Bytes)
**Target:** NVIDIA H200 (141GB) / B200
**Total Size:** ~118 GB.

| Offset | Size | Field | Notes |
| :--- | :--- | :--- | :--- |
| 0 | 16 | Balance | `uint128` |
| 16 | 8 | Nonce | `uint64` |
| 24 | 32 | CodeHash | Full Keccak Hash (No lookup needed) |
| 56 | 8 | Padding | Reserved |

**Storage Items:** In both schemas, Storage Slots use the same row size.
*   **Key:** Implicit in Cuckoo Index (derived from `Address . SlotKey`).
*   **Value:** 32 bytes (`uint256`).
*   **Compact:** Fits exactly in 32B row.
*   **Full:** Fits in 64B row (32B value + 32B padding).

## 2. The Scan Engines

### 2.1 CPU JIT Engine (AVX-512)
This is the fallback path. We bypass safe iterators to manually unroll loops and manage registers directly.

**Logic:**
1.  Load 64 bytes (512 bits) of Matrix Data.
2.  Generate AES keystream for all 3 queries (`q1, q2, q3`).
3.  Mask data with keystream.
4.  XOR into 3 separate accumulation registers.

```rust
#[target_feature(enable = "avx512f,avx512vl,vaes")]
pub unsafe fn scan_kernel(...)
```

### 2.2 GPU Fused Kernel (CUDA)
This is the primary production path for scale. The entire DPF evaluation, masking, and XOR reduction is fused into a single CUDA kernel to maximize HBM bandwidth utilization.

- **Throughput:** ~1.8 TB/s (on H200/B200).
- **Latency:** ~60ms for 108GB.
- **Batching:** Supports processing multiple queries (up to 16) in a single pass.

## 3. The Delta-PIR Implementation

To avoid the overhead of locking the Delta Buffer, we use a `crossbeam::SegQueue` or a simple `RwLock` (since contention is low, only 1 writer).

```rust
pub fn scan_delta(&self, keys: &[DpfKey; 3], partial_results: &mut [u8]) {
    // Locking here is acceptable because this runs AFTER the heavy main scan
    // and processes very few items (<500).
    let delta = self.delta.read(); 
    
    for (row_idx, diff_data) in delta.iter() {
        for k in 0..3 {
            // Check if DPF evaluates to 1 for this row_idx
            if keys[k].eval_bit(*row_idx) {
                // XOR the diff into the result buffer
                xor_bytes(&mut partial_results[k], diff_data);
            }
        }
    }
}
```

## 4. Hybrid CPU/GPU Architecture

To support high-frequency updates (12s block time) without rebuilding the massive 108GB GPU database constantly, we use a hybrid approach:

1.  **Main Matrix (GPU HBM):**
    *   Holds the "Base Snapshot" (Epoch $N$).
    *   Read-only during queries.
    *   Size: ~108 GB.
2.  **Delta Buffer (CPU RAM):**
    *   Accumulates live updates (writes) for Epoch $N$.
    *   Size: Small (<100 MB).
3.  **Query Execution Flow:**
    *   **Step A (GPU):** `GpuScanner` scans the 108GB Main Matrix. Latency: ~60ms.
    *   **Step B (CPU):** Server scans the `DeltaBuffer`. Latency: <1ms.
    *   **Step C (Merge):** Server XORs the GPU results with the CPU Delta results.
    *   **Consistency:** The `EpochManager` ensures the CPU Delta and GPU Snapshot correspond to the same Epoch ID using optimistic concurrency control.

## 5. Memory Management: Striped Copy-on-Write (CoW)

Instead of relying on OS `fork()` or complex `mmap` hacks, we implement a **Chunked Matrix** in userspace.

### Structure
The 75GB matrix is split into **Chunks** of 1GB.
`pub struct ChunkedMatrix { chunks: Vec<Arc<AlignedMatrix>> }`

### Merge Logic (Background Thread)
When Epoch $N$ ends:
1.  **Identify Dirty Chunks:** The Delta Buffer tells us which Row IDs changed. We calculate which 1GB chunks contain those rows.
2.  **Clone & Patch:**
    * For **Clean Chunks**: We simply `Arc::clone()` the pointer. (Zero cost).
    * For **Dirty Chunks**: We allocate a new 1GB buffer, `memcpy` the old data, and apply the XOR patches.
3.  **Create New Snapshot:** Construct a new `EpochSnapshot` pointing to the mix of old and new chunks.
4.  **Swap:** `global_state.store(new_snapshot)`.

**Why 1GB Chunks?**
* Too small (4KB pages) = Too much metadata overhead.
* Too large (75GB) = Copying takes too long.
* 1GB is the "Goldilocks" zone: Copying 1GB takes ~50ms, well within the 12s block time.

## 5. Network Layer

### 5.1 WebSocket Epoch Streaming

Clients subscribe to epoch updates via persistent WebSocket connection:

```rust
enum WsMessage {
    // Client -> Server
    Subscribe { channel: String },           // "epochs" or "queries"
    Query { keys: [DpfKeyBytes; 3] },        // DPF keys for 3 positions
    
    // Server -> Client  
    EpochUpdate(EpochMetadata),              // ~80 bytes, pushed every ~12s
    QueryResponse { results: [Vec<u8>; 3] }, // 768 bytes (Privacy) or 6KB (Trustless)
}

struct EpochMetadata {
    epoch_id: u64,
    num_rows: usize,
    seeds: [u64; 3],
    block_number: u64,
    state_root: B256,
}
```

**Epoch stream bandwidth:** ~80 bytes every 12 seconds = **~7 bytes/sec**

### 5.2 Query Protocol

```
Client                     Server A                    Server B
   │                          │                            │
   │── Query { keys_a } ──────►                            │
   │                          │                            │
   │──────────────────── Query { keys_b } ─────────────────►
   │                          │                            │
   │◄─ QueryResponse ─────────│                            │
   │                          │                            │
   │◄───────────────── QueryResponse ──────────────────────│
   │                          │                            │
   └──── XOR responses locally ────►  payload
```

### 5.3 Connection Management

- **Primary:** WebSocket (persistent, low latency)
- **Fallback:** HTTP/gRPC (stateless, firewall-friendly)
- **TLS:** Required for all connections
- **Reconnect:** Auto-reconnect with exponential backoff

## 6. Crate Dependencies

| Crate | Purpose |
| :--- | :--- |
| `std::arch` | AVX-512 and VAES intrinsics. |
| `arc-swap` | Lock-free replacement of the global Epoch pointer. |
| `rayon` | Parallelizing chunk processing across cores. |
| `rand` | Key generation for AesDpfKey. |
| `tokio-tungstenite` | WebSocket server/client. |
| `ubt` | Handling the Unified Binary Tree and Merkle Proof generation. |
| `ethereum-types` | For RLP encoding and Primitives (`H256`, `Address`). |

## 7. Benchmark Results (Jan 18, 2026)

### 7.1 Test Environment
- **Server:** AMD EPYC 9375F 32-Core (64 threads)
- **RAM:** 1.1 TB DDR5
- **Throughput:** 393 GB/s (8-row unroll + rayon + AesDpfKey)

### 7.2 Query Mode Performance

| Mode | Row Size | Matrix (78M @ 85%) | Scan Time | Concurrent Clients (<600ms) |
|------|----------|-------------------|-----------|----------------------------|
| **Privacy-Only** | 256 bytes | 22 GB | **~66ms** | **~9** |
| Trustless | 2 KB | 175 GB | ~439ms | 1 |

### 7.3 Cuckoo Hash Table Performance

| Load Factor | Table Size (78M accounts) | Status |
|-------------|---------------------------|--------|
| 50% (naive deterministic) | 156M rows | Suboptimal |
| **85% (random-walk)** | **92M rows** | Production |
| 91.8% (theoretical limit) | 85M rows | Stash overflow |

### 7.4 Production Implications (Privacy-Only Mode)
- **Single client:** 66ms per query (9.1x under 600ms target)
- **Concurrent clients:** ~9 clients can be served within 600ms
- **Scaling strategy:** Read replicas for even higher concurrency
- **Cuckoo overhead:** 1.18x (vs 2x with naive 50% load factor)