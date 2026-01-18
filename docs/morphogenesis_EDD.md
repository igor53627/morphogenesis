# Part 3: Engineering Design Document (EDD) v1.1

**Component:** `morphogen-server`
**Target Architecture:** x86_64 (AVX-512 required)
**Rust Version:** `nightly` (for `std::simd` and intrinsics stability)

## 0. Query Modes

The system supports two security modes with different row sizes:

```rust
pub enum QueryMode {
    PrivacyOnly,  // 256-byte rows, no UBT proof
    Trustless,    // 2KB rows, includes UBT Merkle proof
}
```

| Mode | Row Size | Trust Model | Use Case |
|------|----------|-------------|----------|
| **Privacy-Only** (default) | 256 bytes | Honest-but-curious | Fast queries, trusted operators |
| **Trustless** | 2 KB | Fully adversarial | Full verification against block headers |

### 0.1 Client Addressing Requirements

For clients to compute correct query positions, each epoch publishes:

```rust
pub struct EpochMetadata {
    pub epoch_id: u64,
    pub num_rows: usize,                        // Cuckoo table size
    pub seeds: [u64; 3],                        // Hash function seeds
    pub block_root: B256,                       // For Trustless mode verification
}
```

**Stash handling:** At build-time, the server rehashes with new seeds until stash is empty (guaranteed at 85% load). Runtime additions (new accounts) go to Delta buffer at their first candidate position ($h_1$), which Delta-PIR scans automatically.

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

## 2. The JIT Interference Engine (Unsafe Kernel)

This is the critical path. We bypass safe iterators to manually unroll loops and manage registers directly.



**Logic:**
1.  Load 64 bytes (512 bits) of Matrix Data.
2.  Generate AES keystream for all 3 queries (`q1, q2, q3`).
3.  Mask data with keystream.
4.  XOR into 3 separate accumulation registers.

```rust
#[target_feature(enable = "avx512f,avx512vl,vaes")]
pub unsafe fn scan_kernel(
    matrix_ptr: *const u8,
    num_rows: usize,
    keys: &[DpfKey; 3] // The 3 queries
) -> [u8; 1024] { // Returns 3 x 1KB payloads
    
    // 1. Initialize Accumulators (Zeroed ZMM registers)
    let mut acc_1 = _mm512_setzero_si512();
    let mut acc_2 = _mm512_setzero_si512();
    let mut acc_3 = _mm512_setzero_si512();

    // 2. The Hot Loop (Unrolled)
    // We process the matrix in 64-byte strides
    for i in (0..num_rows).step_by(64) {
        // A. Load Data (Bottleneck)
        let data_vec = _mm512_load_si512(matrix_ptr.add(i) as *const _);

        // B. AES Expansion (Pipelined)
        // Note: Actual AES-NI logic omitted for brevity, but happens here.
        // We generate 512 bits of pseudorandomness for each key.
        let stream_1 = aes_expand(keys[0], i); 
        let stream_2 = aes_expand(keys[1], i);
        let stream_3 = aes_expand(keys[2], i);

        // C. Conditional Accumulation (AND + XOR)
        // If the stream bit is 1, we XOR the data.
        acc_1 = _mm512_xor_si512(acc_1, _mm512_and_si512(data_vec, stream_1));
        acc_2 = _mm512_xor_si512(acc_2, _mm512_and_si512(data_vec, stream_2));
        acc_3 = _mm512_xor_si512(acc_3, _mm512_and_si512(data_vec, stream_3));
    }

    // 3. Finalize
    // Write registers back to stack-allocated buffer
    let mut result = [0u8; 1024];
    // ... store logic ...
    result
}
```

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

## 4. Memory Management: Striped Copy-on-Write (CoW)

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