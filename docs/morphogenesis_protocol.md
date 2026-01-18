# Morphogenesis Project Documentation

**Version:** 3.3
**Date:** January 18, 2026
**Status:** Ready for Implementation

---

# Part 1: Product Requirements Document (PRD) v3.3

**Project Name:** Morphogenesis PIR
**Target Platform:** Ethereum (Light Clients / Helios)
**Implementation Language:** Rust (`nightly` for AVX-512/VAES)

## 1. Executive Summary
Morphogenesis v3.3 is a 2-Server DPF-PIR protocol enabling private, stateless access to Ethereum state. It utilizes **Parallel Cuckoo Retrieval** to eliminate access-pattern leakage and **Epoch-Based Delta-PIR** with **Copy-on-Write (CoW)** memory management to guarantee wait-free snapshot isolation during live updates without doubling RAM requirements.

### 1.1 Query Modes
The protocol supports two security modes:

| Mode | Trust Model | Row Size | Use Case |
|------|-------------|----------|----------|
| **Privacy-Only** (default) | Honest-but-curious | 256 bytes | Fast queries, trusted operators |
| **Trustless** | Fully adversarial | 2 KB | Full verification with UBT proofs |

**Privacy-Only Mode:** Servers are trusted not to lie about data, but client hides which account is queried. No Merkle proof overhead.

**Trustless Mode:** Full UBT Merkle proofs included for verification against block headers. Suitable for untrusted server operators.

## 2. System Architecture

### 2.1 Client Requirements

**Epoch Metadata (streamed via WebSocket, ~80 bytes per epoch):**
```rust
struct EpochMetadata {
    epoch_id: u64,
    num_rows: usize,          // Cuckoo table size (~92M for 78M accounts)
    seeds: [u64; 3],          // Hash function seeds (24 bytes)
    block_number: u64,        // Epoch's block reference
    state_root: B256,         // For Trustless mode verification
}
```

**Per-Query Bandwidth:**
| Mode | Upload (DPF keys) | Download (responses) | Total |
|------|-------------------|---------------------|-------|
| Privacy-Only | 2 × 150 bytes | 2 × 768 bytes | ~1.8 KB |
| Trustless | 2 × 150 bytes | 2 × 6 KB | ~12.3 KB |

**Client computes locally:**
- Cuckoo positions: `[h1, h2, h3] = hash(account_key, seeds) % num_rows`
- DPF key pairs for each position

### 2.2 Storage Model: The "Epoch-Sharded" Matrix
To achieve sub-second latency on a 300GB state, we partition the global state into shards.
* **Cluster:** 4 Nodes per Server (Total 8 Nodes).
* **Storage Structure:**
    * **Frozen Matrix ($M_e$):** Immutable snapshot for Epoch $e$.
    * **Delta Buffer ($\Delta_e$):** Append-only log of updates for Epoch $e$.

### 2.3 Components
1.  **`morphogen-server`**:
    * **Epoch Manager:** Handles the lifecycle (Active $\to$ Merge $\to$ Reclaim) using CoW.
    * **JIT Engine:** AVX-512 kernel scanning $M_e$ and $\Delta_e$ in constant time.
2.  **`morphogen-client`**:
    * **Parallel Addresser:** Generates 3 simultaneous queries per request.
    * **Integrity Verifier:** Reconstructs payload and verifies UBT Merkle Proofs.

## 3. Functional Requirements

### FR-01: Addressing & Privacy
* **Strategy:** **Parallel Cuckoo Retrieval**.
    * Client **MUST** generate 3 independent DPF queries ($Q_1, Q_2, Q_3$).
    * **Constraint:** Queries must be sent simultaneously as a fixed batch.
    * **Cuckoo Load Factor:** 85% achieved via random-walk insertion (1.18x overhead vs 2x with naive).
    * **Client Requirements:** Epoch metadata (seeds, num_rows) required for address computation.
* **Leakage Mitigation:** Server scan loop must be **Constant-Time (O(N))**, executing identical instruction sequences regardless of key values.
* **Stash Handling:** Build-time rehash ensures stash=0. Runtime additions use Delta-PIR at $h_1$ position.

### FR-02: Concurrency (Epoch-Based Delta-PIR)
* **Consistency Model:** **Wait-Free Snapshot Isolation**.
    * Readers access Snapshot $S_e = M_e \oplus \Delta_e$.
    * Writers buffer updates into $\Delta_{pending}$.
* **Client Experience:** **Zero-Retries.** The client always receives a valid snapshot of Epoch $e$.

### FR-03: Memory Management (The Merge Lifecycle)
* **Strategy:** **Striped Copy-on-Write (CoW)**.
    * The system **MUST NOT** perform a full 300GB `memcpy` every 12 seconds.
    * **Merge Worker:** Identifies the specific 64-byte stripes modified in the pending Delta. It allocates a new logical view $M_{e+1}$ that points to:
        * Old memory pages for unmodified stripes (Shared).
        * New memory pages for modified stripes (Unique).
* **Reclamation:**
    * Old Epochs ($M_e$) are ref-counted.
    * Once the last query for Epoch $e$ completes, the specific memory pages belonging to $\Delta_e$ and the unique CoW patches are freed.

### FR-04: JIT Interference Engine
* **Execution Path:**
    1.  `Scan_Main_Matrix(M)`: AVX-512 Linear Scan.
    2.  `Scan_Delta_Buffer($\Delta$)`: Linear Scan of the update list.
    3.  `Combine`: $Result = Result_M \oplus Result_\Delta$.
* **Performance:** Delta scan overhead must be negligible (< 0.5ms).

### FR-05: Helios Integration
* **Payload:** `[AccountData | UBT_Proof]`.
* **Verification:** Client validates the proof against the Block Header corresponding to the *Server's Epoch* (returned in metadata).

## 4. Performance Targets
* **Latency:** < 0.6 seconds (p95) using 4-shard architecture.
* **Throughput:** Saturate cluster memory bandwidth (> 560 GB/s).
* **Hardware:** 4x `r6i.16xlarge` (512GB RAM each) per logical server.

## 5. Benchmark Results (Jan 18, 2026)

### 5.1 Single-Node Performance (AMD EPYC 9375F, 64 threads)

**Privacy-Only Mode (default, 256-byte rows):**
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Scan throughput | 140 GB/s | 393 GB/s | 2.8x target |
| Matrix size (78M @ 85% Cuckoo) | - | 22 GB | - |
| Single query latency | <600ms | **~66ms** | 9.1x target |

**Trustless Mode (2KB rows with UBT proofs):**
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Scan throughput | 140 GB/s | 393 GB/s | 2.8x target |
| Matrix size (78M @ 85% Cuckoo) | - | 175 GB | - |
| Single query latency | <600ms | **~439ms** | 1.4x target |

### 5.2 Cuckoo Hash Table Load Factor
| Load Factor | Table Size (78M) | Status |
|-------------|------------------|--------|
| 50% (naive deterministic) | 156M rows | Suboptimal |
| **85% (random-walk)** | **92M rows** | Production |
| 91.8% (theoretical limit) | 85M rows | Stash overflow |

### 5.3 Query Mode Comparison
| Mode | Row Size | Matrix (78M) | Scan Time | Concurrent Clients (<600ms) |
|------|----------|--------------|-----------|----------------------------|
| **Privacy-Only** | 256 bytes | 22 GB | ~66ms | **~9** |
| Trustless | 2 KB | 175 GB | ~439ms | 1 |

### 5.4 Projected 4-Shard Cluster Performance (Privacy-Only)
| Metric | Target | Projected | Status |
|--------|--------|-----------|--------|
| Cluster bandwidth | 560 GB/s | 1,572 GB/s | 2.8x target |
| 88GB full scan (4x22GB) | <600ms | ~66ms | [PASS] |
| Concurrent clients | 1 | ~9 | [PASS] |