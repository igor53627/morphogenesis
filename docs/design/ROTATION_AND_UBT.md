# Design: Cuckoo Seed Rotation & UBT Integration (Major Epochs)

## Problem Statement
To prevent query linkability (privacy erosion over time), the mapping between Ethereum Accounts and PIR Database Indices must change periodically. This requires rotating the Cuckoo Hash seeds.
However, changing seeds reshuffles the entire 108GB database globally.
This cannot be handled by the `DeltaBuffer` (Minor Epoch) mechanism, which assumes stable indices.

## Proposed Architecture: Major vs. Minor Epochs

We introduce a distinction between update types:

1.  **Minor Epoch (Delta Update):**
    *   **Trigger:** Every block (~12s).
    *   **Source:** `ubt-exex` provides `(RowIndex, XOR_Diff)`.
    *   **Action:** In-place XOR update on CPU and GPU.
    *   **Cost:** Low latency (<100ms).

2.  **Major Epoch (Seed Rotation):**
    *   **Trigger:** Configurable interval (e.g., 24 hours).
    *   **Source:** `ubt-exex` rebuilds the **Unified Binary Tree (UBT)** and generates a full 108GB Cuckoo Matrix with *new seeds*.
    *   **Action:** Full state swap.
    *   **Cost:** High bandwidth (108GB upload), requires coordination.

## UBT Integration (`ubt-exex`)

The `ubt-exex` service (external) is responsible for:
1.  Ingesting Reth state updates.
2.  Maintaining the UBT structure to generate Merkle inclusion proofs (for Trustless Mode).
3.  Generating the flat Cuckoo Matrix.
4.  Pushing updates to `morphogen-server`.

### Interface
`morphogen-server` will expose a privileged control API:
*   `POST /control/snapshot`: Upload/Signal availability of a new full matrix + seeds.

## Zero-Downtime Rotation (The 108GB VRAM Constraints)

A single H200 (141GB VRAM) holding a 108GB database cannot allocate a second 108GB buffer for the new epoch (Total 216GB > 141GB).

### Strategy A: Blue/Green GPUs (Recommended)
Requires 2x GPUs (e.g., 2x H100 or 2x H200).
1.  **Steady State:** GPU 0 serves Epoch $N$. GPU 1 is idle/standby.
2.  **Rotation Start:** Server loads Epoch $N+1$ into GPU 1.
3.  **Switch:** Server updates global routing to point to GPU 1.
4.  **Cleanup:** GPU 0 drops Epoch $N$ and becomes standby.

### Strategy B: Maintenance Window (Single GPU)
1.  **Lock:** Server pauses query processing.
2.  **Drop:** Drop Old Matrix (freeing VRAM).
3.  **Upload:** Upload New Matrix (~60s over PCIe Gen5).
4.  **Unlock:** Resume queries.
*Impact:* 1-minute downtime per day. Acceptable for some use cases, but not high-availability.

### Strategy C: Host-Memory Paging (Not Recommended)
Use Unified Memory to oversubscribe VRAM.
*Impact:* Performance tanks during rotation.

## Implementation Plan (Morphogen Server)

1.  **EpochManager:** Add `submit_snapshot(new_matrix, new_seeds)` method.
2.  **API:** Add control endpoint.
3.  **GPU Manager:** Implement "Matrix Swap" logic.
    *   If 2 GPUs available: Load to secondary, then swap handle.
    *   If 1 GPU available: Drop then Load (Downtime).

## Data Flow

```mermaid
graph TD
    Reth[Reth Node] -->|State Changes| UBT[ubt-exex Service]
    UBT -->|Deltas (Block 1..N)| Server[Morphogen Server]
    UBT -->|Full Snapshot (Day 1)| Server
    
    subgraph Server
        EM[Epoch Manager]
        GPU[GPU Worker]
    end
    
    Server -->|Queries| Client
```
