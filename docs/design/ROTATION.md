# Design: Cuckoo Seed Rotation (Major Epochs)

## Problem Statement
To prevent query linkability (privacy erosion over time), the mapping between Ethereum Accounts and PIR Database Indices must change periodically. This requires rotating the Cuckoo Hash seeds.
Changing seeds reshuffles the entire database globally. This constitutes a **Major Epoch**.

## Architecture: Major vs. Minor Epochs

1.  **Minor Epoch (Delta Update):**
    *   **Trigger:** Every block (~12s).
    *   **Action:** In-place XOR update on CPU DeltaBuffer.
    *   **Source:** Real-time RPC from `reth-adapter`.

2.  **Major Epoch (Seed Rotation):**
    *   **Trigger:** Daily / Weekly.
    *   **Source:** `reth-adapter` generates a new `mainnet_compact.bin` offline with new seeds.
    *   **Action:** Full state swap on the GPU.

## Rotation Workflow

1.  **Generation:**
    *   `reth-adapter` runs on a high-memory server (CPU).
    *   Generates `matrix_epoch_N.bin` (60GB) with new random seeds.
    *   Uploads to R2 (`pir/matrix/epoch_N/`).

2.  **Notification:**
    *   Operator calls `POST /admin/snapshot` on `morphogen-server` with the new R2 URL.

3.  **Deployment (Zero-Downtime Strategy):**
    *   **Background:** Server downloads the new matrix to a local NVMe volume.
    *   **Swap:**
        *   *Multi-GPU:* Load new matrix to GPU 1. Switch traffic. Drop GPU 0.
        *   *Single-GPU:* Requires a maintenance window (~60s) to drop the old matrix and load the new one into VRAM.

## State Management

The `GlobalState` tracks the current active epoch.
When a Major Epoch switches:
1.  `EpochSnapshot` is replaced with the new matrix.
2.  `DeltaBuffer` is **cleared** (reset to empty).
3.  Clients are notified via WebSocket to fetch new seeds/metadata.