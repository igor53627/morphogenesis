# Morphogenesis Protocol Specification v4.0

**Status:** Production Ready
**Date:** Jan 21, 2026
**Target:** Single-Server GPU PIR (NVIDIA H100)

## 1. Overview

Morphogenesis is a **Private Information Retrieval (PIR)** protocol for the Ethereum State.
It allows a client to query the balance and nonce of *any* Ethereum address without revealing the target address to the server.

### Key Features
*   **Privacy:** The server learns *nothing* about the query target (Metadata Privacy).
*   **Scale:** Supports the full Mainnet state (1.85 Billion items).
*   **Latency:** **53ms** query time (faster than many non-private RPCs).
*   **Simplicity:** Runs on a single commodity GPU server (H100).

## 2. Architecture

### 2.1 The Data Model
The database is a flattened, linearized **Cuckoo Hash Table** stored in GPU VRAM.
*   **Capacity:** 2.15 Billion Rows.
*   **Row Size:** 32 Bytes (Compact Schema).
*   **Total Size:** 68.8 GB.

### 2.2 The Query Flow
1.  **Addressing:** Client locally computes 3 indices $[h_1, h_2, h_3]$ using the Cuckoo seeds.
2.  **DPF Generation:** Client generates Distributed Point Function keys for these indices.
3.  **Request:** Client sends keys to 2 non-colluding servers.
4.  **Scan:** Servers scan the **entire 68GB matrix** using a fused AES+XOR CUDA kernel.
5.  **Response:** Servers return 3 encrypted blocks.
6.  **Reconstruction:** Client XORs the responses to recover the row.

### 2.3 Code Resolution (Sidecar)
Since the 32-byte row is too small for contract bytecode, we use a sidecar CAS.
1.  PIR query returns `Balance`, `Nonce`, and `CodeID`.
2.  Client resolves `CodeID` $	o$ `CodeHash` via a public Dictionary (HTTP Range Request).
3.  Client fetches bytecode from a public Content Addressable Storage (CAS).

## 3. Protocol Parameters

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Domain** | $2^{32}$ | Covers 2.15B rows |
| **Row Size** | 32 Bytes | Balance (16), Nonce (8), CodeID (4) |
| **Hash Functions** | 3 | Salted SipHash |
| **Load Factor** | 85% | Minimizes stash probability |

## 4. Updates & Consistency

### 4.1 Minor Epochs (Real-Time)
*   **Frequency:** Every block (12s).
*   **Mechanism:** Delta-PIR.
*   **Implementation:** Client query includes a "Pending Epoch" ID. Server XORs the main matrix result with a CPU-resident Delta Buffer containing recent updates.

### 4.2 Major Epochs (Rotation)
*   **Frequency:** Daily/Weekly.
*   **Mechanism:** Full Matrix Swap.
*   **Purpose:** Rotate Cuckoo seeds to prevent long-term statistical leakage.
*   **Implementation:** Offline ETL generates new matrix -> Upload to R2 -> Server hot-swaps.

## 5. Security Model

**Trust Assumption:** 2-Server Semi-Honest.
*   The two servers do not collude to combine queries.
*   The servers follow the protocol (they execute the scan correctly).
*   **Privacy:** Information-Theoretic (IT-PIR).
*   **Integrity:** Trusted Server (Privacy-Only).

*Note: Verifiable integrity (Trustless Mode) is currently iceboxed.*
