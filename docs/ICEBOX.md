# Project Icebox

Features and architectures that have been deprioritized or archived in favor of the **Privacy-Only GPU** path.

## 1. Trustless Mode (Verifiable PIR)

**Goal:** Allow clients to cryptographically verify that the PIR response matches the Ethereum State Root using Merkle Proofs.

**Architecture:**
- **Row Size:** 2 KB (vs 32 bytes for Privacy-Only).
- **Structure:** Each row contains the account data *plus* a Merkle inclusion proof.
- **UBT (Unified Binary Tree):** A static Merkle tree over the Cuckoo table buckets.
- **Client:** Requires a light client (e.g., Helios) to verify the proof against the block header.

**Why Iceboxed?**
- **Hardware Cost:** The dataset size expands from ~60GB to **~3.7 TB**. This requires an 8-GPU H100 cluster (~$30k/month) vs a single H100.
- **Latency:** Scan time increases by ~8x.
- **User Demand:** Most users prioritize privacy (IP protection) over trustlessness (they already trust Infura/RPCs today).

## 2. CPU JIT Engine (AVX-512)

**Goal:** Run PIR on standard commodity servers (AMD EPYC) without GPUs.

**Architecture:**
- **SIMD:** Hand-written AVX-512 assembly/intrinsics to evaluate AES-DPF.
- **Throughput:** Achieved ~16-20 GB/s on 64 cores.
- **Latency:** ~6.6 seconds for Mainnet scale.

**Why Iceboxed?**
- **Performance Gap:** GPUs provide **1,300 GB/s** (75x faster).
- **UX:** 6.6s latency is too slow for a wallet backend.
- **Cost/Query:** GPU is more cost-effective per query at scale.

## 3. "ExEx" Live Streaming (Phase 1 Idea)

**Goal:** Stream updates directly from Reth into the PIR Server memory via an ExEx (Execution Extension) channel.

**Status:**
- Replaced by the **Hybrid Architecture** (Static Matrix on R2 + Delta RPC).
- We now treat the Matrix as a "Static Snapshot" that rotates daily/weekly, rather than a continuously mutating live structure.
- **Reason:** Complexity of managing live Cuckoo displacements in real-time on GPU memory.
