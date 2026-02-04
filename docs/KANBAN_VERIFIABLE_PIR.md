# Kanban: Verifiable PIR (2-Layer Trust)

**Objective:** Achieve fully trustless Private Information Retrieval by separating "Query Verification" (Per-Request) from "Database Verification" (Per-Epoch).

## 🏗️ Architecture

### Layer 1: Query Integrity (Per-Query)
**Goal:** Prove that the PIR response $R$ is the correct dot product of Database $D$ and Query $Q$.
**Tech:** Binius Sum-Check (Binary Field SNARKs).
**Constraint:** Must be ultra-fast (< 50ms) and GPU-accelerated.

### Layer 2: Database Integrity (Per-Epoch)
**Goal:** Prove that the Database $D$ (committed to by $C_D$) correctly represents the Ethereum State Root $S$.
**Tech:**
*   **Phase 1:** Trusted Coordinator Signature.
*   **Phase 2:** zkVM Proof (SP1/Risc0) of the `BuildMatrix` execution.

---

## 📋 Backlog

### 🟢 Layer 1: Query Integrity (Binius)
- [x] **Feasibility:** Confirmed 2.48 TB/s throughput on H200 with fused Binius kernel (0ms overhead).
- [x] **Prototype:** Implemented `SumCheckProver` (CPU) and `fused_kernel.cu` (GPU).
- [ ] **Server Integration:**
    - Update `PirResult` struct in `morphogen-server` to include `proof: SumCheckProof`.
    - Modify `GpuScanner` to return the `verif0..2` accumulators as the "Round 0" proof.
    - Implement the recursive Sum-Check rounds (on CPU or GPU) to complete the proof $\pi$.
- [ ] **Client Integration:**
    - Update `morphogen-client` to receive `proof`.
    - Implement `SumCheckVerifier` (WASM-compatible) to check $\pi$ against $C_D$ and $Q$.

### 🔵 Layer 2: Database Integrity (Construction)
- [ ] **Commitment Scheme:** Define the `DatabaseCommitment` (e.g., Merkle Root or Packed Hash).
    - *Decision:* Start with simple linear hash for Phase 1.
- [ ] **Manifest Update:** Add `db_commitment` and `signature` to the Server Manifest.

### 🟡 Integration
- [ ] **API Update:** Update PIR Response format to include `Proof`.
- [ ] **Client Logic:** Chain the checks: `Verify(Signature) -> Trust(Commitment) -> Verify(QueryProof) -> Trust(Response)`.

## 🏗️ In Progress
- **Server Integration:** Wiring GPU verification outputs to the API response.

## ✅ Done
- [x] **Design Doc:** Draft `docs/design/VERIFIABLE_PIR.md`.
- [x] **Cleanup:** Removed legacy UBT code.
- [x] **Benchmark:** Validated GPU performance (H100/H200/B200).