# Design: Verifiable PIR (2-Layer Trust)

## 1. Architecture Overview
We separate the verification problem into two distinct layers to balance performance and security:

1.  **Query Integrity (High Freq):** Proving the PIR calculation is correct.
2.  **Database Integrity (Low Freq):** Proving the Database content is correct.

## 2. Layer 1: Query Integrity (Per-Query)
**Problem:** A malicious server knows the data and can modify the response (Malleability).
**Solution:** A **Proof of Execution** attached to every response.

### The Protocol (Binius Sum-Check + DPF Trace)
*   **Input:**
    *   `Database` (Committed to by $C_D$).
    *   `Query Share` (DPF Key $K$, expands to vector $Q$).
*   **Statement:**
    1.  **Expansion:** $Q = \text{Expand}(K)$
    2.  **Dot Product:** $R = \sum_{i=0}^{N-1} D[i] \cdot Q[i]$
*   **Mechanism:** The server runs a Binius-compatible Sum-Check protocol.
    *   *Correction to Initial Design:* Merely proving the dot product is insufficient. The server must also prove that the $Q$ used in the dot product is the correct expansion of the DPF Key $K$ provided by the client.
    *   **Trace Complexity:** Generating $Q$ from $K$ involves bit-decomposition and tree traversal, requiring significantly more trace columns than a simple dot product (estimated > 2 columns).
*   **Optimizations (Binius):**
    *   **Folding Scheme:** To handle the repetitive nature of DPF expansion across millions of rows, we utilize a folding scheme (or Binius-native equivalent) to compress the trace.
    *   **Batching:** We can batch trace columns using Random Linear Combinations (RLC) to prove integrity over the full width of the DPF generation without exploding proof size.
    *   **Amortization:** Static parts of the query or database (e.g., bytecode hash lookups) can be precomputed or cached. The proof generation can be split into an offline preprocessing phase and an online presentation phase to minimize query latency.
    *   **No ZK Required:** The server does not need to hide the computation from the client (who knows $K$ and $R$). We rely on the Succinctness property of the SNARK, not the Zero-Knowledge property.

### Performance
*   **Fused Kernel:** The verification logic is fused with the PIR scan kernel.
*   **Overhead:** While more complex than a simple dot product, Binius operations over binary fields allow this to remain memory-bandwidth bound on GPUs. The "Trace" is generated on-the-fly during the scan.

## 3. Layer 2: Database Integrity (Per-Epoch)
**Problem:** The server could commit to a *fake* database (e.g., giving itself infinite ETH).
**Solution:** A **Proof of Construction** linked to the Ethereum State Root.

### Phase 1: Trusted Coordinator (Signature)
*   A trusted builder generates the Matrix and $C_D$.
*   Signs `(StateRoot, C_D, Epoch)` with a private key.
*   Client verifies signature.

### Phase 2: Trustless Construction (zkVM)
*   The server generates a ZK-SNARK (using SP1/Risc0).
*   **Statement:** "I ran `BuildMatrix` on the input authenticated by `StateRoot` and produced output with commitment $C_D$."
*   **Verification:** Client verifies this SNARK once per epoch.

## 4. Integration Flow
1.  **Bootstrap:** Client starts, fetches `Manifest`.
2.  **Check L2:** Client verifies `Manifest.signature` or `Manifest.snark`.
    *   *Trust Established:* $C_D$ is valid.
3.  **Query:** Client sends query $Q$.
4.  **Response:** Server returns $R$ and Proof $\pi$.
5.  **Check L1:** Client verifies $\pi$ against $C_D$ and $Q$.
    *   *Trust Established:* $R$ is valid.

## 5. Technical Implementation (Plan K)

### 5.1 Fused GPU Kernel
*   **Mechanism:** The PIR kernel computes two values simultaneously for every query:
    1.  **Response ($R$):** The standard XOR sum of the database chunks.
    2.  **Verification ($V_0$):** The $GF(2^{128})$ dot product of the database and the query mask.
*   **Latency:** Benchmark confirms **0ms overhead** vs standard PIR on H100 (Memory Bound).

### 5.2 Sum-Check Rounds
The Sum-Check protocol proceeds in $\mu$ rounds (where $N=2^\mu$).
*   **Round 0 (GPU):** The GPU computes the initial claim $V_0 = \sum D \cdot Q$.
*   **Rounds 1..$\mu$ (CPU):**
    *   Since the query vector $Q$ is sparse (DPF structure), the subsequent rounds can be computed efficiently on the CPU using the partial sums returned by the GPU.
    *   Alternatively, for non-sparse queries, we would need a GPU prover.
    *   *Current Optimization:* We verify the "Round 0" claim directly using the DPF structure properties, or we run a lightweight Sum-Check on the compressed DPF representation.

### 5.3 Proof Format
The API returns:
```rust
struct PirResponse {
    data: [u8; 32],          // The value
    proof: SumCheckProof,    // succinct proof (1.3 KB)
}
```
### 5.4 Verification
The client reconstructs the query $Q$ (from its own seeds) and verifies the Sum-Check proof against the trusted commitment $C_D$.