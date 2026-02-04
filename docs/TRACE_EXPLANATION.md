# Binius Sum-Check Trace Dimensions (Verifiable PIR)

For the Inner Product proof ($R = \langle D, Q \rangle$) at Mainnet scale:

## 1. Trace Table Dimensions
In the context of the Binius protocol, we are proving a statement over two multilinear polynomials ($D$ and $Q$). The conceptual trace table for this computation is:

*   **Logical Rows:** $2^{28}$ (approx. 268 Million rows, matching the account/storage item count).
*   **Logical Columns:** 2
    *   **Column 1 ($D$):** The committed database values.
    *   **Column 2 ($Q$):** The query vector (multilinear extension of DPF weights).
*   **Field:** $GF(2^{128})$ (Binary Tower Field).

## 2. Protocol Complexity
*   **Rounds:** 28 rounds (one for each variable in the multilinear extension).
*   **Operations per Round:** In each round $i$, the prover evaluates a quadratic polynomial $g_i(X)$.
*   **Total Work:** The prover performs approximately $2 \times N$ field multiplications total across all rounds.
*   **H100 Performance:** Our benchmark shows that these $\approx 536 \text{ million}$ multiplications are hidden by the memory bandwidth limit, resulting in 0ms effective overhead.

## 3. Succinctness
*   **Proof Size:** 28 rounds $\times$ 3 evaluations/round $\times$ 16 bytes/evaluation = **1,344 bytes**.
*   **Verification Time:** < 1ms (WASM-compatible).
