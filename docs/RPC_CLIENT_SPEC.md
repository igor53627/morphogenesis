# Morphogenesis RPC Adapter Specification

**Status:** Draft (v0.1)
**Scope:** Client-side Privacy Gateway

## 1. Overview

The **Morphogenesis RPC Adapter** is a client-side middleware that enables existing Ethereum wallets (MetaMask, Rabby, Frame) and dApps to utilize Morphogenesis PIR without code changes. It acts as a local proxy, intercepting specific JSON-RPC calls and translating them into privacy-preserving DPF queries.

### Deployment Models
1.  **Local Proxy:** A lightweight background process (Rust/Go) running on `localhost:8545`.
2.  **In-Browser:** A WASM-based library integrated directly into Wallet Extensions.

---

## 2. Architecture

```mermaid
graph LR
    Wallet[Wallet / dApp] -->|JSON-RPC| Adapter[RPC Adapter]
    
    subgraph "RPC Adapter"
        Router[Request Router]
        PIR[PIR Engine]
        EVM[Local EVM (Future)]
        CAS[CAS Resolver]
    end
    
    Adapter -->|Standard RPC (Passthrough)| Infura[Public Node]
    Adapter -->|PIR Query (Privacy)| PIR_Server[Morphogenesis Cluster]
    Adapter -->|Static Content| CDN[Code/Storage CAS]
```

### 2.1 Request Routing
The adapter inspects the JSON-RPC method:
*   **Private Methods (`eth_getBalance`, `eth_getTransactionCount`, `eth_getCode`):** Intercepted and routed to the **PIR Engine**.
*   **Public Methods (`eth_blockNumber`, `eth_sendRawTransaction`, `eth_estimateGas`):** Passed through to a standard upstream provider (e.g., Infura/Alchemy) as they do not leak state-specific targets (though `sendRawTransaction` leaks intent, which is out of scope for *read* privacy).

---

## 3. Phase 1: Account State (Current)

Support for basic account information.

### 3.1 `eth_getBalance` & `eth_getTransactionCount`
**Flow:**
1.  **Input:** `(Address, BlockTag)`
2.  **Metadata Check:** Ensure Client Epoch matches `BlockTag` (or fetch latest).
3.  **Addressing:** Compute 3 Cuckoo indices for `Address`.
4.  **DPF Gen:** Generate 3 key pairs.
5.  **Query:** Send to Servers A & B.
6.  **Reconstruct:** XOR results -> `AccountData { balance, nonce, code_id }`.
7.  **Response:** Return hex-encoded `balance` or `nonce`.

### 3.2 `eth_getCode`
**Flow:**
1.  **PIR Step:** Perform flow 3.1 to get `AccountData`.
2.  **Lookup:** Extract `code_id`.
3.  **Resolution:** 
    *   If `code_id == 0`: Return `0x`.
    *   Map `code_id` -> `code_hash` via public Dictionary (HTTP Range Request).
4.  **Fetch:** Download bytecode from CAS (`/cas/../hash.bin`).
5.  **Response:** Return bytecode.

---

## 4. Phase 2: Future Direction (`eth_call`)

Supporting `eth_call` (executing read-only contract functions) allows for private token balance checks (ERC20 `balanceOf`) and logic execution.

### The Challenge
`eth_call` requires executing EVM bytecode. The execution flow depends on contract state (Storage Slots). We cannot download the entire state of a contract (too large) or ask the server "give me slot X" (leaks access pattern).

### Proposed Solution: Iterative Lazy-Loading

The RPC Adapter embeds a lightweight Rust EVM (e.g., `revm`).

**Execution Flow:**
1.  **Setup:** Wallet sends `eth_call({ to: Contract, data: ... })`.
2.  **Code Fetch:** Adapter performs `eth_getCode(Contract)` via PIR+CAS (Section 3.2).
3.  **Local Execution (Start):** Load code into Local EVM and begin execution.
4.  **Storage Miss:** 
    *   The EVM attempts to read `SLOAD(key)`.
    *   Local DB misses. Execution **Pauses**.
5.  **Storage PIR Query:**
    *   Adapter calculates the Cuckoo index for the storage slot.
    *   *Note:* Requires a second PIR Matrix dedicated to `(Address + SlotKey) -> Value`.
    *   Perform PIR Query for the slot value.
6.  **Resume:** 
    *   Feed the retrieved value back into the Local EVM.
    *   Resume execution.
7.  **Repeat:** Until execution completes.
8.  **Return:** Result of the execution.

### Requirements for Phase 2
1.  **Storage Matrix:** A flattened database of all Ethereum Storage Slots (Address+Key), indexed via Cuckoo Hashing.
2.  **Batching:** If the EVM predicts multiple reads (prefetching), queries can be batched to reduce round-trips.
3.  **Hardware:** The storage matrix is significantly larger (~10TB vs 68GB). This may require:
    *   **Sharded PIR:** Query routed to specific shard servers.
    *   **Disk-Based DPF:** Slower scanning (SSD), acceptable for specific slot lookups vs full account scans.

### Optimization: Access Lists
If the dApp knows which slots it needs (via `eth_createAccessList`), the client can fetch all slots in one parallel PIR batch before execution starts, minimizing latency.
