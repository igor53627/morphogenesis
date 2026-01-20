# Phase 77: Code Serving & Full Integration

## Objective
To support full wallet functionality (`eth_getCode`, `eth_call`) while maintaining the 32-byte PIR row size, we offload bytecode storage to a public Content Addressable Storage (CAS). The PIR row stores a 4-byte `CodeID` which maps to the full bytecode.

## Sub-Tasks

### 1. Extraction (Server-Side)
- [ ] **Dictionary Export:**
  - `reth_dump` already builds `HashMap<CodeHash, u32>`.
  - Ensure `code_dictionary.bin` (Array of 32B Hashes) is saved.
- [ ] **Bytecode Dump:**
  - Iterate the dictionary.
  - For each Hash, fetch bytecode from Reth DB.
  - Write to disk: `code_store/{id}.bin` or `code_store/{hash}.bin`.
  - Optimization: Use a single KV store (e.g. RocksDB/MDBX) for the code server instead of millions of files.

### 2. Code Service (Public API)
- [ ] **Service:** `morphogen-code-server` (Rust/Axum).
- [ ] **Endpoints:**
  - `GET /code/id/:id` -> Returns bytecode (lookup ID -> Hash -> Code).
  - `GET /code/hash/:hash` -> Returns bytecode.
- [ ] **Privacy:** This service is PUBLIC. Queries do not need to be private (checking code of Uniswap is common).
- [ ] **Cache:** Highly cacheable (bytecode is immutable).

### 3. Client Integration
- [ ] **Schema Update:**
  - Update `morphogen-client` to understand the 32-byte row layout:
    - `[Balance(16) | Nonce(8) | CodeID(4) | Padding(4)]`.
- [ ] **Code Resolution:**
  - If user asks for `getCode(address)`:
    1.  Perform PIR Query for `address`.
    2.  Extract `CodeID`.
    3.  If `CodeID == 0`, return `0x` (EOA).
    4.  Fetch bytecode from `Code Service` using `CodeID`.
    5.  Return bytecode.

### 4. Integration Testing
- [ ] **Local:**
  - Mock `Code Service` serving dummy code.
  - Client retrieves balance + code.
- [ ] **Mainnet:**
  - Verify against real contracts (e.g. `USDT`, `WETH`).
  - Ensure `CodeHash` derived from fetched code matches the one on-chain.
