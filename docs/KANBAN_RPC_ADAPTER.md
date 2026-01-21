# Morphogenesis RPC Adapter - Kanban

**Goal:** Create a drop-in JSON-RPC replacement that translates Ethereum requests into privacy-preserving DPF queries, enabling standard tools like `cast` and `metamask` to work with Morphogenesis.

## [DONE]

### 1. Infrastructure Scaffolding (Phase 80a)
- [x] Create `crates/morphogen-rpc-adapter` crate.
- [x] Add dependencies: `jsonrpsee` (server), `reqwest`, `tokio`, `morphogen-client`.
- [x] Implement `main.rs` with basic CLI args (`--upstream`, `--pir-server-a`, `--pir-server-b`).
- [x] Setup `AdapterState` struct to hold upstream provider and PIR clients.

### 2. Upstream Proxying (Phase 80b)
- [x] Implement generic fallback handler for non-private methods (`eth_blockNumber`, `eth_gasPrice`, etc.).
- [x] Forward requests to the `--upstream` URL (e.g., Anvil/Infura).
- [x] Test with `cast block-number`.

### 3. Core Privacy Integration (Phase 80c)
- [x] Initialize `morphogen-client::PirClient` on startup.
- [x] Fetch initial `EpochMetadata` from PIR servers.
- [x] Implement background task to refresh epoch metadata.

### 4. Account RPC Methods (Phase 80d)
- [x] Implement `eth_getBalance(address, block)`.
  - [x] Intercept request.
  - [x] Validate block tag (must be `latest` or match current epoch).
  - [x] Generate DPF query via `morphogen-client`.
  - [x] Send to Servers A/B.
  - [x] Aggregate and return hex value.
- [x] Implement `eth_getTransactionCount(address, block)`.
  - [x] Similar flow, extract nonce.

### 5. Local E2E Environment (Phase 80e)
- [x] Create `scripts/test_rpc_e2e.sh` which:
  - [x] Starts `test_server` (CPU mode, synthetic data).
  - [x] Starts `morphogen-rpc-adapter` pointing to server.
- [x] Verify with `cast balance <random_addr>`.

### 6. Contract Code Support (Phase 81)
- [x] Implement `CodeResolver` logic in adapter.
- [x] Implement `eth_getCode`.
  - [x] PIR query for `AccountData`.
  - [x] Extract `code_id`.
  - [x] Resolve hash and fetch from CAS (verified with mock server).

## [TODO]



### 7. Integration Testing

- [ ] CI workflow: Spin up full stack -> `cast` calls -> Assert outputs.

- [ ] Test error handling (server down, epoch mismatch).



### 8. Phase 82: Wallet Compatibility Verification

Ensure seamless UX with MetaMask/Rabby.

- [ ] **Connection:** Verify `eth_chainId`, `net_version` return correct values from upstream.

- [ ] **Transaction Flow:** Verify `eth_estimateGas` -> `eth_sendRawTransaction` -> `eth_getTransactionReceipt`.

- [ ] **Block Data:** Verify `eth_getBlockByNumber`, `eth_feeHistory`.

- [ ] **Logs:** Verify `eth_getLogs` returns token transfer history.



### 9. Phase 83: Advanced Privacy (Future)

Methods that currently leak privacy via proxying:

- [ ] **`eth_call`:** Implement Local Lazy-Loading EVM (See `RPC_CLIENT_SPEC.md`).

  - [ ] Fetch code via PIR.

  - [ ] Fetch storage slots via PIR on demand.

  - [ ] Execute locally.

- [ ] **`eth_getLogs`:** Implement Private Log Retrieval (Bloom Filter PIR or Scan).



## [IN PROGRESS]
