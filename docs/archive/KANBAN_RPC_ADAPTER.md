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

## 🎯 Priority Roadmap

### Immediate (for basic wallet support)
1. **Fix missing standard methods** (at least as passthroughs)
   - `eth_getStorageAt`, `eth_getBlockByHash`, `eth_getProof`
   - Wallet signing methods (`eth_accounts`, `eth_sign`, `eth_signTransaction`)
   - Filter APIs (`eth_newFilter`, `eth_getFilterChanges`)
2. **Integration testing with real wallets**
   - MetaMask/Rabby connection tests
   - End-to-end transaction flow verification
3. **Error handling hardening**
   - Server down scenarios
   - Epoch mismatch recovery
   - Timeout handling

### High Priority (for meaningful privacy)

**Context:** Current database (68GB) ALREADY contains 1.85B items: 350M accounts + 1.5B storage slots
mixed in the same Cuckoo table. Storage slots are keyed as `Address.SlotKey` (52 bytes) vs accounts
at `Address` (20 bytes). The data exists on the server - we just need client/adapter support to query it.

1. **`eth_getStorageAt` implementation** - enables private ERC20 balance checks
   - **Data status:** ✅ Storage slots already in database (1.5B slots in 68GB matrix)
   - **What's needed:** RPC adapter method to query storage slots
   - **Implementation:**
     - Construct 52-byte key: `Address (20) . SlotKey (32)`
     - SlotKey calculation: `keccak256(abi.encode(userAddress, mappingSlot))` for mappings
     - PIR query using 52-byte key instead of 20-byte address
     - May need client library updates to support variable-length keys
   - **Use case:** Check USDT balance (`balanceOf[address]`), allowances, NFT ownership privately

2. **Client library support for storage queries**
   - Current: `generate_query()` takes 20-byte address
   - Needed: Support 52-byte keys for storage slot queries
   - Cuckoo addressing already supports arbitrary byte keys
   - Should be straightforward extension

3. **Local EVM + lazy storage loading** - enables private `eth_call`
   - Embed local EVM (e.g., `revm`) in adapter
   - Execute contract calls locally
   - On `SLOAD(key)` miss: pause → PIR query storage slot → resume
   - Iterate until execution completes
   - **Use case:** Uniswap price quotes, DeFi position checks without leaking intent
   - **Note:** Storage data already available, just need EVM integration

### Medium Priority
1. **Private log retrieval (`eth_getLogs`)**
   - Current passthrough leaks address/topic interest to upstream
   - Implement Bloom filter PIR or full log database scan
   - Client-side filtering
2. **Caching layer for repeated queries**
   - Cache PIR results by (epoch_id, address)
   - TTL based on epoch rotation interval
   - Reduces query cost for frequently accessed accounts
3. **Batch query support**
   - Accept multiple addresses in single request
   - Single database scan for all queries
   - Reduce per-query overhead

### Research/Future
1. **WASM build for browser integration**
   - Compile adapter to WebAssembly
   - Integrate directly into browser wallet extensions
   - Eliminate localhost proxy requirement
2. **Access list optimization for `eth_call`**
   - Use `eth_createAccessList` to prefetch storage slots
   - Batch PIR queries before execution starts
   - Minimize execution pauses
3. **Storage slot prefetching/prediction**
   - ML-based prediction of likely storage accesses
   - Speculative PIR queries during execution
   - Hide latency with parallelism

---

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
