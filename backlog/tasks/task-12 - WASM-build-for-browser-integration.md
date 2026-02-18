---
id: TASK-12
title: WASM build for browser integration
status: Done
assignee: []
created_date: '2026-02-03 14:38'
updated_date: '2026-02-18 09:08'
labels:
  - rpc
  - research
  - wasm
dependencies: []
references:
  - c9264ec - initial wasm gateway implementation (roborev job 748 addressed)
  - 167033a - passthrough/block-tag hardening (roborev job 751 addressed)
  - 3623013 - demo fallback tests + docs alignment (roborev job 754 addressed)
  - 227319f - demo method validation hardening (roborev job 755 clean)
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MVP: ship a browser-runnable privacy gateway so wallet extensions can call Morphogenesis without a localhost proxy.

### Scope (MVP)

1. Build a new WASM-target client facade (not the full `morphogen-rpc-adapter` HTTP server) for `wasm32-unknown-unknown`.
2. Expose an EIP-1193-style `request({ method, params })` API from WASM via `wasm-bindgen`.
3. Support private read methods in-browser:
   - `eth_getBalance`
   - `eth_getTransactionCount`
   - `eth_getStorageAt`
   - `eth_getCode`
4. Support basic passthrough methods to upstream via browser `fetch` (`eth_chainId`, `eth_blockNumber`, `eth_gasPrice`, and generic fallback for safe read-only methods).
5. Keep privacy boundary explicit: private methods must hit PIR/CAS only; passthrough methods may call upstream.
6. Provide a minimal extension/demo integration that swaps provider routing to the WASM gateway.

### Non-goals (MVP)

- No in-browser JSON-RPC HTTP server.
- No `eth_call` / `eth_estimateGas` local EVM execution.
- No block cache, tx cache, log/filter APIs.
- No signing flow changes (`eth_sendRawTransaction`/wallet signing stay unchanged).

### Implementation notes

- Reuse existing `morphogen-client` query logic where possible; isolate transport behind interfaces so native + WASM share core logic.
- Add clear error mapping from Rust errors to JSON-RPC error objects in the WASM boundary.
- Document required CORS and endpoint configuration for PIR servers and upstream RPC.
- Early build check indicates `getrandom` needs wasm JS support (`js` feature or target-specific config) on `wasm32-unknown-unknown`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 cargo build -p morphogen-client --target wasm32-unknown-unknown succeeds, or a dedicated WASM crate build succeeds if introduced.
- [x] #2 WASM export provides request(method, params) compatible with EIP-1193 call shape.
- [x] #3 Private methods eth_getBalance, eth_getTransactionCount, eth_getStorageAt, and eth_getCode execute successfully in WASM tests against mocked PIR and upstream endpoints.
- [x] #4 Routing tests verify private methods do not call upstream on successful private path.
- [x] #5 Demo integration (example extension page or script) performs at least one private call and one passthrough call end-to-end.
- [x] #6 README/docs include setup instructions, supported method matrix, and explicit MVP non-goals.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All roborev findings addressed. Final review job 755 reported no issues.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Code merged with WASM target build passing in CI, or local verification command documented if CI target is deferred.
- [x] #2 Unit/integration tests cover method routing and error mapping at the WASM boundary.
- [x] #3 Demo integration committed under examples/ or scripts/ with run instructions.
- [x] #4 Backlog task updated with implementation notes/final summary when completed.
<!-- DOD:END -->
