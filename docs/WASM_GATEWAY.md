# WASM Gateway (MVP)

`morphogen-wasm-gateway` is an EIP-1193-compatible browser facade exported via `wasm-bindgen`.
It routes private reads to PIR/CAS and safe reads to upstream RPC.

## Build & Test

```bash
# Native unit tests
cargo test -p morphogen-wasm-gateway

# wasm32 build
cargo build -p morphogen-wasm-gateway --target wasm32-unknown-unknown

# wasm test harness (Node via wasm-pack)
wasm-pack test --node crates/morphogen-wasm-gateway
```

## JavaScript API

```js
import init, { WasmGateway } from "./pkg/morphogen_wasm_gateway.js";

await init();
const gateway = new WasmGateway({
  upstreamUrl: "https://ethereum-rpc.publicnode.com",
  pirServerA: "http://127.0.0.1:3000",
  pirServerB: "http://127.0.0.1:3001",
  dictUrl: "http://127.0.0.1:8080/mainnet_compact.dict",
  casUrl: "http://127.0.0.1:8080/cas",
  requestTimeoutMs: 15000,
});

const balance = await gateway.request({
  method: "eth_getBalance",
  params: ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "latest"],
});
```

`request(payload)` returns a Promise that resolves with the JSON-RPC result value or rejects with a JSON-RPC error object:

```json
{ "code": -32602, "message": "...", "data": null }
```

## Method Routing Matrix (MVP)

| Method | Route | Notes |
|---|---|---|
| `eth_getBalance` | Private PIR | Never proxied upstream; only `latest` block tag currently supported |
| `eth_getTransactionCount` | Private PIR | Never proxied upstream; only `latest` block tag currently supported |
| `eth_getStorageAt` | Private PIR | Never proxied upstream; only `latest` block tag currently supported |
| `eth_getCode` | Private PIR + Dictionary + CAS | Never proxied upstream; only `latest` block tag currently supported |
| `eth_chainId` | Upstream passthrough | Browser fetch JSON-RPC |
| `eth_blockNumber` | Upstream passthrough | Browser fetch JSON-RPC |
| `eth_gasPrice` | Upstream passthrough | Browser fetch JSON-RPC |
| Safe read-only fallback (`eth_get*`, `net_*`, `web3_*`) | Upstream passthrough | Blocked for write/sign/filter methods |
| Write/sign/filter methods (`eth_send*`, `eth_sign*`, `personal_sign`, filter state methods) | Rejected | Returns `-32601` unsupported method |

## CORS / Endpoint Configuration

For browser usage, all endpoints must allow your extension/page origin.

- PIR server A/B:
  - `POST /query` (and `POST /query/batch` if batching is enabled later)
  - `GET /epoch`
- Dictionary endpoint:
  - `GET` with `Range` header
  - CORS should allow request headers: `Range`
  - CORS should expose response headers: `Accept-Ranges`, `Content-Range`, `Content-Length`
- CAS endpoint:
  - `GET /cas/<shard>/<shard>/<hash>.bin`
- Upstream RPC endpoint:
  - `POST` with `Content-Type: application/json`

## MVP Non-Goals

- No in-browser JSON-RPC HTTP server
- No local EVM execution for `eth_call` / `eth_estimateGas`
- No block cache/tx cache/log filter APIs in WASM
- No signing flow changes (`eth_sendRawTransaction` and wallet signing remain wallet-owned)
