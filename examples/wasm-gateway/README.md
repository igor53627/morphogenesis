# WASM Gateway Demo

This demo builds the `morphogen-wasm-gateway` package and installs a provider bridge that routes:

- Private methods to PIR/CAS via WASM (`eth_getBalance`, `eth_getTransactionCount`, `eth_getStorageAt`, `eth_getCode`)
- Base passthrough methods to upstream via WASM (`eth_chainId`, `eth_blockNumber`, `eth_gasPrice`)
- Unsupported/unsafe methods to the original wallet provider (if present)

## Run

1. Build the wasm package into this demo directory:

```bash
wasm-pack build crates/morphogen-wasm-gateway --target web --out-dir ../../examples/wasm-gateway/pkg
```

2. Serve the repo root with a static server (example):

```bash
python3 -m http.server 4173
```

3. Open `http://127.0.0.1:4173/examples/wasm-gateway/`.

4. Click:

- `Initialize Gateway + Provider`
- `Run Private eth_getBalance`
- `Run Passthrough eth_chainId`

The output panel will show one private call and one passthrough call end-to-end through the routed provider.

## Endpoint Requirements

- PIR servers (`pirServerA`, `pirServerB`) and dictionary/CAS endpoints must allow browser `GET`/`POST` from your page origin.
- Upstream RPC must allow browser `POST` with JSON content type from your page origin.
