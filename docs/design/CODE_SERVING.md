# Design: Code Serving via Static CAS

## Problem Statement
The "Compact" PIR schema stores a 4-byte `CodeID` instead of the full bytecode. The client needs to resolve `CodeID -> CodeHash -> Bytecode` to execute local EVM calls or verify contracts.
The dataset consists of:
1.  **Mapping:** ~50M entries (`ID -> Hash`). ~1.6 GB.
2.  **Bytecode:** ~50M files. ~250 GB.

## Architecture: Append-Only Static CAS

We utilize an **Append-Only** strategy to serve this data via standard HTTP CDNs (S3/R2/CloudFront), avoiding complex servers and enabling aggressive caching.

### 1. The Mapping File (`mapping.bin`)
A flat binary file where the 32-byte `CodeHash` for `ID=k` is located at offset `k * 32`.

#### Live Updates (The "Delta" Problem)
Since uploading a 1.6GB file every block (12s) is impossible, we split the mapping:
*   **Base Snapshot (`mapping_v1.bin`):** Contains IDs `0` to `N`. Immutable. Cached forever.
*   **Delta Log (`mapping_delta.bin`):** Contains IDs `N+1` upwards. Appended to by the `reth-adapter`.
    *   *Optimization:* The server can expose a simple endpoint `GET /code/latest` that returns just the newest hashes, or the client can Range Request the delta file.

### 2. The Bytecode Store (`/blob/`)
Content Addressable Storage (CAS) keyed by the hash.
To handle 2M+ files efficiently, the store is **sharded** by the first two bytes of the hash.
*   Path: `/blob/{aa}/{bb}/{hash}.bin`
*   Example: `0xabcd...` -> `/blob/ab/cd/0xabcd...bin`
*   **Immutability:** Contract code never changes. Once written, it is cacheable forever.

## Client Retrieval Logic (Pseudocode)

```javascript
async function getCode(codeId) {
    // 1. Determine location
    const BASE_LIMIT = 50_000_000; // Snapshot cutoff
    let url;
    let offset;

    if (codeId < BASE_LIMIT) {
        url = "https://cdn.morphogen.xyz/mapping_v1.bin";
        offset = codeId * 32;
    } else {
        url = "https://cdn.morphogen.xyz/mapping_delta.bin";
        offset = (codeId - BASE_LIMIT) * 32;
    }

    // 2. Fetch Hash (Range Request)
    const hashBytes = await http.get(url, { 
        headers: { "Range": `bytes=${offset}-${offset + 31}` } 
    });
    const hashHex = toHex(hashBytes);

    // 3. Fetch Code (CAS)
    const code = await http.get(`https://cdn.morphogen.xyz/blob/${hashHex}.bin`);
    return code;
}
```

## Update Cycle (Server Side)

The `reth-adapter` (running as an ExEx) performs:
1.  **Block Import:** Detect new contract deployments.
2.  **Assign ID:** `ID = GlobalCount++`.
3.  **Write Hash:** Append 32 bytes to `mapping_delta.bin`.
4.  **Write Code:** Write bytecode to `blob/{hash}.bin` (if not exists).
5.  **Sync:** Upload `mapping_delta.bin` (range-append) and new blobs to S3/R2.

## Advantages
1.  **Cost:** Static storage is 10x cheaper than active servers.
2.  **Performance:** Range requests are parallelizable and edge-cacheable.
3.  **Scalability:** Zero server load; scaling is handled by the CDN.
4.  **Simplicity:** No database required for the code service.
