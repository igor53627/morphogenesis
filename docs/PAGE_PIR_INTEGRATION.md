# Page-Level PIR Integration Plan

## Overview

Replace insecure `AesDpfKey` (servers see target index) with proper `PageDpfKey` (2-server computational privacy).

## Architecture Changes

### Current: Row-Level PIR (Insecure)

```
Client                          Server
  |                                |
  |-- 3x AesDpfKey (25 bytes) --->|  Server can read target from key!
  |                                |
  |                           scan_main_matrix()
  |                           for each row:
  |                             mask = key.eval_bit(row_idx)
  |                             result ^= row_data & mask
  |                                |
  |<--- 3x row payloads (256B) ---|
```

### New: Page-Level PIR (Private)

```
Client                          Server A                    Server B
  |                                |                            |
  |-- PageDpfKey (party=0) ------>|                            |
  |-- PageDpfKey (party=1) ---------------------------------->|
  |                                |                            |
  |                           eval_and_accumulate_chunked()    |
  |                           (same on both servers)           |
  |                                |                            |
  |<--- page response (4KB) ------|                            |
  |<----------------------------------------- page response ---|
  |                                                             
  | XOR responses locally                                       
  | Extract row from page                                       
```

## Key Differences

| Aspect | Row-Level | Page-Level |
|--------|-----------|------------|
| Privacy | None (target in key) | Computational (2-server) |
| DPF domain | 250M rows | 27M pages (16x smaller) |
| Response size | 256 bytes/query | 4KB/query (16x larger) |
| Client work | None | Extract row from page |
| Keys per query | 3 (Cuckoo) | 3 (Cuckoo) |
| Key size | 25 bytes | ~500 bytes (BGI DPF) |

## Migration Phases

### Phase 61a: Add PageDpfKey serialization
- Add `to_bytes()`/`from_bytes()` to PageDpfKey
- Key format: params + share data
- Estimated size: ~500 bytes for 25-bit domain

### Phase 61b: Create page-level query types
- `PageQueryRequest`: 3 serialized PageDpfKeys + params
- `PageQueryResponse`: 3 page payloads (4KB each) + epoch_id
- New endpoint: POST /query/page

### Phase 61c: Add page-level scan function
- `scan_pages_chunked()`: uses eval_and_accumulate_chunked
- Input: page matrix, PageDpfKey array
- Output: 3 page payloads

### Phase 61d: Wire up page query handler
- Parse PageDpfKeys from request
- Call scan_pages_chunked
- Return page payloads

### Phase 61e: Update client library
- Add page-level query generation
- Row index → (page_index, row_offset)
- XOR responses, extract target row

### Phase 61f: Reorganize data as pages [COMPLETE]
- **No physical reorganization needed**: pages are 16 consecutive rows
- Current row-contiguous storage already supports page-level access
- `scan_pages_chunked()` computes page boundaries within existing matrix chunks

## API Design

### New Endpoint: POST /query/page

Request:
```json
{
  "params": {
    "prg_keys": ["<hex>", "<hex>"],
    "domain_bits": 25
  },
  "keys": ["<hex PageDpfKey 0>", "<hex PageDpfKey 1>", "<hex PageDpfKey 2>"]
}
```

Response:
```json
{
  "epoch_id": 12345,
  "pages": ["<hex 4KB page 0>", "<hex 4KB page 1>", "<hex 4KB page 2>"]
}
```

### Client Usage

```rust
// Client generates keys for target row
let target_row = 12345678;
let addr = PageAddress::from_row_index(target_row);

// PRG keys are shared public parameters (generated once, used by all clients)
let params = PageDpfParams::new(25)?; // 27M pages

// Generate key pair for this query
let (key_a, key_b) = generate_page_dpf_keys(&params, addr.page_index)?;

// Send key_a to server A, key_b to server B
let response_a = server_a.query_page(&params, &key_a);
let response_b = server_b.query_page(&params, &key_b);

// XOR responses to get plaintext page
let page = xor_pages(&response_a, &response_b);

// Extract target row from page
let row = extract_row_from_page(&page, addr.row_offset);
```

## Backward Compatibility

Options:
1. **Breaking change**: Replace /query with /query/page, clients must upgrade
2. **Dual endpoints**: Keep /query (deprecated), add /query/page
3. **Version header**: X-PIR-Version: 1 (row) or 2 (page)

Recommendation: Option 2 (dual endpoints) for gradual migration.

## Performance Impact

| Metric | Row-Level | Page-Level | Delta |
|--------|-----------|------------|-------|
| DPF domain | 250M | 27M | 9.3x smaller |
| DPF eval time | ~250ms (insecure) | ~840ms (fss-rs) | 3.4x slower |
| Response size | 768B (3×256B) | 12KB (3×4KB) | 16x larger |
| Client bandwidth | 768B down | 12KB down | Acceptable |

## Security Considerations

1. **PRG keys**: Must be identical on both servers (public parameter)
2. **Non-collusion**: Privacy requires servers don't share queries
3. **Key freshness**: Generate new DPF keys per query (no reuse)
4. **Timing attacks**: Chunked eval is constant-time per chunk

## Testing Plan

1. Unit tests: PageDpfKey serialization roundtrip
2. Integration tests: Page query endpoint
3. E2E tests: Full flow with 2 servers
4. Privacy test: Verify keys reveal nothing about target
