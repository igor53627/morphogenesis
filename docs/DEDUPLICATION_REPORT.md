# Morphogenesis Deduplication Analysis

**Date:** Jan 21, 2026
**Target:** Ethereum Mainnet State (Extracted via `reth-adapter`)
**Method:** Full scan of 68.8GB compact matrix

## 1. Executive Summary

By using Content Addressable Storage (CAS) for contract bytecode, Morphogenesis achieves a **34.8x reduction** in storage requirements for code. This allows the primary PIR matrix to remain small (68.8 GB) and fit entirely in GPU VRAM, while the bulk of the raw data (bytecodes) is offloaded to cheap, static storage (Cloudflare R2 / S3).

| Metric | Value |
| :--- | :--- |
| **Total Contract Accounts** | **123.8 Million** |
| **Unique Bytecodes** | **2.05 Million** |
| **Deduplication Ratio** | **34.8 : 1** |
| **Space Saved** | **424.08 GB** |

## 2. Detailed Findings

### Account Distribution
*   **Total Database Rows:** 2,152,596,252 (Accounts + Storage Slots)
*   **Accounts with Code:** 123,802,496
*   **Deduplicated Entries:** 121,747,695 (Contracts sharing code with others)

### Storage Savings
*   **Raw Bytecode Size (Redundant):** 436.64 GB
    *   *If we stored full bytecode for every contract account.*
*   **CAS Bytecode Size (Unique):** 12.56 GB
    *   *Actual storage used by the deduplicated `code_store`.*
*   **Net Savings:** 424.08 GB

### Impact on PIR Performance
If we did not deduplicate, the PIR matrix would grow by ~424 GB, reaching **~493 GB**.
*   **Hardware Requirement:** This would require 8x H100 GPUs (640GB VRAM) just to hold the data.
*   **Current Hardware:** Fits on 1x H100 (80GB) or 1x B200.
*   **Cost Impact:** Reduces hardware CAPEX by **87%**.

---

## 3. Replication Instructions

To reproduce these numbers, run the analysis script on a server with the full extracted dataset (e.g., `hsiao`).

### Prerequisites
*   **Matrix File:** `mainnet_compact.bin` (68.8 GB)
*   **Dictionary:** `mainnet_compact.dict` (~64 MB)
*   **Code Store:** `code_store/` directory (Optional, for exact size calc)
*   **Python:** with `numpy` installed

### Script (`calculate_dedup.py`)

```python
import os
import struct
import numpy as np

# Configuration
matrix_path = 'mainnet_compact.bin'
dict_path = 'mainnet_compact.dict'
code_store_path = 'code_store' # Optional

if not os.path.exists(matrix_path):
    print(f'Error: {matrix_path} not found')
    exit(1)

# 1. Load Dictionary (Get Unique Count)
print('Reading code dictionary...')
with open(dict_path, 'rb') as f:
    dict_data = f.read()
unique_contracts = len(dict_data) // 32
print(f'Unique Contracts: {unique_contracts:,}')

# 2. Scan Matrix (Count References)
print('Scanning matrix...')
file_size = os.path.getsize(matrix_path)
num_rows = file_size // 32
counts = np.zeros(unique_contracts + 1000, dtype=np.uint64)
chunk_size = 10_000_000

with open(matrix_path, 'rb') as f:
    for i in range(0, num_rows, chunk_size):
        data = f.read(min(chunk_size, num_rows - i) * 32)
        if not data: break
        
        # Extract CodeID (Bytes 24-28)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)
        c0 = arr[:, 24].astype(np.uint32)
        c1 = arr[:, 25].astype(np.uint32)
        c2 = arr[:, 26].astype(np.uint32)
        c3 = arr[:, 27].astype(np.uint32)
        code_ids = (c0 << 24) | (c1 << 16) | (c2 << 8) | c3
        
        # Filter valid IDs (exclude EOA=0 and Storage Noise)
        mask = (code_ids > 0) & (code_ids < unique_contracts)
        active = code_ids[mask]
        
        if active.size > 0:
            batch = np.bincount(active, minlength=len(counts)).astype(np.uint64)
            counts[:len(batch)] += batch

# 3. Calculate Sizes
total_refs = np.sum(counts[1:])
unique_refs = np.count_nonzero(counts[1:])
unique_size = 0
redundant_size = 0

if os.path.exists(code_store_path):
    print('Calculating sizes...')
    for i in range(1, unique_contracts):
        if counts[i] == 0: continue
        hex_hash = dict_data[i*32 : (i+1)*32].hex()
        path = os.path.join(code_store_path, hex_hash[0:2], hex_hash[2:4], f'{hex_hash}.bin')
        if os.path.exists(path):
            s = os.path.getsize(path)
            unique_size += s
            redundant_size += s * counts[i]
else:
    # Fallback Estimate
    avg_size = 3500 # 3.5KB average
    unique_size = unique_refs * avg_size
    redundant_size = total_refs * avg_size

print(f'\nTotal Contracts: {total_refs:,}')
print(f'Unique Code:     {unique_refs:,}')
print(f'Redundant Size:  {redundant_size / 1e9:.2f} GB')
print(f'Unique Size:     {unique_size / 1e9:.2f} GB')
print(f'Savings:         {(redundant_size - unique_size) / 1e9:.2f} GB')
```

### Execution
Run directly on the data server:
```bash
python3 calculate_dedup.py
```
