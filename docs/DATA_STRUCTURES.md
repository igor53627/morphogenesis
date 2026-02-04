# Morphogenesis Data Structures

## 1. The Matrix (PIR Database)

The core data structure is a linearized **Cuckoo Hash Table** stored as a flat binary file (`.bin`).

### Parameters (Mainnet Jan 2026)
- **Rows:** 2,152,596,252 (Fits 350M Accounts + 1.5B Storage Slots)
- **Load Factor:** 85%
- **Domain:** 32 bits ($2^{32}$)
- **Hash Functions:** 3 (Salted SipHash)

### Row Schemas

We support two row layouts. The 32-byte "Compact" schema is the production standard.

#### Schema A: Compact (32 Bytes)
**Target:** NVIDIA H100 (80GB VRAM)
**Total Size:** 68.8 GB

| Offset | Size | Type | Field | Description |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 16 | `u128` | **Balance** | Account Balance (Wei) |
| 16 | 8 | `u64` | **Nonce** | Transaction Count |
| 24 | 4 | `u32` | **CodeID** | Index into the Code Dictionary (0 = EOA) |
| 28 | 4 | `u8[4]` | **Padding** | Reserved / Flags |

#### Schema B: Full (64 Bytes)
**Target:** NVIDIA H200 (141GB VRAM)
**Total Size:** ~137 GB

| Offset | Size | Type | Field | Description |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 16 | `u128` | **Balance** | Account Balance |
| 16 | 8 | `u64` | **Nonce** | Transaction Count |
| 24 | 32 | `u256` | **CodeHash** | Full Keccak256 Code Hash |
| 56 | 8 | `u8[8]` | **Padding** | Reserved |

### Storage Slots

Storage slots are mixed into the same table as Accounts. Addressing differs by schema:

#### Legacy Schemas (Compact / Full)
- **Cuckoo Key:** Full 52-byte composite key `address || slot_key`
- **Payload:** 32-byte storage value only (no tag)
- The entire 52-byte key is hashed to determine Cuckoo bucket positions

#### Optimized48 Schema
- **Cuckoo Key:** 8-byte tag derived as `keccak256(address || slot_key)[0..8]`
- **Payload:** 8-byte tag (same derivation) + 32-byte storage value + 8-byte padding
- The full 52-byte composite key is used only for:
  1. Tag derivation (first 8 bytes of keccak hash)
  2. Client-side verification (proves slot belongs to address)
- Cuckoo addressing uses only the 8-byte tag, not the full 52-byte key

---

## 2. Code Serving (Static CAS)

To achieve the 32-byte row size, we offload bytecode to a static Content Addressable Storage.

### Dictionary (`mainnet_compact.dict`)
A flat binary file mapping `CodeID` to `CodeHash`.
- **Size:** ~64 MB (2M entries).
- **Layout:** Array of 32-byte Keccak hashes.
- **Access:** Client performs HTTP Range Request: `GET /dict bytes=(ID*32)-((ID*32)+31)`.

### Bytecode Store (`/pir/cas/`)
A sharded directory structure containing raw contract bytecode.
- **Path:** `/{aa}/{bb}/{hash_hex}.bin`
- **Sharding:** First 2 bytes of the hex hash (2 levels).
- **Example:** Hash `0xd0a0...` -> `/d0/a0/d0a0...bin`.

---

## 3. GPU Memory Layout (`GpuPageMatrix`)

To maximize memory bandwidth, the Matrix is not stored as a single contiguous array on the GPU, but as a collection of **Pages**.

- **Page Size:** 4 KB (4096 bytes).
- **Alignment:** Data loaded from disk is padded to the nearest 4KB boundary.
- **Access Pattern:** The fused CUDA kernel loads 128-bit vectors from these pages using the DPF index.
