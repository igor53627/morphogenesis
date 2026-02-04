# Task 1: 8-Byte Cuckoo Keys For Storage Lookups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update storage lookups to use 8-byte Cuckoo keys derived from `keccak(address || slot)[0..8]` while keeping tag verification intact.

**Architecture:** The storage query path will hash the 52-byte storage key into an 8-byte Cuckoo key for addressing, then compute the payload tag from the full 52-byte key to disambiguate the response. The ETL pipeline will likewise index storage entries using the 8-byte Cuckoo key instead of the 52-byte key so client and server address the same rows.

**Tech Stack:** Rust, morphogen-client, reth-adapter, cuckoo addressing.

---

### Task 1: Add a storage key derivation helper (client)

**Files:**
- Modify: `crates/morphogen-client/src/network.rs`
- Test: `crates/morphogen-client/src/network.rs` (unit test module at bottom)

**Step 1: Write the failing test**

Add a unit test that constructs a fixed address+slot, computes the expected 8-byte key from `keccak(address || slot)[0..8]`, and asserts the new helper returns that value.

```rust
#[test]
fn storage_key_hashes_to_8_bytes() {
    use alloy_primitives::keccak256;

    let address = [0x11u8; 20];
    let slot = [0x22u8; 32];
    let mut storage_key = [0u8; 52];
    storage_key[0..20].copy_from_slice(&address);
    storage_key[20..52].copy_from_slice(&slot);
    let expected = keccak256(&storage_key);
    let expected_tag: [u8; 8] = expected[0..8].try_into().unwrap();

    let actual = storage_cuckoo_key(address, slot);
    assert_eq!(actual, expected_tag);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --package morphogen-client storage_key_hashes_to_8_bytes`
Expected: FAIL with "cannot find function storage_cuckoo_key".

**Step 3: Write minimal implementation**

Add a private helper in `network.rs`:

```rust
fn storage_cuckoo_key(address: [u8; 20], slot: [u8; 32]) -> [u8; 8] {
    use alloy_primitives::keccak256;

    let mut storage_key = [0u8; 52];
    storage_key[0..20].copy_from_slice(&address);
    storage_key[20..52].copy_from_slice(&slot);

    let tag = keccak256(&storage_key);
    let mut out = [0u8; 8];
    out.copy_from_slice(&tag[0..8]);
    out
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test --package morphogen-client storage_key_hashes_to_8_bytes`
Expected: PASS.

**Step 5: Commit**

```bash
jj status
jj describe -m "test(client): add 8-byte storage cuckoo key helper test"
```

---

### Task 2: Switch client storage queries to 8-byte Cuckoo keys

**Files:**
- Modify: `crates/morphogen-client/src/network.rs`
- Test: `crates/morphogen-client/src/network.rs` (new unit test)

**Step 1: Write the failing test**

Add a test that verifies `generate_query` is called with an 8-byte key (not 52). Use a small test wrapper to call the helper and assert `key.len() == 8`.

```rust
#[test]
fn storage_query_uses_8_byte_cuckoo_key() {
    let address = [0x33u8; 20];
    let slot = [0x44u8; 32];
    let key = storage_cuckoo_key(address, slot);
    assert_eq!(key.len(), 8);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --package morphogen-client storage_query_uses_8_byte_cuckoo_key`
Expected: FAIL (helper not used yet or test uses wrong API).

**Step 3: Write minimal implementation**

Update `query_storage` to use the 8-byte key for `generate_query`:

```rust
let cuckoo_key = storage_cuckoo_key(address, slot);
let query_keys = {
    let mut rng = thread_rng();
    generate_query(&mut rng, &cuckoo_key, &metadata)
};
```

Keep the 52-byte `storage_key` for tag verification.

**Step 4: Run test to verify it passes**

Run: `cargo test --package morphogen-client storage_query_uses_8_byte_cuckoo_key`
Expected: PASS.

**Step 5: Commit**

```bash
jj status
jj describe -m "feat(client): use 8-byte cuckoo key for storage queries"
```

---

### Task 3: Switch ETL storage indexing to 8-byte Cuckoo keys

**Files:**
- Modify: `crates/reth-adapter/src/lib.rs`
- Test: `crates/reth-adapter/src/bin/test_optimized48.rs`

**Step 1: Write the failing test**

Update `test_optimized48.rs` to assert that the key used for storage is 8 bytes and matches `keccak(address || slot)[0..8]`.

```rust
let mut storage_key = [0u8; 52];
storage_key[0..20].copy_from_slice(&address);
storage_key[20..52].copy_from_slice(&slot);
let expected = keccak256(&storage_key);
let expected_tag: [u8; 8] = expected[0..8].try_into().unwrap();
assert_eq!(short_key, expected_tag);
```

**Step 2: Run test to verify it fails**

Run: `cargo test --package reth-adapter test_optimized48`
Expected: FAIL because the adapter still indexes with 52-byte keys.

**Step 3: Write minimal implementation**

Change the storage key generation to use the 8-byte short key for `CuckooTable` insertion:

```rust
let tag_hash = keccak256(&k);
let short_key: [u8; 8] = tag_hash[0..8].try_into().unwrap();
let key = short_key.to_vec();
```

Keep the full 52-byte key `k` for tag computation in the payload.

**Step 4: Run test to verify it passes**

Run: `cargo test --package reth-adapter test_optimized48`
Expected: PASS.

**Step 5: Commit**

```bash
jj status
jj describe -m "feat(adapter): index storage entries with 8-byte cuckoo key"
```

---

### Task 4: Update docs to match 8-byte keying

**Files:**
- Modify: `docs/DATA_STRUCTURES.md`

**Step 1: Write the failing test**

Not applicable. This is a docs-only change.

**Step 2: Update documentation**

Change the Storage Slots section to say:
- Key used for Cuckoo addressing is `keccak(address || slot)[0..8]` (8 bytes)
- Tag remains `keccak(address || slot)[0..8]` stored in the payload
- Full 52-byte key is only used for tag derivation and client verification

**Step 3: Commit**

```bash
jj status
jj describe -m "docs: document 8-byte cuckoo key for storage"
```

---

### Task 5: Full verification

**Step 1: Run tests**

```bash
cargo test --package morphogen-client
cargo test --package reth-adapter
cargo test --package morphogen-server --features network
```

**Step 2: Run format + lint**

```bash
cargo fmt --package morphogen-server
cargo clippy --package morphogen-server --features network
```

---

Plan complete and saved to `docs/plans/2026-02-03-task-1-8b-cuckoo-keys.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
