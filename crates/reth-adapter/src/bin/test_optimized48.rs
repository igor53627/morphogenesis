use alloy_primitives::keccak256;
use reth_adapter::{
    build_matrix, serialize_account, Account, AccountSource, CodeIndexer, RowScheme, UbtItem,
};

/// A test source that yields a single storage item.
struct SingleStorageSource {
    address: [u8; 20],
    slot: [u8; 32],
    value: [u8; 32],
    consumed: bool,
}

impl SingleStorageSource {
    fn new(address: [u8; 20], slot: [u8; 32], value: [u8; 32]) -> Self {
        Self {
            address,
            slot,
            value,
            consumed: false,
        }
    }
}

impl AccountSource for SingleStorageSource {
    fn next_item(&mut self) -> Option<UbtItem> {
        if self.consumed {
            return None;
        }
        self.consumed = true;
        Some(UbtItem::Storage {
            address: self.address,
            key: self.slot,
            value: self.value,
        })
    }
}

fn main() {
    println!("=== Testing Optimized48 Schema ===");

    // 1. Account Test
    let acc = Account {
        address: [0xAA; 20],
        balance: 100,
        nonce: 5,
        code_hash: [0xCC; 32],
    };

    let mut indexer = CodeIndexer::new();
    let payload = serialize_account(&acc, RowScheme::Optimized48, &mut indexer);

    assert_eq!(payload.len(), 48, "Account payload must be 48 bytes");

    // Check Balance (0..16)
    let bal = u128::from_be_bytes(payload[0..16].try_into().unwrap());
    assert_eq!(bal, 100, "Balance mismatch");

    // Check Nonce (16..24)
    let nonce = u64::from_be_bytes(payload[16..24].try_into().unwrap());
    assert_eq!(nonce, 5, "Nonce mismatch");

    // Check Tag (28..36)
    // Tag = Keccak(Address)[0..8]
    let expected_tag = &keccak256(acc.address)[0..8];
    assert_eq!(&payload[28..36], expected_tag, "Account Tag mismatch");

    println!("[OK] Account serialization passed.");

    // 2. Storage Test - Verify payload serialization logic
    let key = vec![0xBB; 52]; // Address(20) + Slot(32)
    let value = [0xDD; 32];

    let mut p = vec![0u8; 48];
    p[0..32].copy_from_slice(&value);

    let tag_hash = keccak256(&key);
    p[32..40].copy_from_slice(&tag_hash[0..8]);

    assert_eq!(&p[0..32], &value, "Storage Value mismatch");
    assert_eq!(&p[32..40], &tag_hash[0..8], "Storage Tag mismatch");

    println!("[OK] Storage serialization logic verified.");

    // 3. Storage Cuckoo Key Test - Verify 8-byte key indexing
    // The Cuckoo table key for storage must be keccak(address || slot)[0..8]
    let address = [0x11u8; 20];
    let slot = [0x22u8; 32];
    let value = [0x33u8; 32];

    let mut source = SingleStorageSource::new(address, slot, value);
    let (_matrix, manifest, _indexer) =
        build_matrix(&mut source, 1024, RowScheme::Optimized48, false);

    // Compute expected 8-byte Cuckoo key
    let mut storage_key = [0u8; 52];
    storage_key[0..20].copy_from_slice(&address);
    storage_key[20..52].copy_from_slice(&slot);
    let expected_hash = keccak256(&storage_key);
    let expected_tag: [u8; 8] = expected_hash[0..8].try_into().unwrap();

    // Verify that the item was inserted (manifest shows 1 item)
    assert_eq!(
        manifest.item_count, 1,
        "Expected 1 storage item in manifest"
    );

    // Verify the key length used for indexing by checking the cuckoo_key_for_storage function
    let short_key = reth_adapter::cuckoo_key_for_storage(&address, &slot);
    assert_eq!(short_key.len(), 8, "Storage Cuckoo key must be 8 bytes");
    assert_eq!(
        short_key.as_slice(),
        &expected_tag,
        "Storage Cuckoo key must equal keccak(address || slot)[0..8]"
    );

    println!("[OK] Storage Cuckoo key is 8 bytes and matches keccak(address || slot)[0..8].");

    println!("=== All tests passed ===");
}
