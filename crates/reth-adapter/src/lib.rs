use alloy_primitives::keccak256;
use morphogen_core::cuckoo::CuckooTable;
use morphogen_storage::ChunkedMatrix;

#[cfg(feature = "reth")]
use reth_db::{cursor::DbCursorRO, transaction::DbTx, Database, DatabaseEnv};
#[cfg(feature = "reth")]
use std::sync::Arc;

pub struct Account {
    pub address: [u8; 20],
    pub balance: u128,
    pub nonce: u64,
    pub code_hash: [u8; 32],
}

pub enum UbtItem {
    Account(Account),
    Storage {
        address: [u8; 20],
        key: [u8; 32],
        value: [u8; 32],
    },
}

pub trait AccountSource {
    fn next_item(&mut self) -> Option<UbtItem>;

    fn count_all(&mut self) -> usize {
        let mut count = 0;
        while self.next_item().is_some() {
            count += 1;
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morphogen_core::cuckoo::CuckooAddresser;

    const STORAGE_ADDRESS: [u8; 20] = [0x11; 20];
    const STORAGE_SLOT: [u8; 32] = [0x22; 32];
    const STORAGE_VALUE: [u8; 32] = [0x33; 32];

    struct SingleStorageSource {
        consumed: bool,
    }

    impl SingleStorageSource {
        fn new() -> Self {
            Self { consumed: false }
        }
    }

    impl AccountSource for SingleStorageSource {
        fn next_item(&mut self) -> Option<UbtItem> {
            if self.consumed {
                return None;
            }
            self.consumed = true;
            Some(UbtItem::Storage {
                address: STORAGE_ADDRESS,
                key: STORAGE_SLOT,
                value: STORAGE_VALUE,
            })
        }
    }

    #[test]
    fn cuckoo_key_for_storage_matches_keccak_prefix() {
        let short_key = cuckoo_key_for_storage(&STORAGE_ADDRESS, &STORAGE_SLOT);
        assert_eq!(short_key.len(), 8);

        let mut storage_key = [0u8; 52];
        storage_key[0..20].copy_from_slice(&STORAGE_ADDRESS);
        storage_key[20..52].copy_from_slice(&STORAGE_SLOT);
        let expected_hash = keccak256(&storage_key);
        let expected_tag: [u8; 8] = expected_hash[0..8].try_into().unwrap();
        assert_eq!(short_key.as_slice(), &expected_tag);
    }

    #[test]
    fn build_matrix_uses_8_byte_key_for_optimized48_storage() {
        let mut source = SingleStorageSource::new();
        let (matrix, manifest, _indexer) =
            build_matrix(&mut source, 1024, RowScheme::Optimized48, false);
        assert_eq!(manifest.item_count, 1);

        let expected_key = cuckoo_key_for_storage(&STORAGE_ADDRESS, &STORAGE_SLOT);
        let addresser = CuckooAddresser::with_seeds(1024, [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98]);
        let short_indices = addresser.hash_indices(&expected_key);

        let mut storage_key = [0u8; 52];
        storage_key[0..20].copy_from_slice(&STORAGE_ADDRESS);
        storage_key[20..52].copy_from_slice(&STORAGE_SLOT);
        let long_indices = addresser.hash_indices(&storage_key);

        let mut expected_payload = vec![0u8; 48];
        expected_payload[0..32].copy_from_slice(&STORAGE_VALUE);
        let tag_hash = keccak256(&storage_key);
        expected_payload[32..40].copy_from_slice(&tag_hash[0..8]);

        let chunk = matrix.chunk(0);
        let chunk_slice = chunk.as_slice();
        let row_size = 48;
        let row_at = |row: usize| -> &[u8] {
            let offset = row * row_size;
            &chunk_slice[offset..offset + row_size]
        };

        let found_short = short_indices
            .iter()
            .any(|&idx| row_at(idx) == expected_payload.as_slice());
        assert!(found_short, "expected 8-byte cuckoo key to be indexed");

        let long_only = long_indices
            .iter()
            .filter(|idx| !short_indices.contains(idx));
        for &idx in long_only {
            assert_ne!(
                row_at(idx),
                expected_payload.as_slice(),
                "unexpected payload at 52-byte key index"
            );
        }
    }
}

/// Compute the 8-byte Cuckoo key for a storage entry.
/// Key = keccak256(address || slot)[0..8]
pub fn cuckoo_key_for_storage(address: &[u8; 20], slot: &[u8; 32]) -> Vec<u8> {
    let mut k = [0u8; 52];
    k[0..20].copy_from_slice(address);
    k[20..52].copy_from_slice(slot);
    let tag_hash = keccak256(&k);
    tag_hash[0..8].to_vec()
}

// ...

#[cfg(feature = "reth")]
pub struct RethSource {
    db: Arc<DatabaseEnv>,
}

#[cfg(feature = "reth")]
impl RethSource {
    pub fn new(path: &str) -> Self {
        let db = Arc::new(
            reth_db::open_db_read_only(std::path::Path::new(path), Default::default()).unwrap(),
        );
        Self { db }
    }

    pub fn count_items(&self) -> usize {
        let tx = self.db.tx().expect("Failed to start transaction");
        let accounts = tx
            .entries::<reth_db::tables::PlainAccountState>()
            .expect("Failed to count accounts");
        let storage = tx
            .entries::<reth_db::tables::PlainStorageState>()
            .expect("Failed to count storage");
        accounts + storage
    }

    pub fn sample_data(&self, limit: usize) {
        let tx = self.db.tx().expect("Failed to start transaction");
        println!("--- Account Sample ---");

        // Try WETH
        let weth_addr = alloy_primitives::address!("C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2");
        let mut acc_cursor = tx
            .cursor_read::<reth_db::tables::PlainAccountState>()
            .expect("Cursor failed");
        if let Ok(Some((addr, acc))) = acc_cursor.seek_exact(weth_addr) {
            println!(
                "WETH {:?}: Nonce={}, Balance={}, CodeHash={:?}",
                addr, acc.nonce, acc.balance, acc.bytecode_hash
            );
        }

        let mut cursor = tx
            .cursor_read::<reth_db::tables::PlainAccountState>()
            .expect("Cursor failed");
        let mut count = 0;
        while let Some((addr, acc)) = cursor.next().unwrap() {
            if count >= limit {
                break;
            }
            println!(
                "Account {:?}: Nonce={}, Balance={}, CodeHash={:?}",
                addr, acc.nonce, acc.balance, acc.bytecode_hash
            );
            count += 1;
        }
    }

    pub fn verify_compression(&self) {
        let tx = self.db.tx().expect("Failed to start transaction");
        println!("Verifying 16-byte Balance Safety...");

        let mut cursor = tx
            .cursor_read::<reth_db::tables::PlainAccountState>()
            .expect("Cursor failed");
        let mut checked = 0;

        while let Some((addr, acc)) = cursor.next().unwrap() {
            if acc.balance.bit_len() > 128 {
                println!(
                    "CRITICAL: Account {:?} has balance bits {} > 128! Value: {}",
                    addr,
                    acc.balance.bit_len(),
                    acc.balance
                );
            }

            // Check Nonce compression too? Nonce is u64, fits in 8B naturally.

            checked += 1;
            if checked % 10_000_000 == 0 {
                println!("Checked {} accounts...", checked);
            }
        }
        println!(
            "Verification Complete. Checked {} accounts. All fit in 128 bits.",
            checked
        );
    }

    pub fn estimate_proof_sizes(&self, sample_size: usize) {
        use alloy_primitives::keccak256;
        use reth_db::cursor::DbCursorRO;
        use reth_db::tables::AccountsTrie;
        use reth_trie::{Nibbles, StoredNibbles};

        let tx = self.db.tx().expect("Failed to start transaction");
        let mut acc_cursor = tx
            .cursor_read::<reth_db::tables::PlainAccountState>()
            .expect("Cursor failed");

        let mut accounts = Vec::new();
        let mut count = 0;

        println!("Sampling {} accounts for proof estimation...", sample_size);
        while let Some((addr, _)) = acc_cursor.next().unwrap() {
            accounts.push(addr);
            count += 1;
            if count >= sample_size {
                break;
            }
        }

        let mut trie_cursor = tx
            .cursor_read::<AccountsTrie>()
            .expect("Trie cursor failed");
        let mut total_size = 0;
        let mut min_size = usize::MAX;
        let mut max_size = 0;
        let mut found_proofs = 0;

        for addr in &accounts {
            let hash = keccak256(addr);
            let nibbles = Nibbles::unpack(hash);

            let mut proof_size = 0;
            let mut nodes_found = 0;

            // Iterate all prefixes. MPT depth is up to 64 nibbles.
            for len in 0..=64 {
                // Slice the nibbles.
                // Note: Nibbles slice might return a new Nibbles or slice.
                // Assuming standard API. If not, we might need to adjust.
                // reth_trie::Nibbles usually implements typical vec-like methods.
                // If slice is not available, we might try taking first 'len' nibbles.
                let prefix = if len == 0 {
                    Nibbles::default()
                } else {
                    // Try slice
                    // If this fails compile, we'll fix it.
                    // Nibbles usually has `slice` taking range.
                    nibbles.slice(0..len)
                };

                let key: StoredNibbles = prefix.into();

                if let Ok(Some((_k, node))) = trie_cursor.seek_exact(key) {
                    // Estimate BranchNode size: hashes (32B each) + overhead
                    // BranchNodeCompact doesn't implement Encodable directly in this version.
                    proof_size += node.hashes.len() * 33 + 16;
                    nodes_found += 1;
                }
            }

            if nodes_found > 0 {
                // Add leaf account size approx (RLP of nonce, balance, storageRoot, codeHash) ~100 bytes
                proof_size += 100;

                total_size += proof_size;
                if proof_size < min_size {
                    min_size = proof_size;
                }
                if proof_size > max_size {
                    max_size = proof_size;
                }
                found_proofs += 1;
            }
        }

        println!(
            "Checked {} accounts. Found complete/partial proofs for {}.",
            accounts.len(),
            found_proofs
        );
        if found_proofs > 0 {
            let avg = total_size as f64 / found_proofs as f64;
            println!("Average MPT Proof Size: {:.2} bytes", avg);
            println!("Min Proof Size: {} bytes", min_size);
            println!("Max Proof Size: {} bytes", max_size);
        } else {
            println!("Warning: No trie nodes found. Is the AccountsTrie populated? (Requires hashed state and trie stages)");
        }
    }
}

use std::collections::HashMap;

pub struct CodeIndexer {
    map: HashMap<[u8; 32], u32>,
    pub list: Vec<[u8; 32]>,
}

impl CodeIndexer {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            list: Vec::new(),
        }
    }

    pub fn get_or_insert(&mut self, hash: Option<[u8; 32]>) -> u32 {
        // Map None (EOA) to all-zeros. Real contracts have Keccak hashes.
        let key = hash.unwrap_or([0u8; 32]);

        if let Some(&id) = self.map.get(&key) {
            return id;
        }

        let id = self.list.len() as u32;
        self.list.push(key);
        self.map.insert(key, id);
        id
    }
}

pub fn serialize_account(
    acc: &Account,
    scheme: RowScheme,
    code_indexer: &mut CodeIndexer,
) -> Vec<u8> {
    let row_size = match scheme {
        RowScheme::Compact => 32,
        RowScheme::Full => 64,
        RowScheme::Optimized48 => 48,
    };
    let mut payload = vec![0u8; row_size];

    // Balance (16 bytes)
    let balance_bytes = acc.balance.to_be_bytes();
    payload[0..16].copy_from_slice(&balance_bytes);

    // Nonce (8 bytes)
    payload[16..24].copy_from_slice(&acc.nonce.to_be_bytes());

    match scheme {
        RowScheme::Full => {
            payload[24..56].copy_from_slice(&acc.code_hash);
        }
        RowScheme::Compact => {
            // Use 4-byte ID
            let hash = if acc.code_hash == [0u8; 32] {
                None
            } else {
                Some(acc.code_hash)
            };
            let id = code_indexer.get_or_insert(hash);
            payload[24..28].copy_from_slice(&id.to_be_bytes());
            // 28..32 is padding
        }
        RowScheme::Optimized48 => {
            // Code ID (4 bytes)
            let hash = if acc.code_hash == [0u8; 32] {
                None
            } else {
                Some(acc.code_hash)
            };
            let id = code_indexer.get_or_insert(hash);
            payload[24..28].copy_from_slice(&id.to_be_bytes());

            // Tag (8 bytes) = Keccak(Address)[0..8]
            use alloy_primitives::keccak256;
            let tag_hash = keccak256(acc.address);
            payload[28..36].copy_from_slice(&tag_hash[0..8]);

            // Padding (12 bytes) is implicitly zero
        }
    }

    payload
}

#[cfg(feature = "reth")]
pub fn dump_reth_to_matrix(
    db_path: &str,
    num_rows: usize,
    scheme: RowScheme,
    _trustless: bool,
) -> (ChunkedMatrix, Manifest, CodeIndexer) {
    use reth_db::cursor::DbCursorRO;
    use reth_db::table::Table;
    use reth_db::Database;

    type AccTable = reth_db::tables::PlainAccountState;
    type StorageTable = reth_db::tables::PlainStorageState;

    let db = Arc::new(
        reth_db::open_db_read_only(std::path::Path::new(db_path), Default::default()).unwrap(),
    );
    let mut indexer = CodeIndexer::new();
    let row_size = match scheme {
        RowScheme::Compact => 32,
        RowScheme::Full => 64,
        RowScheme::Optimized48 => 48,
    };

    // 1. Load Cuckoo Table
    let seeds = [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98];
    let mut table = CuckooTable::<Vec<u8>>::with_seeds(num_rows, seeds);

    // 2. Stream Accounts (Batching)
    println!("Streaming Accounts...");
    let mut last_addr: Option<<AccTable as Table>::Key> = None;
    let mut account_count = 0;

    loop {
        let tx = db.tx().expect("Failed to start transaction");
        let mut acc_cursor = tx.cursor_read::<AccTable>().expect("Cursor failed");

        let mut batch_count = 0;
        let mut found_any = false;

        let mut it: Box<
            dyn Iterator<
                Item = Result<
                    (<AccTable as Table>::Key, <AccTable as Table>::Value),
                    reth_db::DatabaseError,
                >,
            >,
        > = if let Some(last) = last_addr {
            Box::new(acc_cursor.walk_range(last..).expect("Walk failed"))
        } else {
            Box::new(acc_cursor.walk(None).expect("Walk failed"))
        };

        // If we sought to last_addr, skip it
        if last_addr.is_some() {
            it.next();
        }

        while let Some(result) = it.next() {
            let (addr, acc): (<AccTable as Table>::Key, <AccTable as Table>::Value) =
                result.expect("Read failed");
            found_any = true;
            last_addr = Some(addr);

            let key = addr.to_vec();
            let my_acc = Account {
                address: addr.into(),
                balance: acc.balance.to::<u128>(),
                nonce: acc.nonce,
                code_hash: acc.bytecode_hash.map(|h| h.0).unwrap_or([0u8; 32]),
            };
            let p = serialize_account(&my_acc, scheme, &mut indexer);
            if let Err(_) = table.insert(key, p) {
                println!("Table Full!");
                break;
            }

            account_count += 1;
            batch_count += 1;
            if account_count % 1_000_000 == 0 {
                println!("  Processed {} accounts...", account_count);
            }

            if batch_count >= 5_000_000 {
                break;
            } // Refresh TX
        }

        if !found_any || batch_count < 5_000_000 {
            break;
        }
    }

    // 3. Stream Storage (Batching)
    println!("Streaming Storage...");
    use reth_db::cursor::DbDupCursorRO;

    let mut last_storage_addr: Option<<StorageTable as Table>::Key> = None;
    let mut last_storage_subkey: Option<alloy_primitives::B256> = None;
    let mut storage_count = 0;

    loop {
        let tx = db.tx().expect("Failed to start transaction");
        let mut storage_cursor = tx.cursor_dup_read::<StorageTable>().expect("Cursor failed");

        let mut batch_count = 0;
        let mut found_any = false;

        // Position cursor
        let mut current_item =
            if let (Some(addr), Some(subkey)) = (last_storage_addr, last_storage_subkey) {
                if let Ok(Some(_)) = storage_cursor.seek_by_key_subkey(addr, subkey) {
                    // Successfully positioned at last item. Move to next.
                    storage_cursor.next()
                } else {
                    // Fallback
                    storage_cursor.seek(addr)
                }
            } else {
                storage_cursor.first()
            };

        while let Ok(Some((addr, entry))) = current_item {
            found_any = true;
            last_storage_addr = Some(addr);
            last_storage_subkey = Some(entry.key);

            let mut k = Vec::with_capacity(52);
            k.extend_from_slice(addr.as_slice());
            k.extend_from_slice(entry.key.as_slice());

            let mut p = vec![0u8; row_size];
            p[0..32].copy_from_slice(entry.value.to_be_bytes::<32>().as_slice());

            if let Err(_) = table.insert(k, p) {
                println!("Table Full (Storage)!");
                break;
            }

            storage_count += 1;
            batch_count += 1;
            if storage_count % 10_000_000 == 0 {
                println!("  Processed {} storage slots...", storage_count);
            }

            if batch_count >= 5_000_000 {
                break;
            }

            current_item = storage_cursor.next();
        }

        if !found_any || batch_count < 5_000_000 {
            break;
        }
    }

    // 4. Flatten to Matrix
    println!("Flattening to ChunkedMatrix...");
    let chunk_size = 1024 * 1024 * 1024; // 1GB
    let total_size = num_rows * row_size;
    let mut matrix = ChunkedMatrix::new(total_size, chunk_size);

    for (idx, _key, val) in table.iter_enumerated() {
        matrix.write_row(idx, row_size, val);
    }

    let manifest = Manifest {
        block_number: 0,
        state_root: [0u8; 32],
        item_count: table.len(),
        cuckoo_seeds: seeds,
    };

    (matrix, manifest, indexer)
}

// ... SyntheticSource ...
pub struct SyntheticSource {
    count: usize,
    total: usize,
}

impl SyntheticSource {
    pub fn new(total: usize) -> Self {
        Self { count: 0, total }
    }
}

impl AccountSource for SyntheticSource {
    fn next_item(&mut self) -> Option<UbtItem> {
        if self.count >= self.total {
            return None;
        }
        let mut addr = [0u8; 20];
        let seed = self.count as u64;
        let bytes = seed.to_be_bytes();
        addr[12..20].copy_from_slice(&bytes);

        self.count += 1;

        // Demo: Every 5th item is storage
        if self.count % 5 == 0 {
            Some(UbtItem::Storage {
                address: addr,
                key: [0xAA; 32],
                value: [0xBB; 32],
            })
        } else {
            Some(UbtItem::Account(Account {
                address: addr,
                balance: seed as u128 * 1000,
                nonce: seed,
                code_hash: [0u8; 32],
            }))
        }
    }

    fn count_all(&mut self) -> usize {
        if self.count >= self.total {
            0
        } else {
            self.total - self.count
        }
    }
}

#[derive(serde::Serialize)]
pub struct Manifest {
    pub block_number: u64,
    pub state_root: [u8; 32], // UBT root
    pub item_count: usize,
    pub cuckoo_seeds: [u64; 3],
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum RowScheme {
    Compact,     // 32 bytes: Balance (16) + Nonce (8) + Padding (8) - Legacy
    Full,        // 64 bytes: Balance (16) + Nonce (8) + CodeHash (32) + Padding (8) - Legacy
    Optimized48, // 48 bytes: Bal(16) + Nonce(8) + ID(4) + Tag(8) + Pad(12)
}

/// Build the Cuckoo Matrix and Manifest from an account source.
pub fn build_matrix(
    source: &mut dyn AccountSource,
    num_rows: usize,
    scheme: RowScheme,
    trustless: bool,
) -> (ChunkedMatrix, Manifest, CodeIndexer) {
    let mut indexer = CodeIndexer::new();
    let row_size = match scheme {
        RowScheme::Compact => 32,
        RowScheme::Full => 64,
        RowScheme::Optimized48 => 48,
    };

    // 1. Load Cuckoo Table
    let seeds = [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98];
    let mut table = CuckooTable::<Vec<u8>>::with_seeds(num_rows, seeds);

    // 2. Insert Items
    while let Some(item) = source.next_item() {
        if trustless {
            // TODO: Implementation for trustless mode (2KB rows)
        } else {
            let (key, payload) = match item {
                UbtItem::Account(acc) => {
                    let p = serialize_account(&acc, scheme, &mut indexer);
                    (acc.address.to_vec(), p)
                }
                UbtItem::Storage {
                    address,
                    key: slot,
                    value,
                } => {
                    let mut full_key = Vec::with_capacity(52);
                    full_key.extend_from_slice(&address);
                    full_key.extend_from_slice(&slot);

                    let mut p = vec![0u8; row_size];

                    match scheme {
                        RowScheme::Optimized48 => {
                            // [Value (32) | Tag (8) | Pad (8)]
                            // Tag = keccak(address || slot)[0..8]
                            let cuckoo_key = cuckoo_key_for_storage(&address, &slot);
                            p[0..32].copy_from_slice(&value);
                            p[32..40].copy_from_slice(&cuckoo_key);
                            (cuckoo_key, p)
                        }
                        _ => {
                            // Legacy logic (Compact/Full): keep full 52-byte key
                            p[0..32].copy_from_slice(&value);
                            (full_key, p)
                        }
                    }
                }
            };

            if let Err(_) = table.insert(key, payload) {
                break;
            }
        }
    }

    // 3. Flatten to Matrix
    println!("Flattening to ChunkedMatrix...");
    let chunk_size = 1024 * 1024 * 1024; // 1GB
    let total_size = num_rows * row_size;
    let mut matrix = ChunkedMatrix::new(total_size, chunk_size);

    for (idx, _key, val) in table.iter_enumerated() {
        matrix.write_row(idx, row_size, val);
    }

    let manifest = Manifest {
        block_number: 0,
        state_root: [0u8; 32],
        item_count: table.len(),
        cuckoo_seeds: seeds,
    };

    (matrix, manifest, indexer)
}

#[cfg(feature = "reth")]
pub fn extract_code_from_dict(
    db_path: &str,
    dict_path: &std::path::Path,
    out_dir: &std::path::Path,
) {
    use reth_db::cursor::DbCursorRO;
    use reth_db::Database;
    use std::io::Read;
    use std::io::Write;

    // 1. Load Dictionary
    let mut file = std::fs::File::open(dict_path).expect("Failed to open dictionary");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Failed to read dictionary");

    if buffer.len() % 32 != 0 {
        panic!("Dictionary size not multiple of 32 bytes");
    }
    let num_hashes = buffer.len() / 32;
    println!("Loaded {} hashes from dictionary.", num_hashes);

    // 2. Open DB
    let db = Arc::new(
        reth_db::open_db_read_only(std::path::Path::new(db_path), Default::default()).unwrap(),
    );
    let tx = db.tx().expect("Failed to start transaction");
    let mut cursor = tx
        .cursor_read::<reth_db::tables::Bytecodes>()
        .expect("Cursor failed");

    std::fs::create_dir_all(out_dir).expect("Failed to create output dir");

    // 3. Iterate hashes and fetch code
    let mut found = 0;
    for i in 0..num_hashes {
        let start = i * 32;
        let hash_slice = &buffer[start..start + 32];
        let hash = alloy_primitives::B256::from_slice(hash_slice);

        if hash == alloy_primitives::B256::ZERO {
            continue;
        }

        if let Ok(Some((_, code))) = cursor.seek_exact(hash) {
            let hex_name = hex::encode(hash);

            // Sharding: out_dir/aa/bb/aabb...bin
            let shard1 = &hex_name[0..2];
            let shard2 = &hex_name[2..4];
            let shard_dir = out_dir.join(shard1).join(shard2);

            if i % 1000 == 0 {
                std::fs::create_dir_all(&shard_dir).expect("Failed to create shard dir");
            }

            let path = shard_dir.join(format!("{}.bin", hex_name));
            if !path.exists() {
                // Double check dir exists (less frequent)
                if !shard_dir.exists() {
                    std::fs::create_dir_all(&shard_dir).expect("Failed to create shard dir");
                }
                let mut f = std::fs::File::create(path).unwrap();
                f.write_all(code.original_byte_slice()).unwrap();
            }
            found += 1;
        }

        if i % 100_000 == 0 {
            println!("Processed {}/{} hashes...", i, num_hashes);
        }
    }
    println!("Extracted {}/{} contracts.", found, num_hashes);
}
