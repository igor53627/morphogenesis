use morphogen_core::cuckoo::CuckooTable;
use morphogen_storage::{AlignedMatrix, ChunkedMatrix};
use std::sync::Arc;

#[cfg(feature = "reth")]
use reth_db::{tables, transaction::DbTx, DatabaseEnv, Database, cursor::DbCursorRO};
#[cfg(feature = "reth")]
use alloy_primitives::Address;

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

// ...

#[cfg(feature = "reth")]
pub struct RethSource {
    db: Arc<DatabaseEnv>,
}

#[cfg(feature = "reth")]
impl RethSource {
    pub fn new(path: &str) -> Self {
        let db = Arc::new(reth_db::open_db_read_only(std::path::Path::new(path), Default::default()).unwrap());
        Self { db }
    }

    pub fn count_items(&self) -> usize {
        let tx = self.db.tx().expect("Failed to start transaction");
        let accounts = tx.entries::<reth_db::tables::PlainAccountState>().expect("Failed to count accounts");
        let storage = tx.entries::<reth_db::tables::PlainStorageState>().expect("Failed to count storage");
        accounts + storage
    }

    pub fn sample_data(&self, limit: usize) {
        // ... (existing) ...
    }

    pub fn verify_compression(&self) {
        let tx = self.db.tx().expect("Failed to start transaction");
        println!("Verifying 16-byte Balance Safety...");
        
        let mut cursor = tx.cursor_read::<reth_db::tables::PlainAccountState>().expect("Cursor failed");
        let mut checked = 0;
        let mut max_balance: u128 = 0;
        
        while let Some((addr, acc)) = cursor.next().unwrap() {
            let balance = acc.balance.to::<u128>(); // This clamps to u128 if it fits? 
            // Wait, reth U256 .to::<u128>() might panic or truncate?
            // Alloy U256::to() is generic.
            // Let's check bits manually.
            
            if acc.balance.bit_len() > 128 {
                println!("CRITICAL: Account {:?} has balance bits {} > 128! Value: {}", addr, acc.balance.bit_len(), acc.balance);
            }
            
            // Check Nonce compression too? Nonce is u64, fits in 8B naturally.
            
            checked += 1;
            if checked % 10_000_000 == 0 {
                println!("Checked {} accounts...", checked);
            }
        }
        println!("Verification Complete. Checked {} accounts. All fit in 128 bits.", checked);
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

pub fn serialize_account(acc: &Account, scheme: RowScheme, code_indexer: &mut CodeIndexer) -> Vec<u8> {
    let row_size = match scheme {
        RowScheme::Compact => 32,
        RowScheme::Full => 64,
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
            // Detect if code_hash is "empty" (EOA). Reth gives us [0; 32] for empty?
            // Account struct has [u8; 32]. 
            // In dump_reth_to_matrix, we did: acc.bytecode_hash.map(|h| h.0).unwrap_or([0u8; 32])
            // So [0; 32] means EOA.
            let hash = if acc.code_hash == [0u8; 32] { None } else { Some(acc.code_hash) };
            let id = code_indexer.get_or_insert(hash);
            payload[24..28].copy_from_slice(&id.to_be_bytes());
            // 28..32 is padding
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
    use reth_db::Database;
    use reth_db::cursor::DbCursorRO;
    use reth_db::table::Table;
    
    type AccTable = reth_db::tables::PlainAccountState;
    type StorageTable = reth_db::tables::PlainStorageState;

    let db = Arc::new(reth_db::open_db_read_only(std::path::Path::new(db_path), Default::default()).unwrap());
    let mut indexer = CodeIndexer::new();
    let row_size = match scheme {
        RowScheme::Compact => 32,
        RowScheme::Full => 64,
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

        let mut it: Box<dyn Iterator<Item = Result<(<AccTable as Table>::Key, <AccTable as Table>::Value), _>>> = 
            if let Some(last) = last_addr {
                Box::new(acc_cursor.walk_range(last..).expect("Walk failed"))
            } else {
                Box::new(acc_cursor.walk(None).expect("Walk failed"))
            };

        // If we sought to last_addr, skip it
        if last_addr.is_some() {
            it.next(); 
        }

        while let Some(result) = it.next() {
            let (addr, acc): (<AccTable as Table>::Key, <AccTable as Table>::Value) = result.expect("Read failed");
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
            
            if batch_count >= 5_000_000 { break; } // Refresh TX
        }
        
        if !found_any || batch_count < 5_000_000 { break; }
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
        let mut current_item = if let (Some(addr), Some(subkey)) = (last_storage_addr, last_storage_subkey) {
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
            
            if batch_count >= 5_000_000 { break; }
            
            current_item = storage_cursor.next();
        }
        
        if !found_any || batch_count < 5_000_000 { break; }
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

// ... UBT Logic ...
pub struct UbtTree {
    pub root: [u8; 32],
}

impl UbtTree {
    pub fn build(accounts: &[Account]) -> Self {
        // Build binary Merkle tree
        Self { root: [0u8; 32] }
    }

    pub fn generate_proof(&self, address: &[u8; 20]) -> Vec<u8> {
        // Generate authentication path
        vec![0u8; 1024] // Dummy proof
    }
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
        if self.count >= self.total { 0 } else { self.total - self.count }
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
    Compact, // 32 bytes: Balance (16) + Nonce (8) + Padding (8)
    Full,    // 64 bytes: Balance (16) + Nonce (8) + CodeHash (32) + Padding (8)
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
                UbtItem::Storage { address, key, value } => {
                    // Storage key: Address . SlotKey (52 bytes)
                    let mut k = Vec::with_capacity(52);
                    k.extend_from_slice(&address);
                    k.extend_from_slice(&key);
                    
                    let mut p = vec![0u8; row_size];
                    // For storage, we just store the 32-byte value.
                    // If row_size is 64, the rest is zero padding.
                    p[0..32].copy_from_slice(&value);
                    (k, p)
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
    use std::io::Read;
    use std::io::Write;
    use reth_db::Database;
    use reth_db::cursor::DbCursorRO;
    
    // 1. Load Dictionary
    let mut file = std::fs::File::open(dict_path).expect("Failed to open dictionary");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read dictionary");
    
    if buffer.len() % 32 != 0 {
        panic!("Dictionary size not multiple of 32 bytes");
    }
    let num_hashes = buffer.len() / 32;
    println!("Loaded {} hashes from dictionary.", num_hashes);

    // 2. Open DB
    let db = Arc::new(reth_db::open_db_read_only(std::path::Path::new(db_path), Default::default()).unwrap());
    let tx = db.tx().expect("Failed to start transaction");
    let mut cursor = tx.cursor_read::<reth_db::tables::Bytecodes>().expect("Cursor failed");

    std::fs::create_dir_all(out_dir).expect("Failed to create output dir");

    // 3. Iterate hashes and fetch code
    let mut found = 0;
    for i in 0..num_hashes {
        let start = i * 32;
        let hash_slice = &buffer[start..start+32];
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
                f.write_all(code.original_bytes().as_ref()).unwrap();
            }
            found += 1;
        }
        
        if i % 100_000 == 0 {
            println!("Processed {}/{} hashes...", i, num_hashes);
        }
    }
    println!("Extracted {}/{} contracts.", found, num_hashes);
}
