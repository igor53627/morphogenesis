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

    pub fn sample_data(&self, limit: usize) {
        let tx = self.db.tx().expect("Failed to start transaction");
        
        // Sample Accounts
        println!("--- Account Sample ---");
        let mut cursor = tx.cursor_read::<reth_db::tables::PlainAccountState>().expect("Cursor failed");
        let mut count = 0;
        while let Some((addr, acc)) = cursor.next().unwrap() {
            if count >= limit { break; }
            println!("Account {:?}: Nonce={}, Balance={}, CodeHash={:?}", addr, acc.nonce, acc.balance, acc.bytecode_hash);
            // Analyze byte usage
            let b_bytes = (128 - acc.balance.leading_zeros() + 7) / 8;
            println!("  Balance bits: {}, Bytes needed: {}", 128 - acc.balance.leading_zeros(), b_bytes);
            count += 1;
        }

        // Sample Storage
        println!("--- Storage Sample ---");
        let mut cursor = tx.cursor_read::<reth_db::tables::PlainStorageState>().expect("Cursor failed");
        let mut count = 0;
        while let Some((addr, entry)) = cursor.next().unwrap() {
            if count >= limit { break; }
            println!("Storage {:?}: Key={:?}, Value={:?}", addr, entry.key, entry.value);
            count += 1;
        }
    }
}

#[cfg(feature = "reth")]
pub fn dump_reth_to_matrix(
    db_path: &str,
    num_rows: usize,
    scheme: RowScheme,
    trustless: bool,
) -> (ChunkedMatrix, Manifest) {
    use reth_db::Database;
    
    let db = Arc::new(reth_db::open_db_read_only(std::path::Path::new(db_path), Default::default()).unwrap());
    let tx = db.tx().expect("Failed to start transaction");
    
    let row_size = match scheme {
        RowScheme::Compact => 32,
        RowScheme::Full => 64,
    };

    // 1. Load Cuckoo Table
    let seeds = [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98];
    let mut table = CuckooTable::<Vec<u8>>::with_seeds(num_rows, seeds);

    // 2. Stream Accounts
    println!("Streaming Accounts...");
    let mut acc_cursor = tx.cursor_read::<reth_db::tables::PlainAccountState>().expect("Cursor failed");
    while let Some((addr, acc)) = acc_cursor.next().unwrap() {
        let key = addr.to_vec();
        // Convert reth account to our Account struct
        let my_acc = Account {
            address: addr.into(),
            balance: acc.balance.to::<u128>(),
            nonce: acc.nonce,
            code_hash: acc.bytecode_hash.map(|h| h.0).unwrap_or([0u8; 32]),
        };
        let p = serialize_account(&my_acc, scheme);
        if let Err(_) = table.insert(key, p) {
            break;
        }
    }

    // 3. Stream Storage
    println!("Streaming Storage...");
    let mut storage_cursor = tx.cursor_read::<reth_db::tables::PlainStorageState>().expect("Cursor failed");
    while let Some((addr, entry)) = storage_cursor.next().unwrap() {
        let mut k = Vec::with_capacity(52);
        k.extend_from_slice(addr.as_slice());
        k.extend_from_slice(entry.key.as_slice());
        
        let mut p = vec![0u8; row_size];
        p[0..32].copy_from_slice(entry.value.to_be_bytes::<32>().as_slice());
        
        if let Err(_) = table.insert(k, p) {
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
    
    (matrix, manifest)
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

pub fn serialize_account(acc: &Account, scheme: RowScheme) -> Vec<u8> {
    let row_size = match scheme {
        RowScheme::Compact => 32,
        RowScheme::Full => 64,
    };
    let mut payload = vec![0u8; row_size];
    
    // Balance (clamped to 16 bytes for 32B rows, which is safe for ETH supply)
    let balance_bytes = acc.balance.to_be_bytes();
    payload[0..16].copy_from_slice(&balance_bytes);
    
    // Nonce (8 bytes)
    payload[16..24].copy_from_slice(&acc.nonce.to_be_bytes());
    
    // Fields specific to Full scheme
    if let RowScheme::Full = scheme {
        payload[24..56].copy_from_slice(&acc.code_hash);
        // Bytes 56..64 are zeros (padding)
    } else {
        // Bytes 24..32 are zeros (padding)
    }
    
    payload
}

/// Build the Cuckoo Matrix and Manifest from an account source.
pub fn build_matrix(
    source: &mut dyn AccountSource,
    num_rows: usize,
    scheme: RowScheme,
    trustless: bool,
) -> (ChunkedMatrix, Manifest) {
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
                    let p = serialize_account(&acc, scheme);
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
    
    (matrix, manifest)
}
