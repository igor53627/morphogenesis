use morphogen_core::cuckoo::CuckooTable;
use morphogen_storage::{AlignedMatrix, ChunkedMatrix};
use std::sync::Arc;

#[cfg(feature = "reth")]
use reth_db::{tables, transaction::DbTx, DatabaseEnv, cursor::DbCursorRO};
#[cfg(feature = "reth")]
use reth_primitives::Address;

pub struct Account {
    pub address: [u8; 20],
    pub balance: u128,
    pub nonce: u64,
    pub code_hash: [u8; 32],
}

pub trait AccountSource {
    fn next_account(&mut self) -> Option<Account>;
}

#[cfg(feature = "reth")]
pub struct RethSource {
    db: Arc<DatabaseEnv>,
    // In a real implementation, we would store the transaction and cursor here
    // But for a trait-based source, we might need a different approach (e.g. streaming)
}

#[cfg(feature = "reth")]
impl AccountSource for RethSource {
    fn next_account(&mut self) -> Option<Account> {
        // Implementation that walks PlainAccountState
        None 
    }
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
    count: usize,
    total: usize,
}

impl SyntheticSource {
    pub fn new(total: usize) -> Self {
        Self { count: 0, total }
    }
}

impl AccountSource for SyntheticSource {
    fn next_account(&mut self) -> Option<Account> {
        if self.count >= self.total {
            return None;
        }
        let mut addr = [0u8; 20];
        // Deterministic address generation
        let seed = self.count as u64;
        let bytes = seed.to_be_bytes();
        addr[12..20].copy_from_slice(&bytes);
        
        self.count += 1;
        Some(Account {
            address: addr,
            balance: seed as u128 * 1000,
            nonce: seed,
        })
    }
}

/// Build the Cuckoo Matrix and EpochMetadata from an account source.
pub fn build_matrix(
    source: &mut dyn AccountSource,
    num_rows: usize,
    row_size: usize,
    trustless: bool,
) -> (ChunkedMatrix, [u64; 3]) {
    // 1. Load Cuckoo Table
    let seeds = [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98];
    let mut table = CuckooTable::<Vec<u8>>::with_seeds(num_rows, seeds);

    // 2. Insert Accounts
    let mut accounts = Vec::new();
    while let Some(acc) = source.next_account() {
        if trustless {
            accounts.push(acc);
        } else {
            let key = acc.address.to_vec();
            let mut payload = vec![0u8; row_size];
            payload[0..16].copy_from_slice(&acc.balance.to_be_bytes());
            payload[16..24].copy_from_slice(&acc.nonce.to_be_bytes());
            payload[24..56].copy_from_slice(&acc.code_hash);
            
            if let Err(_) = table.insert(key, payload) {
                break;
            }
        }
    }

    if trustless {
        let tree = UbtTree::build(&accounts);
        for acc in accounts {
            let key = acc.address.to_vec();
            let proof = tree.generate_proof(&acc.address);
            let mut payload = vec![0u8; row_size];
            payload[0..16].copy_from_slice(&acc.balance.to_be_bytes());
            payload[16..24].copy_from_slice(&acc.nonce.to_be_bytes());
            payload[24..56].copy_from_slice(&acc.code_hash);
            
            // Append proof (assuming row_size is large enough, e.g. 2KB)
            let proof_len = proof.len().min(row_size - 56);
            payload[56..56+proof_len].copy_from_slice(&proof[..proof_len]);

            if let Err(_) = table.insert(key, payload) {
                break;
            }
        }
    }

    // 2. Flatten to Matrix
    println!("Flattening to ChunkedMatrix...");
    let chunk_size = 1024 * 1024 * 1024; // 1GB
    let total_size = num_rows * row_size;
    let mut matrix = ChunkedMatrix::new(total_size, chunk_size);
    
    for (idx, _key, val) in table.iter_enumerated() {
        matrix.write_row(idx, row_size, val);
    }
    
    (matrix, seeds)
}

pub fn calculate_diff(old_payload: &[u8], new_payload: &[u8]) -> Vec<u8> {
    old_payload.iter().zip(new_payload.iter()).map(|(o, n)| o ^ n).collect()
}

pub fn serialize_account(acc: &Account, row_size: usize) -> Vec<u8> {
    let mut payload = vec![0u8; row_size];
    payload[0..16].copy_from_slice(&acc.balance.to_be_bytes());
    payload[16..24].copy_from_slice(&acc.nonce.to_be_bytes());
    payload[24..56].copy_from_slice(&acc.code_hash);
    payload
}
