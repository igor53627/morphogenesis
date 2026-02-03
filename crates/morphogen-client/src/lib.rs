pub mod fixture;
pub mod gpu;
pub mod network;
pub mod page;
mod verification_test;

use morphogen_core::{CuckooAddresser, NUM_HASH_FUNCTIONS};
pub use morphogen_dpf::AesDpfKey;
use rand::Rng;

pub const QUERIES_PER_REQUEST: usize = NUM_HASH_FUNCTIONS;

pub struct EpochMetadata {
    pub epoch_id: u64,
    pub num_rows: usize,
    pub seeds: [u64; NUM_HASH_FUNCTIONS],
    pub block_number: u64,
    pub state_root: [u8; 32],
}

pub struct QueryKeys {
    pub positions: [usize; 3],
    pub keys_a: [AesDpfKey; 3],
    pub keys_b: [AesDpfKey; 3],
}

pub struct ServerResponse {
    pub epoch_id: u64,
    pub payloads: [Vec<u8>; QUERIES_PER_REQUEST],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregatedResult {
    pub epoch_id: u64,
    pub payloads: [Vec<u8>; QUERIES_PER_REQUEST],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregationError {
    EpochMismatch {
        server_a: u64,
        server_b: u64,
    },
    PayloadLengthMismatch {
        index: usize,
        len_a: usize,
        len_b: usize,
    },
}

pub fn aggregate_responses(
    response_a: &ServerResponse,
    response_b: &ServerResponse,
) -> Result<AggregatedResult, AggregationError> {
    if response_a.epoch_id != response_b.epoch_id {
        return Err(AggregationError::EpochMismatch {
            server_a: response_a.epoch_id,
            server_b: response_b.epoch_id,
        });
    }

    let mut payloads: [Vec<u8>; QUERIES_PER_REQUEST] = Default::default();

    for i in 0..QUERIES_PER_REQUEST {
        let a = &response_a.payloads[i];
        let b = &response_b.payloads[i];

        if a.len() != b.len() {
            return Err(AggregationError::PayloadLengthMismatch {
                index: i,
                len_a: a.len(),
                len_b: b.len(),
            });
        }

        payloads[i] = a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect();
    }

    Ok(AggregatedResult {
        epoch_id: response_a.epoch_id,
        payloads,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountData {
    pub nonce: u64,
    pub balance: u128,
    pub code_hash: Option<[u8; 32]>,
    pub code_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageData {
    pub value: [u8; 32],
}

impl AccountData {
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // Prioritize Compact Scheme (default for v4)
        // If it's a full page (4096) or exactly 32 bytes, treat as Compact.
        // Full scheme (64 bytes) is legacy/iceboxed.
        if bytes.len() == 32 || bytes.len() == 4096 {
            // Compact Scheme: [Balance(16) | Nonce(8) | CodeID(4) | Padding(4)]
            let balance = u128::from_be_bytes(bytes[0..16].try_into().ok()?);
            let nonce = u64::from_be_bytes(bytes[16..24].try_into().ok()?);
            let code_id = u32::from_be_bytes(bytes[24..28].try_into().ok()?);
            Some(Self {
                nonce,
                balance,
                code_hash: None,
                code_id: Some(code_id),
            })
        } else if bytes.len() == 64 {
            // Full Scheme: [Balance(16) | Nonce(8) | CodeHash(32) | Padding(8)]
            let balance = u128::from_be_bytes(bytes[0..16].try_into().ok()?);
            let nonce = u64::from_be_bytes(bytes[16..24].try_into().ok()?);
            let code_hash: [u8; 32] = bytes[24..56].try_into().ok()?;
            Some(Self {
                nonce,
                balance,
                code_hash: Some(code_hash),
                code_id: None,
            })
        } else {
            None
        }
    }
}

impl StorageData {
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // Storage payloads: [Value (32) | Tag (8) | Pad (8)] for Optimized48
        // or just [Value (32) | ...] for legacy schemes
        if bytes.len() >= 32 {
            let value: [u8; 32] = bytes[0..32].try_into().ok()?;
            Some(Self { value })
        } else {
            None
        }
    }
}

pub struct CodeResolver {
    pub dict_url: String,
    pub cas_base_url: String,
    pub client: reqwest::Client,
}

impl CodeResolver {
    pub fn new(dict_url: String, cas_base_url: String) -> Self {
        Self {
            dict_url,
            cas_base_url,
            client: reqwest::Client::new(),
        }
    }

    pub async fn resolve_code_hash(&self, code_id: u32) -> Result<[u8; 32], String> {
        let offset = code_id as u64 * 32;
        let range_header = format!("bytes={}-{}", offset, offset + 31);

        let response = self
            .client
            .get(&self.dict_url)
            .header("Range", range_header)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let bytes = response.bytes().await.map_err(|e| e.to_string())?;
        if bytes.len() < 32 {
            return Err("Received less than 32 bytes for hash".to_string());
        }

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[..32]);
        Ok(hash)
    }

    pub async fn fetch_bytecode(&self, code_hash: [u8; 32]) -> Result<Vec<u8>, String> {
        if code_hash == [0u8; 32] {
            return Ok(Vec::new()); // EOA
        }

        let hex_hash = hex::encode(code_hash);
        let shard1 = &hex_hash[0..2];
        let shard2 = &hex_hash[2..4];
        let url = format!(
            "{}/{}/{}/{}.bin",
            self.cas_base_url, shard1, shard2, hex_hash
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()));
        }

        let bytes = response.bytes().await.map_err(|e| e.to_string())?;
        Ok(bytes.to_vec())
    }

    pub async fn resolve_full_data(&self, data: &AccountData) -> Result<AccountData, String> {
        let mut resolved = data.clone();

        if let Some(code_id) = data.code_id {
            if resolved.code_hash.is_none() {
                resolved.code_hash = Some(self.resolve_code_hash(code_id).await?);
            }
        }

        Ok(resolved)
    }
}

pub fn generate_query<R: Rng>(
    rng: &mut R,
    account_key: &[u8],
    metadata: &EpochMetadata,
) -> QueryKeys {
    let addresser = CuckooAddresser::with_seeds(metadata.num_rows, metadata.seeds);
    let positions = addresser.hash_indices(account_key);

    let (key0_a, key0_b) = AesDpfKey::generate_pair(rng, positions[0]);
    let (key1_a, key1_b) = AesDpfKey::generate_pair(rng, positions[1]);
    let (key2_a, key2_b) = AesDpfKey::generate_pair(rng, positions[2]);

    QueryKeys {
        positions,
        keys_a: [key0_a, key1_a, key2_a],
        keys_b: [key0_b, key1_b, key2_b],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morphogen_dpf::DpfKey;

    fn test_metadata() -> EpochMetadata {
        EpochMetadata {
            epoch_id: 1,
            num_rows: 100_000,
            seeds: [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98],
            block_number: 12345,
            state_root: [0u8; 32],
        }
    }

    #[test]
    fn generate_query_returns_three_positions() {
        let mut rng = rand::thread_rng();
        let metadata = test_metadata();
        let account = b"0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045";

        let query = generate_query(&mut rng, account, &metadata);

        assert_eq!(query.positions.len(), 3);
        for &pos in &query.positions {
            assert!(pos < metadata.num_rows, "position {} >= num_rows", pos);
        }
    }

    #[test]
    fn generate_query_positions_match_cuckoo_addresser() {
        let mut rng = rand::thread_rng();
        let metadata = test_metadata();
        let account = b"0xABCDEF1234567890";

        let query = generate_query(&mut rng, account, &metadata);
        let addresser = CuckooAddresser::with_seeds(metadata.num_rows, metadata.seeds);
        let expected = addresser.hash_indices(account);

        assert_eq!(query.positions, expected);
    }

    #[test]
    fn generate_query_key_pairs_xor_to_point_function() {
        let mut rng = rand::thread_rng();
        let metadata = test_metadata();
        let account = b"test_account";

        let query = generate_query(&mut rng, account, &metadata);

        for i in 0..3 {
            let target = query.positions[i];
            let key_a = &query.keys_a[i];
            let key_b = &query.keys_b[i];

            for idx in 0..1000 {
                let a = key_a.eval_bit(idx);
                let b = key_b.eval_bit(idx);
                let xor = a ^ b;

                if idx == target {
                    assert!(xor, "key pair {} should XOR to 1 at target {}", i, target);
                } else {
                    assert!(!xor, "key pair {} should XOR to 0 at non-target {}", i, idx);
                }
            }
        }
    }

    #[test]
    fn generate_query_same_account_same_positions() {
        let mut rng1 = rand::thread_rng();
        let mut rng2 = rand::thread_rng();
        let metadata = test_metadata();
        let account = b"consistent_account";

        let query1 = generate_query(&mut rng1, account, &metadata);
        let query2 = generate_query(&mut rng2, account, &metadata);

        assert_eq!(query1.positions, query2.positions);
    }

    #[test]
    fn generate_query_different_accounts_different_positions() {
        let mut rng = rand::thread_rng();
        let metadata = test_metadata();

        let query1 = generate_query(&mut rng, b"account_1", &metadata);
        let query2 = generate_query(&mut rng, b"account_2", &metadata);

        assert_ne!(query1.positions, query2.positions);
    }

    #[test]
    fn generate_query_different_seeds_different_positions() {
        let mut rng = rand::thread_rng();
        let account = b"same_account";

        let metadata1 = EpochMetadata {
            epoch_id: 1,
            num_rows: 100_000,
            seeds: [0x1111, 0x2222, 0x3333],
            block_number: 100,
            state_root: [0u8; 32],
        };

        let metadata2 = EpochMetadata {
            epoch_id: 2,
            num_rows: 100_000,
            seeds: [0x4444, 0x5555, 0x6666],
            block_number: 200,
            state_root: [0u8; 32],
        };

        let query1 = generate_query(&mut rng, account, &metadata1);
        let query2 = generate_query(&mut rng, account, &metadata2);

        assert_ne!(query1.positions, query2.positions);
    }

    #[test]
    fn generate_query_keys_are_independent() {
        let mut rng = rand::thread_rng();
        let metadata = test_metadata();
        let account = b"independence_test";

        let query = generate_query(&mut rng, account, &metadata);

        let mut masks_a = [[0u8; 100]; 3];
        let mut masks_b = [[0u8; 100]; 3];

        for i in 0..3 {
            query.keys_a[i].eval_range_masks(0, &mut masks_a[i]);
            query.keys_b[i].eval_range_masks(0, &mut masks_b[i]);
        }

        assert_ne!(
            masks_a[0], masks_a[1],
            "key_a[0] should differ from key_a[1]"
        );
        assert_ne!(
            masks_a[1], masks_a[2],
            "key_a[1] should differ from key_a[2]"
        );
    }

    #[test]
    fn aggregate_responses_xors_payloads() {
        let response_a = ServerResponse {
            epoch_id: 1,
            payloads: [
                vec![0xAA, 0xBB, 0xCC],
                vec![0x11, 0x22, 0x33],
                vec![0xFF, 0x00, 0xFF],
            ],
        };
        let response_b = ServerResponse {
            epoch_id: 1,
            payloads: [
                vec![0x55, 0x44, 0x33],
                vec![0x11, 0x22, 0x33],
                vec![0x00, 0xFF, 0x00],
            ],
        };

        let result = aggregate_responses(&response_a, &response_b).unwrap();

        assert_eq!(result.epoch_id, 1);
        assert_eq!(
            result.payloads[0],
            vec![0xAA ^ 0x55, 0xBB ^ 0x44, 0xCC ^ 0x33]
        );
        assert_eq!(result.payloads[1], vec![0x00, 0x00, 0x00]);
        assert_eq!(result.payloads[2], vec![0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn aggregate_responses_recovers_original_payload() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let mask: Vec<u8> = vec![0x12, 0x34, 0x56, 0x78];
        let masked: Vec<u8> = original.iter().zip(&mask).map(|(&o, &m)| o ^ m).collect();

        let response_a = ServerResponse {
            epoch_id: 42,
            payloads: [mask.clone(), vec![0; 4], vec![0; 4]],
        };
        let response_b = ServerResponse {
            epoch_id: 42,
            payloads: [masked.clone(), vec![0; 4], vec![0; 4]],
        };

        let result = aggregate_responses(&response_a, &response_b).unwrap();
        assert_eq!(result.payloads[0], original);
    }

    #[test]
    fn aggregate_responses_epoch_mismatch_error() {
        let response_a = ServerResponse {
            epoch_id: 1,
            payloads: [vec![0], vec![0], vec![0]],
        };
        let response_b = ServerResponse {
            epoch_id: 2,
            payloads: [vec![0], vec![0], vec![0]],
        };

        let result = aggregate_responses(&response_a, &response_b);
        assert_eq!(
            result,
            Err(AggregationError::EpochMismatch {
                server_a: 1,
                server_b: 2
            })
        );
    }

    #[test]
    fn aggregate_responses_length_mismatch_error() {
        let response_a = ServerResponse {
            epoch_id: 1,
            payloads: [vec![0, 1, 2], vec![0], vec![0]],
        };
        let response_b = ServerResponse {
            epoch_id: 1,
            payloads: [vec![0, 1], vec![0], vec![0]],
        };

        let result = aggregate_responses(&response_a, &response_b);
        assert_eq!(
            result,
            Err(AggregationError::PayloadLengthMismatch {
                index: 0,
                len_a: 3,
                len_b: 2
            })
        );
    }

    #[test]
    fn aggregate_responses_empty_payloads() {
        let response_a = ServerResponse {
            epoch_id: 1,
            payloads: [vec![], vec![], vec![]],
        };
        let response_b = ServerResponse {
            epoch_id: 1,
            payloads: [vec![], vec![], vec![]],
        };

        let result = aggregate_responses(&response_a, &response_b).unwrap();
        assert!(result.payloads[0].is_empty());
    }

    #[test]
    fn aggregate_responses_large_payloads() {
        let size = 2048;
        let payload_a: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let payload_b: Vec<u8> = (0..size).map(|i| ((i * 7) % 256) as u8).collect();
        let expected: Vec<u8> = payload_a
            .iter()
            .zip(&payload_b)
            .map(|(&a, &b)| a ^ b)
            .collect();

        let response_a = ServerResponse {
            epoch_id: 99,
            payloads: [payload_a, vec![0; size], vec![0; size]],
        };
        let response_b = ServerResponse {
            epoch_id: 99,
            payloads: [payload_b, vec![0; size], vec![0; size]],
        };

        let result = aggregate_responses(&response_a, &response_b).unwrap();
        assert_eq!(result.payloads[0], expected);
    }

    #[test]
    fn account_data_parses_compact_row() {
        let mut row = vec![0u8; 32];
        // Balance = 1
        row[15] = 1;
        // Nonce = 2
        row[23] = 2;
        // CodeID = 0xDEADBEEF
        row[24] = 0xDE;
        row[25] = 0xAD;
        row[26] = 0xBE;
        row[27] = 0xEF;

        let data = AccountData::from_bytes(&row).unwrap();
        assert_eq!(data.balance, 1);
        assert_eq!(data.nonce, 2);
        assert_eq!(data.code_id, Some(0xDEADBEEF));
        assert!(data.code_hash.is_none());
    }

    #[test]
    fn account_data_parses_full_row() {
        let mut row = vec![0u8; 64];
        row[15] = 100;
        row[23] = 5;
        // CodeHash = all 0xAA
        for i in 24..56 {
            row[i] = 0xAA;
        }

        let data = AccountData::from_bytes(&row).unwrap();
        assert_eq!(data.balance, 100);
        assert_eq!(data.nonce, 5);
        assert_eq!(data.code_hash, Some([0xAA; 32]));
        assert!(data.code_id.is_none());
    }

    #[tokio::test]
    async fn test_code_resolver_resolve_hash() {
        let mut server = mockito::Server::new_async().await;
        let dict_url = format!("{}/dict.bin", server.url());

        let hash = [0x11u8; 32];
        let _m = server
            .mock("GET", "/dict.bin")
            .match_header("Range", "bytes=32-63")
            .with_body(&hash)
            .create_async()
            .await;

        let resolver = CodeResolver::new(dict_url, "http://localhost/cas".to_string());
        let resolved_hash = resolver.resolve_code_hash(1).await.unwrap();

        assert_eq!(resolved_hash, hash);
    }

    #[tokio::test]
    async fn test_code_resolver_fetch_bytecode() {
        let mut server = mockito::Server::new_async().await;
        let cas_base_url = server.url();

        let hash = [0xAAu8; 32];
        let hex_hash = hex::encode(hash);
        let shard1 = &hex_hash[0..2];
        let shard2 = &hex_hash[2..4];
        let path = format!("/{}/{}/{}.bin", shard1, shard2, hex_hash);

        let bytecode = vec![0x60, 0x80, 0x60, 0x40];
        let _m = server
            .mock("GET", path.as_str())
            .with_body(&bytecode)
            .create_async()
            .await;

        let resolver = CodeResolver::new("http://localhost/dict".to_string(), cas_base_url);
        let fetched_bytecode = resolver.fetch_bytecode(hash).await.unwrap();

        assert_eq!(fetched_bytecode, bytecode);
    }

    #[tokio::test]
    async fn test_code_resolver_e2e_flow() {
        let mut server = mockito::Server::new_async().await;
        let base_url = server.url();

        let code_id = 628088;
        let weth_hash =
            hex::decode("d0a06b12ac47863b5c7be4185c2deaad1c61557033f56c7d4ea74429cbb25e23")
                .unwrap();
        let weth_bytecode = vec![0x60, 0x80, 0x60, 0x40]; // Mock bytecode

        // Mock dictionary range request
        let _m1 = server
            .mock("GET", "/mainnet_compact.dict")
            .match_header(
                "Range",
                format!("bytes={}-{}", code_id * 32, code_id * 32 + 31).as_str(),
            )
            .with_body(&weth_hash)
            .create_async()
            .await;

        // Mock CAS request
        let shard1 = "d0";
        let shard2 = "a0";
        let hex_hash = "d0a06b12ac47863b5c7be4185c2deaad1c61557033f56c7d4ea74429cbb25e23";
        let cas_path = format!("/cas/{}/{}/{}.bin", shard1, shard2, hex_hash);
        let _m2 = server
            .mock("GET", cas_path.as_str())
            .with_body(&weth_bytecode)
            .create_async()
            .await;

        let resolver = CodeResolver::new(
            format!("{}/mainnet_compact.dict", base_url),
            format!("{}/cas", base_url),
        );

        // 1. Resolve hash
        let resolved_hash = resolver.resolve_code_hash(code_id).await.unwrap();
        assert_eq!(resolved_hash.as_slice(), weth_hash.as_slice());

        // 2. Fetch bytecode
        let fetched_bytecode = resolver.fetch_bytecode(resolved_hash).await.unwrap();
        assert_eq!(fetched_bytecode, weth_bytecode);
    }

    #[test]
    fn storage_data_parses_value() {
        let mut payload = vec![0u8; 48];
        // Storage value = 0x123
        payload[31] = 0x23;
        payload[30] = 0x01;

        let data = StorageData::from_bytes(&payload).unwrap();
        assert_eq!(data.value[31], 0x23);
        assert_eq!(data.value[30], 0x01);
        for i in 0..30 {
            assert_eq!(data.value[i], 0);
        }
    }

    #[test]
    fn storage_data_parses_zero_value() {
        let payload = vec![0u8; 48];
        let data = StorageData::from_bytes(&payload).unwrap();
        assert_eq!(data.value, [0u8; 32]);
    }

    #[test]
    fn storage_data_parses_full_value() {
        let mut payload = vec![0xFFu8; 48];
        for i in 0..32 {
            payload[i] = i as u8;
        }

        let data = StorageData::from_bytes(&payload).unwrap();
        for i in 0..32 {
            assert_eq!(data.value[i], i as u8);
        }
    }

    #[test]
    fn storage_data_rejects_short_payload() {
        let payload = vec![0u8; 16];
        assert!(StorageData::from_bytes(&payload).is_none());
    }

    #[test]
    fn storage_data_accepts_32_byte_payload() {
        let mut payload = vec![0u8; 32];
        payload[31] = 0x42;

        let data = StorageData::from_bytes(&payload).unwrap();
        assert_eq!(data.value[31], 0x42);
    }
}
