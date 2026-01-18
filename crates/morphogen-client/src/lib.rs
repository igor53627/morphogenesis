pub mod fixture;

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
}
