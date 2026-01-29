//! Page-level PIR client for privacy-preserving queries.
//!
//! This module provides the client-side API for page-level PIR, which offers
//! true 2-server computational privacy (unlike row-level AesDpfKey where servers
//! can see the target index).
//!
//! # Flow
//! 1. Client computes page addresses from account key using CuckooAddresser
//! 2. Client generates PageDpfKey pairs for each page address
//! 3. Client sends one key from each pair to each server
//! 4. Client XORs server responses to get plaintext pages
//! 5. Client extracts target row from each page locally

use morphogen_core::{sumcheck::SumCheckProof, CuckooAddresser, NUM_HASH_FUNCTIONS};
pub use morphogen_dpf::page::{
    extract_row_from_page, generate_page_dpf_keys, xor_pages, PageAddress, PageDpfError,
    PageDpfKey, PageDpfParams, PAGE_SIZE_BYTES, ROWS_PER_PAGE, ROW_SIZE_BYTES,
};

pub const QUERIES_PER_REQUEST: usize = NUM_HASH_FUNCTIONS;

#[derive(Clone)]
pub struct PageEpochMetadata {
    pub epoch_id: u64,
    pub num_pages: usize,
    pub seeds: [u64; NUM_HASH_FUNCTIONS],
    pub block_number: u64,
    pub state_root: [u8; 32],
    pub params: PageDpfParams,
}

pub struct PageQueryKeys {
    pub page_addresses: [PageAddress; 3],
    pub keys_a: [PageDpfKey; 3],
    pub keys_b: [PageDpfKey; 3],
}

pub struct PageServerResponse {
    pub epoch_id: u64,
    pub pages: [Vec<u8>; QUERIES_PER_REQUEST],
    pub proof: Option<SumCheckProof>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageAggregatedResult {
    pub epoch_id: u64,
    pub pages: [Vec<u8>; QUERIES_PER_REQUEST],
    pub proofs: [Option<SumCheckProof>; 2], // Proof from A and B
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageAggregationError {
    EpochMismatch {
        server_a: u64,
        server_b: u64,
    },
    PageLengthMismatch {
        index: usize,
        len_a: usize,
        len_b: usize,
    },
}

pub fn aggregate_page_responses(
    response_a: &PageServerResponse,
    response_b: &PageServerResponse,
) -> Result<PageAggregatedResult, PageAggregationError> {
    if response_a.epoch_id != response_b.epoch_id {
        return Err(PageAggregationError::EpochMismatch {
            server_a: response_a.epoch_id,
            server_b: response_b.epoch_id,
        });
    }

    let mut pages: [Vec<u8>; QUERIES_PER_REQUEST] = Default::default();

    for i in 0..QUERIES_PER_REQUEST {
        let a = &response_a.pages[i];
        let b = &response_b.pages[i];

        if a.len() != b.len() {
            return Err(PageAggregationError::PageLengthMismatch {
                index: i,
                len_a: a.len(),
                len_b: b.len(),
            });
        }

        pages[i] = vec![0u8; a.len()];
        xor_pages(a, b, &mut pages[i]);
    }

    Ok(PageAggregatedResult {
        epoch_id: response_a.epoch_id,
        pages,
        proofs: [response_a.proof.clone(), response_b.proof.clone()],
    })
}

pub fn generate_page_query(
    account_key: &[u8],
    metadata: &PageEpochMetadata,
) -> Result<PageQueryKeys, PageDpfError> {
    let num_rows =
        metadata
            .num_pages
            .checked_mul(ROWS_PER_PAGE)
            .ok_or(PageDpfError::InvalidDomainBits {
                domain_bits: metadata.params.domain_bits,
                reason: "num_pages * ROWS_PER_PAGE would overflow",
            })?;
    let addresser = CuckooAddresser::with_seeds(num_rows, metadata.seeds);
    let row_positions = addresser.hash_indices(account_key);

    let page_addresses: [PageAddress; 3] = [
        PageAddress::from_row_index(row_positions[0]),
        PageAddress::from_row_index(row_positions[1]),
        PageAddress::from_row_index(row_positions[2]),
    ];

    let (key0_a, key0_b) = generate_page_dpf_keys(&metadata.params, page_addresses[0].page_index)?;
    let (key1_a, key1_b) = generate_page_dpf_keys(&metadata.params, page_addresses[1].page_index)?;
    let (key2_a, key2_b) = generate_page_dpf_keys(&metadata.params, page_addresses[2].page_index)?;

    Ok(PageQueryKeys {
        page_addresses,
        keys_a: [key0_a, key1_a, key2_a],
        keys_b: [key0_b, key1_b, key2_b],
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractError {
    InvalidRowOffset { index: usize, row_offset: usize },
}

pub fn extract_rows_from_pages(
    result: &PageAggregatedResult,
    addresses: &[PageAddress; 3],
) -> Result<[Vec<u8>; 3], ExtractError> {
    let row0 = extract_row_from_page(&result.pages[0], addresses[0].row_offset)
        .ok_or(ExtractError::InvalidRowOffset {
            index: 0,
            row_offset: addresses[0].row_offset,
        })?
        .to_vec();
    let row1 = extract_row_from_page(&result.pages[1], addresses[1].row_offset)
        .ok_or(ExtractError::InvalidRowOffset {
            index: 1,
            row_offset: addresses[1].row_offset,
        })?
        .to_vec();
    let row2 = extract_row_from_page(&result.pages[2], addresses[2].row_offset)
        .ok_or(ExtractError::InvalidRowOffset {
            index: 2,
            row_offset: addresses[2].row_offset,
        })?
        .to_vec();
    Ok([row0, row1, row2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> PageEpochMetadata {
        let params = PageDpfParams::new(8).unwrap(); // 256 pages
        PageEpochMetadata {
            epoch_id: 1,
            num_pages: 256,
            seeds: [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98],
            block_number: 12345,
            state_root: [0u8; 32],
            params,
        }
    }

    #[test]
    fn generate_page_query_returns_three_page_addresses() {
        let metadata = test_metadata();
        let account = b"0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045";

        let query = generate_page_query(account, &metadata).unwrap();

        for addr in &query.page_addresses {
            assert!(
                addr.page_index < metadata.num_pages,
                "page_index {} >= num_pages {}",
                addr.page_index,
                metadata.num_pages
            );
            assert!(
                addr.row_offset < ROWS_PER_PAGE,
                "row_offset {} >= ROWS_PER_PAGE",
                addr.row_offset
            );
        }
    }

    #[test]
    fn generate_page_query_key_pairs_xor_to_point_function() {
        let metadata = test_metadata();
        let account = b"test_account";

        let query = generate_page_query(account, &metadata).unwrap();

        use fss_rs::group::byte::ByteGroup;
        use fss_rs::group::Group;

        for i in 0..3 {
            let target_page = query.page_addresses[i].page_index;
            let key_a = &query.keys_a[i];
            let key_b = &query.keys_b[i];

            let mut output_a = vec![ByteGroup::zero(); metadata.num_pages];
            let mut output_b = vec![ByteGroup::zero(); metadata.num_pages];

            key_a.full_eval(&mut output_a).unwrap();
            key_b.full_eval(&mut output_b).unwrap();

            for idx in 0..metadata.num_pages {
                let mut xor = [0u8; 16];
                for j in 0..16 {
                    xor[j] = output_a[idx].0[j] ^ output_b[idx].0[j];
                }

                if idx == target_page {
                    assert_eq!(
                        xor, [0xFF; 16],
                        "key pair {} should XOR to 0xFF at target page {}",
                        i, target_page
                    );
                } else {
                    assert_eq!(
                        xor, [0x00; 16],
                        "key pair {} should XOR to 0x00 at non-target page {}",
                        i, idx
                    );
                }
            }
        }
    }

    #[test]
    fn aggregate_page_responses_xors_pages() {
        let page_a = vec![0xAA; PAGE_SIZE_BYTES];
        let page_b = vec![0x55; PAGE_SIZE_BYTES];
        let expected: Vec<u8> = page_a.iter().zip(&page_b).map(|(&a, &b)| a ^ b).collect();

        let response_a = PageServerResponse {
            epoch_id: 1,
            pages: [
                page_a.clone(),
                vec![0; PAGE_SIZE_BYTES],
                vec![0; PAGE_SIZE_BYTES],
            ],
            proof: None,
        };
        let response_b = PageServerResponse {
            epoch_id: 1,
            pages: [
                page_b.clone(),
                vec![0; PAGE_SIZE_BYTES],
                vec![0; PAGE_SIZE_BYTES],
            ],
            proof: None,
        };

        let result = aggregate_page_responses(&response_a, &response_b).unwrap();
        assert_eq!(result.epoch_id, 1);
        assert_eq!(result.pages[0], expected);
    }

    #[test]
    fn aggregate_page_responses_epoch_mismatch() {
        let response_a = PageServerResponse {
            epoch_id: 1,
            pages: [
                vec![0; PAGE_SIZE_BYTES],
                vec![0; PAGE_SIZE_BYTES],
                vec![0; PAGE_SIZE_BYTES],
            ],
            proof: None,
        };
        let response_b = PageServerResponse {
            epoch_id: 2,
            pages: [
                vec![0; PAGE_SIZE_BYTES],
                vec![0; PAGE_SIZE_BYTES],
                vec![0; PAGE_SIZE_BYTES],
            ],
            proof: None,
        };

        let result = aggregate_page_responses(&response_a, &response_b);
        assert_eq!(
            result,
            Err(PageAggregationError::EpochMismatch {
                server_a: 1,
                server_b: 2
            })
        );
    }

    #[test]
    fn extract_rows_from_aggregated_pages() {
        let mut pages: [Vec<u8>; 3] = Default::default();
        for i in 0..3 {
            let mut page = vec![0u8; PAGE_SIZE_BYTES];
            for row in 0..ROWS_PER_PAGE {
                let start = row * ROW_SIZE_BYTES;
                page[start..start + ROW_SIZE_BYTES].fill((row + i * 16) as u8);
            }
            pages[i] = page;
        }

        let result = PageAggregatedResult {
            epoch_id: 42,
            pages,
            proofs: [None, None],
        };

        let addresses = [
            PageAddress {
                page_index: 0,
                row_offset: 5,
            },
            PageAddress {
                page_index: 1,
                row_offset: 10,
            },
            PageAddress {
                page_index: 2,
                row_offset: 15,
            },
        ];

        let rows = extract_rows_from_pages(&result, &addresses).unwrap();

        assert!(
            rows[0].iter().all(|&b| b == 5),
            "row0 should be filled with 5"
        );
        assert!(
            rows[1].iter().all(|&b| b == 26),
            "row1 should be filled with 26"
        );
        assert!(
            rows[2].iter().all(|&b| b == 47),
            "row2 should be filled with 47"
        );
    }

    #[test]
    fn same_account_same_page_addresses() {
        let metadata = test_metadata();
        let account = b"consistent_account";

        let query1 = generate_page_query(account, &metadata).unwrap();
        let query2 = generate_page_query(account, &metadata).unwrap();

        assert_eq!(query1.page_addresses, query2.page_addresses);
    }

    #[test]
    fn different_accounts_different_page_addresses() {
        let metadata = test_metadata();

        let query1 = generate_page_query(b"account_1", &metadata).unwrap();
        let query2 = generate_page_query(b"account_2", &metadata).unwrap();

        assert_ne!(query1.page_addresses, query2.page_addresses);
    }
}
