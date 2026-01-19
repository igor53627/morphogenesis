//! GPU-based Page PIR client.
//!
//! Uses ChaCha8-based DPF keys for high performance on GPU.

use morphogen_core::{CuckooAddresser, NUM_HASH_FUNCTIONS};
pub use morphogen_gpu_dpf::dpf::{
    generate_chacha_dpf_keys, ChaChaKey, ChaChaParams,
};
pub use morphogen_dpf::page::{
    extract_row_from_page, PageAddress, PAGE_SIZE_BYTES, ROWS_PER_PAGE, ROW_SIZE_BYTES,
};
use crate::page::{PageAggregatedResult, PageServerResponse, PageAggregationError, PageAggregatedResult as GpuPageAggregatedResult};

pub struct GpuPageQueryKeys {
    pub page_addresses: [PageAddress; 3],
    pub keys_a: [ChaChaKey; 3],
    pub keys_b: [ChaChaKey; 3],
}

pub fn generate_gpu_page_query(
    account_key: &[u8],
    num_pages: usize,
    seeds: [u64; 3],
    domain_bits: usize,
) -> Result<GpuPageQueryKeys, morphogen_gpu_dpf::dpf::GpuDpfError> {
    let num_rows = num_pages
        .checked_mul(ROWS_PER_PAGE)
        .ok_or(morphogen_gpu_dpf::dpf::GpuDpfError::InvalidDomainBits {
            bits: domain_bits,
        })?;
    let addresser = CuckooAddresser::with_seeds(num_rows, seeds);
    let row_positions = addresser.hash_indices(account_key);

    let page_addresses: [PageAddress; 3] = [
        PageAddress::from_row_index(row_positions[0]),
        PageAddress::from_row_index(row_positions[1]),
        PageAddress::from_row_index(row_positions[2]),
    ];

    let params = ChaChaParams::new(domain_bits)?;

    let (key0_a, key0_b) = generate_chacha_dpf_keys(&params, page_addresses[0].page_index)?;
    let (key1_a, key1_b) = generate_chacha_dpf_keys(&params, page_addresses[1].page_index)?;
    let (key2_a, key2_b) = generate_chacha_dpf_keys(&params, page_addresses[2].page_index)?;

    Ok(GpuPageQueryKeys {
        page_addresses,
        keys_a: [key0_a, key1_a, key2_a],
        keys_b: [key0_b, key1_b, key2_b],
    })
}

// Re-use aggregation logic from page module
pub use crate::page::aggregate_page_responses as aggregate_gpu_page_responses;
pub use crate::page::extract_rows_from_pages as extract_rows_from_gpu_pages;
