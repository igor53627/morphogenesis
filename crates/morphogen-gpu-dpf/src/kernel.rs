//! Fused DPF+DB kernel implementation.
//!
//! This module provides both a CPU reference implementation and
//! a GPU kernel stub for the fused DPF evaluation.
//!
//! The fused kernel performs:
//! 1. DPF evaluation to generate 16-byte masks for each page
//! 2. AND-mask with database pages
//! 3. XOR-accumulate into output
//!
//! Processing is done in tiles (subtrees) of SUBTREE_SIZE pages,
//! with all 3 Cuckoo DPF keys evaluated in a single pass.

use crate::chacha_prg::Seed128;
use crate::dpf::ChaChaKey;

/// Size of each page in bytes.
pub const PAGE_SIZE_BYTES: usize = 4096;

/// Default subtree size for kernel processing.
/// Each block processes this many pages.
pub const SUBTREE_SIZE: usize = 1024;

/// Timing breakdown for fused kernel execution.
#[derive(Debug, Clone, Default)]
pub struct KernelTiming {
    pub dpf_eval_ns: u64,
    pub xor_accumulate_ns: u64,
    pub total_ns: u64,
}

/// Result of fused PIR query (3 pages for 3 Cuckoo keys).
#[derive(Debug, Clone)]
pub struct PirResult {
    pub page0: Vec<u8>,
    pub page1: Vec<u8>,
    pub page2: Vec<u8>,
    pub timing: KernelTiming,
}

/// Apply a 16-byte mask to a page and XOR into accumulator.
#[inline(always)]
fn xor_page_masked(acc: &mut [u8], page: &[u8], mask: &Seed128) {
    debug_assert_eq!(acc.len(), PAGE_SIZE_BYTES);
    debug_assert_eq!(page.len(), PAGE_SIZE_BYTES);

    let mask_bytes = mask.to_bytes();

    // Simple byte-by-byte XOR with mask cycling
    // Compiler optimizes this well for modern CPUs
    for (i, (a, p)) in acc.iter_mut().zip(page.iter()).enumerate() {
        let m = mask_bytes[i % 16];
        *a ^= *p & m;
    }
}

/// CPU reference implementation of fused 3-key PIR evaluation.
///
/// Processes pages in tiles and evaluates all 3 DPF keys in a single pass.
pub fn eval_fused_3dpf_cpu(
    keys: [&ChaChaKey; 3],
    pages: &[&[u8]],
) -> Result<PirResult, &'static str> {
    let num_pages = pages.len();
    let domain_size = keys[0].max_pages();

    if num_pages > domain_size {
        return Err("More pages than domain size");
    }

    for page in pages {
        if page.len() != PAGE_SIZE_BYTES {
            return Err("Invalid page size");
        }
    }

    let start = std::time::Instant::now();

    // Initialize accumulators
    let mut acc0 = vec![0u8; PAGE_SIZE_BYTES];
    let mut acc1 = vec![0u8; PAGE_SIZE_BYTES];
    let mut acc2 = vec![0u8; PAGE_SIZE_BYTES];

    let mut dpf_ns: u64 = 0;
    let mut xor_ns: u64 = 0;

    // Process in tiles
    let num_tiles = (num_pages + SUBTREE_SIZE - 1) / SUBTREE_SIZE;

    for tile in 0..num_tiles {
        let tile_start = tile * SUBTREE_SIZE;
        let tile_end = (tile_start + SUBTREE_SIZE).min(num_pages);

        // Evaluate DPF masks for this tile
        let dpf_start = std::time::Instant::now();

        // Pre-compute masks for all pages in tile
        let mut masks0 = Vec::with_capacity(tile_end - tile_start);
        let mut masks1 = Vec::with_capacity(tile_end - tile_start);
        let mut masks2 = Vec::with_capacity(tile_end - tile_start);

        for page_idx in tile_start..tile_end {
            masks0.push(keys[0].eval(page_idx));
            masks1.push(keys[1].eval(page_idx));
            masks2.push(keys[2].eval(page_idx));
        }

        dpf_ns += dpf_start.elapsed().as_nanos() as u64;

        // Apply masks and accumulate
        let xor_start = std::time::Instant::now();

        for (i, page_idx) in (tile_start..tile_end).enumerate() {
            let page = pages[page_idx];
            xor_page_masked(&mut acc0, page, &masks0[i]);
            xor_page_masked(&mut acc1, page, &masks1[i]);
            xor_page_masked(&mut acc2, page, &masks2[i]);
        }

        xor_ns += xor_start.elapsed().as_nanos() as u64;
    }

    let total_ns = start.elapsed().as_nanos() as u64;

    Ok(PirResult {
        page0: acc0,
        page1: acc1,
        page2: acc2,
        timing: KernelTiming {
            dpf_eval_ns: dpf_ns,
            xor_accumulate_ns: xor_ns,
            total_ns,
        },
    })
}

/// Evaluate single DPF key over all pages (for testing/comparison).
pub fn eval_single_dpf_cpu(key: &ChaChaKey, pages: &[&[u8]]) -> Vec<u8> {
    let mut acc = vec![0u8; PAGE_SIZE_BYTES];

    for (i, page) in pages.iter().enumerate() {
        let mask = key.eval(i);
        xor_page_masked(&mut acc, page, &mask);
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpf::{generate_chacha_dpf_keys, ChaChaParams};

    fn make_test_pages(num_pages: usize) -> Vec<Vec<u8>> {
        (0..num_pages)
            .map(|i| {
                let mut page = vec![0u8; PAGE_SIZE_BYTES];
                // Fill with identifiable pattern
                page[0] = (i & 0xFF) as u8;
                page[1] = ((i >> 8) & 0xFF) as u8;
                for j in 2..PAGE_SIZE_BYTES {
                    page[j] = ((i + j) & 0xFF) as u8;
                }
                page
            })
            .collect()
    }

    #[test]
    fn single_dpf_recovers_target_page() {
        let params = ChaChaParams::new(8).unwrap();
        let target = 42;
        let (k0, k1) = generate_chacha_dpf_keys(&params, target).unwrap();

        let pages_data = make_test_pages(256);
        let pages: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

        let result0 = eval_single_dpf_cpu(&k0, &pages);
        let result1 = eval_single_dpf_cpu(&k1, &pages);

        // XOR results to recover target page
        let mut recovered = vec![0u8; PAGE_SIZE_BYTES];
        for i in 0..PAGE_SIZE_BYTES {
            recovered[i] = result0[i] ^ result1[i];
        }

        assert_eq!(recovered, pages_data[target]);
    }

    #[test]
    fn fused_3dpf_recovers_all_targets() {
        let params = ChaChaParams::new(8).unwrap();
        let targets = [10, 100, 200];

        // Generate 3 key pairs (one per Cuckoo hash)
        let (k0_0, k0_1) = generate_chacha_dpf_keys(&params, targets[0]).unwrap();
        let (k1_0, k1_1) = generate_chacha_dpf_keys(&params, targets[1]).unwrap();
        let (k2_0, k2_1) = generate_chacha_dpf_keys(&params, targets[2]).unwrap();

        let pages_data = make_test_pages(256);
        let pages: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

        // Server 0 evaluation
        let result0 = eval_fused_3dpf_cpu([&k0_0, &k1_0, &k2_0], &pages).unwrap();

        // Server 1 evaluation
        let result1 = eval_fused_3dpf_cpu([&k0_1, &k1_1, &k2_1], &pages).unwrap();

        // XOR to recover pages
        let mut recovered0 = vec![0u8; PAGE_SIZE_BYTES];
        let mut recovered1 = vec![0u8; PAGE_SIZE_BYTES];
        let mut recovered2 = vec![0u8; PAGE_SIZE_BYTES];

        for i in 0..PAGE_SIZE_BYTES {
            recovered0[i] = result0.page0[i] ^ result1.page0[i];
            recovered1[i] = result0.page1[i] ^ result1.page1[i];
            recovered2[i] = result0.page2[i] ^ result1.page2[i];
        }

        assert_eq!(recovered0, pages_data[targets[0]]);
        assert_eq!(recovered1, pages_data[targets[1]]);
        assert_eq!(recovered2, pages_data[targets[2]]);
    }

    #[test]
    fn fused_timing_is_recorded() {
        let params = ChaChaParams::new(8).unwrap();
        let (k0, _) = generate_chacha_dpf_keys(&params, 0).unwrap();
        let (k1, _) = generate_chacha_dpf_keys(&params, 1).unwrap();
        let (k2, _) = generate_chacha_dpf_keys(&params, 2).unwrap();

        let pages_data = make_test_pages(256);
        let pages: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

        let result = eval_fused_3dpf_cpu([&k0, &k1, &k2], &pages).unwrap();

        assert!(result.timing.total_ns > 0);
        assert!(result.timing.dpf_eval_ns > 0);
        assert!(result.timing.xor_accumulate_ns > 0);
    }

    #[test]
    fn fused_handles_non_power_of_two_pages() {
        let params = ChaChaParams::new(10).unwrap();
        let targets = [5, 150, 300];

        let (k0_0, k0_1) = generate_chacha_dpf_keys(&params, targets[0]).unwrap();
        let (k1_0, k1_1) = generate_chacha_dpf_keys(&params, targets[1]).unwrap();
        let (k2_0, k2_1) = generate_chacha_dpf_keys(&params, targets[2]).unwrap();

        // Use only 500 pages (not full 1024)
        let pages_data = make_test_pages(500);
        let pages: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

        let result0 = eval_fused_3dpf_cpu([&k0_0, &k1_0, &k2_0], &pages).unwrap();
        let result1 = eval_fused_3dpf_cpu([&k0_1, &k1_1, &k2_1], &pages).unwrap();

        // XOR to recover pages
        let mut recovered0 = vec![0u8; PAGE_SIZE_BYTES];
        for i in 0..PAGE_SIZE_BYTES {
            recovered0[i] = result0.page0[i] ^ result1.page0[i];
        }

        assert_eq!(recovered0, pages_data[targets[0]]);
    }

    #[test]
    fn tile_processing_correctness() {
        // Test that tile boundaries don't break correctness
        let params = ChaChaParams::new(12).unwrap(); // 4096 pages
        let target = 2048; // In second tile

        let (k0, k1) = generate_chacha_dpf_keys(&params, target).unwrap();
        let (k1_0, k1_1) = generate_chacha_dpf_keys(&params, 100).unwrap();
        let (k2_0, k2_1) = generate_chacha_dpf_keys(&params, 3000).unwrap();

        let pages_data = make_test_pages(4096);
        let pages: Vec<&[u8]> = pages_data.iter().map(|p| p.as_slice()).collect();

        let result0 = eval_fused_3dpf_cpu([&k0, &k1_0, &k2_0], &pages).unwrap();
        let result1 = eval_fused_3dpf_cpu([&k1, &k1_1, &k2_1], &pages).unwrap();

        let mut recovered = vec![0u8; PAGE_SIZE_BYTES];
        for i in 0..PAGE_SIZE_BYTES {
            recovered[i] = result0.page0[i] ^ result1.page0[i];
        }

        assert_eq!(recovered, pages_data[target]);
    }
}
