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
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use crate::storage::GpuPageMatrix;
#[cfg(feature = "cuda")]
use cudarc::driver::{
    sys::CUfunction_attribute, CudaDevice, CudaFunction, CudaSlice, DriverError, LaunchAsync,
    LaunchConfig,
};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Size of each page in bytes.
pub const PAGE_SIZE_BYTES: usize = 4096;

/// Default subtree size for kernel processing.
/// Each block processes this many pages.
pub const SUBTREE_SIZE: usize = 2048;

/// Threads per block for CUDA kernel.
pub const THREADS_PER_BLOCK: usize = 256;

/// Maximum domain bits supported by the GPU kernel.
pub const MAX_DOMAIN_BITS: usize = 25;

/// CUDA-compatible DPF key structure.
/// Matches the struct in fused_kernel.cu.
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DpfKeyGpu {
    pub root_seed: [u32; 4],
    pub root_t: u8,
    pub domain_bits: u8,
    pub _pad: [u8; 2], // Alignment
    pub cw_seed: [[u32; 4]; MAX_DOMAIN_BITS],
    pub cw_t_left: [u8; MAX_DOMAIN_BITS],
    pub cw_t_right: [u8; MAX_DOMAIN_BITS],
    pub _pad2: [u8; 2], // Alignment for final_cw
    pub final_cw: [u32; 4],
}

#[cfg(feature = "cuda")]
impl DpfKeyGpu {
    pub fn from_chacha_key(key: &ChaChaKey) -> Self {
        let mut cw_seed = [[0u32; 4]; MAX_DOMAIN_BITS];
        let mut cw_t_left = [0u8; MAX_DOMAIN_BITS];
        let mut cw_t_right = [0u8; MAX_DOMAIN_BITS];

        for (i, cw) in key
            .correction_words
            .iter()
            .enumerate()
            .take(MAX_DOMAIN_BITS)
        {
            cw_seed[i] = cw.seed_cw.words;
            cw_t_left[i] = cw.t_cw_left;
            cw_t_right[i] = cw.t_cw_right;
        }

        Self {
            root_seed: key.root_seed.words,
            root_t: key.root_t,
            domain_bits: key.domain_bits as u8,
            _pad: [0; 2],
            cw_seed,
            cw_t_left,
            cw_t_right,
            _pad2: [0; 2],
            final_cw: key.final_cw.words,
        }
    }
}

/// GPU implementation of the PIR scanner.
#[cfg(feature = "cuda")]
pub struct GpuScanner {
    pub device: Arc<CudaDevice>,
    kernels: HashMap<usize, CudaFunction>,
    kernels_transposed: HashMap<usize, CudaFunction>,
}

#[cfg(feature = "cuda")]
impl GpuScanner {
    /// Create a new GpuScanner by loading the fused kernel.
    pub fn new(device_ord: usize) -> Result<Self, DriverError> {
        let device = CudaDevice::new(device_ord)?;

        // Load the PTX/CUBIN compiled by build.rs
        let ptx_path = concat!(env!("OUT_DIR"), "/fused_kernel.ptx");
        let ptx = std::fs::read_to_string(ptx_path).map_err(|e| {
            eprintln!("Failed to read PTX: {}", e);
            panic!("Failed to read PTX: {}", e);
        })?;

        let module_name = "fused_pir".to_string();
        // Load all specialized kernels
        let kernel_names = [
            "fused_batch_pir_kernel_1",
            "fused_batch_pir_kernel_2",
            "fused_batch_pir_kernel_4",
            "fused_batch_pir_kernel_8",
            "fused_batch_pir_kernel_16",
            "fused_batch_pir_kernel_transposed_1",
            "fused_batch_pir_kernel_transposed_2",
            "fused_batch_pir_kernel_transposed_4",
            "fused_batch_pir_kernel_transposed_8",
            "fused_batch_pir_kernel_transposed_16",
        ];

        device.load_ptx(ptx.into(), &module_name, &kernel_names)?;

        let mut kernels = HashMap::new();
        let mut kernels_transposed = HashMap::new();

        let setup_kernel = |name: &str, batch_size: usize| -> Result<CudaFunction, DriverError> {
            let func = device
                .get_func(&module_name, name)
                .expect("Kernel not found");

            let accum_bytes = batch_size * 3 * PAGE_SIZE_BYTES;
            let verif_bytes = batch_size * 3 * 16; // uint4 per query/key
            let seed_bytes = batch_size * 3 * 32;
            let mask_bytes = THREADS_PER_BLOCK * batch_size * 3 * 16;
            let required_shmem = accum_bytes + verif_bytes + seed_bytes + mask_bytes;

            unsafe {
                let _ = func.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    required_shmem as i32,
                );
            }
            Ok(func)
        };

        for (i, &name) in [1, 2, 4, 8, 16].iter().zip(kernel_names.iter().take(5)) {
            kernels.insert(*i, setup_kernel(name, *i)?);
        }

        for (i, &name) in [1, 2, 4, 8, 16].iter().zip(kernel_names.iter().skip(5)) {
            kernels_transposed.insert(*i, setup_kernel(name, *i)?);
        }

        Ok(Self {
            device,
            kernels,
            kernels_transposed,
        })
    }

    /// Execute a fused PIR scan on the GPU for a batch of queries.
    ///
    /// Each query consists of 3 keys (for 3-way Cuckoo hashing).
    /// Returns a vector of PirResults, one for each query in the batch.
    pub unsafe fn scan_batch(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
    ) -> Result<Vec<PirResult>, DriverError> {
        self.scan_batch_opts(db, queries, false)
    }

    /// Execute a fused PIR scan with options (e.g. transposed layout).
    pub unsafe fn scan_batch_opts(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
        transposed: bool,
    ) -> Result<Vec<PirResult>, DriverError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let num_pages = db.num_pages();
        let db_pages = db.as_slice();
        let total_queries = queries.len();
        let start = std::time::Instant::now();

        // Flatten all keys first?
        // No, we process in chunks. We allocate output for ALL queries at once to minimize host overhead?
        // Or allocate per chunk?
        // Allocating one big output buffer is better.
        let total_output_bytes = total_queries * 3 * PAGE_SIZE_BYTES;
        let mut out_accumulators = self.device.alloc_zeros::<u8>(total_output_bytes)?;

        let total_verif_bytes = total_queries * 3 * 16;
        let mut out_verifiers = self.device.alloc_zeros::<u8>(total_verif_bytes)?;

        let mut processed = 0;
        while processed < total_queries {
            let remaining = total_queries - processed;

            // Greedily pick largest batch size
            let batch_size = if remaining >= 16 {
                16
            } else if remaining >= 8 {
                8
            } else if remaining >= 4 {
                4
            } else if remaining >= 2 {
                2
            } else {
                1
            };

            // Prepare keys for this batch
            let mut gpu_keys = Vec::with_capacity(batch_size * 3);
            for i in 0..batch_size {
                let q = &queries[processed + i];
                gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[0]));
                gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[1]));
                gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[2]));
            }
            let keys_slice: CudaSlice<u8> = self
                .device
                .htod_sync_copy(bytemuck::cast_slice(&gpu_keys))?;

            // Get slice of output buffer for this batch
            let out_offset = processed * 3 * PAGE_SIZE_BYTES;
            let out_len = batch_size * 3 * PAGE_SIZE_BYTES;
            let mut out_slice = out_accumulators.slice_mut(out_offset..out_offset + out_len);

            let verif_offset = processed * 3 * 16;
            let verif_len = batch_size * 3 * 16;
            let mut verif_slice = out_verifiers.slice_mut(verif_offset..verif_offset + verif_len);

            // Launch kernel
            let kernel_map = if transposed {
                &self.kernels_transposed
            } else {
                &self.kernels
            };
            let func = kernel_map.get(&batch_size).expect("Kernel not found");

            // Shared memory: Accumulators + Verifiers + Tile Seeds + Mask Buffer
            let accum_bytes = batch_size * 3 * PAGE_SIZE_BYTES;
            let verif_bytes_sh = batch_size * 3 * 16;
            let seed_bytes = batch_size * 3 * 32;
            let mask_bytes = THREADS_PER_BLOCK * batch_size * 3 * 16;
            let shared_mem_bytes = accum_bytes + verif_bytes_sh + seed_bytes + mask_bytes;

            let cfg = LaunchConfig {
                grid_dim: (((num_pages + SUBTREE_SIZE - 1) / SUBTREE_SIZE) as u32, 1, 1),
                block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
                shared_mem_bytes: shared_mem_bytes as u32,
            };

            let params = (
                db_pages,
                &keys_slice,
                &mut out_slice,
                &mut verif_slice,
                num_pages as i32,
                batch_size as i32,
            );
            func.clone().launch(cfg, params)?;

            processed += batch_size;
        }

        // 4. Copy results back
        let flat_results = self.device.sync_reclaim(out_accumulators)?;
        let flat_verifs = self.device.sync_reclaim(out_verifiers)?;

        let total_ns = start.elapsed().as_nanos() as u64;
        let avg_ns = total_ns / total_queries as u64;

        // 5. Unflatten results
        let mut results = Vec::with_capacity(total_queries);
        for i in 0..total_queries {
            let base = i * 3 * PAGE_SIZE_BYTES;
            let p0 = flat_results[base..base + PAGE_SIZE_BYTES].to_vec();
            let p1 = flat_results[base + PAGE_SIZE_BYTES..base + 2 * PAGE_SIZE_BYTES].to_vec();
            let p2 = flat_results[base + 2 * PAGE_SIZE_BYTES..base + 3 * PAGE_SIZE_BYTES].to_vec();

            let v_base = i * 3 * 16;
            let v0 = flat_verifs[v_base..v_base + 16].to_vec();
            let v1 = flat_verifs[v_base + 16..v_base + 32].to_vec();
            let v2 = flat_verifs[v_base + 32..v_base + 48].to_vec();

            results.push(PirResult {
                page0: p0,
                page1: p1,
                page2: p2,
                verif0: v0,
                verif1: v1,
                verif2: v2,
                timing: KernelTiming {
                    dpf_eval_ns: 0,
                    xor_accumulate_ns: 0,
                    total_ns: avg_ns,
                },
            });
        }

        Ok(results)
    }

    /// Execute a fused PIR scan on the GPU (single query wrapper).
    pub unsafe fn scan(
        &self,
        db: &GpuPageMatrix,
        keys: [&ChaChaKey; 3],
    ) -> Result<PirResult, DriverError> {
        let query = [keys[0].clone(), keys[1].clone(), keys[2].clone()];
        let results = self.scan_batch(db, &[query])?;
        Ok(results.into_iter().next().unwrap())
    }
}

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
    pub verif0: Vec<u8>,
    pub verif1: Vec<u8>,
    pub verif2: Vec<u8>,
    pub timing: KernelTiming,
}

/// Apply a 16-byte mask to a page and XOR into accumulator.
#[inline(always)]
fn xor_page_masked(acc: &mut [u8], page: &[u8], mask: &Seed128) {
    debug_assert_eq!(acc.len(), PAGE_SIZE_BYTES);
    debug_assert_eq!(page.len(), PAGE_SIZE_BYTES);

    // Cast slices to Seed128 for 128-bit operations
    let acc_seeds: &mut [Seed128] = bytemuck::cast_slice_mut(acc);
    let page_seeds: &[Seed128] = bytemuck::cast_slice(page);

    for (a, p) in acc_seeds.iter_mut().zip(page_seeds.iter()) {
        // DPF mask acts as an AND gate: result = acc ^ (page & mask)
        *a = a.xor(&p.and(mask));
    }
}

/// XOR source page into destination page.
#[inline(always)]
fn xor_pages(dest: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dest.len(), PAGE_SIZE_BYTES);
    debug_assert_eq!(src.len(), PAGE_SIZE_BYTES);

    let dest_seeds: &mut [Seed128] = bytemuck::cast_slice_mut(dest);
    let src_seeds: &[Seed128] = bytemuck::cast_slice(src);

    for (d, s) in dest_seeds.iter_mut().zip(src_seeds.iter()) {
        *d = d.xor(s);
    }
}

/// CPU reference implementation of fused 3-key PIR evaluation.
///
/// Processes pages in tiles and evaluates all 3 DPF keys in a single pass.
/// Uses Rayon for parallel processing.
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

    // Process in tiles
    let effective_subtree_size = SUBTREE_SIZE.min(domain_size);
    let num_tiles = (num_pages + effective_subtree_size - 1) / effective_subtree_size;

    let (acc0, acc1, acc2) = (0..num_tiles)
        .into_par_iter()
        .map(|tile| {
            let mut acc0 = vec![0u8; PAGE_SIZE_BYTES];
            let mut acc1 = vec![0u8; PAGE_SIZE_BYTES];
            let mut acc2 = vec![0u8; PAGE_SIZE_BYTES];

            let tile_start = tile * effective_subtree_size;
            let tile_end = (tile_start + effective_subtree_size).min(num_pages);
            let current_tile_size = tile_end - tile_start;

            // Pre-compute masks for all pages in tile using optimized O(effective_subtree_size) expansion
            let mut masks0 = vec![Seed128::ZERO; effective_subtree_size];
            let mut masks1 = vec![Seed128::ZERO; effective_subtree_size];
            let mut masks2 = vec![Seed128::ZERO; effective_subtree_size];

            // Note: error handling inside map is tricky, we panic for simplicity in this bench
            keys[0]
                .eval_subtree(tile_start, &mut masks0)
                .expect("eval_subtree failed");
            keys[1]
                .eval_subtree(tile_start, &mut masks1)
                .expect("eval_subtree failed");
            keys[2]
                .eval_subtree(tile_start, &mut masks2)
                .expect("eval_subtree failed");

            for i in 0..current_tile_size {
                let page_idx = tile_start + i;
                let page = pages[page_idx];
                xor_page_masked(&mut acc0, page, &masks0[i]);
                xor_page_masked(&mut acc1, page, &masks1[i]);
                xor_page_masked(&mut acc2, page, &masks2[i]);
            }

            (acc0, acc1, acc2)
        })
        .reduce(
            || {
                (
                    vec![0u8; PAGE_SIZE_BYTES],
                    vec![0u8; PAGE_SIZE_BYTES],
                    vec![0u8; PAGE_SIZE_BYTES],
                )
            },
            |mut a, b| {
                xor_pages(&mut a.0, &b.0);
                xor_pages(&mut a.1, &b.1);
                xor_pages(&mut a.2, &b.2);
                a
            },
        );

    // Note: Timing is less granular with parallelism, we estimate based on total wall time
    let total_ns = start.elapsed().as_nanos() as u64;

    Ok(PirResult {
        page0: acc0,
        page1: acc1,
        page2: acc2,
        verif0: vec![0u8; 16],
        verif1: vec![0u8; 16],
        verif2: vec![0u8; 16],
        timing: KernelTiming {
            dpf_eval_ns: 0, // Not easily measured in parallel
            xor_accumulate_ns: 0,
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

    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn gpu_fused_3dpf_recovers_all_targets() {
        let scanner = GpuScanner::new(0).expect("Failed to create GpuScanner");
        let device = scanner.device.clone();

        let params = crate::dpf::ChaChaParams::new(8).unwrap();
        let targets = [10, 100, 200];

        let (k0_0, k0_1) = crate::dpf::generate_chacha_dpf_keys(&params, targets[0]).unwrap();
        let (k1_0, k1_1) = crate::dpf::generate_chacha_dpf_keys(&params, targets[1]).unwrap();
        let (k2_0, k2_1) = crate::dpf::generate_chacha_dpf_keys(&params, targets[2]).unwrap();

        let num_pages = 256;
        let pages_data = make_test_pages(num_pages);
        let mut pages_flat = Vec::with_capacity(num_pages * PAGE_SIZE_BYTES);
        for p in &pages_data {
            pages_flat.extend_from_slice(p);
        }

        let db = GpuPageMatrix::new(device, &pages_flat).expect("Failed to create GpuPageMatrix");

        let result0 = unsafe { scanner.scan(&db, [&k0_0, &k1_0, &k2_0]) }.expect("Scan 0 failed");
        let result1 = unsafe { scanner.scan(&db, [&k0_1, &k1_1, &k2_1]) }.expect("Scan 1 failed");

        // XOR to recover and verify
        for i in 0..PAGE_SIZE_BYTES {
            assert_eq!(
                result0.page0[i] ^ result1.page0[i],
                pages_data[targets[0]][i],
                "Target 0 mismatch at byte {}",
                i
            );
            assert_eq!(
                result0.page1[i] ^ result1.page1[i],
                pages_data[targets[1]][i],
                "Target 1 mismatch at byte {}",
                i
            );
            assert_eq!(
                result0.page2[i] ^ result1.page2[i],
                pages_data[targets[2]][i],
                "Target 2 mismatch at byte {}",
                i
            );
        }
        println!("GPU fused scan verified successfully!");
    }
}
