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
    sys::{self, CUfunction_attribute},
    CudaDevice, CudaFunction, CudaSlice, DevicePtr, DriverError, LaunchAsync, LaunchConfig,
};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::mem::MaybeUninit;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

/// Size of each page in bytes.
pub const PAGE_SIZE_BYTES: usize = 4096;

/// Default subtree size for kernel processing.
/// Each block processes this many pages.
pub const SUBTREE_SIZE: usize = 2048;

/// Threads per block for CUDA kernel.
pub const THREADS_PER_BLOCK: usize = 256;

/// Maximum domain bits supported by the GPU kernel.
pub const MAX_DOMAIN_BITS: usize = 25;

#[cfg(any(feature = "cuda", test))]
const MAX_KERNEL_BATCH_SIZE: usize = 16;
#[cfg(feature = "cuda")]
const KEYS_PER_QUERY: usize = 3;
#[cfg(feature = "cuda")]
const VERIF_BYTES_PER_KEY: usize = 16;

#[cfg(any(feature = "cuda", test))]
fn clamp_tiled_launch_limit(tile_size: usize) -> usize {
    tile_size.clamp(1, MAX_KERNEL_BATCH_SIZE)
}

#[cfg(any(feature = "cuda", test))]
fn select_tiled_launch_batch_size(remaining_queries: usize, tile_size: usize) -> usize {
    let capped_remaining = remaining_queries.min(clamp_tiled_launch_limit(tile_size));
    if capped_remaining >= 16 {
        16
    } else if capped_remaining >= 8 {
        8
    } else if capped_remaining >= 4 {
        4
    } else if capped_remaining >= 2 {
        2
    } else {
        1
    }
}

#[cfg(any(feature = "cuda", test))]
fn plan_tiled_launch_batch_sizes(total_queries: usize, tile_size: usize) -> Vec<usize> {
    let mut remaining = total_queries;
    let mut plan = Vec::new();
    while remaining > 0 {
        let launch = select_tiled_launch_batch_size(remaining, tile_size);
        plan.push(launch);
        remaining -= launch;
    }
    plan
}

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
    /// Optimized kernels v1 - fast PRG, reduced shared memory
    kernels_optimized: HashMap<usize, CudaFunction>,
    /// Optimized kernels v2 - minimal shared memory (sequential queries)
    kernels_optimized_v2: HashMap<usize, CudaFunction>,
    /// Optimized kernels v3 - hybrid approach (query grouping)
    kernels_optimized_v3: HashMap<usize, CudaFunction>,
    /// Reusable batch buffers to avoid per-request CUDA allocations.
    batch_workspace: Mutex<Option<BatchWorkspace>>,
}

#[cfg(feature = "cuda")]
struct BatchWorkspace {
    capacity_queries: usize,
    out_accumulators: CudaSlice<u8>,
    out_verifiers: CudaSlice<u8>,
    key_buffer: CudaSlice<u8>,
    host_results: Vec<u8>,
    host_verifs: Vec<u8>,
    graph_cache: Option<CudaGraphCache>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaGraphSignature {
    kernel_kind: u8,
    batch_size: usize,
    num_pages: usize,
    shared_mem_bytes: u32,
    db_ptr: u64,
    key_ptr: u64,
    out_ptr: u64,
    verif_ptr: u64,
}

#[cfg(feature = "cuda")]
struct CudaGraphCache {
    device: Arc<CudaDevice>,
    signature: CudaGraphSignature,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGraphCache {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGraphCache {}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphCache {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        if !self.exec.is_null() {
            // Best-effort cleanup; cache eviction should not panic request path.
            let _ = unsafe { sys::lib().cuGraphExecDestroy(self.exec).result() };
        }
        if !self.graph.is_null() {
            let _ = unsafe { sys::lib().cuGraphDestroy(self.graph).result() };
        }
    }
}

#[cfg(feature = "cuda")]
impl BatchWorkspace {
    fn new(device: Arc<CudaDevice>, capacity_queries: usize) -> Result<Self, DriverError> {
        let capacity_queries = capacity_queries.max(1);
        let output_len = capacity_queries * KEYS_PER_QUERY * PAGE_SIZE_BYTES;
        let verif_len = capacity_queries * KEYS_PER_QUERY * VERIF_BYTES_PER_KEY;
        let max_key_bytes =
            MAX_KERNEL_BATCH_SIZE * KEYS_PER_QUERY * std::mem::size_of::<DpfKeyGpu>();
        Ok(Self {
            capacity_queries,
            out_accumulators: device.alloc_zeros::<u8>(output_len)?,
            out_verifiers: device.alloc_zeros::<u8>(verif_len)?,
            key_buffer: device.alloc_zeros::<u8>(max_key_bytes)?,
            host_results: vec![0u8; output_len],
            host_verifs: vec![0u8; verif_len],
            graph_cache: None,
        })
    }

    fn ensure_host_lengths(&mut self, required_queries: usize) {
        let output_len = required_queries * KEYS_PER_QUERY * PAGE_SIZE_BYTES;
        let verif_len = required_queries * KEYS_PER_QUERY * VERIF_BYTES_PER_KEY;
        if self.host_results.len() != output_len {
            self.host_results.resize(output_len, 0);
        }
        if self.host_verifs.len() != verif_len {
            self.host_verifs.resize(verif_len, 0);
        }
    }
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

        // Load optimized kernels v1
        let mut kernels_optimized = HashMap::new();
        let opt_ptx_path = concat!(env!("OUT_DIR"), "/fused_kernel_optimized.ptx");
        if let Ok(opt_ptx) = std::fs::read_to_string(opt_ptx_path) {
            let opt_module_name = "fused_pir_optimized".to_string();
            let opt_kernel_names = [
                "fused_batch_pir_kernel_optimized_1",
                "fused_batch_pir_kernel_optimized_2",
                "fused_batch_pir_kernel_optimized_4",
                "fused_batch_pir_kernel_optimized_8",
                "fused_batch_pir_kernel_optimized_16",
            ];

            if device
                .load_ptx(opt_ptx.into(), &opt_module_name, &opt_kernel_names)
                .is_ok()
            {
                let setup_opt_kernel =
                    |name: &str, batch_size: usize| -> Result<CudaFunction, DriverError> {
                        let func = device
                            .get_func(&opt_module_name, name)
                            .expect("Optimized kernel not found");

                        let accum_bytes = batch_size * 3 * PAGE_SIZE_BYTES;
                        let verif_bytes = batch_size * 3 * 16;
                        let seed_bytes = batch_size * 3 * 32;
                        let required_shmem = accum_bytes + verif_bytes + seed_bytes;

                        unsafe {
                            let _ = func.set_attribute(
                            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            required_shmem as i32,
                        );
                        }
                        Ok(func)
                    };

                for (i, &name) in [1, 2, 4, 8, 16].iter().zip(opt_kernel_names.iter()) {
                    if let Ok(kernel) = setup_opt_kernel(name, *i) {
                        kernels_optimized.insert(*i, kernel);
                    }
                }
            }
        }

        // Load optimized kernels v2 - minimal shared memory
        let mut kernels_optimized_v2 = HashMap::new();
        let opt_v2_ptx_path = concat!(env!("OUT_DIR"), "/fused_kernel_optimized_v2.ptx");
        if let Ok(opt_v2_ptx) = std::fs::read_to_string(opt_v2_ptx_path) {
            let opt_v2_module_name = "fused_pir_optimized_v2".to_string();
            let opt_v2_kernel_names = [
                "fused_batch_pir_kernel_v2_1",
                "fused_batch_pir_kernel_v2_2",
                "fused_batch_pir_kernel_v2_4",
                "fused_batch_pir_kernel_v2_8",
                "fused_batch_pir_kernel_v2_16",
            ];

            if device
                .load_ptx(opt_v2_ptx.into(), &opt_v2_module_name, &opt_v2_kernel_names)
                .is_ok()
            {
                let setup_opt_v2_kernel =
                    |name: &str, batch_size: usize| -> Result<CudaFunction, DriverError> {
                        let func = device
                            .get_func(&opt_v2_module_name, name)
                            .expect("Optimized v2 kernel not found");

                        let seed_bytes = batch_size * 3 * 32;
                        let required_shmem = seed_bytes;

                        unsafe {
                            let _ = func.set_attribute(
                            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            required_shmem as i32,
                        );
                        }
                        Ok(func)
                    };

                for (i, &name) in [1, 2, 4, 8, 16].iter().zip(opt_v2_kernel_names.iter()) {
                    if let Ok(kernel) = setup_opt_v2_kernel(name, *i) {
                        kernels_optimized_v2.insert(*i, kernel);
                    }
                }
            }
        }

        // Load optimized kernels v3 - hybrid approach
        let mut kernels_optimized_v3 = HashMap::new();
        let opt_v3_ptx_path = concat!(env!("OUT_DIR"), "/fused_kernel_optimized_v3.ptx");
        if let Ok(opt_v3_ptx) = std::fs::read_to_string(opt_v3_ptx_path) {
            let opt_v3_module_name = "fused_pir_optimized_v3".to_string();
            let opt_v3_kernel_names = [
                "fused_batch_pir_kernel_v3_1",
                "fused_batch_pir_kernel_v3_2",
                "fused_batch_pir_kernel_v3_4",
                "fused_batch_pir_kernel_v3_8",
                "fused_batch_pir_kernel_v3_16",
            ];

            if device
                .load_ptx(opt_v3_ptx.into(), &opt_v3_module_name, &opt_v3_kernel_names)
                .is_ok()
            {
                let setup_opt_v3_kernel =
                    |name: &str, batch_size: usize| -> Result<CudaFunction, DriverError> {
                        let func = device
                            .get_func(&opt_v3_module_name, name)
                            .expect("Optimized v3 kernel not found");

                        // v3: seeds only in shared memory (accumulators in registers)
                        let seed_bytes = batch_size * 3 * 32;
                        let required_shmem = seed_bytes;

                        unsafe {
                            let _ = func.set_attribute(
                            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            required_shmem as i32,
                        );
                        }
                        Ok(func)
                    };

                for (i, &name) in [1, 2, 4, 8, 16].iter().zip(opt_v3_kernel_names.iter()) {
                    if let Ok(kernel) = setup_opt_v3_kernel(name, *i) {
                        kernels_optimized_v3.insert(*i, kernel);
                    }
                }
            }
        }

        Ok(Self {
            device,
            kernels,
            kernels_transposed,
            kernels_optimized,
            kernels_optimized_v2,
            kernels_optimized_v3,
            batch_workspace: Mutex::new(None),
        })
    }

    fn ensure_batch_workspace<'a>(
        &self,
        slot: &'a mut Option<BatchWorkspace>,
        min_queries: usize,
    ) -> Result<&'a mut BatchWorkspace, DriverError> {
        let needs_realloc = slot
            .as_ref()
            .map(|workspace| workspace.capacity_queries < min_queries)
            .unwrap_or(true);
        if needs_realloc {
            *slot = Some(BatchWorkspace::new(self.device.clone(), min_queries)?);
        }
        Ok(slot.as_mut().expect("batch workspace must be initialized"))
    }

    fn select_kernel_for_batch(
        &self,
        batch_size: usize,
        transposed: bool,
        optimized: bool,
        optimized_v2: bool,
        optimized_v3: bool,
    ) -> (&CudaFunction, u8, u32) {
        let (func, kernel_kind, use_opt_v1, use_opt_v2, use_opt_v3) =
            if optimized_v3 && !self.kernels_optimized_v3.is_empty() {
                (
                    self.kernels_optimized_v3
                        .get(&batch_size)
                        .expect("Optimized v3 kernel not found"),
                    4u8,
                    false,
                    false,
                    true,
                )
            } else if optimized_v2 && !self.kernels_optimized_v2.is_empty() {
                (
                    self.kernels_optimized_v2
                        .get(&batch_size)
                        .expect("Optimized v2 kernel not found"),
                    3u8,
                    false,
                    true,
                    false,
                )
            } else if optimized && !self.kernels_optimized.is_empty() {
                (
                    self.kernels_optimized
                        .get(&batch_size)
                        .expect("Optimized kernel not found"),
                    2u8,
                    true,
                    false,
                    false,
                )
            } else if transposed {
                (
                    self.kernels_transposed
                        .get(&batch_size)
                        .expect("Kernel not found"),
                    1u8,
                    false,
                    false,
                    false,
                )
            } else {
                (
                    self.kernels.get(&batch_size).expect("Kernel not found"),
                    0u8,
                    false,
                    false,
                    false,
                )
            };

        let accum_bytes = batch_size * 3 * PAGE_SIZE_BYTES;
        let verif_bytes_sh = batch_size * 3 * 16;
        let seed_bytes = batch_size * 3 * 32;
        let mask_bytes = if use_opt_v1 || use_opt_v2 || use_opt_v3 {
            0
        } else {
            THREADS_PER_BLOCK * batch_size * 3 * 16
        };
        let shared_mem_bytes = if use_opt_v2 || use_opt_v3 {
            seed_bytes
        } else {
            accum_bytes + verif_bytes_sh + seed_bytes + mask_bytes
        };

        (func, kernel_kind, shared_mem_bytes as u32)
    }

    fn graph_eligible_batch_size(total_queries: usize) -> bool {
        matches!(total_queries, 1 | 2 | 4 | 8 | 16)
    }

    unsafe fn launch_single_batch_with_cuda_graph(
        &self,
        graph_cache: &mut Option<CudaGraphCache>,
        signature: CudaGraphSignature,
        func: &CudaFunction,
        cfg: LaunchConfig,
        db_pages: &CudaSlice<u8>,
        keys_slice: &mut cudarc::driver::CudaViewMut<u8>,
        out_slice: &mut cudarc::driver::CudaViewMut<u8>,
        verif_slice: &mut cudarc::driver::CudaViewMut<u8>,
        num_pages: i32,
        batch_size: i32,
    ) -> Result<(), DriverError> {
        self.device.bind_to_thread()?;
        let stream = *self.device.cu_stream();
        let mut launch_direct = || {
            let params = (
                db_pages,
                &mut *keys_slice,
                &mut *out_slice,
                &mut *verif_slice,
                num_pages,
                batch_size,
            );
            func.clone().launch(cfg, params)
        };

        if let Some(cache) = graph_cache.as_ref() {
            if cache.signature == signature {
                if sys::lib()
                    .cuGraphLaunch(cache.exec, stream)
                    .result()
                    .is_ok()
                {
                    return Ok(());
                }
                *graph_cache = None;
                return launch_direct();
            }
        }

        *graph_cache = None;

        if sys::lib()
            .cuStreamBeginCapture_v2(
                stream,
                sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
            .result()
            .is_err()
        {
            return launch_direct();
        }

        if launch_direct().is_err() {
            let mut stale_graph = MaybeUninit::uninit();
            if sys::lib()
                .cuStreamEndCapture(stream, stale_graph.as_mut_ptr())
                .result()
                .is_ok()
            {
                let graph = stale_graph.assume_init();
                if !graph.is_null() {
                    let _ = sys::lib().cuGraphDestroy(graph).result();
                }
            }
            return launch_direct();
        }

        let mut graph_handle = MaybeUninit::uninit();
        if sys::lib()
            .cuStreamEndCapture(stream, graph_handle.as_mut_ptr())
            .result()
            .is_err()
        {
            return launch_direct();
        }
        let graph = graph_handle.assume_init();
        if graph.is_null() {
            return launch_direct();
        }

        let mut exec_handle = MaybeUninit::uninit();
        if sys::lib()
            .cuGraphInstantiateWithFlags(exec_handle.as_mut_ptr(), graph, 0)
            .result()
            .is_err()
        {
            let _ = sys::lib().cuGraphDestroy(graph).result();
            return launch_direct();
        }
        let exec = exec_handle.assume_init();

        if sys::lib().cuGraphUpload(exec, stream).result().is_err() {
            let _ = sys::lib().cuGraphExecDestroy(exec).result();
            let _ = sys::lib().cuGraphDestroy(graph).result();
            return launch_direct();
        }

        *graph_cache = Some(CudaGraphCache {
            device: self.device.clone(),
            signature,
            graph,
            exec,
        });

        let cache = graph_cache.as_ref().expect("graph cache must be populated");
        if sys::lib()
            .cuGraphLaunch(cache.exec, stream)
            .result()
            .is_err()
        {
            *graph_cache = None;
            return launch_direct();
        }

        Ok(())
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
        self.scan_batch_opts(
            db,
            queries,
            false,
            false,
            false,
            false,
            false,
            MAX_KERNEL_BATCH_SIZE,
        )
    }

    /// Execute a fused PIR scan with options (e.g. transposed layout).
    ///
    /// # Arguments
    /// * `db` - The database matrix on GPU
    /// * `queries` - Batch of queries (each with 3 DPF keys for Cuckoo)
    /// * `transposed` - Whether to use transposed memory layout
    /// * `optimized` - Whether to use Plinko-optimized kernels v1 (fast PRG)
    /// * `optimized_v2` - Whether to use optimized kernels v2 (minimal shared memory, sequential)
    /// * `optimized_v3` - Whether to use optimized kernels v3 (hybrid query grouping)
    pub unsafe fn scan_batch_opts(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
        transposed: bool,
        optimized: bool,
        optimized_v2: bool,
        optimized_v3: bool,
        use_cuda_graph: bool,
        max_tile_size: usize,
    ) -> Result<Vec<PirResult>, DriverError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let num_pages = db.num_pages();
        let db_pages = db.as_slice();
        let total_queries = queries.len();
        let total_start = std::time::Instant::now();
        let total_output_bytes = total_queries * KEYS_PER_QUERY * PAGE_SIZE_BYTES;
        let total_verif_bytes = total_queries * KEYS_PER_QUERY * VERIF_BYTES_PER_KEY;

        let mut workspace_guard = self
            .batch_workspace
            .lock()
            .expect("batch workspace lock poisoned");
        let workspace = self.ensure_batch_workspace(&mut workspace_guard, total_queries)?;
        workspace.ensure_host_lengths(total_queries);

        let mut gpu_keys = Vec::with_capacity(MAX_KERNEL_BATCH_SIZE * KEYS_PER_QUERY);
        let mut h2d_ns = 0u64;
        let clear_start = std::time::Instant::now();
        {
            let mut out_view = workspace.out_accumulators.slice_mut(..total_output_bytes);
            self.device.memset_zeros(&mut out_view)?;
        }
        {
            let mut verif_view = workspace.out_verifiers.slice_mut(..total_verif_bytes);
            self.device.memset_zeros(&mut verif_view)?;
        }
        h2d_ns = h2d_ns.saturating_add(clear_start.elapsed().as_nanos() as u64);
        let kernel_start = std::time::Instant::now();
        let launch_tile_size = clamp_tiled_launch_limit(max_tile_size);
        if use_cuda_graph
            && total_queries <= launch_tile_size
            && Self::graph_eligible_batch_size(total_queries)
        {
            let batch_size = total_queries;
            gpu_keys.clear();
            for q in queries {
                gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[0]));
                gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[1]));
                gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[2]));
            }
            let key_bytes = bytemuck::cast_slice(gpu_keys.as_slice());
            let (func, kernel_kind, shared_mem_bytes) = self.select_kernel_for_batch(
                batch_size,
                transposed,
                optimized,
                optimized_v2,
                optimized_v3,
            );
            let cfg = LaunchConfig {
                grid_dim: (((num_pages + SUBTREE_SIZE - 1) / SUBTREE_SIZE) as u32, 1, 1),
                block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
                shared_mem_bytes,
            };
            let signature = CudaGraphSignature {
                kernel_kind,
                batch_size,
                num_pages,
                shared_mem_bytes,
                db_ptr: *db_pages.device_ptr(),
                key_ptr: *workspace.key_buffer.device_ptr(),
                out_ptr: *workspace.out_accumulators.device_ptr(),
                verif_ptr: *workspace.out_verifiers.device_ptr(),
            };

            let mut keys_slice = workspace.key_buffer.slice_mut(..key_bytes.len());
            let h2d_start = std::time::Instant::now();
            self.device
                .htod_sync_copy_into(key_bytes, &mut keys_slice)?;
            h2d_ns = h2d_ns.saturating_add(h2d_start.elapsed().as_nanos() as u64);
            let mut out_slice = workspace.out_accumulators.slice_mut(..total_output_bytes);
            let mut verif_slice = workspace.out_verifiers.slice_mut(..total_verif_bytes);

            self.launch_single_batch_with_cuda_graph(
                &mut workspace.graph_cache,
                signature,
                func,
                cfg,
                db_pages,
                &mut keys_slice,
                &mut out_slice,
                &mut verif_slice,
                num_pages as i32,
                batch_size as i32,
            )?;
        } else {
            let mut processed = 0usize;
            for batch_size in plan_tiled_launch_batch_sizes(total_queries, launch_tile_size) {
                // Prepare keys for this batch
                gpu_keys.clear();
                for i in 0..batch_size {
                    let q = &queries[processed + i];
                    gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[0]));
                    gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[1]));
                    gpu_keys.push(DpfKeyGpu::from_chacha_key(&q[2]));
                }
                let key_bytes = bytemuck::cast_slice(gpu_keys.as_slice());
                let mut keys_slice = workspace.key_buffer.slice_mut(..key_bytes.len());
                let h2d_start = std::time::Instant::now();
                self.device
                    .htod_sync_copy_into(key_bytes, &mut keys_slice)?;
                h2d_ns = h2d_ns.saturating_add(h2d_start.elapsed().as_nanos() as u64);

                // Get slice of output buffer for this batch
                let out_offset = processed * 3 * PAGE_SIZE_BYTES;
                let out_len = batch_size * 3 * PAGE_SIZE_BYTES;
                let mut out_slice = workspace
                    .out_accumulators
                    .slice_mut(out_offset..out_offset + out_len);

                let verif_offset = processed * 3 * 16;
                let verif_len = batch_size * 3 * 16;
                let mut verif_slice = workspace
                    .out_verifiers
                    .slice_mut(verif_offset..verif_offset + verif_len);

                let (func, _kernel_kind, shared_mem_bytes) = self.select_kernel_for_batch(
                    batch_size,
                    transposed,
                    optimized,
                    optimized_v2,
                    optimized_v3,
                );
                let cfg = LaunchConfig {
                    grid_dim: (((num_pages + SUBTREE_SIZE - 1) / SUBTREE_SIZE) as u32, 1, 1),
                    block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
                    shared_mem_bytes,
                };
                let params = (
                    db_pages,
                    &mut keys_slice,
                    &mut out_slice,
                    &mut verif_slice,
                    num_pages as i32,
                    batch_size as i32,
                );
                func.clone().launch(cfg, params)?;

                processed += batch_size;
            }
        }

        self.device.synchronize()?;
        let kernel_ns = kernel_start.elapsed().as_nanos() as u64;

        let d2h_start = std::time::Instant::now();
        {
            let out_view = workspace.out_accumulators.slice(..total_output_bytes);
            self.device
                .dtoh_sync_copy_into(&out_view, workspace.host_results.as_mut_slice())?;
        }
        {
            let verif_view = workspace.out_verifiers.slice(..total_verif_bytes);
            self.device
                .dtoh_sync_copy_into(&verif_view, workspace.host_verifs.as_mut_slice())?;
        }
        let d2h_ns = d2h_start.elapsed().as_nanos() as u64;

        let total_ns = total_start.elapsed().as_nanos() as u64;
        let query_divisor = total_queries as u64;
        let avg_h2d_ns = h2d_ns / query_divisor;
        let avg_kernel_ns = kernel_ns / query_divisor;
        let avg_d2h_ns = d2h_ns / query_divisor;
        let avg_total_ns = total_ns / query_divisor;

        // 5. Unflatten results
        let mut results = Vec::with_capacity(total_queries);
        for i in 0..total_queries {
            let base = i * 3 * PAGE_SIZE_BYTES;
            let p0 = workspace.host_results[base..base + PAGE_SIZE_BYTES].to_vec();
            let p1 =
                workspace.host_results[base + PAGE_SIZE_BYTES..base + 2 * PAGE_SIZE_BYTES].to_vec();
            let p2 = workspace.host_results[base + 2 * PAGE_SIZE_BYTES..base + 3 * PAGE_SIZE_BYTES]
                .to_vec();

            let v_base = i * 3 * 16;
            let v0 = workspace.host_verifs[v_base..v_base + 16].to_vec();
            let v1 = workspace.host_verifs[v_base + 16..v_base + 32].to_vec();
            let v2 = workspace.host_verifs[v_base + 32..v_base + 48].to_vec();

            results.push(PirResult {
                page0: p0,
                page1: p1,
                page2: p2,
                verif0: v0,
                verif1: v1,
                verif2: v2,
                timing: KernelTiming {
                    dpf_eval_ns: avg_kernel_ns,
                    xor_accumulate_ns: 0,
                    h2d_ns: avg_h2d_ns,
                    kernel_ns: avg_kernel_ns,
                    d2h_ns: avg_d2h_ns,
                    total_ns: avg_total_ns,
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

    /// Execute queries as single-query kernel launches distributed across multiple CUDA streams.
    ///
    /// This avoids large in-kernel batch sizes while still providing host-side overlap.
    pub unsafe fn scan_batch_single_query_multistream_optimized(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
        stream_count: usize,
    ) -> Result<Vec<PirResult>, DriverError> {
        self.scan_batch_single_query_multistream_optimized_with_graph(
            db,
            queries,
            stream_count,
            false,
        )
    }

    /// Multistream optimized scan with optional CUDA Graph replay on single-stream fallback.
    pub unsafe fn scan_batch_single_query_multistream_optimized_with_graph(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
        stream_count: usize,
        use_cuda_graph: bool,
    ) -> Result<Vec<PirResult>, DriverError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let total_queries = queries.len();
        let stream_count = stream_count.max(1).min(total_queries);
        if stream_count <= 1 {
            return self.scan_batch_optimized_with_graph(db, queries, use_cuda_graph);
        }

        let num_pages = db.num_pages();
        let db_pages = db.as_slice();
        let total_start = std::time::Instant::now();

        // Prepare all per-query buffers on default stream first, then fork streams.
        let h2d_start = std::time::Instant::now();
        let mut key_slices: Vec<CudaSlice<u8>> = Vec::with_capacity(total_queries);
        let mut out_accumulators: Vec<CudaSlice<u8>> = Vec::with_capacity(total_queries);
        let mut out_verifiers: Vec<CudaSlice<u8>> = Vec::with_capacity(total_queries);
        for q in queries {
            let gpu_keys = [
                DpfKeyGpu::from_chacha_key(&q[0]),
                DpfKeyGpu::from_chacha_key(&q[1]),
                DpfKeyGpu::from_chacha_key(&q[2]),
            ];
            key_slices.push(
                self.device
                    .htod_sync_copy(bytemuck::cast_slice(&gpu_keys))?,
            );
            out_accumulators.push(self.device.alloc_zeros::<u8>(3 * PAGE_SIZE_BYTES)?);
            out_verifiers.push(self.device.alloc_zeros::<u8>(3 * 16)?);
        }
        let h2d_ns = h2d_start.elapsed().as_nanos() as u64;

        let streams = (0..stream_count)
            .map(|_| self.device.fork_default_stream())
            .collect::<Result<Vec<_>, _>>()?;

        let using_optimized_batch1 = self.kernels_optimized.contains_key(&1);
        let func = if using_optimized_batch1 {
            self.kernels_optimized.get(&1).expect("Kernel not found")
        } else {
            self.kernels.get(&1).expect("Kernel not found")
        };

        let accum_bytes = 3 * PAGE_SIZE_BYTES;
        let verif_bytes_sh = 3 * 16;
        let seed_bytes = 3 * 32;
        let mask_bytes = if using_optimized_batch1 {
            0
        } else {
            THREADS_PER_BLOCK * 3 * 16
        };
        let shared_mem_bytes = accum_bytes + verif_bytes_sh + seed_bytes + mask_bytes;

        let cfg = LaunchConfig {
            grid_dim: (((num_pages + SUBTREE_SIZE - 1) / SUBTREE_SIZE) as u32, 1, 1),
            block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        let kernel_start = std::time::Instant::now();
        for idx in 0..total_queries {
            let stream = &streams[idx % stream_count];
            let mut out_slice = out_accumulators[idx].slice_mut(..);
            let mut verif_slice = out_verifiers[idx].slice_mut(..);
            let params = (
                db_pages,
                &key_slices[idx],
                &mut out_slice,
                &mut verif_slice,
                num_pages as i32,
                1i32,
            );
            func.clone().launch_on_stream(stream, cfg, params)?;
        }

        // Make default stream wait on all worker streams before reclaiming buffers.
        for stream in &streams {
            self.device.wait_for(stream)?;
        }
        drop(streams);
        let kernel_ns = kernel_start.elapsed().as_nanos() as u64;

        let d2h_start = std::time::Instant::now();

        let mut results = Vec::with_capacity(total_queries);
        for (out_acc, out_verif) in out_accumulators.into_iter().zip(out_verifiers.into_iter()) {
            let flat_results = self.device.sync_reclaim(out_acc)?;
            let flat_verifs = self.device.sync_reclaim(out_verif)?;

            let p0 = flat_results[0..PAGE_SIZE_BYTES].to_vec();
            let p1 = flat_results[PAGE_SIZE_BYTES..2 * PAGE_SIZE_BYTES].to_vec();
            let p2 = flat_results[2 * PAGE_SIZE_BYTES..3 * PAGE_SIZE_BYTES].to_vec();

            let v0 = flat_verifs[0..16].to_vec();
            let v1 = flat_verifs[16..32].to_vec();
            let v2 = flat_verifs[32..48].to_vec();

            results.push(PirResult {
                page0: p0,
                page1: p1,
                page2: p2,
                verif0: v0,
                verif1: v1,
                verif2: v2,
                timing: KernelTiming {
                    dpf_eval_ns: kernel_ns / total_queries as u64,
                    xor_accumulate_ns: 0,
                    h2d_ns: h2d_ns / total_queries as u64,
                    kernel_ns: kernel_ns / total_queries as u64,
                    d2h_ns: 0,
                    total_ns: 0,
                },
            });
        }

        let d2h_ns = d2h_start.elapsed().as_nanos() as u64;
        let total_ns = total_start.elapsed().as_nanos() as u64;
        let avg_h2d_ns = h2d_ns / total_queries as u64;
        let avg_kernel_ns = kernel_ns / total_queries as u64;
        let avg_d2h_ns = d2h_ns / total_queries as u64;
        let avg_total_ns = total_ns / total_queries as u64;
        for result in &mut results {
            result.timing = KernelTiming {
                dpf_eval_ns: avg_kernel_ns,
                xor_accumulate_ns: 0,
                h2d_ns: avg_h2d_ns,
                kernel_ns: avg_kernel_ns,
                d2h_ns: avg_d2h_ns,
                total_ns: avg_total_ns,
            };
        }

        Ok(results)
    }

    /// Execute a fused PIR scan using optimized kernels v1 (Plinko-style fast PRG).
    ///
    /// This uses the optimized CUDA kernels with:
    /// - Fast PRG expansion (fewer ChaCha rounds computed)
    /// - Warp-level DPF sharing
    /// - Better instruction scheduling
    pub unsafe fn scan_batch_optimized(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
    ) -> Result<Vec<PirResult>, DriverError> {
        self.scan_batch_optimized_with_graph(db, queries, false)
    }

    /// Execute a fused PIR scan using optimized kernels v1 with optional CUDA Graph replay.
    pub unsafe fn scan_batch_optimized_with_graph(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
        use_cuda_graph: bool,
    ) -> Result<Vec<PirResult>, DriverError> {
        self.scan_batch_opts(
            db,
            queries,
            false,
            true,
            false,
            false,
            use_cuda_graph,
            MAX_KERNEL_BATCH_SIZE,
        )
    }

    /// Execute fused optimized scan while capping each kernel launch to `tile_size`.
    ///
    /// For example, with `queries.len()=16` and `tile_size=4`, this issues four batch-4 launches.
    pub unsafe fn scan_batch_optimized_tiled(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
        tile_size: usize,
    ) -> Result<Vec<PirResult>, DriverError> {
        self.scan_batch_optimized_tiled_with_graph(db, queries, tile_size, false)
    }

    /// Tiled optimized scan with optional CUDA Graph replay when the whole request is one tile.
    pub unsafe fn scan_batch_optimized_tiled_with_graph(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
        tile_size: usize,
        use_cuda_graph: bool,
    ) -> Result<Vec<PirResult>, DriverError> {
        self.scan_batch_opts(
            db,
            queries,
            false,
            true,
            false,
            false,
            use_cuda_graph,
            tile_size,
        )
    }

    /// Execute a fused PIR scan using optimized kernels v2 (minimal shared memory).
    ///
    /// This uses the optimized CUDA kernels with:
    /// - Register-based accumulation (no shared memory for accumulators)
    /// - Warp-level reduction using shuffle
    /// - Minimal shared memory usage (only for tile seeds)
    /// - Supports much larger batch sizes without shared memory overflow
    pub unsafe fn scan_batch_optimized_v2(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
    ) -> Result<Vec<PirResult>, DriverError> {
        self.scan_batch_opts(
            db,
            queries,
            false,
            false,
            true,
            false,
            false,
            MAX_KERNEL_BATCH_SIZE,
        )
    }

    /// Execute a fused PIR scan using optimized kernels v3 (hybrid query grouping).
    ///
    /// This uses the optimized CUDA kernels with:
    /// - Query grouping (process 3 queries at a time to reduce register pressure)
    /// - Parallel processing within each group
    /// - Accumulates in registers, flushes via warp shuffle
    /// - Intended to balance parallelism and resource usage
    pub unsafe fn scan_batch_optimized_v3(
        &self,
        db: &GpuPageMatrix,
        queries: &[[ChaChaKey; 3]],
    ) -> Result<Vec<PirResult>, DriverError> {
        self.scan_batch_opts(
            db,
            queries,
            false,
            false,
            false,
            true,
            false,
            MAX_KERNEL_BATCH_SIZE,
        )
    }
}

/// Timing breakdown for fused kernel execution.
#[derive(Debug, Clone, Default)]
pub struct KernelTiming {
    /// Legacy field retained for compatibility.
    pub dpf_eval_ns: u64,
    /// Legacy field retained for compatibility.
    pub xor_accumulate_ns: u64,
    /// Host-to-device transfer time (average per query).
    pub h2d_ns: u64,
    /// Kernel execution time (average per query).
    pub kernel_ns: u64,
    /// Device-to-host transfer time (average per query).
    pub d2h_ns: u64,
    /// End-to-end elapsed time (average per query).
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
    let num_tiles = num_pages.div_ceil(effective_subtree_size);

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
            h2d_ns: 0,
            kernel_ns: 0,
            d2h_ns: 0,
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
    fn tiled_launch_batch_sizes_respect_requested_tile_limit() {
        assert_eq!(plan_tiled_launch_batch_sizes(17, 4), vec![4, 4, 4, 4, 1]);
        assert_eq!(plan_tiled_launch_batch_sizes(6, 3), vec![2, 2, 2]);
    }

    #[test]
    fn tiled_launch_batch_sizes_default_to_supported_minimum_when_zero() {
        assert_eq!(plan_tiled_launch_batch_sizes(5, 0), vec![1, 1, 1, 1, 1]);
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

    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn gpu_tiled_batch_matches_default_optimized_outputs() {
        let scanner = GpuScanner::new(0).expect("Failed to create GpuScanner");
        let device = scanner.device.clone();
        let params = crate::dpf::ChaChaParams::new(8).unwrap();

        let num_pages = 256;
        let pages_data = make_test_pages(num_pages);
        let mut pages_flat = Vec::with_capacity(num_pages * PAGE_SIZE_BYTES);
        for p in &pages_data {
            pages_flat.extend_from_slice(p);
        }
        let db = GpuPageMatrix::new(device, &pages_flat).expect("Failed to create GpuPageMatrix");

        let mut queries = Vec::with_capacity(32);
        for i in 0..32usize {
            let t0 = i % num_pages;
            let t1 = (i * 7 + 13) % num_pages;
            let t2 = (i * 11 + 29) % num_pages;
            let (k0, _) = crate::dpf::generate_chacha_dpf_keys(&params, t0).unwrap();
            let (k1, _) = crate::dpf::generate_chacha_dpf_keys(&params, t1).unwrap();
            let (k2, _) = crate::dpf::generate_chacha_dpf_keys(&params, t2).unwrap();
            queries.push([k0, k1, k2]);
        }

        for &q in &[1usize, 2, 4, 8, 16, 32] {
            let query_slice = &queries[..q];
            let baseline =
                unsafe { scanner.scan_batch_optimized(&db, query_slice) }.expect("baseline failed");
            let tiled = unsafe { scanner.scan_batch_optimized_tiled(&db, query_slice, 4) }
                .expect("tiled failed");

            assert_eq!(
                baseline.len(),
                tiled.len(),
                "result length mismatch for q={q}"
            );
            for (idx, (left, right)) in baseline.iter().zip(tiled.iter()).enumerate() {
                assert_eq!(
                    left.page0, right.page0,
                    "page0 mismatch for q={q}, idx={idx}"
                );
                assert_eq!(
                    left.page1, right.page1,
                    "page1 mismatch for q={q}, idx={idx}"
                );
                assert_eq!(
                    left.page2, right.page2,
                    "page2 mismatch for q={q}, idx={idx}"
                );
                assert_eq!(
                    left.verif0, right.verif0,
                    "verif0 mismatch for q={q}, idx={idx}"
                );
                assert_eq!(
                    left.verif1, right.verif1,
                    "verif1 mismatch for q={q}, idx={idx}"
                );
                assert_eq!(
                    left.verif2, right.verif2,
                    "verif2 mismatch for q={q}, idx={idx}"
                );
            }
        }
    }
}
