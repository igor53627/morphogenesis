//! GPU scan-execution helpers: matrix-lock handling, dispatch fan-out, page
//! collection, and the test-injection hooks for the GPU page-query handlers.
//!
//! Extracted from `network/api.rs` in TASK-54.12. These helpers depend on
//! `morphogen_gpu_dpf` runtime types and on the policy types in
//! [`super::gpu_batch`]. The two DTO-coupled helpers (`parse_gpu_query_keys`,
//! which reads `GpuPageQueryRequest`, and `cpu_eval_gpu_page_batch`, which
//! writes `BatchGpuPageQueryResult`) stay in `api.rs` until the DTOs themselves
//! are extracted; `cpu_eval_gpu_page_batch` reaches [`collect_gpu_page_refs`]
//! via the glob re-export from `api.rs`.
//!
//! Private to the `network` module — NOT part of the crate's public API.
//! Referenced from `network/api.rs` via `use super::gpu_scan::*;`.

#[cfg(any(feature = "cuda", test))]
use axum::http::StatusCode;

#[cfg(any(feature = "cuda", test))]
use super::gpu_batch::{ensure_gpu_result_count, gpu_micro_batch_ranges, GpuBatchDispatch};

#[cfg(any(feature = "cuda", test))]
pub(crate) fn with_gpu_matrix_ref<T, R, F>(
    matrix_mutex: &std::sync::Mutex<Option<T>>,
    scan: F,
) -> Result<Option<R>, StatusCode>
where
    F: FnOnce(&T) -> Result<R, StatusCode>,
{
    let matrix_guard = matrix_mutex
        .lock()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    matrix_guard.as_ref().map(scan).transpose()
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn run_gpu_scan_branches_with<FMulti, FFull, FMicro>(
    all_keys: &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
    dispatch: GpuBatchDispatch,
    mut scan_multistream: FMulti,
    mut scan_full_batch: FFull,
    mut scan_micro_batch: FMicro,
) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>
where
    FMulti: FnMut(
        &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
        usize,
    ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>,
    FFull: FnMut(
        &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
    ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>,
    FMicro: FnMut(
        &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
    ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>,
{
    let n = all_keys.len();
    let gpu_results = match dispatch {
        GpuBatchDispatch::MultiStream { stream_count } => scan_multistream(all_keys, stream_count)?,
        GpuBatchDispatch::FullBatch => {
            let results = scan_full_batch(all_keys)?;
            ensure_gpu_result_count(n, results.len())?;
            results
        }
        GpuBatchDispatch::MicroBatch2 => {
            let mut gpu_results = Vec::with_capacity(n);
            for (start, end) in gpu_micro_batch_ranges(n) {
                let key_batch = &all_keys[start..end];
                let mut chunk_results = scan_micro_batch(key_batch)?;
                ensure_gpu_result_count(key_batch.len(), chunk_results.len())?;
                gpu_results.append(&mut chunk_results);
            }
            gpu_results
        }
    };
    ensure_gpu_result_count(n, gpu_results.len())?;
    Ok(gpu_results)
}

#[cfg(test)]
pub(crate) type TestGpuBatchMultistreamScan = std::sync::Arc<
    dyn Fn(
            &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
            usize,
        ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>
        + Send
        + Sync,
>;

#[cfg(test)]
pub(crate) type TestGpuBatchFullScan = std::sync::Arc<
    dyn Fn(
            &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
        ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>
        + Send
        + Sync,
>;

#[cfg(test)]
pub(crate) type TestGpuBatchMicroScan = std::sync::Arc<
    dyn Fn(
            &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
        ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>
        + Send
        + Sync,
>;

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct TestGpuBatchHooks {
    pub(crate) dispatch: GpuBatchDispatch,
    pub(crate) multistream_scan: TestGpuBatchMultistreamScan,
    pub(crate) full_batch_scan: TestGpuBatchFullScan,
    pub(crate) micro_batch_scan: TestGpuBatchMicroScan,
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_GPU_BATCH_HOOKS: std::cell::RefCell<Option<TestGpuBatchHooks>> =
        const { std::cell::RefCell::new(None) };
}

pub(crate) fn collect_gpu_page_refs(matrix: &morphogen_storage::ChunkedMatrix) -> Vec<&[u8]> {
    let page_size = morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;
    let num_pages = matrix.total_size_bytes() / page_size;
    let mut pages_refs = Vec::with_capacity(num_pages);
    for i in 0..num_pages {
        let start = i * page_size;
        let (chunk_idx, chunk_offset) = (
            start / matrix.chunk_size_bytes(),
            start % matrix.chunk_size_bytes(),
        );
        let chunk = matrix.chunk(chunk_idx);
        pages_refs.push(&chunk.as_slice()[chunk_offset..chunk_offset + page_size]);
    }
    pages_refs
}
