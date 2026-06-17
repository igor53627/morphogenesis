//! GPU page-query batching policy and env-var parsing.
//!
//! Extracted from `network/api.rs` in TASK-54.9. Pure config/policy code:
//! constants, the policy/dispatch enums, env-var parsers, dispatch-shape
//! selection, and a result-count sanity check. No dependency on `AppState`
//! or on `morphogen_gpu_dpf` runtime types — only `axum::http::StatusCode`.
//!
//! Private to the `network` module — NOT part of the crate's public API.
//! Referenced from `network/api.rs` via `use super::gpu_batch::*;`.
//!
//! The scan-execution helpers (`parse_gpu_query_keys`, `with_gpu_matrix_ref`,
//! `run_gpu_scan_branches_with`, `GpuTimingTotals`, `record_gpu_*`,
//! `collect_gpu_page_refs`, `cpu_eval_gpu_page_batch`) stay in `api.rs` for
//! now — they reference `morphogen_gpu_dpf` runtime types and will move in a
//! later scan-execution extraction.

#[cfg(any(feature = "cuda", test))]
use axum::http::StatusCode;

// --- Constants -------------------------------------------------------------
// cfg-gated: only referenced by the cuda/test helpers below.

#[cfg(any(feature = "cuda", test))]
pub(crate) const GPU_MICRO_BATCH_SIZE: usize = 2;

// Env-var names are read only by `configured_*` below, which are cuda-only.
#[cfg(feature = "cuda")]
pub(crate) const GPU_STREAM_COUNT_ENV: &str = "MORPHOGEN_GPU_STREAMS";
#[cfg(feature = "cuda")]
pub(crate) const GPU_BATCH_POLICY_ENV: &str = "MORPHOGEN_GPU_BATCH_POLICY";
#[cfg(feature = "cuda")]
pub(crate) const GPU_BATCH_ADAPTIVE_THRESHOLD_ENV: &str = "MORPHOGEN_GPU_BATCH_ADAPTIVE_THRESHOLD";
#[cfg(feature = "cuda")]
pub(crate) const GPU_CUDA_GRAPH_ENV: &str = "MORPHOGEN_GPU_CUDA_GRAPH";
#[cfg(feature = "cuda")]
pub(crate) const GPU_BATCH_TILE_SIZE_ENV: &str = "MORPHOGEN_GPU_BATCH_TILE_SIZE";

#[cfg(any(feature = "cuda", test))]
pub(crate) const DEFAULT_GPU_STREAM_COUNT: usize = 1;
#[cfg(any(feature = "cuda", test))]
pub(crate) const MAX_GPU_STREAM_COUNT: usize = 8;
#[cfg(any(feature = "cuda", test))]
pub(crate) const DEFAULT_GPU_BATCH_ADAPTIVE_THRESHOLD: usize = 4;
#[cfg(any(feature = "cuda", test))]
pub(crate) const DEFAULT_GPU_BATCH_TILE_SIZE: usize = 16;
#[cfg(any(feature = "cuda", test))]
pub(crate) const MAX_GPU_BATCH_TILE_SIZE: usize = 16;

// --- Policy / dispatch types ----------------------------------------------

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuBatchPolicy {
    Adaptive,
    Throughput,
    Latency,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuBatchPolicyConfig {
    pub(crate) policy: GpuBatchPolicy,
    pub(crate) adaptive_threshold: usize,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuBatchDispatch {
    MultiStream { stream_count: usize },
    FullBatch,
    MicroBatch2,
}

#[cfg(any(feature = "cuda", test))]
impl GpuBatchDispatch {
    #[cfg(any(feature = "metrics", test))]
    pub(crate) fn mode_label(self) -> &'static str {
        match self {
            GpuBatchDispatch::MultiStream { .. } => "multistream",
            GpuBatchDispatch::FullBatch => "full_batch",
            GpuBatchDispatch::MicroBatch2 => "micro_batch2",
        }
    }
}

// --- Range / dispatch helpers ---------------------------------------------

#[cfg(any(feature = "cuda", test))]
pub(crate) fn gpu_micro_batch_ranges(total_queries: usize) -> Vec<(usize, usize)> {
    if total_queries == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::with_capacity(total_queries.div_ceil(GPU_MICRO_BATCH_SIZE));
    let mut start = 0usize;
    while start < total_queries {
        let end = (start + GPU_MICRO_BATCH_SIZE).min(total_queries);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

// --- Env-var parsers ------------------------------------------------------

#[cfg(any(feature = "cuda", test))]
pub(crate) fn parse_gpu_batch_policy(raw: Option<&str>) -> GpuBatchPolicy {
    match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("throughput") => GpuBatchPolicy::Throughput,
        Some("latency") => GpuBatchPolicy::Latency,
        _ => GpuBatchPolicy::Adaptive,
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn parse_gpu_batch_adaptive_threshold(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GPU_BATCH_ADAPTIVE_THRESHOLD)
        .clamp(1, super::api::MAX_BATCH_SIZE)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn parse_gpu_batch_tile_size(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GPU_BATCH_TILE_SIZE)
        .clamp(1, MAX_GPU_BATCH_TILE_SIZE)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn choose_gpu_batch_dispatch(
    total_queries: usize,
    stream_count: usize,
    cfg: GpuBatchPolicyConfig,
) -> GpuBatchDispatch {
    if stream_count > 1 {
        return GpuBatchDispatch::MultiStream { stream_count };
    }

    match cfg.policy {
        GpuBatchPolicy::Throughput => GpuBatchDispatch::MicroBatch2,
        GpuBatchPolicy::Latency => GpuBatchDispatch::FullBatch,
        GpuBatchPolicy::Adaptive => {
            if total_queries <= cfg.adaptive_threshold {
                GpuBatchDispatch::FullBatch
            } else {
                GpuBatchDispatch::MicroBatch2
            }
        }
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn parse_gpu_stream_count(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GPU_STREAM_COUNT)
        .clamp(DEFAULT_GPU_STREAM_COUNT, MAX_GPU_STREAM_COUNT)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn parse_gpu_cuda_graph_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn ensure_gpu_result_count(expected: usize, actual: usize) -> Result<(), StatusCode> {
    if actual == expected {
        return Ok(());
    }
    Err(StatusCode::INTERNAL_SERVER_ERROR)
}

// --- Env-var readers (with defaults) --------------------------------------
// Used only by the cuda build's page-query handlers; no unit-test coverage,
// so gated to `feature = "cuda"` rather than `any(feature = "cuda", test)` to
// avoid dead-code warnings in plain test builds.

#[cfg(feature = "cuda")]
pub(crate) fn configured_gpu_stream_count() -> usize {
    let raw = std::env::var(GPU_STREAM_COUNT_ENV).ok();
    parse_gpu_stream_count(raw.as_deref())
}

#[cfg(feature = "cuda")]
pub(crate) fn configured_gpu_batch_policy() -> GpuBatchPolicyConfig {
    let policy = parse_gpu_batch_policy(std::env::var(GPU_BATCH_POLICY_ENV).ok().as_deref());
    let adaptive_threshold = parse_gpu_batch_adaptive_threshold(
        std::env::var(GPU_BATCH_ADAPTIVE_THRESHOLD_ENV)
            .ok()
            .as_deref(),
    );
    GpuBatchPolicyConfig {
        policy,
        adaptive_threshold,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn configured_gpu_cuda_graph_enabled() -> bool {
    let raw = std::env::var(GPU_CUDA_GRAPH_ENV).ok();
    parse_gpu_cuda_graph_enabled(raw.as_deref())
}

#[cfg(feature = "cuda")]
pub(crate) fn configured_gpu_batch_tile_size() -> usize {
    let raw = std::env::var(GPU_BATCH_TILE_SIZE_ENV).ok();
    parse_gpu_batch_tile_size(raw.as_deref())
}
