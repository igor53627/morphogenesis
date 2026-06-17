//! GPU scan timing/metrics recording helpers.
//!
//! Extracted from `network/api.rs` in TASK-54.10. Pure metrics glue — derive
//! per-request H2D/kernel/D2H totals from a `KernelTiming`, and emit them to
//! the `metrics` crate. Depends on `morphogen_gpu_dpf::kernel::KernelTiming`
//! only (no `AppState`).
//!
//! Private to the `network` module — NOT part of the crate's public API.
//! Referenced from `network/api.rs` via `use super::gpu_metrics::*;`.

#[cfg(any(test, all(feature = "metrics", feature = "cuda")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuTimingTotals {
    pub(crate) h2d_ns: u64,
    pub(crate) kernel_ns: u64,
    pub(crate) d2h_ns: u64,
}

#[cfg(any(test, all(feature = "metrics", feature = "cuda")))]
pub(crate) fn gpu_timing_totals_for_request(
    timing: &morphogen_gpu_dpf::kernel::KernelTiming,
    query_count: usize,
) -> GpuTimingTotals {
    let multiplier = query_count.max(1) as u64;
    GpuTimingTotals {
        h2d_ns: timing.h2d_ns.saturating_mul(multiplier),
        kernel_ns: timing.kernel_ns.saturating_mul(multiplier),
        d2h_ns: timing.d2h_ns.saturating_mul(multiplier),
    }
}

#[cfg(all(feature = "metrics", feature = "cuda"))]
pub(crate) fn ns_to_secs(ns: u64) -> f64 {
    ns as f64 / 1_000_000_000.0
}

#[cfg(all(feature = "metrics", feature = "cuda"))]
pub(crate) fn record_gpu_phase_duration(
    endpoint: &'static str,
    phase: &'static str,
    duration_ns: u64,
) {
    metrics::histogram!(
        "gpu_query_phase_duration_seconds",
        "endpoint" => endpoint,
        "phase" => phase
    )
    .record(ns_to_secs(duration_ns));
}

#[cfg(all(feature = "metrics", feature = "cuda"))]
pub(crate) fn record_gpu_transfer_and_kernel_metrics(
    endpoint: &'static str,
    timing: &morphogen_gpu_dpf::kernel::KernelTiming,
    query_count: usize,
) {
    let totals = gpu_timing_totals_for_request(timing, query_count);
    record_gpu_phase_duration(endpoint, "transfer_h2d", totals.h2d_ns);
    record_gpu_phase_duration(endpoint, "kernel", totals.kernel_ns);
    record_gpu_phase_duration(endpoint, "transfer_d2h", totals.d2h_ns);
    metrics::histogram!("gpu_scan_duration_seconds", "endpoint" => endpoint)
        .record(ns_to_secs(totals.kernel_ns));
}
