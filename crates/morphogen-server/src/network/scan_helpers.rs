//! Scan-result post-processing helpers used by the PIR query handlers.
//!
//! Extracted from `network/api.rs` in TASK-54.13. These helpers bridge the
//! `crate::scan` kernel output to the API DTOs: convert scan errors to HTTP
//! statuses, move `[Vec<u8>; 3]` payload arrays into `Vec<Vec<u8>>` without
//! copying, apply delta-buffer XOR entries to scan results, and assemble
//! per-snapshot batch query results (fused or per-key).
//!
//! Private to the `network` module — NOT part of the crate's public API.
//! Referenced from `network/api.rs` via `use super::scan_helpers::*;`.

use axum::http::StatusCode;

use crate::scan::ScanError;

pub(crate) fn scan_error_to_status(e: ScanError) -> StatusCode {
    match e {
        ScanError::TooManyRetries { .. } => StatusCode::SERVICE_UNAVAILABLE,
        ScanError::LockPoisoned
        | ScanError::MatrixNotAligned { .. }
        | ScanError::ChunkNotAligned { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

pub(crate) fn payload_array_into_vec(payloads: [Vec<u8>; 3]) -> Vec<Vec<u8>> {
    Vec::from(payloads)
}

pub(crate) fn apply_delta_entries_to_payloads<K: morphogen_dpf::DpfKey>(
    payloads: &mut [Vec<u8>; 3],
    keys: &[K; 3],
    entries: &[morphogen_core::DeltaEntry],
) -> Result<(), StatusCode> {
    for entry in entries {
        for (k, key) in keys.iter().enumerate() {
            if key.eval_bit(entry.row_idx) {
                if entry.diff.len() != payloads[k].len() {
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
                for (d, s) in payloads[k].iter_mut().zip(entry.diff.iter()) {
                    *d ^= s;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn scan_batch_results_from_snapshot<K: morphogen_dpf::DpfKey>(
    matrix: &morphogen_storage::ChunkedMatrix,
    all_keys: &[[K; 3]],
    entries: &[morphogen_core::DeltaEntry],
    row_size_bytes: usize,
) -> Result<Vec<super::api::BatchQueryResult>, StatusCode> {
    #[cfg(feature = "fused-batch-scan")]
    {
        let payload_sets = crate::scan::scan_main_matrix_multi(matrix, all_keys, row_size_bytes);
        let mut results = Vec::with_capacity(all_keys.len());
        for (keys, mut payloads) in all_keys.iter().zip(payload_sets.into_iter()) {
            apply_delta_entries_to_payloads(&mut payloads, keys, entries)?;
            results.push(super::api::BatchQueryResult {
                payloads: payload_array_into_vec(payloads),
            });
        }
        return Ok(results);
    }

    #[cfg(not(feature = "fused-batch-scan"))]
    {
        let mut results = Vec::with_capacity(all_keys.len());
        for keys in all_keys {
            let mut payloads = crate::scan::scan_main_matrix(matrix, keys, row_size_bytes);
            apply_delta_entries_to_payloads(&mut payloads, keys, entries)?;
            results.push(super::api::BatchQueryResult {
                payloads: payload_array_into_vec(payloads),
            });
        }
        Ok(results)
    }
}
