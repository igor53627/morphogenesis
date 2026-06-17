//! HTTP and WebSocket API implementation.

// Wire-format DTOs (request/response shapes) extracted to
// `network::dto` (pub) in TASK-54.15. Re-exported here so the existing
// `network::api::<DTO>` path keeps resolving at all external call sites.
pub use super::dto::*;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        DefaultBodyLimit, State, WebSocketUpgrade,
    },
    http::{HeaderMap, HeaderName, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
#[cfg(feature = "metrics")]
use metrics::counter;
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::epoch::EpochManager;
#[cfg(feature = "verifiable-pir")]
use morphogen_core::sumcheck::SumCheckProof;
use morphogen_core::GlobalState;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::{watch, Mutex};

/// Optimal DPF chunk size for eval_and_accumulate_chunked.
/// Larger chunks amortize tree-traversal overhead. 65536 = 64K elements.
/// At 16 bytes per DPF output, this uses 1MB of buffer per evaluation.
/// Benchmarks show this is ~1.4x faster than chunk_size=4096.
pub const OPTIMAL_DPF_CHUNK_SIZE: usize = 65536;

#[derive(Clone)]
pub struct PagePirConfig {
    pub domain_bits: usize,
    pub rows_per_page: usize,
    pub prg_keys: [[u8; 16]; 2],
}

#[derive(Clone)]
pub struct AppState {
    pub global: Arc<GlobalState>,
    pub epoch_manager: Arc<EpochManager>,
    pub epoch_tx: watch::Sender<EpochMetadata>,
    pub snapshot_rotation_lock: Arc<Mutex<()>>,
    pub admin_snapshot_token: Option<String>,
    pub admin_mtls_subject_header: HeaderName,
    pub admin_mtls_allowed_subjects: Vec<String>,
    pub admin_mtls_trust_proxy_headers: bool,
    pub admin_snapshot_allow_local_paths: bool,
    pub admin_snapshot_allowed_hosts: Vec<String>,
    pub admin_snapshot_max_bytes: usize,
    pub row_size_bytes: usize,
    pub num_rows: usize,
    pub seeds: [u64; 3],
    pub block_number: u64,
    pub state_root: [u8; 32],
    pub epoch_rx: watch::Receiver<EpochMetadata>,
    pub page_config: Option<PagePirConfig>,
    #[cfg(feature = "cuda")]
    pub gpu_scanner: Option<Arc<morphogen_gpu_dpf::kernel::GpuScanner>>,
    #[cfg(feature = "cuda")]
    pub gpu_matrix:
        Option<Arc<std::sync::Mutex<Option<morphogen_gpu_dpf::storage::GpuPageMatrix>>>>,
}

#[cfg(any(feature = "cuda", test))]
// GPU batching policy/parse/configured helpers extracted to
// `network::gpu_batch` (private) in TASK-54.9.
#[cfg(any(feature = "cuda", test))]
use super::gpu_batch::*;

// Hex serde helpers (`hex_bytes`, `hex_bytes_vec`, `hex_bytes_array`) now live
// next to the DTOs in `network::dto` (TASK-54.15); not needed in api.rs.

pub async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let snapshot = state.global.load();
    let metadata = state.epoch_rx.borrow().clone();
    Json(HealthResponse {
        status: "ok".to_string(),
        epoch_id: snapshot.epoch_id,
        block_number: metadata.block_number,
    })
}

pub async fn epoch_handler(State(state): State<Arc<AppState>>) -> Json<EpochMetadataResponse> {
    let snapshot = state.global.load();
    let metadata = state.epoch_rx.borrow().clone();
    let page_pir = state.page_config.as_ref().map(|cfg| PagePirResponse {
        domain_bits: cfg.domain_bits,
        rows_per_page: cfg.rows_per_page,
        num_pages: 1usize << cfg.domain_bits,
        prg_keys: cfg.prg_keys,
    });
    Json(EpochMetadataResponse {
        epoch_id: snapshot.epoch_id,
        num_rows: metadata.num_rows,
        seeds: metadata.seeds,
        block_number: metadata.block_number,
        state_root: metadata.state_root,
        page_pir,
    })
}

// Admin-snapshot auth primitives (header parsing, constant-time compare,
// snapshot source) extracted to `network::admin_auth` (private) in TASK-54.11.
#[cfg(feature = "network")]
use super::admin_auth::*;

// Compares secrets without early exit on mismatch.
fn authorize_admin_snapshot(state: &AppState, headers: &HeaderMap) -> Result<(), StatusCode> {
    let token_configured = state.admin_snapshot_token.is_some();
    let mtls_configured =
        state.admin_mtls_trust_proxy_headers && !state.admin_mtls_allowed_subjects.is_empty();
    if !token_configured && !mtls_configured {
        return Err(StatusCode::FORBIDDEN);
    }

    if let Some(expected) = state.admin_snapshot_token.as_deref() {
        let bearer_matches = bearer_token_from_headers(headers)
            .is_some_and(|provided| admin_token_eq_constant_time(expected, provided));
        let legacy_matches = legacy_admin_token_from_headers(headers)
            .is_some_and(|provided| admin_token_eq_constant_time(expected, provided));
        if bearer_matches || legacy_matches {
            return Ok(());
        }
    }

    if mtls_configured {
        if let Some(subject) = headers
            .get(&state.admin_mtls_subject_header)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            if state
                .admin_mtls_allowed_subjects
                .iter()
                .any(|allowed| allowed == subject)
            {
                return Ok(());
            }
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

// Snapshot source-resolution / fetch / decode helpers extracted to
// `network::snapshot_config` (private) in TASK-54.14.
use super::snapshot_config::*;

fn validate_snapshot_page_pir_compatibility(
    state: &AppState,
    matrix: &morphogen_storage::ChunkedMatrix,
    num_rows: usize,
) -> Result<(), StatusCode> {
    let Some(page_config) = state.page_config.as_ref() else {
        return Ok(());
    };

    let page_size_bytes = morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;
    if !matrix.total_size_bytes().is_multiple_of(page_size_bytes) {
        return Err(StatusCode::BAD_REQUEST);
    }
    for chunk_idx in 0..matrix.num_chunks() {
        if !matrix.chunk_size(chunk_idx).is_multiple_of(page_size_bytes) {
            return Err(StatusCode::BAD_REQUEST);
        }
    }

    let Some(num_pages) = 1usize.checked_shl(page_config.domain_bits as u32) else {
        return Err(StatusCode::BAD_REQUEST);
    };
    let Some(max_rows) = num_pages.checked_mul(page_config.rows_per_page) else {
        return Err(StatusCode::BAD_REQUEST);
    };
    if num_rows > max_rows {
        return Err(StatusCode::BAD_REQUEST);
    }

    Ok(())
}

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn admin_snapshot_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<AdminSnapshotRequest>,
) -> Result<StatusCode, StatusCode> {
    authorize_admin_snapshot(state.as_ref(), &headers)?;

    let source = parse_snapshot_source(
        &request.r2_url,
        state.admin_snapshot_allow_local_paths,
        &state.admin_snapshot_allowed_hosts,
    )?;
    let snapshot_bytes = fetch_snapshot_bytes(source, state.admin_snapshot_max_bytes).await?;

    let _rotation_guard = state.snapshot_rotation_lock.lock().await;

    let current = state.global.load();
    let chunk_size_bytes = current.matrix.chunk_size_bytes();
    let matrix = decode_snapshot_matrix(&snapshot_bytes, state.row_size_bytes, chunk_size_bytes)?;
    let num_rows = matrix.total_size_bytes() / state.row_size_bytes;
    if num_rows == 0 {
        return Err(StatusCode::BAD_REQUEST);
    }
    validate_snapshot_page_pir_compatibility(state.as_ref(), &matrix, num_rows)?;

    let current_metadata = state.epoch_rx.borrow().clone();
    let seeds = request.seeds.unwrap_or(current_metadata.seeds);
    let block_number = request
        .block_number
        .unwrap_or(current_metadata.block_number);
    let state_root = request.state_root.unwrap_or(current_metadata.state_root);

    let next_epoch_id = state
        .epoch_manager
        .submit_snapshot(matrix, seeds)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let metadata = EpochMetadata {
        epoch_id: next_epoch_id,
        num_rows,
        seeds,
        block_number,
        state_root,
    };
    let _ = state.epoch_tx.send(metadata);

    Ok(StatusCode::OK)
}

// Scan-result helpers (error mapping, payload move, delta XOR, batch
// assembly) extracted to `network::scan_helpers` (private) in TASK-54.13.
use super::scan_helpers::*;

fn parse_gpu_query_keys(
    request: &GpuPageQueryRequest,
) -> Result<[morphogen_gpu_dpf::dpf::ChaChaKey; 3], StatusCode> {
    use morphogen_gpu_dpf::dpf::ChaChaKey;
    use morphogen_gpu_dpf::kernel::MAX_DOMAIN_BITS;

    if request.keys.len() != 3 {
        return Err(StatusCode::BAD_REQUEST);
    }

    match (
        ChaChaKey::from_bytes(&request.keys[0]),
        ChaChaKey::from_bytes(&request.keys[1]),
        ChaChaKey::from_bytes(&request.keys[2]),
    ) {
        (Ok(k0), Ok(k1), Ok(k2)) => {
            if k0.domain_bits > MAX_DOMAIN_BITS
                || k1.domain_bits > MAX_DOMAIN_BITS
                || k2.domain_bits > MAX_DOMAIN_BITS
            {
                return Err(StatusCode::BAD_REQUEST);
            }
            Ok([k0, k1, k2])
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

// GPU scan-execution helpers (matrix-lock, dispatch fan-out, page
// collection, test hooks) extracted to `network::gpu_scan` (private)
// in TASK-54.12. The glob is ungated: cfg-gated items (with_gpu_matrix_ref,
// run_gpu_scan_branches_with, test hooks) simply aren't imported when their
// cfg is inactive; collect_gpu_page_refs is always available.
use super::gpu_scan::*;

// GPU timing/metrics helpers extracted to `network::gpu_metrics`
// (private) in TASK-54.10.
#[cfg(any(test, all(feature = "metrics", feature = "cuda")))]
use super::gpu_metrics::*;

fn cpu_eval_gpu_page_batch(
    all_keys: &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
    matrix: &morphogen_storage::ChunkedMatrix,
) -> Result<Vec<BatchGpuPageQueryResult>, StatusCode> {
    let pages_refs = collect_gpu_page_refs(matrix);
    let mut results = Vec::with_capacity(all_keys.len());
    for keys in all_keys {
        let result = morphogen_gpu_dpf::kernel::eval_fused_3dpf_cpu(
            [&keys[0], &keys[1], &keys[2]],
            &pages_refs,
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        results.push(BatchGpuPageQueryResult {
            pages: vec![result.page0, result.page1, result.page2],
            #[cfg(feature = "verifiable-pir")]
            proof: None,
        });
    }
    Ok(results)
}

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn query_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, StatusCode> {
    #[cfg(feature = "metrics")]
    counter!("pir_query_count_total", "type" => "row").increment(1);

    use crate::scan::scan_consistent;
    use morphogen_dpf::AesDpfKey;

    if request.keys.len() != 3 {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Parse DPF keys from bytes
    let keys: [AesDpfKey; 3] = match (
        AesDpfKey::from_bytes(&request.keys[0]),
        AesDpfKey::from_bytes(&request.keys[1]),
        AesDpfKey::from_bytes(&request.keys[2]),
    ) {
        (Ok(k0), Ok(k1), Ok(k2)) => [k0, k1, k2],
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    let global = Arc::clone(&state.global);
    let row_size_bytes = state.row_size_bytes;
    let (results, epoch_id) = tokio::task::spawn_blocking(move || {
        let pending = global.load_pending();
        scan_consistent(global.as_ref(), pending.as_ref(), &keys, row_size_bytes)
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(scan_error_to_status)?;

    Ok(Json(QueryResponse {
        epoch_id,
        payloads: payload_array_into_vec(results),
    }))
}

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn batch_query_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchQueryRequest>,
) -> Result<Json<BatchQueryResponse>, StatusCode> {
    #[cfg(feature = "metrics")]
    counter!("pir_query_count_total", "type" => "batch").increment(1);

    use morphogen_dpf::AesDpfKey;

    let n = request.queries.len();
    if n == 0 || n > MAX_BATCH_SIZE {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Validate all queries have exactly 3 keys and parse DPF keys upfront
    let mut all_keys: Vec<[AesDpfKey; 3]> = Vec::with_capacity(n);
    for query in &request.queries {
        if query.keys.len() != 3 {
            return Err(StatusCode::BAD_REQUEST);
        }
        let keys: [AesDpfKey; 3] = match (
            AesDpfKey::from_bytes(&query.keys[0]),
            AesDpfKey::from_bytes(&query.keys[1]),
            AesDpfKey::from_bytes(&query.keys[2]),
        ) {
            (Ok(k0), Ok(k1), Ok(k2)) => [k0, k1, k2],
            _ => return Err(StatusCode::BAD_REQUEST),
        };
        all_keys.push(keys);
    }

    // Take a consistent snapshot once for all queries
    let snapshot1 = state.global.load();
    let epoch1 = snapshot1.epoch_id;
    let pending = state.global.load_pending();
    let (pending_epoch, entries) = pending
        .snapshot_with_epoch()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let snapshot2 = state.global.load();

    if snapshot2.epoch_id != epoch1 || pending_epoch != epoch1 {
        // Epoch changed during snapshot — fall back to per-query scan_consistent
        let mut results = Vec::with_capacity(n);
        let mut batch_epoch: Option<u64> = None;
        let global = Arc::clone(&state.global);
        let row_size_bytes = state.row_size_bytes;
        for keys in &all_keys {
            let keys = keys.clone();
            let global_for_scan = Arc::clone(&global);
            let (payloads, query_epoch) = tokio::task::spawn_blocking(move || {
                let pending_for_scan = global_for_scan.load_pending();
                crate::scan::scan_consistent(
                    global_for_scan.as_ref(),
                    pending_for_scan.as_ref(),
                    &keys,
                    row_size_bytes,
                )
            })
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .map_err(scan_error_to_status)?;
            match batch_epoch {
                None => batch_epoch = Some(query_epoch),
                Some(e) if e != query_epoch => {
                    return Err(StatusCode::SERVICE_UNAVAILABLE);
                }
                _ => {}
            }
            results.push(BatchQueryResult {
                payloads: payload_array_into_vec(payloads),
            });
        }
        let epoch_id = batch_epoch.unwrap_or_else(|| state.global.load().epoch_id);
        return Ok(Json(BatchQueryResponse { epoch_id, results }));
    }

    // All queries share the same consistent snapshot
    let matrix = Arc::clone(&snapshot1.matrix);
    let row_size_bytes = state.row_size_bytes;
    let results = tokio::task::spawn_blocking(move || {
        scan_batch_results_from_snapshot(matrix.as_ref(), &all_keys, &entries, row_size_bytes)
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    Ok(Json(BatchQueryResponse {
        epoch_id: epoch1,
        results,
    }))
}

pub async fn ws_epoch_handler(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    let epoch_rx = state.epoch_rx.clone();
    ws.max_message_size(MAX_WS_MESSAGE_BYTES)
        .max_frame_size(MAX_WS_MESSAGE_BYTES)
        .on_upgrade(move |socket| handle_ws_epoch(socket, epoch_rx))
}

async fn handle_ws_epoch(mut socket: WebSocket, mut epoch_rx: watch::Receiver<EpochMetadata>) {
    let metadata = epoch_rx.borrow().clone();
    if let Ok(json) = serde_json::to_string(&metadata) {
        if socket.send(Message::Text(json.into())).await.is_err() {
            return;
        }
    }

    while epoch_rx.changed().await.is_ok() {
        let metadata = epoch_rx.borrow().clone();
        if let Ok(json) = serde_json::to_string(&metadata) {
            if socket.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    }
}

#[derive(Serialize)]
pub struct WsQueryError {
    pub error: String,
    pub code: String,
}

pub async fn ws_query_handler(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.max_message_size(MAX_WS_MESSAGE_BYTES)
        .max_frame_size(MAX_WS_MESSAGE_BYTES)
        .on_upgrade(move |socket| handle_ws_query(socket, state))
}

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn page_query_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PageQueryRequest>,
) -> Result<Json<PageQueryResponse>, StatusCode> {
    #[cfg(feature = "metrics")]
    counter!("pir_query_count_total", "type" => "page").increment(1);

    use crate::scan::scan_pages_consistent;
    use morphogen_dpf::page::{PageDpfKey, PAGE_SIZE_BYTES};

    let page_config = state.page_config.as_ref().ok_or(StatusCode::NOT_FOUND)?;

    if request.keys.len() != 3 {
        return Err(StatusCode::BAD_REQUEST);
    }

    let keys: [PageDpfKey; 3] = match (
        PageDpfKey::from_bytes(&request.keys[0]),
        PageDpfKey::from_bytes(&request.keys[1]),
        PageDpfKey::from_bytes(&request.keys[2]),
    ) {
        (Ok(k0), Ok(k1), Ok(k2)) => {
            // Validate domain_bits matches server config
            if k0.domain_bits() != page_config.domain_bits
                || k1.domain_bits() != page_config.domain_bits
                || k2.domain_bits() != page_config.domain_bits
            {
                return Err(StatusCode::BAD_REQUEST);
            }
            // Validate PRG keys match server config (prevents silent wrong answers)
            if k0.prg_keys() != &page_config.prg_keys
                || k1.prg_keys() != &page_config.prg_keys
                || k2.prg_keys() != &page_config.prg_keys
            {
                return Err(StatusCode::BAD_REQUEST);
            }
            [k0, k1, k2]
        }
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    let global = Arc::clone(&state.global);
    let (results, epoch_id) = tokio::task::spawn_blocking(move || {
        scan_pages_consistent(
            global.as_ref(),
            &keys,
            PAGE_SIZE_BYTES,
            OPTIMAL_DPF_CHUNK_SIZE,
        )
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(scan_error_to_status)?;

    Ok(Json(PageQueryResponse {
        epoch_id,
        pages: payload_array_into_vec(results),
        #[cfg(feature = "verifiable-pir")]
        proof: None,
    }))
}

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn page_query_gpu_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GpuPageQueryRequest>,
) -> Result<Json<PageQueryResponse>, StatusCode> {
    #[cfg(feature = "metrics")]
    counter!("pir_query_count_total", "type" => "gpu").increment(1);

    let keys = parse_gpu_query_keys(&request)?;

    #[cfg(feature = "cuda")]
    if let (Some(scanner), Some(matrix_mutex)) = (&state.gpu_scanner, &state.gpu_matrix) {
        use crate::scan::scan_delta_for_gpu;
        #[cfg(feature = "verifiable-pir")]
        use morphogen_core::sumcheck::SumCheckProof;
        use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

        for _ in 0..10 {
            let snapshot1 = state.global.load();
            let pending = state.global.load_pending();
            let pending_epoch = pending.pending_epoch();
            if pending_epoch != snapshot1.epoch_id {
                std::thread::yield_now();
                continue;
            }

            let mut results = match tokio::task::block_in_place(|| {
                with_gpu_matrix_ref(matrix_mutex.as_ref(), |matrix| {
                    unsafe { scanner.scan(matrix, [&keys[0], &keys[1], &keys[2]]) }
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
                })
            })? {
                Some(result) => result,
                None => {
                    std::thread::yield_now();
                    continue;
                }
            };

            #[cfg(feature = "metrics")]
            record_gpu_transfer_and_kernel_metrics("gpu", &results.timing, 1);

            let merge_start = std::time::Instant::now();
            let delta_results = scan_delta_for_gpu(pending.as_ref(), &keys, PAGE_SIZE_BYTES)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let snapshot2 = state.global.load();
            let pending_epoch_after = pending.pending_epoch();
            if snapshot1.epoch_id != snapshot2.epoch_id || pending_epoch_after != snapshot1.epoch_id
            {
                std::thread::yield_now();
                continue;
            }

            for k in 0..3 {
                let gpu_page = match k {
                    0 => &mut results.page0,
                    1 => &mut results.page1,
                    _ => &mut results.page2,
                };
                for (i, b) in delta_results[k].iter().enumerate() {
                    gpu_page[i] ^= *b;
                }
            }
            #[cfg(feature = "metrics")]
            record_gpu_phase_duration("gpu", "merge", merge_start.elapsed().as_nanos() as u64);

            // Construct Proof (Round 0)
            // In Binius Sum-Check for Dot Product, the round polynomial g(X) is quadratic.
            // We need evaluations at 0, 1, and alpha.
            // The GPU kernel returns the sum for each key.
            // Since standard PIR is just sum(D * Q) where Q is 0/1,
            // the "verification" result is the GF(2^128) dot product.
            // We assume verif0..2 are the dot products for the 3 keys.
            // For a full proof, we'd need to combine them or prove each separately.
            // Here we package them as a single "proof" for the client to verify against C_D.

            #[cfg(feature = "verifiable-pir")]
            let proof = {
                let v0 = u128::from_le_bytes(results.verif0.try_into().unwrap_or([0; 16]));
                let v1 = u128::from_le_bytes(results.verif1.try_into().unwrap_or([0; 16]));
                let v2 = u128::from_le_bytes(results.verif2.try_into().unwrap_or([0; 16]));

                SumCheckProof {
                    round_polynomials: vec![], // Populated in later rounds (on CPU/Client or next step)
                    sum: v0 ^ v1 ^ v2,         // Aggregate for simple check
                }
            };

            return Ok(Json(PageQueryResponse {
                epoch_id: snapshot1.epoch_id,
                pages: vec![results.page0, results.page1, results.page2],
                #[cfg(feature = "verifiable-pir")]
                proof: Some(proof),
            }));
        }
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    // CPU fallback (or if CUDA is disabled)
    let snapshot = state.global.load();
    let mut batch = cpu_eval_gpu_page_batch(std::slice::from_ref(&keys), snapshot.matrix.as_ref())?;
    let result = batch.pop().ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(PageQueryResponse {
        epoch_id: snapshot.epoch_id,
        pages: result.pages,
        #[cfg(feature = "verifiable-pir")]
        proof: result.proof, // No proof for CPU fallback path
    }))
}

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn page_query_gpu_batch_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchGpuPageQueryRequest>,
) -> Result<Json<BatchGpuPageQueryResponse>, StatusCode> {
    #[cfg(feature = "metrics")]
    counter!("pir_query_count_total", "type" => "gpu_batch").increment(1);

    let n = request.queries.len();
    if n == 0 || n > MAX_BATCH_SIZE {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut all_keys = Vec::with_capacity(n);
    for query in &request.queries {
        all_keys.push(parse_gpu_query_keys(query)?);
    }

    #[cfg(test)]
    if let Some(hooks) = TEST_GPU_BATCH_HOOKS.with(|slot| slot.borrow().clone()) {
        let gpu_results = run_gpu_scan_branches_with(
            &all_keys,
            hooks.dispatch,
            |keys, count| (hooks.multistream_scan)(keys, count),
            |keys| (hooks.full_batch_scan)(keys),
            |key_batch| (hooks.micro_batch_scan)(key_batch),
        )?;
        let results = gpu_results
            .into_iter()
            .map(|gpu_result| BatchGpuPageQueryResult {
                pages: vec![gpu_result.page0, gpu_result.page1, gpu_result.page2],
                #[cfg(feature = "verifiable-pir")]
                proof: None,
            })
            .collect();
        return Ok(Json(BatchGpuPageQueryResponse {
            epoch_id: state.global.load().epoch_id,
            results,
        }));
    }

    #[cfg(feature = "cuda")]
    if let (Some(scanner), Some(matrix_mutex)) = (&state.gpu_scanner, &state.gpu_matrix) {
        use crate::scan::scan_delta_for_gpu;
        #[cfg(feature = "verifiable-pir")]
        use morphogen_core::sumcheck::SumCheckProof;
        use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

        for _ in 0..10 {
            let snapshot1 = state.global.load();
            let pending = state.global.load_pending();
            let pending_epoch = pending.pending_epoch();
            if pending_epoch != snapshot1.epoch_id {
                std::thread::yield_now();
                continue;
            }

            let stream_count = configured_gpu_stream_count();
            let policy_cfg = configured_gpu_batch_policy();
            let cuda_graph_enabled = configured_gpu_cuda_graph_enabled();
            let tile_size = configured_gpu_batch_tile_size();
            let dispatch = choose_gpu_batch_dispatch(n, stream_count, policy_cfg);
            #[cfg(feature = "metrics")]
            counter!("gpu_batch_dispatch_mode_total", "mode" => dispatch.mode_label()).increment(1);
            let gpu_results = match tokio::task::block_in_place(|| {
                with_gpu_matrix_ref(matrix_mutex.as_ref(), |matrix| {
                    run_gpu_scan_branches_with(
                        &all_keys,
                        dispatch,
                        |keys, count| {
                            unsafe {
                                scanner.scan_batch_single_query_multistream_optimized_with_graph(
                                    matrix,
                                    keys,
                                    count,
                                    cuda_graph_enabled,
                                )
                            }
                            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
                        },
                        |keys| {
                            unsafe {
                                scanner.scan_batch_optimized_tiled_with_graph(
                                    matrix,
                                    keys,
                                    tile_size,
                                    cuda_graph_enabled,
                                )
                            }
                            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
                        },
                        |key_batch| {
                            unsafe {
                                scanner.scan_batch_optimized_tiled_with_graph(
                                    matrix,
                                    key_batch,
                                    tile_size,
                                    cuda_graph_enabled,
                                )
                            }
                            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
                        },
                    )
                })
            })? {
                Some(results) => results,
                None => {
                    std::thread::yield_now();
                    continue;
                }
            };

            #[cfg(feature = "metrics")]
            if let Some(first) = gpu_results.first() {
                record_gpu_transfer_and_kernel_metrics("gpu_batch", &first.timing, n);
            }

            let merge_start = std::time::Instant::now();
            let mut results = Vec::with_capacity(n);
            for (mut gpu_result, keys) in gpu_results.into_iter().zip(all_keys.iter()) {
                let delta_results = scan_delta_for_gpu(pending.as_ref(), keys, PAGE_SIZE_BYTES)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

                for k in 0..3 {
                    let gpu_page = match k {
                        0 => &mut gpu_result.page0,
                        1 => &mut gpu_result.page1,
                        _ => &mut gpu_result.page2,
                    };
                    for (i, b) in delta_results[k].iter().enumerate() {
                        gpu_page[i] ^= *b;
                    }
                }

                #[cfg(feature = "verifiable-pir")]
                let proof = {
                    let v0 = u128::from_le_bytes(gpu_result.verif0.try_into().unwrap_or([0; 16]));
                    let v1 = u128::from_le_bytes(gpu_result.verif1.try_into().unwrap_or([0; 16]));
                    let v2 = u128::from_le_bytes(gpu_result.verif2.try_into().unwrap_or([0; 16]));

                    Some(SumCheckProof {
                        round_polynomials: vec![],
                        sum: v0 ^ v1 ^ v2,
                    })
                };

                results.push(BatchGpuPageQueryResult {
                    pages: vec![gpu_result.page0, gpu_result.page1, gpu_result.page2],
                    #[cfg(feature = "verifiable-pir")]
                    proof,
                });
            }

            let snapshot2 = state.global.load();
            let pending_epoch_after = pending.pending_epoch();
            if snapshot1.epoch_id != snapshot2.epoch_id || pending_epoch_after != snapshot1.epoch_id
            {
                std::thread::yield_now();
                continue;
            }

            #[cfg(feature = "metrics")]
            record_gpu_phase_duration(
                "gpu_batch",
                "merge",
                merge_start.elapsed().as_nanos() as u64,
            );

            return Ok(Json(BatchGpuPageQueryResponse {
                epoch_id: snapshot1.epoch_id,
                results,
            }));
        }

        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    // CPU fallback (or if CUDA is disabled)
    let snapshot = state.global.load();
    let results = cpu_eval_gpu_page_batch(&all_keys, snapshot.matrix.as_ref())?;
    Ok(Json(BatchGpuPageQueryResponse {
        epoch_id: snapshot.epoch_id,
        results,
    }))
}

// WebSocket query message handlers (single + batch + session loop)
// extracted to `network::ws_query` (private) in TASK-54.16.
use super::ws_query::handle_ws_query;
#[cfg(test)]
use super::ws_query::{handle_ws_batch_query, handle_ws_single_query};

/// Maximum request body size: 64KB
/// Batch queries: up to 32 queries × 3 keys × ~55 hex chars ≈ 5KB
/// Using 64KB as a safe upper bound with room for batch queries
pub const MAX_REQUEST_BODY_SIZE: usize = 64 * 1024;

/// Maximum WebSocket message size (same limit as HTTP body)
pub const MAX_WS_MESSAGE_BYTES: usize = MAX_REQUEST_BODY_SIZE;

/// Maximum number of queued websocket text messages while one request is in-flight.
pub const MAX_WS_QUEUED_MESSAGES: usize = 32;

/// Maximum concurrent PIR scans (query + page query endpoints)
/// Limits CPU usage under load; additional requests get 503
pub const MAX_CONCURRENT_SCANS: usize = 32;

pub fn create_router(state: Arc<AppState>) -> Router {
    #[cfg(feature = "metrics")]
    {
        create_router_with_concurrency(state, MAX_CONCURRENT_SCANS, None)
    }
    #[cfg(not(feature = "metrics"))]
    {
        create_router_with_concurrency(state, MAX_CONCURRENT_SCANS)
    }
}

pub fn create_router_with_concurrency(
    state: Arc<AppState>,
    max_concurrent: usize,
    #[cfg(feature = "metrics")] metrics_handle: Option<
        metrics_exporter_prometheus::PrometheusHandle,
    >,
) -> Router {
    use tower::limit::ConcurrencyLimitLayer;

    let scan_routes = Router::new()
        .route("/query", post(query_handler))
        .route("/query/batch", post(batch_query_handler))
        .route("/query/page", post(page_query_handler))
        .route("/query/page/gpu", post(page_query_gpu_handler))
        .route("/query/page/gpu/batch", post(page_query_gpu_batch_handler))
        .layer(ConcurrencyLimitLayer::new(max_concurrent));

    let mut app = Router::new()
        .route("/health", get(health_handler))
        .route("/epoch", get(epoch_handler))
        .route("/admin/snapshot", post(admin_snapshot_handler))
        .merge(scan_routes)
        .route("/ws/epoch", get(ws_epoch_handler))
        .route("/ws/query", get(ws_query_handler))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_SIZE))
        .with_state(state);

    #[cfg(feature = "metrics")]
    if let Some(handle) = metrics_handle {
        app = app.route("/metrics", get(move || async move { handle.render() }));
    }

    app
}

#[cfg(test)]
mod tests;
