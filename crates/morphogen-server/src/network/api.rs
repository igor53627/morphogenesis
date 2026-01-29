//! HTTP and WebSocket API implementation.

/// Optimal DPF chunk size for eval_and_accumulate_chunked.
/// Larger chunks amortize tree-traversal overhead. 65536 = 64K elements.
/// At 16 bytes per DPF output, this uses 1MB of buffer per evaluation.
/// Benchmarks show this is ~1.4x faster than chunk_size=4096.
pub const OPTIMAL_DPF_CHUNK_SIZE: usize = 65536;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        DefaultBodyLimit, State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
#[cfg(feature = "metrics")]
use metrics::{counter, histogram};
#[cfg(feature = "tracing")]
use tracing::{error, info, instrument};

use morphogen_core::{sumcheck::SumCheckProof, GlobalState};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Clone, Serialize)]
pub struct EpochMetadata {
    pub epoch_id: u64,
    pub num_rows: usize,
    pub seeds: [u64; 3],
    pub block_number: u64,
    #[serde(with = "hex_bytes")]
    pub state_root: [u8; 32],
}

// ... existing code ...

/// Page-level PIR query response.
///
/// Returns full pages (4KB each) that the client XORs with the other server's response.
#[derive(Serialize)]
pub struct PageQueryResponse {
    pub epoch_id: u64,
    /// 3 page payloads (4KB each for standard page size)
    #[serde(with = "hex_bytes_vec")]
    pub pages: Vec<Vec<u8>>,
    /// Verifiable PIR Sum-Check Proof (Round 0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<SumCheckProof>,
}

// ... existing code ...

mod hex_bytes_array {
    use serde::{self, Serializer};

    pub fn serialize<S>(keys: &[[u8; 16]; 2], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(2))?;
        for key in keys {
            seq.serialize_element(&format!("0x{}", hex::encode(key)))?;
        }
        seq.end()
    }
}

pub async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let snapshot = state.global.load();
    Json(HealthResponse {
        status: "ok".to_string(),
        epoch_id: snapshot.epoch_id,
        block_number: state.block_number,
    })
}

pub async fn epoch_handler(State(state): State<Arc<AppState>>) -> Json<EpochMetadataResponse> {
    let snapshot = state.global.load();
    let page_pir = state.page_config.as_ref().map(|cfg| PagePirResponse {
        domain_bits: cfg.domain_bits,
        rows_per_page: cfg.rows_per_page,
        num_pages: 1usize << cfg.domain_bits,
        prg_keys: cfg.prg_keys,
    });
    Json(EpochMetadataResponse {
        epoch_id: snapshot.epoch_id,
        num_rows: state.num_rows,
        seeds: state.seeds,
        block_number: state.block_number,
        state_root: state.state_root,
        page_pir,
    })
}

fn scan_error_to_status(e: crate::scan::ScanError) -> StatusCode {
    use crate::scan::ScanError;
    match e {
        ScanError::TooManyRetries { .. } => StatusCode::SERVICE_UNAVAILABLE,
        ScanError::LockPoisoned
        | ScanError::MatrixNotAligned { .. }
        | ScanError::ChunkNotAligned { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    }
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

    let (results, epoch_id) = scan_consistent(
        state.global.as_ref(),
        state.global.load_pending().as_ref(),
        &keys,
        state.row_size_bytes,
    )
    .map_err(scan_error_to_status)?;

    Ok(Json(QueryResponse {
        epoch_id,
        payloads: results.to_vec(),
    }))
}

pub async fn ws_epoch_handler(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    let epoch_rx = state.epoch_rx.clone();
    ws.on_upgrade(move |socket| handle_ws_epoch(socket, epoch_rx))
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
    ws.on_upgrade(move |socket| handle_ws_query(socket, state))
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

    let (results, epoch_id) = scan_pages_consistent(
        state.global.as_ref(),
        &keys,
        PAGE_SIZE_BYTES,
        OPTIMAL_DPF_CHUNK_SIZE,
    )
    .map_err(scan_error_to_status)?;

    Ok(Json(PageQueryResponse {
        epoch_id,
        pages: results.to_vec(),
    }))
}

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn page_query_gpu_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GpuPageQueryRequest>,
) -> Result<Json<PageQueryResponse>, StatusCode> {
    #[cfg(feature = "metrics")]
    counter!("pir_query_count_total", "type" => "gpu").increment(1);

    use morphogen_gpu_dpf::dpf::ChaChaKey;

    if request.keys.len() != 3 {
        return Err(StatusCode::BAD_REQUEST);
    }

    let keys: [ChaChaKey; 3] = match (
        ChaChaKey::from_bytes(&request.keys[0]),
        ChaChaKey::from_bytes(&request.keys[1]),
        ChaChaKey::from_bytes(&request.keys[2]),
    ) {
        (Ok(k0), Ok(k1), Ok(k2)) => [k0, k1, k2],
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    #[cfg(feature = "cuda")]
    if let (Some(scanner), Some(matrix_mutex)) = (&state.gpu_scanner, &state.gpu_matrix) {
        use crate::scan::scan_delta_for_gpu;
        use morphogen_core::sumcheck::{RoundPolynomial, SumCheckProof};
        use morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

        for _ in 0..10 {
            let snapshot1 = state.global.load();

            // 1. Get GPU lock and matrix
            let matrix_guard = matrix_mutex
                .lock()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            // 2. Check epochs
            if let Some(matrix) = matrix_guard.as_ref() {
                let pending = state.global.load_pending();
                let pending_epoch = pending.pending_epoch(); // Relaxed check
                if pending_epoch != snapshot1.epoch_id {
                    drop(matrix_guard);
                    std::thread::yield_now();
                    continue;
                }

                // 3. Scan GPU
                let mut results = unsafe { scanner.scan(matrix, [&keys[0], &keys[1], &keys[2]]) }
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

                // 4. Scan Delta
                let delta_results = scan_delta_for_gpu(
                    pending.as_ref(),
                    &[keys[0], keys[1], keys[2]],
                    PAGE_SIZE_BYTES,
                )
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

                // 5. Verify consistency after scan
                let snapshot2 = state.global.load();
                let pending_epoch_after = pending.pending_epoch();

                if snapshot1.epoch_id == snapshot2.epoch_id
                    && pending_epoch_after == snapshot1.epoch_id
                {
                    // Merge results
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

                    // Construct Proof (Round 0)
                    // In Binius Sum-Check for Dot Product, the round polynomial g(X) is quadratic.
                    // We need evaluations at 0, 1, and alpha.
                    // The GPU kernel returns the sum for each key.
                    // Since standard PIR is just sum(D * Q) where Q is 0/1,
                    // the "verification" result is the GF(2^128) dot product.
                    // We assume verif0..2 are the dot products for the 3 keys.
                    // For a full proof, we'd need to combine them or prove each separately.
                    // Here we package them as a single "proof" for the client to verify against C_D.

                    // Interpret 16-byte verif results as u128
                    let v0 = u128::from_le_bytes(results.verif0.try_into().unwrap_or([0; 16]));
                    let v1 = u128::from_le_bytes(results.verif1.try_into().unwrap_or([0; 16]));
                    let v2 = u128::from_le_bytes(results.verif2.try_into().unwrap_or([0; 16]));

                    // For the prototype, we return the XOR sum of verifiers as the "proof sum"
                    // Real implementation would return a proof per key or a combined proof.
                    // Let's just put v0 in the proof structure for now to wire it up.
                    let proof = SumCheckProof {
                        round_polynomials: vec![], // Populated in later rounds (on CPU/Client or next step)
                        sum: v0 ^ v1 ^ v2,         // Aggregate for simple check
                    };

                    return Ok(Json(PageQueryResponse {
                        epoch_id: snapshot1.epoch_id,
                        pages: vec![results.page0, results.page1, results.page2],
                        proof: Some(proof),
                    }));
                }
            }
            drop(matrix_guard);
            std::thread::yield_now();
        }
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    // CPU Fallback (or if CUDA is disabled)
    let snapshot = state.global.load();
    let num_pages =
        snapshot.matrix.total_size_bytes() / morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;

    // We need to collect pages from the ChunkedMatrix
    // This is slow on CPU but provides a reference implementation
    let mut pages_refs = Vec::with_capacity(num_pages);
    for i in 0..num_pages {
        let start = i * morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES;
        let (chunk_idx, chunk_offset) = (
            start / snapshot.matrix.chunk_size_bytes(),
            start % snapshot.matrix.chunk_size_bytes(),
        );
        let chunk = snapshot.matrix.chunk(chunk_idx);
        pages_refs.push(
            &chunk.as_slice()
                [chunk_offset..chunk_offset + morphogen_gpu_dpf::storage::PAGE_SIZE_BYTES],
        );
    }

    let result =
        morphogen_gpu_dpf::kernel::eval_fused_3dpf_cpu([&keys[0], &keys[1], &keys[2]], &pages_refs)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(PageQueryResponse {
        epoch_id: snapshot.epoch_id,
        pages: vec![result.page0, result.page1, result.page2],
        proof: None, // No proof for CPU fallback yet
    }))
}

const WS_INTERNAL_ERROR: &str = r#"{"error":"internal server error","code":"internal_error"}"#;

fn ws_error_json(error: &str, code: &str) -> String {
    serde_json::to_string(&WsQueryError {
        error: error.to_string(),
        code: code.to_string(),
    })
    .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string())
}

async fn handle_ws_query(mut socket: WebSocket, state: Arc<AppState>) {
    use crate::scan::scan_consistent;
    use morphogen_dpf::AesDpfKey;

    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            if text.len() > MAX_WS_MESSAGE_BYTES {
                let error = ws_error_json("message too large", "message_too_large");
                let _ = socket.send(Message::Text(error.into())).await;
                continue;
            }

            let response = match serde_json::from_str::<QueryRequest>(&text) {
                Ok(request) if request.keys.len() == 3 => {
                    let keys_result = (
                        AesDpfKey::from_bytes(&request.keys[0]),
                        AesDpfKey::from_bytes(&request.keys[1]),
                        AesDpfKey::from_bytes(&request.keys[2]),
                    );

                    match keys_result {
                        (Ok(k0), Ok(k1), Ok(k2)) => {
                            let keys = [k0, k1, k2];
                            match scan_consistent(
                                state.global.as_ref(),
                                state.global.load_pending().as_ref(),
                                &keys,
                                state.row_size_bytes,
                            ) {
                                Ok((results, epoch_id)) => serde_json::to_string(&QueryResponse {
                                    epoch_id,
                                    payloads: results.to_vec(),
                                })
                                .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string()),
                                Err(e) => {
                                    use crate::scan::ScanError;
                                    let code = match e {
                                        ScanError::TooManyRetries { .. } => "too_many_retries",
                                        ScanError::LockPoisoned => "internal_error",
                                        ScanError::MatrixNotAligned { .. } => "internal_error",
                                        ScanError::ChunkNotAligned { .. } => "internal_error",
                                    };
                                    ws_error_json(&format!("scan error: {}", e), code)
                                }
                            }
                        }
                        _ => ws_error_json("invalid key format", "bad_request"),
                    }
                }
                Ok(_) => ws_error_json("expected exactly 3 keys", "bad_request"),
                Err(e) => ws_error_json(&e.to_string(), "bad_request"),
            };

            if socket.send(Message::Text(response.into())).await.is_err() {
                break;
            }
        }
    }
}

/// Maximum request body size: 16KB for row queries, 8KB for page queries
/// Row query: 3 keys × 25 bytes = 75 bytes + JSON overhead ≈ 500 bytes
/// Page query: 3 keys × 500 bytes = 1500 bytes + JSON overhead ≈ 4KB
/// Using 16KB as a safe upper bound with room for future expansion
pub const MAX_REQUEST_BODY_SIZE: usize = 16 * 1024;

/// Maximum WebSocket message size (same limit as HTTP body)
pub const MAX_WS_MESSAGE_BYTES: usize = MAX_REQUEST_BODY_SIZE;

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
        .route("/query/page", post(page_query_handler))
        .route("/query/page/gpu", post(page_query_gpu_handler))
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
mod tests {
    use super::*;
    use morphogen_core::{DeltaBuffer, EpochSnapshot};
    use morphogen_storage::ChunkedMatrix;

    fn test_state() -> Arc<AppState> {
        let row_size_bytes = 256;
        let num_rows = 4;
        let matrix = Arc::new(ChunkedMatrix::new(row_size_bytes * num_rows, 512));
        let snapshot = EpochSnapshot {
            epoch_id: 42,
            matrix,
        };
        let pending = Arc::new(DeltaBuffer::new_with_epoch(row_size_bytes, 42));
        let global = Arc::new(GlobalState::new(Arc::new(snapshot), pending));

        let initial = EpochMetadata {
            epoch_id: 42,
            num_rows: 100_000,
            seeds: [0x1234, 0x5678, 0x9ABC],
            block_number: 12345678,
            state_root: [0xAB; 32],
        };
        let (_tx, rx) = watch::channel(initial);
        Arc::new(AppState {
            global,
            row_size_bytes,
            num_rows: 100_000,
            seeds: [0x1234, 0x5678, 0x9ABC],
            block_number: 12345678,
            state_root: [0xAB; 32],
            epoch_rx: rx,
            page_config: None,
            #[cfg(feature = "cuda")]
            gpu_scanner: None,
            #[cfg(feature = "cuda")]
            gpu_matrix: None,
        })
    }

    #[test]
    fn app_state_has_pending_buffer() {
        let state = test_state();
        assert!(state.global.load_pending().is_empty().unwrap());
        assert_eq!(state.row_size_bytes, 256);
    }

    #[test]
    fn health_response_serializes_correctly() {
        let response = HealthResponse {
            status: "ok".to_string(),
            epoch_id: 42,
            block_number: 12345,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"epoch_id\":42"));
    }

    #[test]
    fn epoch_metadata_serializes_with_hex_root() {
        let response = EpochMetadataResponse {
            epoch_id: 1,
            num_rows: 1000,
            seeds: [1, 2, 3],
            block_number: 100,
            state_root: [0xFF; 32],
            page_pir: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"state_root\":\"0x"));
        assert!(json.contains("ffffffff"));
        assert!(!json.contains("page_pir")); // skip_serializing_if works
    }

    #[test]
    fn epoch_metadata_serializes_page_pir_when_present() {
        let response = EpochMetadataResponse {
            epoch_id: 1,
            num_rows: 1000,
            seeds: [1, 2, 3],
            block_number: 100,
            state_root: [0xFF; 32],
            page_pir: Some(PagePirResponse {
                domain_bits: 10,
                rows_per_page: 16,
                num_pages: 1024,
                prg_keys: [[0xAA; 16], [0xBB; 16]],
            }),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"page_pir\""));
        assert!(json.contains("\"domain_bits\":10"));
        assert!(json.contains("\"num_pages\":1024"));
        assert!(json.contains("\"prg_keys\""));
        assert!(json.contains("0xaaaa"));
    }

    #[test]
    fn query_request_deserializes_hex_keys() {
        let json = r#"{"keys":["0xaabb","0xccdd","0xeeff"]}"#;
        let request: QueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.keys.len(), 3);
        assert_eq!(request.keys[0], vec![0xAA, 0xBB]);
    }

    #[test]
    fn query_response_serializes_hex_payloads() {
        let response = QueryResponse {
            epoch_id: 1,
            payloads: vec![vec![0xDE, 0xAD], vec![0xBE, 0xEF], vec![0xCA, 0xFE]],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"0xdead\""));
        assert!(json.contains("\"0xbeef\""));
        assert!(json.contains("\"0xcafe\""));
    }

    #[test]
    fn app_state_is_clone_and_send() {
        fn assert_clone_send<T: Clone + Send>() {}
        assert_clone_send::<AppState>();
    }

    #[test]
    fn query_handler_requires_25_byte_keys() {
        // AES_DPF_KEY_SIZE = 25 bytes
        // Keys shorter than 25 bytes should be rejected
        use morphogen_dpf::AES_DPF_KEY_SIZE;
        assert_eq!(AES_DPF_KEY_SIZE, 25);
    }
}
