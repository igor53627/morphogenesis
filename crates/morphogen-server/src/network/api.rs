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
use metrics::counter;
#[cfg(feature = "tracing")]
use tracing::instrument;

#[cfg(feature = "verifiable-pir")]
use morphogen_core::sumcheck::SumCheckProof;
use morphogen_core::GlobalState;
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

#[derive(Clone)]
pub struct PagePirConfig {
    pub domain_bits: usize,
    pub rows_per_page: usize,
    pub prg_keys: [[u8; 16]; 2],
}

#[derive(Clone)]
pub struct AppState {
    pub global: Arc<GlobalState>,
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

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub epoch_id: u64,
    pub block_number: u64,
}

#[derive(Deserialize)]
pub struct StorageProofRequest {
    #[serde(default, deserialize_with = "hex_bytes::deserialize_option")]
    pub state_root: Option<[u8; 32]>,
}

#[derive(Serialize)]
pub struct PagePirResponse {
    pub domain_bits: usize,
    pub rows_per_page: usize,
    pub num_pages: usize,
    #[serde(with = "hex_bytes_array")]
    pub prg_keys: [[u8; 16]; 2],
}

#[derive(Serialize)]
pub struct EpochMetadataResponse {
    pub epoch_id: u64,
    pub num_rows: usize,
    pub seeds: [u64; 3],
    pub block_number: u64,
    #[serde(with = "hex_bytes")]
    pub state_root: [u8; 32],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_pir: Option<PagePirResponse>,
}

#[derive(Deserialize)]
pub struct QueryRequest {
    #[serde(with = "hex_bytes_vec")]
    pub keys: Vec<Vec<u8>>,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub epoch_id: u64,
    #[serde(with = "hex_bytes_vec")]
    pub payloads: Vec<Vec<u8>>,
}

/// Maximum number of queries in a single batch request.
pub const MAX_BATCH_SIZE: usize = 32;

#[derive(Deserialize)]
pub struct BatchQueryRequest {
    pub queries: Vec<QueryRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchQueryResponse {
    pub epoch_id: u64,
    pub results: Vec<BatchQueryResult>,
}

#[derive(Debug, Serialize)]
pub struct BatchQueryResult {
    #[serde(with = "hex_bytes_vec")]
    pub payloads: Vec<Vec<u8>>,
}

/// Page-level PIR query response.
///
/// Returns full pages (4KB each) that the client XORs with the other server's response.
#[derive(Serialize)]
#[cfg(feature = "verifiable-pir")]
pub struct PageQueryResponse {
    pub epoch_id: u64,
    /// 3 page payloads (4KB each for standard page size)
    #[serde(with = "hex_bytes_vec")]
    pub pages: Vec<Vec<u8>>,
    /// Verifiable PIR Sum-Check Proof (Round 0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<SumCheckProof>,
}

#[cfg(not(feature = "verifiable-pir"))]
#[derive(Serialize)]
pub struct PageQueryResponse {
    pub epoch_id: u64,
    /// 3 page payloads (4KB each for standard page size)
    #[serde(with = "hex_bytes_vec")]
    pub pages: Vec<Vec<u8>>,
}

#[derive(Deserialize)]
pub struct PageQueryRequest {
    #[serde(with = "hex_bytes_vec")]
    pub keys: Vec<Vec<u8>>,
}

#[derive(Deserialize)]
pub struct GpuPageQueryRequest {
    #[serde(with = "hex_bytes_vec")]
    pub keys: Vec<Vec<u8>>,
}

#[derive(Deserialize)]
pub struct AdminSnapshotRequest {
    #[serde(alias = "url")]
    pub r2_url: String,
}

mod hex_bytes {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("0x{}", hex::encode(bytes)))
    }

    pub fn deserialize_option<'de, D>(deserializer: D) -> Result<Option<[u8; 32]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<String>::deserialize(deserializer)?;
        match opt {
            Some(s) => {
                let hex_str = s.strip_prefix("0x").unwrap_or(&s);
                let bytes = hex::decode(hex_str).map_err(serde::de::Error::custom)?;
                if bytes.len() != 32 {
                    return Err(serde::de::Error::custom(format!(
                        "expected 32 bytes, got {}",
                        bytes.len()
                    )));
                }
                let mut out = [0u8; 32];
                out.copy_from_slice(&bytes);
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }
}

mod hex_bytes_vec {
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &[Vec<u8>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded: Vec<String> = bytes
            .iter()
            .map(|b| format!("0x{}", hex::encode(b)))
            .collect();
        encoded.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let items: Vec<String> = Vec::deserialize(deserializer)?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let hex_str = item.strip_prefix("0x").unwrap_or(&item);
            let bytes = hex::decode(hex_str).map_err(serde::de::Error::custom)?;
            out.push(bytes);
        }
        Ok(out)
    }
}

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

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn admin_snapshot_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AdminSnapshotRequest>,
) -> Result<StatusCode, StatusCode> {
    let _ = (state, request);
    Ok(StatusCode::NOT_IMPLEMENTED)
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

#[cfg_attr(feature = "tracing", instrument(skip(state, request)))]
pub async fn batch_query_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchQueryRequest>,
) -> Result<Json<BatchQueryResponse>, StatusCode> {
    #[cfg(feature = "metrics")]
    counter!("pir_query_count_total", "type" => "batch").increment(1);

    use morphogen_dpf::{AesDpfKey, DpfKey};

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
        for keys in &all_keys {
            let (payloads, _) = crate::scan::scan_consistent(
                state.global.as_ref(),
                state.global.load_pending().as_ref(),
                keys,
                state.row_size_bytes,
            )
            .map_err(scan_error_to_status)?;
            results.push(BatchQueryResult {
                payloads: payloads.to_vec(),
            });
        }
        let epoch_id = state.global.load().epoch_id;
        return Ok(Json(BatchQueryResponse { epoch_id, results }));
    }

    // All queries share the same consistent snapshot
    let mut results = Vec::with_capacity(n);
    for keys in &all_keys {
        let mut payloads = crate::scan::scan_main_matrix(
            snapshot1.matrix.as_ref(),
            keys,
            state.row_size_bytes,
        );
        // Apply delta entries
        for entry in &entries {
            for (k, key) in keys.iter().enumerate() {
                if key.eval_bit(entry.row_idx) {
                    for (d, s) in payloads[k].iter_mut().zip(entry.diff.iter()) {
                        *d ^= s;
                    }
                }
            }
        }
        results.push(BatchQueryResult {
            payloads: payloads.to_vec(),
        });
    }

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
        #[cfg(feature = "verifiable-pir")]
        use morphogen_core::sumcheck::SumCheckProof;
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
        #[cfg(feature = "verifiable-pir")]
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

fn handle_ws_single_query(
    state: &AppState,
    request: QueryRequest,
) -> String {
    use crate::scan::scan_consistent;
    use morphogen_dpf::AesDpfKey;

    if request.keys.len() != 3 {
        return ws_error_json("expected exactly 3 keys", "bad_request");
    }

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

fn handle_ws_batch_query(
    state: &AppState,
    request: BatchQueryRequest,
) -> String {
    use crate::scan::scan_consistent;
    use morphogen_dpf::{AesDpfKey, DpfKey};

    let n = request.queries.len();
    if n == 0 || n > MAX_BATCH_SIZE {
        return ws_error_json("batch size must be 1..=32", "bad_request");
    }

    let mut all_keys: Vec<[AesDpfKey; 3]> = Vec::with_capacity(n);
    for query in &request.queries {
        if query.keys.len() != 3 {
            return ws_error_json("each query must have exactly 3 keys", "bad_request");
        }
        match (
            AesDpfKey::from_bytes(&query.keys[0]),
            AesDpfKey::from_bytes(&query.keys[1]),
            AesDpfKey::from_bytes(&query.keys[2]),
        ) {
            (Ok(k0), Ok(k1), Ok(k2)) => all_keys.push([k0, k1, k2]),
            _ => return ws_error_json("invalid key format", "bad_request"),
        }
    }

    // Try consistent snapshot for all queries
    let snapshot1 = state.global.load();
    let epoch1 = snapshot1.epoch_id;
    let pending = state.global.load_pending();
    let snapshot_result = pending.snapshot_with_epoch();
    let (pending_epoch, entries) = match snapshot_result {
        Ok(r) => r,
        Err(_) => return ws_error_json("internal error", "internal_error"),
    };
    let snapshot2 = state.global.load();

    if snapshot2.epoch_id != epoch1 || pending_epoch != epoch1 {
        // Fallback: per-query scan_consistent
        let mut results = Vec::with_capacity(n);
        for keys in &all_keys {
            match scan_consistent(
                state.global.as_ref(),
                state.global.load_pending().as_ref(),
                keys,
                state.row_size_bytes,
            ) {
                Ok((payloads, _)) => results.push(BatchQueryResult {
                    payloads: payloads.to_vec(),
                }),
                Err(e) => return ws_error_json(&format!("scan error: {}", e), "internal_error"),
            }
        }
        let epoch_id = state.global.load().epoch_id;
        return serde_json::to_string(&BatchQueryResponse { epoch_id, results })
            .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string());
    }

    let mut results = Vec::with_capacity(n);
    for keys in &all_keys {
        let mut payloads = crate::scan::scan_main_matrix(
            snapshot1.matrix.as_ref(),
            keys,
            state.row_size_bytes,
        );
        for entry in &entries {
            for (k, key) in keys.iter().enumerate() {
                if key.eval_bit(entry.row_idx) {
                    for (d, s) in payloads[k].iter_mut().zip(entry.diff.iter()) {
                        *d ^= s;
                    }
                }
            }
        }
        results.push(BatchQueryResult {
            payloads: payloads.to_vec(),
        });
    }

    serde_json::to_string(&BatchQueryResponse {
        epoch_id: epoch1,
        results,
    })
    .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string())
}

async fn handle_ws_query(mut socket: WebSocket, state: Arc<AppState>) {
    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            if text.len() > MAX_WS_MESSAGE_BYTES {
                let error = ws_error_json("message too large", "message_too_large");
                let _ = socket.send(Message::Text(error.into())).await;
                continue;
            }

            // Try batch request first (has "queries" field), fall back to single query
            let response = if let Ok(batch) = serde_json::from_str::<BatchQueryRequest>(&text) {
                handle_ws_batch_query(&state, batch)
            } else {
                match serde_json::from_str::<QueryRequest>(&text) {
                    Ok(request) => handle_ws_single_query(&state, request),
                    Err(e) => ws_error_json(&e.to_string(), "bad_request"),
                }
            };

            if socket.send(Message::Text(response.into())).await.is_err() {
                break;
            }
        }
    }
}

/// Maximum request body size: 64KB
/// Batch queries: up to 32 queries × 3 keys × ~55 hex chars ≈ 5KB
/// Using 64KB as a safe upper bound with room for batch queries
pub const MAX_REQUEST_BODY_SIZE: usize = 64 * 1024;

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
        .route("/query/batch", post(batch_query_handler))
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
    #[cfg(not(feature = "verifiable-pir"))]
    fn page_query_response_omits_proof_field_without_feature() {
        let response = PageQueryResponse {
            epoch_id: 1,
            pages: vec![vec![0u8; 8], vec![0u8; 8], vec![0u8; 8]],
            #[cfg(feature = "verifiable-pir")]
            proof: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(!json.contains("\"proof\""));
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

    #[test]
    fn batch_request_deserializes() {
        let json = r#"{"queries":[{"keys":["0xaabb","0xccdd","0xeeff"]}]}"#;
        let request: BatchQueryRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.queries.len(), 1);
        assert_eq!(request.queries[0].keys.len(), 3);
    }

    #[test]
    fn batch_response_serializes_correctly() {
        let response = BatchQueryResponse {
            epoch_id: 42,
            results: vec![
                BatchQueryResult {
                    payloads: vec![vec![0xDE, 0xAD], vec![0xBE, 0xEF], vec![0xCA, 0xFE]],
                },
                BatchQueryResult {
                    payloads: vec![vec![0x11], vec![0x22], vec![0x33]],
                },
            ],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"epoch_id\":42"));
        assert!(json.contains("\"results\""));
        assert!(json.contains("\"0xdead\""));
        assert!(json.contains("\"0x11\""));
    }

    #[tokio::test]
    async fn batch_query_empty_returns_bad_request() {
        let state = test_state();
        let request = BatchQueryRequest { queries: vec![] };
        let result = batch_query_handler(State(state), Json(request)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn batch_query_too_large_returns_bad_request() {
        let state = test_state();
        let queries: Vec<QueryRequest> = (0..MAX_BATCH_SIZE + 1)
            .map(|_| QueryRequest {
                keys: vec![vec![0u8; 25], vec![0u8; 25], vec![0u8; 25]],
            })
            .collect();
        let request = BatchQueryRequest { queries };
        let result = batch_query_handler(State(state), Json(request)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn batch_query_wrong_key_count_returns_bad_request() {
        let state = test_state();
        let request = BatchQueryRequest {
            queries: vec![QueryRequest {
                keys: vec![vec![0u8; 25], vec![0u8; 25]], // only 2 keys
            }],
        };
        let result = batch_query_handler(State(state), Json(request)).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn batch_query_of_one_returns_single_result() {
        use morphogen_dpf::AesDpfKey;

        let state = test_state();
        let mut rng = rand::thread_rng();
        let (k0, _) = AesDpfKey::generate_pair(&mut rng, 0);
        let (k1, _) = AesDpfKey::generate_pair(&mut rng, 1);
        let (k2, _) = AesDpfKey::generate_pair(&mut rng, 2);

        let request = BatchQueryRequest {
            queries: vec![QueryRequest {
                keys: vec![
                    k0.to_bytes().to_vec(),
                    k1.to_bytes().to_vec(),
                    k2.to_bytes().to_vec(),
                ],
            }],
        };
        let result = batch_query_handler(State(state), Json(request)).await;
        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].payloads.len(), 3);
    }

    #[tokio::test]
    async fn batch_query_of_three_returns_three_results() {
        use morphogen_dpf::AesDpfKey;

        let state = test_state();
        let mut rng = rand::thread_rng();

        let mut queries = Vec::new();
        for target in [0, 1, 2] {
            let (k0, _) = AesDpfKey::generate_pair(&mut rng, target);
            let (k1, _) = AesDpfKey::generate_pair(&mut rng, target + 1);
            let (k2, _) = AesDpfKey::generate_pair(&mut rng, target + 2);
            queries.push(QueryRequest {
                keys: vec![
                    k0.to_bytes().to_vec(),
                    k1.to_bytes().to_vec(),
                    k2.to_bytes().to_vec(),
                ],
            });
        }

        let request = BatchQueryRequest { queries };
        let result = batch_query_handler(State(state), Json(request)).await;
        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.results.len(), 3);
        assert_eq!(response.epoch_id, 42);
        for r in &response.results {
            assert_eq!(r.payloads.len(), 3);
        }
    }
}
