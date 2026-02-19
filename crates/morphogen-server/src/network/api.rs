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
#[cfg(any(feature = "cuda", test))]
const GPU_MICRO_BATCH_SIZE: usize = 2;
#[cfg(feature = "cuda")]
const GPU_STREAM_COUNT_ENV: &str = "MORPHOGEN_GPU_STREAMS";
#[cfg(any(feature = "cuda", test))]
const DEFAULT_GPU_STREAM_COUNT: usize = 1;
#[cfg(any(feature = "cuda", test))]
const MAX_GPU_STREAM_COUNT: usize = 8;

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
pub struct BatchGpuPageQueryRequest {
    pub queries: Vec<GpuPageQueryRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchGpuPageQueryResponse {
    pub epoch_id: u64,
    pub results: Vec<BatchGpuPageQueryResult>,
}

#[derive(Debug, Serialize)]
pub struct BatchGpuPageQueryResult {
    #[serde(with = "hex_bytes_vec")]
    pub pages: Vec<Vec<u8>>,
    #[cfg(feature = "verifiable-pir")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<SumCheckProof>,
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

fn apply_delta_entries_to_payloads<K: morphogen_dpf::DpfKey>(
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

fn scan_batch_results_from_snapshot<K: morphogen_dpf::DpfKey>(
    matrix: &morphogen_storage::ChunkedMatrix,
    all_keys: &[[K; 3]],
    entries: &[morphogen_core::DeltaEntry],
    row_size_bytes: usize,
) -> Result<Vec<BatchQueryResult>, StatusCode> {
    let mut results = Vec::with_capacity(all_keys.len());
    for keys in all_keys {
        let mut payloads = crate::scan::scan_main_matrix(matrix, keys, row_size_bytes);
        apply_delta_entries_to_payloads(&mut payloads, keys, entries)?;
        results.push(BatchQueryResult {
            payloads: payloads.to_vec(),
        });
    }
    Ok(results)
}

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

#[cfg(any(feature = "cuda", test))]
fn gpu_micro_batch_ranges(total_queries: usize) -> Vec<(usize, usize)> {
    if total_queries == 0 {
        return Vec::new();
    }
    let mut ranges =
        Vec::with_capacity((total_queries + GPU_MICRO_BATCH_SIZE - 1) / GPU_MICRO_BATCH_SIZE);
    let mut start = 0usize;
    while start < total_queries {
        let end = (start + GPU_MICRO_BATCH_SIZE).min(total_queries);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[cfg(any(feature = "cuda", test))]
fn parse_gpu_stream_count(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GPU_STREAM_COUNT)
        .clamp(DEFAULT_GPU_STREAM_COUNT, MAX_GPU_STREAM_COUNT)
}

#[cfg(any(feature = "cuda", test))]
fn ensure_gpu_result_count(expected: usize, actual: usize) -> Result<(), StatusCode> {
    if actual == expected {
        return Ok(());
    }
    Err(StatusCode::INTERNAL_SERVER_ERROR)
}

#[cfg(any(feature = "cuda", test))]
fn run_gpu_scan_branches_with<FMulti, FMicro>(
    all_keys: &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
    stream_count: usize,
    mut scan_multistream: FMulti,
    mut scan_micro_batch: FMicro,
) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>
where
    FMulti: FnMut(
        &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
        usize,
    ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>,
    FMicro: FnMut(
        &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
    ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>,
{
    let n = all_keys.len();
    let gpu_results = if stream_count > 1 {
        scan_multistream(all_keys, stream_count)?
    } else {
        let mut gpu_results = Vec::with_capacity(n);
        for (start, end) in gpu_micro_batch_ranges(n) {
            let key_batch = &all_keys[start..end];
            let mut chunk_results = scan_micro_batch(key_batch)?;
            ensure_gpu_result_count(key_batch.len(), chunk_results.len())?;
            gpu_results.append(&mut chunk_results);
        }
        gpu_results
    };
    ensure_gpu_result_count(n, gpu_results.len())?;
    Ok(gpu_results)
}

#[cfg(test)]
type TestGpuBatchMultistreamScan = std::sync::Arc<
    dyn Fn(
            &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
            usize,
        ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>
        + Send
        + Sync,
>;

#[cfg(test)]
type TestGpuBatchMicroScan = std::sync::Arc<
    dyn Fn(
            &[[morphogen_gpu_dpf::dpf::ChaChaKey; 3]],
        ) -> Result<Vec<morphogen_gpu_dpf::kernel::PirResult>, StatusCode>
        + Send
        + Sync,
>;

#[cfg(test)]
#[derive(Clone)]
struct TestGpuBatchHooks {
    stream_count: usize,
    multistream_scan: TestGpuBatchMultistreamScan,
    micro_batch_scan: TestGpuBatchMicroScan,
}

#[cfg(test)]
thread_local! {
    static TEST_GPU_BATCH_HOOKS: std::cell::RefCell<Option<TestGpuBatchHooks>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "cuda")]
fn configured_gpu_stream_count() -> usize {
    let raw = std::env::var(GPU_STREAM_COUNT_ENV).ok();
    parse_gpu_stream_count(raw.as_deref())
}

fn collect_gpu_page_refs<'a>(matrix: &'a morphogen_storage::ChunkedMatrix) -> Vec<&'a [u8]> {
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
        for keys in &all_keys {
            let (payloads, query_epoch) = crate::scan::scan_consistent(
                state.global.as_ref(),
                state.global.load_pending().as_ref(),
                keys,
                state.row_size_bytes,
            )
            .map_err(scan_error_to_status)?;
            match batch_epoch {
                None => batch_epoch = Some(query_epoch),
                Some(e) if e != query_epoch => {
                    return Err(StatusCode::SERVICE_UNAVAILABLE);
                }
                _ => {}
            }
            results.push(BatchQueryResult {
                payloads: payloads.to_vec(),
            });
        }
        let epoch_id = batch_epoch.unwrap_or_else(|| state.global.load().epoch_id);
        return Ok(Json(BatchQueryResponse { epoch_id, results }));
    }

    // All queries share the same consistent snapshot
    let results = scan_batch_results_from_snapshot(
        snapshot1.matrix.as_ref(),
        &all_keys,
        &entries,
        state.row_size_bytes,
    )?;

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

    let keys = parse_gpu_query_keys(&request)?;

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
                let delta_results = scan_delta_for_gpu(pending.as_ref(), &keys, PAGE_SIZE_BYTES)
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
            hooks.stream_count,
            |keys, count| (hooks.multistream_scan)(keys, count),
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
            let matrix_guard = matrix_mutex
                .lock()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            if let Some(matrix) = matrix_guard.as_ref() {
                let pending = state.global.load_pending();
                let pending_epoch = pending.pending_epoch();
                if pending_epoch != snapshot1.epoch_id {
                    drop(matrix_guard);
                    std::thread::yield_now();
                    continue;
                }

                let stream_count = configured_gpu_stream_count();
                let gpu_results = run_gpu_scan_branches_with(
                    &all_keys,
                    stream_count,
                    |keys, count| {
                        unsafe {
                            scanner
                                .scan_batch_single_query_multistream_optimized(matrix, keys, count)
                        }
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
                    },
                    |key_batch| {
                        unsafe { scanner.scan_batch_optimized(matrix, key_batch) }
                            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
                    },
                )?;
                let snapshot2 = state.global.load();
                let pending_epoch_after = pending.pending_epoch();
                if snapshot1.epoch_id != snapshot2.epoch_id
                    || pending_epoch_after != snapshot1.epoch_id
                {
                    drop(matrix_guard);
                    std::thread::yield_now();
                    continue;
                }

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
                        let v0 =
                            u128::from_le_bytes(gpu_result.verif0.try_into().unwrap_or([0; 16]));
                        let v1 =
                            u128::from_le_bytes(gpu_result.verif1.try_into().unwrap_or([0; 16]));
                        let v2 =
                            u128::from_le_bytes(gpu_result.verif2.try_into().unwrap_or([0; 16]));

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

                return Ok(Json(BatchGpuPageQueryResponse {
                    epoch_id: snapshot1.epoch_id,
                    results,
                }));
            }

            drop(matrix_guard);
            std::thread::yield_now();
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

const WS_INTERNAL_ERROR: &str = r#"{"error":"internal server error","code":"internal_error"}"#;

fn ws_error_json(error: &str, code: &str) -> String {
    serde_json::to_string(&WsQueryError {
        error: error.to_string(),
        code: code.to_string(),
    })
    .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string())
}

fn handle_ws_single_query(state: &AppState, request: QueryRequest) -> String {
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

fn handle_ws_batch_query(state: &AppState, request: BatchQueryRequest) -> String {
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
        // Fallback: per-query scan_consistent, tracking epoch consistency
        let mut results = Vec::with_capacity(n);
        let mut batch_epoch: Option<u64> = None;
        for keys in &all_keys {
            match scan_consistent(
                state.global.as_ref(),
                state.global.load_pending().as_ref(),
                keys,
                state.row_size_bytes,
            ) {
                Ok((payloads, query_epoch)) => {
                    match batch_epoch {
                        None => batch_epoch = Some(query_epoch),
                        Some(e) if e != query_epoch => {
                            return ws_error_json(
                                "epoch changed during batch scan",
                                "too_many_retries",
                            );
                        }
                        _ => {}
                    }
                    results.push(BatchQueryResult {
                        payloads: payloads.to_vec(),
                    });
                }
                Err(e) => {
                    use crate::scan::ScanError;
                    let code = match e {
                        ScanError::TooManyRetries { .. } => "too_many_retries",
                        ScanError::LockPoisoned
                        | ScanError::MatrixNotAligned { .. }
                        | ScanError::ChunkNotAligned { .. } => "internal_error",
                    };
                    return ws_error_json(&format!("scan error: {}", e), code);
                }
            }
        }
        let epoch_id = batch_epoch.unwrap_or_else(|| state.global.load().epoch_id);
        return serde_json::to_string(&BatchQueryResponse { epoch_id, results })
            .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string());
    }

    let mut results = Vec::with_capacity(n);
    for keys in &all_keys {
        let mut payloads =
            crate::scan::scan_main_matrix(snapshot1.matrix.as_ref(), keys, state.row_size_bytes);
        for entry in &entries {
            for (k, key) in keys.iter().enumerate() {
                if key.eval_bit(entry.row_idx) {
                    if entry.diff.len() != payloads[k].len() {
                        return ws_error_json("delta length mismatch", "internal_error");
                    }
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
    fn parse_gpu_query_keys_rejects_domain_bits_above_kernel_limit() {
        use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaParams};
        use morphogen_gpu_dpf::kernel::MAX_DOMAIN_BITS;

        let params = ChaChaParams::new(MAX_DOMAIN_BITS + 1).expect("valid params");
        let (k0, _) = generate_chacha_dpf_keys(&params, 0).expect("key generation should succeed");
        let key_bytes = k0.to_bytes().to_vec();
        let request = GpuPageQueryRequest {
            keys: vec![key_bytes.clone(), key_bytes.clone(), key_bytes],
        };

        assert!(matches!(
            parse_gpu_query_keys(&request),
            Err(StatusCode::BAD_REQUEST)
        ));
    }

    #[test]
    fn parse_gpu_query_keys_accepts_kernel_max_domain_bits() {
        use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaParams};
        use morphogen_gpu_dpf::kernel::MAX_DOMAIN_BITS;

        let params = ChaChaParams::new(MAX_DOMAIN_BITS).expect("valid params");
        let (k0, _) = generate_chacha_dpf_keys(&params, 0).expect("key generation should succeed");
        let key_bytes = k0.to_bytes().to_vec();
        let request = GpuPageQueryRequest {
            keys: vec![key_bytes.clone(), key_bytes.clone(), key_bytes],
        };

        assert!(parse_gpu_query_keys(&request).is_ok());
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

    #[test]
    fn batch_snapshot_scan_applies_pending_delta_for_all_queries() {
        use morphogen_dpf::AesDpfKey;

        let state = test_state();
        state
            .global
            .load_pending()
            .push(0, vec![0xAB; state.row_size_bytes])
            .expect("delta insert should succeed");

        let mut rng = rand::thread_rng();
        let mut all_keys: Vec<[AesDpfKey; 3]> = Vec::new();
        for target in [0, 1, 2, 3] {
            let (k0, _) = AesDpfKey::generate_pair(&mut rng, target);
            let (k1, _) = AesDpfKey::generate_pair(&mut rng, target + 1);
            let (k2, _) = AesDpfKey::generate_pair(&mut rng, target + 2);
            all_keys.push([k0, k1, k2]);
        }

        let snapshot = state.global.load();
        let (pending_epoch, entries) = state
            .global
            .load_pending()
            .snapshot_with_epoch()
            .expect("snapshot should succeed");
        assert_eq!(pending_epoch, snapshot.epoch_id);

        let batch_results = scan_batch_results_from_snapshot(
            snapshot.matrix.as_ref(),
            &all_keys,
            &entries,
            state.row_size_bytes,
        )
        .expect("batch path should succeed");

        let mut expected_payloads = Vec::with_capacity(all_keys.len());
        for keys in &all_keys {
            let mut payloads =
                crate::scan::scan_main_matrix(snapshot.matrix.as_ref(), keys, state.row_size_bytes);
            apply_delta_entries_to_payloads(&mut payloads, keys, &entries)
                .expect("delta application should succeed");
            expected_payloads.push(payloads.to_vec());
        }

        let actual_payloads: Vec<Vec<Vec<u8>>> =
            batch_results.into_iter().map(|r| r.payloads).collect();
        assert_eq!(actual_payloads, expected_payloads);
    }

    #[tokio::test]
    async fn batch_query_consistency_fallback_returns_service_unavailable_on_persistent_mismatch() {
        use morphogen_dpf::AesDpfKey;

        let state = test_state();
        state
            .global
            .load_pending()
            .drain_for_epoch(43)
            .expect("force pending epoch mismatch");

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
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn gpu_micro_batch_ranges_caps_each_batch_to_two() {
        assert_eq!(gpu_micro_batch_ranges(1), vec![(0, 1)]);
        assert_eq!(gpu_micro_batch_ranges(2), vec![(0, 2)]);
        assert_eq!(gpu_micro_batch_ranges(3), vec![(0, 2), (2, 3)]);
        assert_eq!(gpu_micro_batch_ranges(4), vec![(0, 2), (2, 4)]);
        assert_eq!(gpu_micro_batch_ranges(5), vec![(0, 2), (2, 4), (4, 5)]);
    }

    #[test]
    fn gpu_micro_batch_ranges_preserves_order_and_full_coverage() {
        let ranges = gpu_micro_batch_ranges(7);
        let mut covered = Vec::new();
        for (start, end) in ranges {
            assert!(end > start);
            assert!(end - start <= 2);
            covered.extend(start..end);
        }
        assert_eq!(covered, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn parse_gpu_stream_count_supports_required_values() {
        assert_eq!(parse_gpu_stream_count(Some("1")), 1);
        assert_eq!(parse_gpu_stream_count(Some("2")), 2);
        assert_eq!(parse_gpu_stream_count(Some("4")), 4);
        assert_eq!(parse_gpu_stream_count(Some("8")), 8);
    }

    #[test]
    fn parse_gpu_stream_count_defaults_and_clamps() {
        assert_eq!(parse_gpu_stream_count(None), 1);
        assert_eq!(parse_gpu_stream_count(Some("0")), 1);
        assert_eq!(parse_gpu_stream_count(Some("999")), 8);
        assert_eq!(parse_gpu_stream_count(Some("bad")), 1);
    }

    #[test]
    fn ensure_gpu_result_count_accepts_matching_lengths() {
        assert!(ensure_gpu_result_count(4, 4).is_ok());
    }

    #[test]
    fn ensure_gpu_result_count_rejects_mismatched_lengths() {
        assert_eq!(
            ensure_gpu_result_count(4, 3),
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        );
    }

    fn test_gpu_keys(count: usize) -> Vec<[morphogen_gpu_dpf::dpf::ChaChaKey; 3]> {
        use morphogen_gpu_dpf::dpf::{generate_chacha_dpf_keys, ChaChaParams};

        let params = ChaChaParams::new(8).expect("valid params");
        let mut keys = Vec::with_capacity(count);
        for i in 0..count {
            let (k0, _) = generate_chacha_dpf_keys(&params, i).expect("key generation should work");
            keys.push([k0.clone(), k0.clone(), k0]);
        }
        keys
    }

    fn test_pir_result() -> morphogen_gpu_dpf::kernel::PirResult {
        morphogen_gpu_dpf::kernel::PirResult {
            page0: Vec::new(),
            page1: Vec::new(),
            page2: Vec::new(),
            verif0: Vec::new(),
            verif1: Vec::new(),
            verif2: Vec::new(),
            timing: morphogen_gpu_dpf::kernel::KernelTiming::default(),
        }
    }

    fn test_pir_result_with_marker(marker: u8) -> morphogen_gpu_dpf::kernel::PirResult {
        morphogen_gpu_dpf::kernel::PirResult {
            page0: vec![marker],
            page1: vec![marker.wrapping_add(1)],
            page2: vec![marker.wrapping_add(2)],
            verif0: Vec::new(),
            verif1: Vec::new(),
            verif2: Vec::new(),
            timing: morphogen_gpu_dpf::kernel::KernelTiming::default(),
        }
    }

    fn test_gpu_batch_request(count: usize) -> BatchGpuPageQueryRequest {
        let all_keys = test_gpu_keys(count);
        let queries = all_keys
            .iter()
            .map(|keys| GpuPageQueryRequest {
                keys: vec![
                    keys[0].to_bytes().to_vec(),
                    keys[1].to_bytes().to_vec(),
                    keys[2].to_bytes().to_vec(),
                ],
            })
            .collect();
        BatchGpuPageQueryRequest { queries }
    }

    struct TestGpuHookResetGuard;

    impl Drop for TestGpuHookResetGuard {
        fn drop(&mut self) {
            TEST_GPU_BATCH_HOOKS.with(|slot| {
                slot.replace(None);
            });
        }
    }

    async fn with_test_gpu_batch_hooks<F, Fut, R>(hooks: TestGpuBatchHooks, f: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        TEST_GPU_BATCH_HOOKS.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none(), "nested test hooks are not supported");
            *slot = Some(hooks);
        });
        let _guard = TestGpuHookResetGuard;
        f().await
    }

    #[test]
    fn run_gpu_scan_branches_with_multistream_rejects_mismatched_lengths() {
        let all_keys = test_gpu_keys(2);
        let result = run_gpu_scan_branches_with(
            &all_keys,
            4,
            |_keys, _stream_count| Ok(vec![test_pir_result()]),
            |_key_batch| unreachable!("micro-batch path should not run"),
        );

        assert!(matches!(result, Err(StatusCode::INTERNAL_SERVER_ERROR)));
    }

    #[test]
    fn run_gpu_scan_branches_with_multistream_collects_full_result_set() {
        let all_keys = test_gpu_keys(3);
        let result = run_gpu_scan_branches_with(
            &all_keys,
            4,
            |keys, _stream_count| Ok(vec![test_pir_result(); keys.len()]),
            |_key_batch| unreachable!("micro-batch path should not run"),
        )
        .expect("multistream path should preserve result count");

        assert_eq!(result.len(), all_keys.len());
    }

    #[test]
    fn run_gpu_scan_branches_with_micro_batch_rejects_mismatched_chunk_lengths() {
        let all_keys = test_gpu_keys(3);
        let result = run_gpu_scan_branches_with(
            &all_keys,
            1,
            |_keys, _stream_count| unreachable!("multistream path should not run"),
            |_key_batch| Ok(vec![test_pir_result()]),
        );

        assert!(matches!(result, Err(StatusCode::INTERNAL_SERVER_ERROR)));
    }

    #[test]
    fn run_gpu_scan_branches_with_micro_batch_collects_full_result_set() {
        let all_keys = test_gpu_keys(3);
        let result = run_gpu_scan_branches_with(
            &all_keys,
            1,
            |_keys, _stream_count| unreachable!("multistream path should not run"),
            |key_batch| Ok(vec![test_pir_result(); key_batch.len()]),
        )
        .expect("micro-batch path should preserve result count");

        assert_eq!(result.len(), all_keys.len());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn page_query_gpu_batch_handler_returns_internal_error_on_multistream_mismatch() {
        let state = test_state();
        let request = test_gpu_batch_request(2);
        let hooks = TestGpuBatchHooks {
            stream_count: 4,
            multistream_scan: std::sync::Arc::new(|_keys, _stream_count| {
                Ok(vec![test_pir_result()])
            }),
            micro_batch_scan: std::sync::Arc::new(|_key_batch| {
                unreachable!("micro-batch path should not run")
            }),
        };

        let result = with_test_gpu_batch_hooks(hooks, move || {
            page_query_gpu_batch_handler(State(state), Json(request))
        })
        .await;

        assert!(matches!(result, Err(StatusCode::INTERNAL_SERVER_ERROR)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn page_query_gpu_batch_handler_returns_internal_error_on_micro_batch_mismatch() {
        let state = test_state();
        let request = test_gpu_batch_request(3);
        let hooks = TestGpuBatchHooks {
            stream_count: 1,
            multistream_scan: std::sync::Arc::new(|_keys, _stream_count| {
                unreachable!("multistream path should not run")
            }),
            micro_batch_scan: std::sync::Arc::new(|_key_batch| Ok(vec![test_pir_result()])),
        };

        let result = with_test_gpu_batch_hooks(hooks, move || {
            page_query_gpu_batch_handler(State(state), Json(request))
        })
        .await;

        assert!(matches!(result, Err(StatusCode::INTERNAL_SERVER_ERROR)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn page_query_gpu_batch_handler_maps_test_hook_results_in_order() {
        let state = test_state();
        let request = test_gpu_batch_request(2);
        let hooks = TestGpuBatchHooks {
            stream_count: 4,
            multistream_scan: std::sync::Arc::new(|keys, _stream_count| {
                Ok(keys
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| test_pir_result_with_marker((idx as u8) * 10))
                    .collect())
            }),
            micro_batch_scan: std::sync::Arc::new(|_key_batch| {
                unreachable!("micro-batch path should not run")
            }),
        };

        let response = with_test_gpu_batch_hooks(hooks, move || {
            page_query_gpu_batch_handler(State(state), Json(request))
        })
        .await
        .expect("test hook success path should return response")
        .0;

        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].pages[0], vec![0u8]);
        assert_eq!(response.results[0].pages[1], vec![1u8]);
        assert_eq!(response.results[0].pages[2], vec![2u8]);
        assert_eq!(response.results[1].pages[0], vec![10u8]);
        assert_eq!(response.results[1].pages[1], vec![11u8]);
        assert_eq!(response.results[1].pages[2], vec![12u8]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn with_test_gpu_batch_hooks_clears_state_after_panic() {
        let hooks = TestGpuBatchHooks {
            stream_count: 4,
            multistream_scan: std::sync::Arc::new(|_keys, _stream_count| Ok(Vec::new())),
            micro_batch_scan: std::sync::Arc::new(|_key_batch| Ok(Vec::new())),
        };

        let join = tokio::spawn(with_test_gpu_batch_hooks(hooks, || async {
            panic!("intentional panic for hook cleanup test");
        }))
        .await;
        assert!(join.is_err());
        assert!(TEST_GPU_BATCH_HOOKS.with(|slot| slot.borrow().is_none()));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn with_test_gpu_batch_hooks_nested_call_panics_without_leaking_state() {
        let hooks = TestGpuBatchHooks {
            stream_count: 4,
            multistream_scan: std::sync::Arc::new(|_keys, _stream_count| Ok(Vec::new())),
            micro_batch_scan: std::sync::Arc::new(|_key_batch| Ok(Vec::new())),
        };

        let join = tokio::spawn(with_test_gpu_batch_hooks(hooks.clone(), move || {
            let nested_hooks = hooks.clone();
            async move {
                let _ =
                    with_test_gpu_batch_hooks(nested_hooks, || async { Ok::<(), StatusCode>(()) })
                        .await;
            }
        }))
        .await;

        assert!(join.is_err());
        assert!(TEST_GPU_BATCH_HOOKS.with(|slot| slot.borrow().is_none()));
    }
}
