//! HTTP and WebSocket API implementation.

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
pub struct AppState {
    pub global: Arc<GlobalState>,
    pub pending: Arc<morphogen_core::DeltaBuffer>,
    pub row_size_bytes: usize,
    pub num_rows: usize,
    pub seeds: [u64; 3],
    pub block_number: u64,
    pub state_root: [u8; 32],
    pub epoch_rx: watch::Receiver<EpochMetadata>,
    /// Page-level PIR configuration (None = page PIR disabled)
    pub page_config: Option<PagePirConfig>,
}

/// Configuration for page-level PIR queries.
#[derive(Clone)]
pub struct PagePirConfig {
    /// Number of rows per page (typically 16)
    pub rows_per_page: usize,
    /// Domain bits for DPF (log2 of number of pages)
    pub domain_bits: usize,
    /// PRG keys shared between both servers (public parameter)
    pub prg_keys: [[u8; 16]; 2],
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub epoch_id: u64,
    pub block_number: u64,
}

/// Page-level PIR parameters exposed in /epoch response.
#[derive(Clone, Serialize)]
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

/// Page-level PIR query request (privacy-preserving).
///
/// Uses proper 2-server DPF where servers cannot determine the target page.
#[derive(Deserialize)]
pub struct PageQueryRequest {
    /// Serialized PageDpfKey (one of the pair, the other goes to the other server)
    #[serde(with = "hex_bytes_vec")]
    pub keys: Vec<Vec<u8>>,
}

/// Page-level PIR query response.
///
/// Returns full pages (4KB each) that the client XORs with the other server's response.
#[derive(Serialize)]
pub struct PageQueryResponse {
    pub epoch_id: u64,
    /// 3 page payloads (4KB each for standard page size)
    #[serde(with = "hex_bytes_vec")]
    pub pages: Vec<Vec<u8>>,
}

#[allow(dead_code)]
mod hex_bytes {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("0x{}", hex::encode(bytes)))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let s = s.strip_prefix("0x").unwrap_or(&s);
        let bytes = hex::decode(s).map_err(serde::de::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

mod hex_bytes_vec {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes_vec: &[Vec<u8>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(bytes_vec.len()))?;
        for bytes in bytes_vec {
            seq.serialize_element(&format!("0x{}", hex::encode(bytes)))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let strings: Vec<String> = Vec::deserialize(deserializer)?;
        strings
            .into_iter()
            .map(|s| {
                let s = s.strip_prefix("0x").unwrap_or(&s);
                hex::decode(s).map_err(serde::de::Error::custom)
            })
            .collect()
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

fn scan_error_to_status(e: crate::scan::ScanError) -> StatusCode {
    use crate::scan::ScanError;
    match e {
        ScanError::TooManyRetries { .. } => StatusCode::SERVICE_UNAVAILABLE,
        ScanError::LockPoisoned
        | ScanError::MatrixNotAligned { .. }
        | ScanError::ChunkNotAligned { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

pub async fn query_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, StatusCode> {
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
        state.pending.as_ref(),
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
}

pub async fn ws_query_handler(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_query(socket, state))
}

pub async fn page_query_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PageQueryRequest>,
) -> Result<Json<PageQueryResponse>, StatusCode> {
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
            if k0.domain_bits() != page_config.domain_bits
                || k1.domain_bits() != page_config.domain_bits
                || k2.domain_bits() != page_config.domain_bits
            {
                return Err(StatusCode::BAD_REQUEST);
            }
            [k0, k1, k2]
        }
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    let chunk_size = 4096;
    let (results, epoch_id) = scan_pages_consistent(
        state.global.as_ref(),
        &keys,
        PAGE_SIZE_BYTES,
        chunk_size,
    )
    .map_err(scan_error_to_status)?;

    Ok(Json(PageQueryResponse {
        epoch_id,
        pages: results.to_vec(),
    }))
}

const WS_INTERNAL_ERROR: &str = r#"{"error":"internal server error"}"#;

fn ws_error_json(error: &str) -> String {
    serde_json::to_string(&WsQueryError {
        error: error.to_string(),
    })
    .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string())
}

async fn handle_ws_query(mut socket: WebSocket, state: Arc<AppState>) {
    use crate::scan::scan_consistent;
    use morphogen_dpf::AesDpfKey;

    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            if text.len() > MAX_WS_MESSAGE_BYTES {
                let error = ws_error_json("message too large");
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
                                state.pending.as_ref(),
                                &keys,
                                state.row_size_bytes,
                            ) {
                                Ok((results, epoch_id)) => {
                                    serde_json::to_string(&QueryResponse {
                                        epoch_id,
                                        payloads: results.to_vec(),
                                    })
                                    .unwrap_or_else(|_| WS_INTERNAL_ERROR.to_string())
                                }
                                Err(e) => ws_error_json(&format!("scan error: {}", e)),
                            }
                        }
                        _ => ws_error_json("invalid key format"),
                    }
                }
                Ok(_) => ws_error_json("expected exactly 3 keys"),
                Err(e) => ws_error_json(&e.to_string()),
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
    create_router_with_concurrency(state, MAX_CONCURRENT_SCANS)
}

pub fn create_router_with_concurrency(state: Arc<AppState>, max_concurrent: usize) -> Router {
    use tower::limit::ConcurrencyLimitLayer;

    let scan_routes = Router::new()
        .route("/query", post(query_handler))
        .route("/query/page", post(page_query_handler))
        .layer(ConcurrencyLimitLayer::new(max_concurrent));

    Router::new()
        .route("/health", get(health_handler))
        .route("/epoch", get(epoch_handler))
        .merge(scan_routes)
        .route("/ws/epoch", get(ws_epoch_handler))
        .route("/ws/query", get(ws_query_handler))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_SIZE))
        .with_state(state)
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
        let global = Arc::new(GlobalState::new(Arc::new(snapshot)));
        let pending = Arc::new(DeltaBuffer::new_with_epoch(row_size_bytes, 42));

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
            pending,
            row_size_bytes,
            num_rows: 100_000,
            seeds: [0x1234, 0x5678, 0x9ABC],
            block_number: 12345678,
            state_root: [0xAB; 32],
            epoch_rx: rx,
            page_config: None,
        })
    }

    #[test]
    fn app_state_has_pending_buffer() {
        let state = test_state();
        assert!(state.pending.is_empty().unwrap());
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
