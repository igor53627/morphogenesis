//! Wire-format DTOs (data transfer objects) for the HTTP and WebSocket API.
//!
//! Extracted from `network/api.rs` in TASK-54.15. These are the
//! `Serialize`/`Deserialize` request and response shapes used by every
//! handler. The shared server state (`AppState`) stays in `api.rs` — it is
//! not a wire type and pulls in `EpochManager` / `GlobalState`.
//!
//! `pub mod` (not private like the other network submodules): the DTOs are
//! part of the crate's public API and are referenced from bench binaries as
//! `morphogen_server::network::api::<DTO>`. `api.rs` re-exports them via
//! `pub use super::dto::*;` so that path keeps resolving without changes
//! at any call site.
//!
//! `#[serde(with = "...")]` attributes resolve via the `use super::...`
//! statements below, mirroring how they resolved when the DTOs lived in
//! `api.rs`.

use serde::{Deserialize, Serialize};

#[cfg(feature = "verifiable-pir")]
use morphogen_core::sumcheck::SumCheckProof;

use super::serde_hex::{hex_bytes, hex_bytes_array, hex_bytes_vec};

#[derive(Clone, Serialize)]
pub struct EpochMetadata {
    pub epoch_id: u64,
    pub num_rows: usize,
    pub seeds: [u64; 3],
    pub block_number: u64,
    #[serde(with = "hex_bytes")]
    pub state_root: [u8; 32],
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
    #[serde(default)]
    pub seeds: Option<[u64; 3]>,
    #[serde(default)]
    pub block_number: Option<u64>,
    #[serde(default, deserialize_with = "hex_bytes::deserialize_option")]
    pub state_root: Option<[u8; 32]>,
}
