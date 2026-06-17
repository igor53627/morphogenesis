//! Snapshot source resolution, fetch, and decode helpers for the admin
//! snapshot-ingestion endpoint.
//!
//! Extracted from `network/api.rs` in TASK-54.14. These three helpers are
//! AppState-independent: they take primitives (URL strings, allowlists,
//! byte buffers, row/chunk sizes) and return [`morphogen_storage::ChunkedMatrix`]
//! or snapshot-source enums. The AppState-aware validator
//! (`validate_snapshot_page_pir_compatibility`) and `admin_snapshot_handler`
//! stay in `api.rs` — they need the full `AppState` shape (page config).
//!
//! Private to the `network` module — NOT part of the crate's public API.
//! Referenced from `network/api.rs` via `use super::snapshot_config::*;`.

use std::path::PathBuf;

use axum::http::StatusCode;
use tokio::io::AsyncReadExt;

use super::admin_auth::{is_allowed_snapshot_host, SnapshotSource};

pub(crate) fn parse_snapshot_source(
    raw: &str,
    allow_local_paths: bool,
    allowed_hosts: &[String],
) -> Result<SnapshotSource, StatusCode> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let url = reqwest::Url::parse(trimmed).map_err(|_| StatusCode::BAD_REQUEST)?;
        let host = url.host_str().ok_or(StatusCode::BAD_REQUEST)?;
        if allowed_hosts.is_empty() || !is_allowed_snapshot_host(host, allowed_hosts) {
            return Err(StatusCode::FORBIDDEN);
        }
        return Ok(SnapshotSource::Http(url));
    }

    if trimmed.starts_with("file://") {
        if !allow_local_paths {
            return Err(StatusCode::FORBIDDEN);
        }
        let url = reqwest::Url::parse(trimmed).map_err(|_| StatusCode::BAD_REQUEST)?;
        let path = url.to_file_path().map_err(|_| StatusCode::BAD_REQUEST)?;
        return Ok(SnapshotSource::Local(path));
    }

    if trimmed.contains("://") {
        return Err(StatusCode::BAD_REQUEST);
    }

    if !allow_local_paths {
        return Err(StatusCode::FORBIDDEN);
    }
    Ok(SnapshotSource::Local(PathBuf::from(trimmed)))
}

pub(crate) async fn fetch_snapshot_bytes(
    source: SnapshotSource,
    max_bytes: usize,
) -> Result<Vec<u8>, StatusCode> {
    if max_bytes == 0 {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    match source {
        SnapshotSource::Http(url) => {
            let client = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(std::time::Duration::from_secs(10))
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let mut response = client
                .get(url)
                .send()
                .await
                .map_err(|_| StatusCode::BAD_GATEWAY)?;
            if !response.status().is_success() {
                return Err(StatusCode::BAD_GATEWAY);
            }
            if let Some(content_length) = response.content_length() {
                if content_length == 0 {
                    return Err(StatusCode::BAD_REQUEST);
                }
                if content_length > max_bytes as u64 {
                    return Err(StatusCode::PAYLOAD_TOO_LARGE);
                }
            }

            let mut bytes = Vec::new();
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|_| StatusCode::BAD_GATEWAY)?
            {
                if bytes.len().saturating_add(chunk.len()) > max_bytes {
                    return Err(StatusCode::PAYLOAD_TOO_LARGE);
                }
                bytes.extend_from_slice(&chunk);
            }

            if bytes.is_empty() {
                return Err(StatusCode::BAD_REQUEST);
            }
            Ok(bytes)
        }
        SnapshotSource::Local(path) => {
            let mut file = tokio::fs::File::open(path)
                .await
                .map_err(|err| match err.kind() {
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidInput => {
                        StatusCode::BAD_REQUEST
                    }
                    std::io::ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                })?;

            let mut bytes = Vec::new();
            let mut chunk = vec![0u8; 64 * 1024];
            loop {
                let read = file
                    .read(&mut chunk)
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                if read == 0 {
                    break;
                }
                if bytes.len().saturating_add(read) > max_bytes {
                    return Err(StatusCode::PAYLOAD_TOO_LARGE);
                }
                bytes.extend_from_slice(&chunk[..read]);
            }

            if bytes.is_empty() {
                return Err(StatusCode::BAD_REQUEST);
            }
            Ok(bytes)
        }
    }
}

pub(crate) fn decode_snapshot_matrix(
    bytes: &[u8],
    row_size_bytes: usize,
    chunk_size_bytes: usize,
) -> Result<morphogen_storage::ChunkedMatrix, StatusCode> {
    if row_size_bytes == 0 || chunk_size_bytes == 0 {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    if bytes.is_empty() || !bytes.len().is_multiple_of(row_size_bytes) {
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(morphogen_storage::ChunkedMatrix::from_bytes(
        bytes,
        chunk_size_bytes,
    ))
}
