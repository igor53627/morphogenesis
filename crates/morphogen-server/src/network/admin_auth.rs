//! Admin-snapshot auth primitives and snapshot-source parsing.
//!
//! Extracted from `network/api.rs` in TASK-54.11. Pure auth/parse helpers
//! that do NOT depend on `AppState`:
//!
//! - bearer / legacy-admin token extraction from request headers
//! - constant-time token comparison (no early exit on mismatch)
//! - `SnapshotSource` enum + URL/host allowlist parsing
//!
//! The `AppState`-aware authorizer (`authorize_admin_snapshot`), the snapshot
//! fetcher/decoder, and the `admin_snapshot_handler` itself stay in `api.rs`
//! — they require the full AppState shape and will move together with a
//! later handler-extraction pass.
//!
//! Private to the `network` module — NOT part of the crate's public API.
//! Referenced from `network/api.rs` via `use super::admin_auth::*;`.

use axum::http::HeaderMap;
use std::path::PathBuf;

pub(crate) const ADMIN_SNAPSHOT_TOKEN_HEADER: &str = "x-admin-token";
pub(crate) const AUTHORIZATION_BEARER_SCHEME: &str = "bearer";

pub(crate) enum SnapshotSource {
    Http(reqwest::Url),
    Local(PathBuf),
}

pub(crate) fn bearer_token_from_headers(headers: &HeaderMap) -> Option<&str> {
    let raw = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())?
        .trim();
    let mut parts = raw.split_whitespace();
    let scheme = parts.next()?;
    let token = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if scheme.eq_ignore_ascii_case(AUTHORIZATION_BEARER_SCHEME) && !token.is_empty() {
        return Some(token);
    }
    None
}

pub(crate) fn legacy_admin_token_from_headers(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(ADMIN_SNAPSHOT_TOKEN_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
}

/// Compares secrets without early exit on mismatch (timing-attack resistant).
pub(crate) fn admin_token_eq_constant_time(expected: &str, provided: &str) -> bool {
    let expected = expected.as_bytes();
    let provided = provided.as_bytes();
    let max_len = expected.len().max(provided.len());
    let mut diff = expected.len() ^ provided.len();
    for idx in 0..max_len {
        let a = expected.get(idx).copied().unwrap_or(0);
        let b = provided.get(idx).copied().unwrap_or(0);
        diff |= usize::from(a ^ b);
    }
    diff == 0
}

/// Check whether a request host matches the operator-configured allowlist.
///
/// An entry `example.com` matches both `example.com` (exact) and
/// `foo.example.com` (subdomain — the `foo.` prefix is what authorizes the
/// suffix match). Comparison is case-insensitive on the request host.
pub(crate) fn is_allowed_snapshot_host(host: &str, allowed_hosts: &[String]) -> bool {
    let host = host.to_ascii_lowercase();
    allowed_hosts.iter().any(|allowed| {
        host == *allowed
            || host
                .strip_suffix(allowed)
                .is_some_and(|prefix| prefix.ends_with('.'))
    })
}
