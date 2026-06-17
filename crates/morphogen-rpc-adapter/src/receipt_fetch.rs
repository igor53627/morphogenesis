//! Receipt-fetching helpers for the block cache poller.
//!
//! Extracted from `block_cache.rs` in TASK-54.19. Pure network/IO helpers:
//! JSON-RPC `rpc_call`, `fetch_receipts` (tries `eth_getBlockReceipts` then
//! falls back to per-tx `eth_getTransactionReceipt`), and the bounded
//! concurrent fan-out `fetch_receipts_fallback_bounded`.
//!
//! Top-level module (sibling of `block_cache`). `block_cache.rs` re-exports
//! them via `pub(crate) use crate::receipt_fetch::*;` so the poller's existing
//! `rpc_call` / `fetch_receipts` call sites keep resolving.

use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, warn};

use crate::block_cache::parse_tx_hash;

pub(crate) const RECEIPT_FALLBACK_MAX_IN_FLIGHT: usize = 8;

pub(crate) type ReceiptFetchFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>>;

/// Fetch receipts for a block. Tries eth_getBlockReceipts first,
/// falls back to individual eth_getTransactionReceipt calls.
pub(crate) async fn fetch_receipts(
    client: &reqwest::Client,
    upstream_url: &str,
    block_hex: &str,
    tx_hashes: &[[u8; 32]],
) -> Result<Vec<([u8; 32], Value)>, String> {
    // Try eth_getBlockReceipts (supported by most modern nodes)
    match rpc_call(
        client,
        upstream_url,
        "eth_getBlockReceipts",
        serde_json::json!([block_hex]),
    )
    .await
    {
        Ok(Value::Array(receipts_array)) => {
            let mut receipts = Vec::with_capacity(receipts_array.len());
            for receipt in receipts_array {
                if let Some(hash_str) = receipt.get("transactionHash").and_then(|h| h.as_str()) {
                    if let Some(hash) = parse_tx_hash(hash_str) {
                        receipts.push((hash, receipt));
                    }
                }
            }
            return Ok(receipts);
        }
        Ok(Value::Null) => {
            debug!("eth_getBlockReceipts returned null, falling back to individual calls");
        }
        Err(e) => {
            debug!(
                "eth_getBlockReceipts not available ({}), falling back to individual calls",
                e
            );
        }
        _ => {
            debug!("eth_getBlockReceipts returned unexpected type, falling back");
        }
    }

    // Fallback: fetch receipts individually
    let upstream_url = Arc::<str>::from(upstream_url);
    let client = client.clone();
    let fetcher: Arc<dyn Fn([u8; 32]) -> ReceiptFetchFuture + Send + Sync> =
        Arc::new(move |hash: [u8; 32]| {
            let client = client.clone();
            let upstream_url = upstream_url.clone();
            Box::pin(async move {
                let hash_hex = format!("0x{}", hex::encode(hash));
                rpc_call(
                    &client,
                    upstream_url.as_ref(),
                    "eth_getTransactionReceipt",
                    serde_json::json!([hash_hex]),
                )
                .await
            })
        });

    Ok(fetch_receipts_fallback_bounded(tx_hashes, fetcher).await)
}

pub(crate) async fn fetch_receipts_fallback_bounded(
    tx_hashes: &[[u8; 32]],
    fetcher: Arc<dyn Fn([u8; 32]) -> ReceiptFetchFuture + Send + Sync>,
) -> Vec<([u8; 32], Value)> {
    if tx_hashes.is_empty() {
        return Vec::new();
    }

    let max_in_flight = RECEIPT_FALLBACK_MAX_IN_FLIGHT.clamp(1, tx_hashes.len());
    let mut join_set = tokio::task::JoinSet::new();
    let mut next_index = 0usize;
    let mut ordered_receipts: Vec<Option<([u8; 32], Value)>> = vec![None; tx_hashes.len()];

    while next_index < max_in_flight {
        let index = next_index;
        let hash = tx_hashes[index];
        let fetcher = fetcher.clone();
        join_set.spawn(async move {
            let result = fetcher(hash).await;
            (index, hash, result)
        });
        next_index += 1;
    }

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok((index, hash, Ok(receipt))) if !receipt.is_null() => {
                ordered_receipts[index] = Some((hash, receipt));
            }
            Ok((_index, _hash, Ok(_))) => {}
            Ok((_index, hash, Err(e))) => {
                warn!("Failed to fetch receipt for 0x{}: {}", hex::encode(hash), e);
            }
            Err(e) => {
                warn!("Receipt fetch task failed: {}", e);
            }
        }

        if next_index < tx_hashes.len() {
            let index = next_index;
            let hash = tx_hashes[index];
            let fetcher = fetcher.clone();
            join_set.spawn(async move {
                let result = fetcher(hash).await;
                (index, hash, result)
            });
            next_index += 1;
        }
    }

    ordered_receipts.into_iter().flatten().collect()
}

/// Make a JSON-RPC call and return the result field.
///
/// Error strings are **generic** (method + error category only). The raw
/// `reqwest::Error` is never formatted into the string because its `Debug`
/// output can include a normalized upstream URL with embedded credentials
/// (basic-auth, API-key query params) that `.replace()` cannot reliably
/// strip. See gemini security-high + greptile P1 on PR #52.
pub(crate) async fn rpc_call(
    client: &reqwest::Client,
    url: &str,
    method: &str,
    params: Value,
) -> Result<Value, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    });

    let resp = client.post(url).json(&request).send().await.map_err(|e| {
        let kind = if e.is_timeout() {
            "timed out"
        } else if e.is_connect() {
            "connection failed"
        } else {
            "request failed"
        };
        format!("{} {}", method, kind)
    })?;

    if !resp.status().is_success() {
        return Err(format!("{} HTTP {}", method, resp.status()));
    }

    let json: Value = resp
        .json()
        .await
        .map_err(|_| format!("{} response parse failed", method))?;

    if let Some(err) = json.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown");
        return Err(format!("{} RPC error: {}", method, msg));
    }

    Ok(json.get("result").cloned().unwrap_or(Value::Null))
}
