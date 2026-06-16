//! Log-filter parsing and block-tag resolution helpers.
//!
//! Extracted from the crate root in TASK-54.4. These helpers translate an
//! `eth_getLogs`/`eth_newFilter` filter object into a [`block_cache::LogFilter`]
//! by resolving block tags (`safe` / `finalized`) against the upstream and
//! computing an effective latest-height floor. The handler functions that wire
//! these helpers into the `RpcModule<AdapterState>` stay in the crate root for
//! now and move with the `methods` module in a follow-up.
//!
//! Visibility: all items are `pub(crate)` — intentional seams per the
//! TASK-54 design constraints (no broad `pub` of internals).

use anyhow::Result;
use jsonrpsee::types::ErrorObjectOwned;
use serde_json::Value;

use crate::proxy::proxy_to_upstream;

/// True if the filter object references the given block tag (`safe` or
/// `finalized`) in either `fromBlock` or `toBlock`.
pub(crate) fn filter_uses_tag(filter_obj: &Value, tag: &str) -> bool {
    ["fromBlock", "toBlock"].iter().any(|field| {
        filter_obj
            .get(field)
            .and_then(Value::as_str)
            .is_some_and(|value| value == tag)
    })
}

/// Parse a `0x`-prefixed hex block number returned by upstream. `tag` is used
/// only to label the error message.
pub(crate) fn parse_block_number_quantity(
    number: &str,
    tag: &str,
) -> Result<u64, ErrorObjectOwned> {
    let hex = number.strip_prefix("0x").ok_or_else(|| {
        ErrorObjectOwned::owned(
            -32000,
            format!("Invalid {} block number format from upstream", tag),
            None::<()>,
        )
    })?;
    u64::from_str_radix(hex, 16).map_err(|_| {
        ErrorObjectOwned::owned(
            -32000,
            format!("Invalid {} block number value from upstream", tag),
            None::<()>,
        )
    })
}

/// Resolve a block tag (`safe` / `finalized`) to a numeric height by querying
/// upstream `eth_getBlockByNumber`.
pub(crate) async fn resolve_block_tag_height(
    client: &reqwest::Client,
    upstream_url: &str,
    tag: &'static str,
) -> Result<u64, ErrorObjectOwned> {
    let result = proxy_to_upstream(
        upstream_url,
        client,
        "eth_getBlockByNumber",
        serde_json::json!([tag, false]),
    )
    .await?;

    let number = result
        .get("number")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ErrorObjectOwned::owned(
                -32000,
                format!("Missing {} block number from upstream", tag),
                None::<()>,
            )
        })?;

    parse_block_number_quantity(number, tag)
}

/// Resolve `safe` / `finalized` heights for a filter object (None for any tag
/// the filter does not reference).
pub(crate) async fn resolve_filter_finality_heights(
    filter_obj: &Value,
    client: &reqwest::Client,
    upstream_url: &str,
) -> Result<(Option<u64>, Option<u64>), ErrorObjectOwned> {
    let safe_height = if filter_uses_tag(filter_obj, "safe") {
        Some(resolve_block_tag_height(client, upstream_url, "safe").await?)
    } else {
        None
    };

    let finalized_height = if filter_uses_tag(filter_obj, "finalized") {
        Some(resolve_block_tag_height(client, upstream_url, "finalized").await?)
    } else {
        None
    };

    Ok((safe_height, finalized_height))
}

/// Effective latest height for a filter: the max of the cache latest, the
/// resolved `safe` height, and the resolved `finalized` height.
pub(crate) fn effective_latest_for_filter(
    cache_latest: u64,
    safe_height: Option<u64>,
    finalized_height: Option<u64>,
) -> u64 {
    cache_latest
        .max(safe_height.unwrap_or(0))
        .max(finalized_height.unwrap_or(0))
}

/// Parse a filter object into a [`block_cache::LogFilter`], resolving block
/// tags against upstream and applying the effective-latest floor.
pub(crate) async fn parse_log_filter_for_rpc(
    filter_obj: &Value,
    cache_latest: u64,
    client: &reqwest::Client,
    upstream_url: &str,
) -> Result<crate::block_cache::LogFilter, ErrorObjectOwned> {
    let (safe_height, finalized_height) =
        resolve_filter_finality_heights(filter_obj, client, upstream_url).await?;
    let effective_latest = effective_latest_for_filter(cache_latest, safe_height, finalized_height);
    crate::block_cache::parse_log_filter_object(
        filter_obj,
        effective_latest,
        safe_height,
        finalized_height,
    )
    .map_err(|e| ErrorObjectOwned::owned(-32602, e, None::<()>))
}
