//! Background block-cache poller.
//!
//! Extracted from `block_cache.rs` in TASK-54.20. Periodically polls upstream
//! for new blocks (with reorg detection) and inserts them into the shared
//! `BlockCache`. Pure networking + cache-mutation glue.
//!
//! Top-level module (sibling of `block_cache`). Referenced from `block_cache.rs`
//! via `pub(crate) use crate::block_poller::start_block_poller;`.

use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::block_cache::{parse_hex_block_number, parse_tx_hash, BlockCache};
use crate::receipt_fetch::{fetch_receipts, rpc_call};

/// Polling interval in seconds.
pub(crate) const POLL_INTERVAL_SECS: u64 = 2;

pub(crate) fn start_block_poller(
    cache: Arc<RwLock<BlockCache>>,
    client: reqwest::Client,
    upstream_url: String,
) {
    tokio::spawn(async move {
        info!("Block cache poller started");
        let mut consecutive_failures = 0u32;

        loop {
            match poll_new_blocks(&cache, &client, &upstream_url).await {
                Ok(new_blocks) => {
                    if consecutive_failures > 0 {
                        info!(
                            "Block cache poller recovered after {} failures",
                            consecutive_failures
                        );
                    }
                    consecutive_failures = 0;
                    if new_blocks > 0 {
                        let cache_read = cache.read().await;
                        debug!(
                            new_blocks,
                            latest = cache_read.latest_block(),
                            cached_txs = cache_read.cached_tx_count(),
                            "Block cache updated"
                        );
                    }
                }
                Err(e) => {
                    consecutive_failures += 1;
                    if consecutive_failures <= 3 {
                        warn!(consecutive_failures, "Block cache poll failed: {}", e);
                    } else if consecutive_failures.is_multiple_of(10) {
                        error!(
                            consecutive_failures,
                            "Block cache poll repeatedly failing: {}", e
                        );
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(POLL_INTERVAL_SECS)).await;
        }
    });
}

/// Poll upstream for new blocks and insert into cache.
/// Detects reorgs by comparing the block hash at our cached latest
/// against what upstream reports, and invalidates stale entries.
/// Returns the number of new blocks fetched.
pub(crate) async fn poll_new_blocks(
    cache: &Arc<RwLock<BlockCache>>,
    client: &reqwest::Client,
    upstream_url: &str,
) -> Result<u64, String> {
    // Get current block number
    let block_num_result = rpc_call(
        client,
        upstream_url,
        "eth_blockNumber",
        serde_json::json!([]),
    )
    .await?;
    let block_num_str = block_num_result
        .as_str()
        .ok_or("eth_blockNumber: expected hex string")?;
    let current_block = parse_hex_block_number(block_num_str)
        .ok_or_else(|| format!("invalid block number: {}", block_num_str))?;

    let cached_info = cache.read().await.latest_block_hash();

    // Reorg detection: compare block hash at our cached latest against upstream
    if let Some((cached_num, cached_hash)) = cached_info {
        if current_block < cached_num {
            // Head moved backwards — clear everything above current
            warn!(
                cached = cached_num,
                chain_head = current_block,
                "Chain reorg detected (head moved backwards), invalidating cache"
            );
            cache.write().await.invalidate_from(current_block);
            return Ok(0);
        }

        // Even if head didn't move backwards, check hash at cached_latest
        // to detect same-height reorgs
        let check_hex = format!("0x{:x}", cached_num);
        let check_block = rpc_call(
            client,
            upstream_url,
            "eth_getBlockByNumber",
            serde_json::json!([check_hex, false]),
        )
        .await?;
        if !check_block.is_null() {
            match check_block
                .get("hash")
                .and_then(|h| h.as_str())
                .and_then(parse_tx_hash)
            {
                Some(upstream_hash) if upstream_hash != cached_hash => {
                    warn!(
                        block = cached_num,
                        "Chain reorg detected (block hash mismatch), invalidating cache"
                    );
                    cache.write().await.invalidate_from(cached_num);
                    return Ok(0);
                }
                None => {
                    warn!(
                        block = cached_num,
                        "Upstream block missing hash, conservatively invalidating cache"
                    );
                    cache.write().await.invalidate_from(cached_num);
                    return Ok(0);
                }
                _ => {} // hash matches, no reorg
            }
        }

        if current_block <= cached_num {
            return Ok(0);
        }
    }

    let cached_latest = cached_info.map_or(0, |(n, _)| n);

    // On first run (cached_latest == 0), just fetch the latest block
    let start_block = if cached_latest == 0 {
        current_block
    } else {
        cached_latest + 1
    };

    // Cap how many blocks we fetch at once to avoid overwhelming upstream
    let end_block = current_block.min(start_block + 4);

    let mut new_blocks = 0u64;
    for block_num in start_block..=end_block {
        let block_hex = format!("0x{:x}", block_num);

        // Fetch block with full tx objects
        let block = rpc_call(
            client,
            upstream_url,
            "eth_getBlockByNumber",
            serde_json::json!([block_hex, true]),
        )
        .await?;

        if block.is_null() {
            debug!(block_num, "Block not found (may not be finalized yet)");
            break;
        }

        let block_hash = match block
            .get("hash")
            .and_then(|h| h.as_str())
            .and_then(parse_tx_hash)
        {
            Some(h) => h,
            None => {
                return Err(format!("block {} missing valid hash", block_num));
            }
        };

        let txs_array = block
            .get("transactions")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut txs = Vec::with_capacity(txs_array.len());
        let mut tx_hashes_for_receipts: Vec<[u8; 32]> = Vec::with_capacity(txs_array.len());

        for tx in &txs_array {
            if let Some(hash_str) = tx.get("hash").and_then(|h| h.as_str()) {
                if let Some(hash) = parse_tx_hash(hash_str) {
                    tx_hashes_for_receipts.push(hash);
                    txs.push((hash, tx.clone()));
                }
            }
        }

        // Try eth_getBlockReceipts first (efficient), fall back to individual calls
        let receipts =
            fetch_receipts(client, upstream_url, &block_hex, &tx_hashes_for_receipts).await?;

        cache
            .write()
            .await
            .insert_block(block_num, block_hash, txs, receipts);
        new_blocks += 1;
    }

    Ok(new_blocks)
}
