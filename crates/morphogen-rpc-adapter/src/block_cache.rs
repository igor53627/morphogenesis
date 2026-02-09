use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Maximum number of blocks to keep in cache.
const MAX_CACHED_BLOCKS: usize = 64;

/// Block polling interval in seconds.
const POLL_INTERVAL_SECS: u64 = 2;

/// Cache of recent block transactions and receipts.
/// Serves eth_getTransactionByHash and eth_getTransactionReceipt
/// from local cache to avoid leaking which txs the user queries.
pub struct BlockCache {
    transactions: HashMap<[u8; 32], Value>,
    receipts: HashMap<[u8; 32], Value>,
    cached_blocks: VecDeque<(u64, Vec<[u8; 32]>)>,
    latest_block: u64,
}

impl BlockCache {
    pub fn new() -> Self {
        Self {
            transactions: HashMap::new(),
            receipts: HashMap::new(),
            cached_blocks: VecDeque::new(),
            latest_block: 0,
        }
    }

    pub fn insert_block(&mut self, block_number: u64, txs: Vec<([u8; 32], Value)>, receipts: Vec<([u8; 32], Value)>) {
        let mut tx_hashes = Vec::with_capacity(txs.len());
        for (hash, tx) in txs {
            tx_hashes.push(hash);
            self.transactions.insert(hash, tx);
        }
        for (hash, receipt) in receipts {
            self.receipts.insert(hash, receipt);
        }
        self.cached_blocks.push_back((block_number, tx_hashes));

        if block_number > self.latest_block {
            self.latest_block = block_number;
        }

        // Evict oldest blocks
        while self.cached_blocks.len() > MAX_CACHED_BLOCKS {
            if let Some((_, old_hashes)) = self.cached_blocks.pop_front() {
                for hash in &old_hashes {
                    self.transactions.remove(hash);
                    self.receipts.remove(hash);
                }
            }
        }
    }

    pub fn get_transaction(&self, hash: &[u8; 32]) -> Option<&Value> {
        self.transactions.get(hash)
    }

    pub fn get_receipt(&self, hash: &[u8; 32]) -> Option<&Value> {
        self.receipts.get(hash)
    }

    pub fn latest_block(&self) -> u64 {
        self.latest_block
    }

    pub fn cached_tx_count(&self) -> usize {
        self.transactions.len()
    }
}

/// Parse a 0x-prefixed hex string into a 32-byte hash.
fn parse_tx_hash(s: &str) -> Option<[u8; 32]> {
    let hex = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(hex).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&bytes);
    Some(hash)
}

/// Parse a hex block number string (e.g. "0x1234") into u64.
fn parse_hex_block_number(s: &str) -> Option<u64> {
    let hex = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(hex, 16).ok()
}

/// Start the background block cache poller.
/// Polls eth_blockNumber every POLL_INTERVAL_SECS, fetches new blocks
/// with full tx objects and receipts, and inserts them into the cache.
pub fn start_block_poller(
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
                        info!("Block cache poller recovered after {} failures", consecutive_failures);
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
                    } else if consecutive_failures % 10 == 0 {
                        error!(consecutive_failures, "Block cache poll repeatedly failing: {}", e);
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(POLL_INTERVAL_SECS)).await;
        }
    });
}

/// Poll upstream for new blocks and insert into cache.
/// Returns the number of new blocks fetched.
async fn poll_new_blocks(
    cache: &Arc<RwLock<BlockCache>>,
    client: &reqwest::Client,
    upstream_url: &str,
) -> Result<u64, String> {
    // Get current block number
    let block_num_result = rpc_call(client, upstream_url, "eth_blockNumber", serde_json::json!([])).await?;
    let block_num_str = block_num_result
        .as_str()
        .ok_or("eth_blockNumber: expected hex string")?;
    let current_block = parse_hex_block_number(block_num_str)
        .ok_or_else(|| format!("invalid block number: {}", block_num_str))?;

    let cached_latest = cache.read().await.latest_block();

    if current_block <= cached_latest {
        return Ok(0);
    }

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
        ).await?;

        if block.is_null() {
            debug!(block_num, "Block not found (may not be finalized yet)");
            break;
        }

        let txs_array = block.get("transactions")
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
        let receipts = fetch_receipts(client, upstream_url, &block_hex, &tx_hashes_for_receipts).await?;

        cache.write().await.insert_block(block_num, txs, receipts);
        new_blocks += 1;
    }

    Ok(new_blocks)
}

/// Fetch receipts for a block. Tries eth_getBlockReceipts first,
/// falls back to individual eth_getTransactionReceipt calls.
async fn fetch_receipts(
    client: &reqwest::Client,
    upstream_url: &str,
    block_hex: &str,
    tx_hashes: &[[u8; 32]],
) -> Result<Vec<([u8; 32], Value)>, String> {
    // Try eth_getBlockReceipts (supported by most modern nodes)
    match rpc_call(client, upstream_url, "eth_getBlockReceipts", serde_json::json!([block_hex])).await {
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
            // Block not found or method returned null
            return Ok(Vec::new());
        }
        Err(e) => {
            debug!("eth_getBlockReceipts not available ({}), falling back to individual calls", e);
        }
        _ => {
            debug!("eth_getBlockReceipts returned unexpected type, falling back");
        }
    }

    // Fallback: fetch receipts individually
    let mut receipts = Vec::with_capacity(tx_hashes.len());
    for hash in tx_hashes {
        let hash_hex = format!("0x{}", hex::encode(hash));
        match rpc_call(client, upstream_url, "eth_getTransactionReceipt", serde_json::json!([hash_hex])).await {
            Ok(receipt) if !receipt.is_null() => {
                receipts.push((*hash, receipt));
            }
            Ok(_) => {} // null receipt, skip
            Err(e) => {
                warn!("Failed to fetch receipt for {}: {}", hash_hex, e);
            }
        }
    }
    Ok(receipts)
}

/// Make a JSON-RPC call and return the result field.
async fn rpc_call(
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

    let resp = client
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("{} request failed: {}", method, e))?;

    if !resp.status().is_success() {
        return Err(format!("{} HTTP {}", method, resp.status()));
    }

    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("{} parse failed: {}", method, e))?;

    if let Some(err) = json.get("error") {
        let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("unknown");
        return Err(format!("{} RPC error: {}", method, msg));
    }

    Ok(json.get("result").cloned().unwrap_or(Value::Null))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_cache_insert_and_get() {
        let mut cache = BlockCache::new();
        let hash = [0xAA; 32];
        let tx = serde_json::json!({"hash": "0xaa", "value": "0x1"});
        let receipt = serde_json::json!({"transactionHash": "0xaa", "status": "0x1"});

        cache.insert_block(100, vec![(hash, tx.clone())], vec![(hash, receipt.clone())]);

        assert_eq!(cache.latest_block(), 100);
        assert_eq!(cache.get_transaction(&hash), Some(&tx));
        assert_eq!(cache.get_receipt(&hash), Some(&receipt));
    }

    #[test]
    fn block_cache_eviction() {
        let mut cache = BlockCache::new();

        // Insert MAX_CACHED_BLOCKS + 1 blocks
        for i in 0..=(MAX_CACHED_BLOCKS as u64) {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            hash[1] = (i >> 8) as u8;
            let tx = serde_json::json!({"block": i});
            cache.insert_block(i, vec![(hash, tx)], vec![]);
        }

        // Oldest block's tx should be evicted
        let old_hash = [0u8; 32]; // block 0's hash
        assert_eq!(cache.get_transaction(&old_hash), None);

        // Latest block's tx should still be there
        let mut new_hash = [0u8; 32];
        new_hash[0] = MAX_CACHED_BLOCKS as u8;
        assert!(cache.get_transaction(&new_hash).is_some());

        assert_eq!(cache.cached_blocks.len(), MAX_CACHED_BLOCKS);
    }

    #[test]
    fn block_cache_miss() {
        let cache = BlockCache::new();
        let hash = [0xFF; 32];
        assert_eq!(cache.get_transaction(&hash), None);
        assert_eq!(cache.get_receipt(&hash), None);
    }

    #[test]
    fn parse_tx_hash_valid() {
        let hash_str = "0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060";
        let hash = parse_tx_hash(hash_str).unwrap();
        assert_eq!(hash[0], 0x5c);
        assert_eq!(hash[31], 0x60);
    }

    #[test]
    fn parse_tx_hash_invalid() {
        assert!(parse_tx_hash("0xshort").is_none());
        assert!(parse_tx_hash("not_hex").is_none());
    }

    #[test]
    fn parse_hex_block_number_valid() {
        assert_eq!(parse_hex_block_number("0x1"), Some(1));
        assert_eq!(parse_hex_block_number("0xff"), Some(255));
        assert_eq!(parse_hex_block_number("0x13b6340"), Some(0x13b6340));
    }

    #[test]
    fn parse_hex_block_number_invalid() {
        assert_eq!(parse_hex_block_number("0xZZZZ"), None);
    }
}
