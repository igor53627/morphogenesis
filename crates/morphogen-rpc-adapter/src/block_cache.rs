use rand::Rng;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Filter expiry timeout (5 minutes).
const FILTER_EXPIRY_SECS: u64 = 300;

/// Maximum number of blocks to keep in cache.
const MAX_CACHED_BLOCKS: usize = 64;

/// Block polling interval in seconds.
const POLL_INTERVAL_SECS: u64 = 2;

/// Per-block tracking for reliable eviction of both txs and receipts.
struct CachedBlock {
    number: u64,
    block_hash: [u8; 32],
    tx_hashes: Vec<[u8; 32]>,
    receipt_hashes: Vec<[u8; 32]>,
}

/// A stored filter with its kind, cursor, and last access time.
pub struct StoredFilter {
    kind: FilterKind,
    last_polled_block: u64,
    last_accessed: Instant,
}

/// The kind of filter installed by the client.
pub enum FilterKind {
    Log(LogFilter),
    Block,
    PendingTransaction,
}

/// Cache of recent block transactions, receipts, and logs.
/// Serves eth_getTransactionByHash, eth_getTransactionReceipt, and
/// eth_getLogs from local cache to avoid leaking query patterns.
/// Also stores filter state for eth_newFilter / eth_getFilterChanges.
pub struct BlockCache {
    transactions: HashMap<[u8; 32], Value>,
    receipts: HashMap<[u8; 32], Value>,
    logs: HashMap<u64, Vec<Value>>,
    cached_blocks: VecDeque<CachedBlock>,
    latest_block: u64,
    filters: HashMap<String, StoredFilter>,
}

impl BlockCache {
    pub fn new() -> Self {
        Self {
            transactions: HashMap::new(),
            receipts: HashMap::new(),
            logs: HashMap::new(),
            cached_blocks: VecDeque::new(),
            latest_block: 0,
            filters: HashMap::new(),
        }
    }

    pub fn insert_block(
        &mut self,
        block_number: u64,
        block_hash: [u8; 32],
        txs: Vec<([u8; 32], Value)>,
        receipts: Vec<([u8; 32], Value)>,
    ) {
        let mut tx_hashes = Vec::with_capacity(txs.len());
        for (hash, tx) in txs {
            tx_hashes.push(hash);
            self.transactions.insert(hash, tx);
        }
        let mut receipt_hashes = Vec::with_capacity(receipts.len());
        let mut block_logs = Vec::new();
        for (hash, receipt) in receipts {
            receipt_hashes.push(hash);
            if let Some(logs_array) = receipt.get("logs").and_then(|l| l.as_array()) {
                block_logs.extend(logs_array.iter().cloned());
            }
            self.receipts.insert(hash, receipt);
        }
        self.logs.insert(block_number, block_logs);
        self.cached_blocks.push_back(CachedBlock {
            number: block_number,
            block_hash,
            tx_hashes,
            receipt_hashes,
        });

        if block_number > self.latest_block {
            self.latest_block = block_number;
        }

        // Evict oldest blocks
        while self.cached_blocks.len() > MAX_CACHED_BLOCKS {
            if let Some(old) = self.cached_blocks.pop_front() {
                for hash in &old.tx_hashes {
                    self.transactions.remove(hash);
                }
                for hash in &old.receipt_hashes {
                    self.receipts.remove(hash);
                }
                self.logs.remove(&old.number);
            }
        }
    }

    /// Invalidate all cached blocks at or above the given block number (reorg).
    pub fn invalidate_from(&mut self, block_number: u64) {
        while let Some(last) = self.cached_blocks.back() {
            if last.number >= block_number {
                let old = self.cached_blocks.pop_back().unwrap();
                for hash in &old.tx_hashes {
                    self.transactions.remove(hash);
                }
                for hash in &old.receipt_hashes {
                    self.receipts.remove(hash);
                }
                self.logs.remove(&old.number);
            } else {
                break;
            }
        }
        self.latest_block = self.cached_blocks.back().map_or(0, |b| b.number);
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

    /// Returns (block_number, block_hash) of the latest cached block, if any.
    pub(crate) fn latest_block_hash(&self) -> Option<(u64, [u8; 32])> {
        self.cached_blocks.back().map(|b| (b.number, b.block_hash))
    }

    /// Check whether all blocks in [from, to] are present in the cache.
    /// O(1) check against the cached window bounds rather than iterating.
    pub fn has_block_range(&self, from: u64, to: u64) -> bool {
        if from > to {
            return false;
        }
        let oldest = match self.cached_blocks.front() {
            Some(b) => b.number,
            None => return false,
        };
        let latest = match self.cached_blocks.back() {
            Some(b) => b.number,
            None => return false,
        };
        from >= oldest && to <= latest
    }

    /// Return all logs in [filter.from_block, filter.to_block] matching the filter.
    /// Only iterates over cached blocks (O(64) max) to avoid DoS from large ranges.
    pub fn get_logs(&self, filter: &LogFilter) -> Vec<Value> {
        let mut result = Vec::new();
        for block in &self.cached_blocks {
            if block.number < filter.from_block {
                continue;
            }
            if block.number > filter.to_block {
                break;
            }
            if let Some(block_logs) = self.logs.get(&block.number) {
                for log in block_logs {
                    if log_matches_filter(log, filter) {
                        result.push(log.clone());
                    }
                }
            }
        }
        result
    }

    // --- Filter API ---

    /// Remove filters that haven't been accessed in FILTER_EXPIRY_SECS.
    pub fn cleanup_expired_filters(&mut self) {
        let now = Instant::now();
        self.filters
            .retain(|_, f| now.duration_since(f.last_accessed).as_secs() < FILTER_EXPIRY_SECS);
    }

    /// Generate an unguessable random hex filter ID (u64 quantity for client compat).
    fn generate_filter_id(&self) -> String {
        let mut rng = rand::thread_rng();
        loop {
            let id: u64 = rng.gen();
            let hex_id = format!("0x{:x}", id);
            if !self.filters.contains_key(&hex_id) {
                return hex_id;
            }
        }
    }

    /// Create a log filter and return its hex ID.
    pub fn create_log_filter(&mut self, filter: LogFilter) -> String {
        self.cleanup_expired_filters();
        let hex_id = self.generate_filter_id();
        self.filters.insert(
            hex_id.clone(),
            StoredFilter {
                kind: FilterKind::Log(filter),
                last_polled_block: self.latest_block,
                last_accessed: Instant::now(),
            },
        );
        hex_id
    }

    /// Create a block filter and return its hex ID.
    pub fn create_block_filter(&mut self) -> String {
        self.cleanup_expired_filters();
        let hex_id = self.generate_filter_id();
        self.filters.insert(
            hex_id.clone(),
            StoredFilter {
                kind: FilterKind::Block,
                last_polled_block: self.latest_block,
                last_accessed: Instant::now(),
            },
        );
        hex_id
    }

    /// Create a pending transaction filter and return its hex ID.
    pub fn create_pending_tx_filter(&mut self) -> String {
        self.cleanup_expired_filters();
        let hex_id = self.generate_filter_id();
        self.filters.insert(
            hex_id.clone(),
            StoredFilter {
                kind: FilterKind::PendingTransaction,
                last_polled_block: self.latest_block,
                last_accessed: Instant::now(),
            },
        );
        hex_id
    }

    /// Get changes since last poll for the given filter.
    /// Returns None if the filter ID doesn't exist.
    pub fn get_filter_changes(&mut self, id: &str) -> Option<Vec<Value>> {
        self.cleanup_expired_filters();
        let filter = self.filters.get_mut(id)?;
        filter.last_accessed = Instant::now();
        let from_exclusive = filter.last_polled_block;

        match &filter.kind {
            FilterKind::Log(log_filter) => {
                let to_inclusive = self.latest_block.min(log_filter.to_block);
                // Only advance cursor forward (never backwards)
                if to_inclusive <= from_exclusive {
                    return Some(vec![]);
                }
                filter.last_polled_block = to_inclusive;
                let scan_filter = LogFilter {
                    from_block: from_exclusive.saturating_add(1),
                    to_block: to_inclusive,
                    addresses: log_filter.addresses.clone(),
                    topics: log_filter
                        .topics
                        .iter()
                        .map(|t| match t {
                            TopicFilter::Any => TopicFilter::Any,
                            TopicFilter::Exact(s) => TopicFilter::Exact(s.clone()),
                            TopicFilter::OneOf(v) => TopicFilter::OneOf(v.clone()),
                        })
                        .collect(),
                };
                Some(self.get_logs(&scan_filter))
            }
            FilterKind::Block => {
                let to_inclusive = self.latest_block;
                filter.last_polled_block = to_inclusive;
                Some(self.get_block_hashes_in_range(from_exclusive, to_inclusive))
            }
            FilterKind::PendingTransaction => {
                filter.last_polled_block = self.latest_block;
                Some(vec![])
            }
        }
    }

    /// Get all matching logs for a log filter (used by eth_getFilterLogs).
    /// Returns None if filter ID not found.
    /// Returns Some(None) if filter exists but is not a log filter.
    /// Returns Some(Some(logs)) with matching logs.
    pub fn get_filter_logs(&mut self, id: &str) -> Option<Option<Vec<Value>>> {
        self.cleanup_expired_filters();
        let filter = self.filters.get_mut(id)?;
        filter.last_accessed = Instant::now();

        match &filter.kind {
            FilterKind::Log(log_filter) => {
                let scan_filter = LogFilter {
                    from_block: log_filter.from_block,
                    to_block: self.latest_block.min(log_filter.to_block),
                    addresses: log_filter.addresses.clone(),
                    topics: log_filter
                        .topics
                        .iter()
                        .map(|t| match t {
                            TopicFilter::Any => TopicFilter::Any,
                            TopicFilter::Exact(s) => TopicFilter::Exact(s.clone()),
                            TopicFilter::OneOf(v) => TopicFilter::OneOf(v.clone()),
                        })
                        .collect(),
                };
                Some(Some(self.get_logs(&scan_filter)))
            }
            _ => Some(None),
        }
    }

    /// Remove a filter. Returns true if it existed.
    pub fn uninstall_filter(&mut self, id: &str) -> bool {
        self.filters.remove(id).is_some()
    }

    /// Collect block hashes for blocks in (from_exclusive, to_inclusive].
    fn get_block_hashes_in_range(&self, from_exclusive: u64, to_inclusive: u64) -> Vec<Value> {
        let mut result = Vec::new();
        for block in &self.cached_blocks {
            if block.number > from_exclusive && block.number <= to_inclusive {
                result.push(Value::String(format!(
                    "0x{}",
                    hex::encode(block.block_hash)
                )));
            }
        }
        result
    }
}

/// Parse an optionally 0x-prefixed hex string into a 32-byte hash.
/// Rejects input that isn't exactly 64 hex chars (with optional 0x prefix)
/// before decoding, preventing DoS via oversized input.
pub fn parse_tx_hash(s: &str) -> Option<[u8; 32]> {
    let hex = s.strip_prefix("0x").unwrap_or(s);
    if hex.len() != 64 {
        return None;
    }
    let bytes = hex::decode(hex).ok()?;
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&bytes);
    Some(hash)
}

/// Parse a hex block number string (e.g. "0x1234") into u64.
pub fn parse_hex_block_number(s: &str) -> Option<u64> {
    let hex = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(hex, 16).ok()
}

/// Filter for matching logs by address and topics (EIP-1474 semantics).
pub struct LogFilter {
    pub from_block: u64,
    pub to_block: u64,
    /// If Some, only match logs from these addresses (lowercased hex with 0x prefix).
    pub addresses: Option<Vec<String>>,
    /// Topic filters by position. Each position is a filter.
    pub topics: Vec<TopicFilter>,
}

/// A single topic position filter.
pub enum TopicFilter {
    /// null = match any topic at this position.
    Any,
    /// Match a single exact topic hash (lowercased hex with 0x prefix).
    Exact(String),
    /// Match any one of these topic hashes (lowercased hex with 0x prefix).
    OneOf(Vec<String>),
}

/// Parse a JSON filter object into a LogFilter, resolving block tags.
/// Returns Err with a human-readable message on invalid input.
pub fn parse_log_filter_object(filter_obj: &Value, latest: u64) -> Result<LogFilter, String> {
    let from_block = match filter_obj.get("fromBlock").and_then(|v| v.as_str()) {
        Some("latest") | Some("pending") | None => latest,
        Some("earliest") => 0,
        Some("safe") | Some("finalized") => {
            return Err("\"safe\" and \"finalized\" block tags are not supported; use an explicit block number".to_string());
        }
        Some(hex) => {
            parse_hex_block_number(hex).ok_or_else(|| format!("invalid fromBlock: {}", hex))?
        }
    };

    let to_block = match filter_obj.get("toBlock").and_then(|v| v.as_str()) {
        Some("latest") | Some("pending") | None => latest,
        Some("earliest") => 0,
        Some("safe") | Some("finalized") => {
            return Err("\"safe\" and \"finalized\" block tags are not supported; use an explicit block number".to_string());
        }
        Some(hex) => {
            parse_hex_block_number(hex).ok_or_else(|| format!("invalid toBlock: {}", hex))?
        }
    };

    if from_block > to_block {
        return Err(format!(
            "invalid block range: fromBlock ({}) > toBlock ({})",
            from_block, to_block
        ));
    }

    let addresses = match filter_obj.get("address") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(vec![s.to_lowercase()]),
        Some(Value::Array(arr)) => {
            let mut addrs = Vec::with_capacity(arr.len());
            for v in arr {
                match v.as_str() {
                    Some(s) => addrs.push(s.to_lowercase()),
                    None => {
                        return Err(format!(
                            "invalid address in array: expected string, got {}",
                            v
                        ))
                    }
                }
            }
            Some(addrs) // empty list = match nothing
        }
        Some(other) => {
            return Err(format!(
                "invalid address field: expected string or array, got {}",
                other
            ))
        }
    };

    let topics = match filter_obj.get("topics") {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(arr)) => {
            let mut result = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::Null => result.push(TopicFilter::Any),
                    Value::String(s) => result.push(TopicFilter::Exact(s.to_lowercase())),
                    Value::Array(alts) => {
                        let mut options = Vec::with_capacity(alts.len());
                        for v in alts {
                            match v.as_str() {
                                Some(s) => options.push(s.to_lowercase()),
                                None => {
                                    return Err(format!(
                                    "invalid topic in alternatives array: expected string, got {}",
                                    v
                                ))
                                }
                            }
                        }
                        if options.is_empty() {
                            return Err("empty topic alternatives array is not valid".to_string());
                        }
                        result.push(TopicFilter::OneOf(options));
                    }
                    other => {
                        return Err(format!(
                            "invalid topic filter: expected null, string, or array, got {}",
                            other
                        ))
                    }
                }
            }
            result
        }
        Some(other) => {
            return Err(format!(
                "invalid topics field: expected array, got {}",
                other
            ))
        }
    };

    Ok(LogFilter {
        from_block,
        to_block,
        addresses,
        topics,
    })
}

/// Check whether a log entry matches the given filter per EIP-1474 semantics.
pub fn log_matches_filter(log: &Value, filter: &LogFilter) -> bool {
    // Address filter
    if let Some(ref addrs) = filter.addresses {
        let log_addr = log
            .get("address")
            .and_then(|a| a.as_str())
            .unwrap_or("")
            .to_lowercase();
        if !addrs.iter().any(|a| a == &log_addr) {
            return false;
        }
    }

    // Topic filters
    let log_topics = log.get("topics").and_then(|t| t.as_array());
    for (i, topic_filter) in filter.topics.iter().enumerate() {
        match topic_filter {
            TopicFilter::Any => {
                // Wildcard still implies the position exists per EIP-1474
                if log_topics.is_none_or(|t| t.get(i).is_none()) {
                    return false;
                }
            }
            TopicFilter::Exact(expected) => {
                let actual = log_topics
                    .and_then(|t| t.get(i))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                if &actual != expected {
                    return false;
                }
            }
            TopicFilter::OneOf(options) => {
                let actual = log_topics
                    .and_then(|t| t.get(i))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                if !options.iter().any(|o| o == &actual) {
                    return false;
                }
            }
        }
    }

    true
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
async fn poll_new_blocks(
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

/// Fetch receipts for a block. Tries eth_getBlockReceipts first,
/// falls back to individual eth_getTransactionReceipt calls.
async fn fetch_receipts(
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
            // Block not found or method returned null
            return Ok(Vec::new());
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
    let mut receipts = Vec::with_capacity(tx_hashes.len());
    for hash in tx_hashes {
        let hash_hex = format!("0x{}", hex::encode(hash));
        match rpc_call(
            client,
            upstream_url,
            "eth_getTransactionReceipt",
            serde_json::json!([hash_hex]),
        )
        .await
        {
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
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown");
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
        let blk_hash = [0x11; 32];
        let tx = serde_json::json!({"hash": "0xaa", "value": "0x1"});
        let receipt = serde_json::json!({"transactionHash": "0xaa", "status": "0x1"});

        cache.insert_block(
            100,
            blk_hash,
            vec![(hash, tx.clone())],
            vec![(hash, receipt.clone())],
        );

        assert_eq!(cache.latest_block(), 100);
        assert_eq!(cache.get_transaction(&hash), Some(&tx));
        assert_eq!(cache.get_receipt(&hash), Some(&receipt));
        assert_eq!(cache.latest_block_hash(), Some((100, blk_hash)));
    }

    #[test]
    fn block_cache_eviction() {
        let mut cache = BlockCache::new();

        // Insert MAX_CACHED_BLOCKS + 1 blocks with both txs and receipts
        for i in 0..=(MAX_CACHED_BLOCKS as u64) {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            hash[1] = (i >> 8) as u8;
            let mut blk_hash = [0u8; 32];
            blk_hash[0] = i as u8;
            let tx = serde_json::json!({"block": i});
            let receipt = serde_json::json!({"status": "0x1"});
            cache.insert_block(i, blk_hash, vec![(hash, tx)], vec![(hash, receipt)]);
        }

        // Oldest block's tx and receipt should be evicted
        let old_hash = [0u8; 32]; // block 0's hash
        assert_eq!(cache.get_transaction(&old_hash), None);
        assert_eq!(cache.get_receipt(&old_hash), None);

        // Latest block's tx and receipt should still be there
        let mut new_hash = [0u8; 32];
        new_hash[0] = MAX_CACHED_BLOCKS as u8;
        assert!(cache.get_transaction(&new_hash).is_some());
        assert!(cache.get_receipt(&new_hash).is_some());

        assert_eq!(cache.cached_blocks.len(), MAX_CACHED_BLOCKS);
    }

    #[test]
    fn block_cache_invalidate_reorg() {
        let mut cache = BlockCache::new();
        let hash_a = [0xAA; 32];
        let hash_b = [0xBB; 32];
        let hash_c = [0xCC; 32];

        cache.insert_block(
            100,
            [0x01; 32],
            vec![(hash_a, serde_json::json!({}))],
            vec![(hash_a, serde_json::json!({}))],
        );
        cache.insert_block(
            101,
            [0x02; 32],
            vec![(hash_b, serde_json::json!({}))],
            vec![(hash_b, serde_json::json!({}))],
        );
        cache.insert_block(
            102,
            [0x03; 32],
            vec![(hash_c, serde_json::json!({}))],
            vec![(hash_c, serde_json::json!({}))],
        );

        assert_eq!(cache.latest_block(), 102);

        // Simulate reorg at block 101 — invalidate 101 and 102
        cache.invalidate_from(101);

        assert_eq!(cache.latest_block(), 100);
        assert!(cache.get_transaction(&hash_a).is_some());
        assert_eq!(cache.get_transaction(&hash_b), None);
        assert_eq!(cache.get_transaction(&hash_c), None);
        assert_eq!(cache.get_receipt(&hash_b), None);
        assert_eq!(cache.get_receipt(&hash_c), None);
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
        // Correct length (64 chars) but invalid hex characters
        let bad_hex = format!("0x{}", "g".repeat(64));
        assert_eq!(bad_hex.strip_prefix("0x").unwrap().len(), 64);
        assert!(parse_tx_hash(&bad_hex).is_none());
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

    // --- Log filter tests ---

    fn make_log(address: &str, topics: &[&str]) -> Value {
        serde_json::json!({
            "address": address,
            "topics": topics,
            "data": "0x",
            "blockNumber": "0x64",
            "logIndex": "0x0"
        })
    }

    fn make_receipt_with_logs(tx_hash: &str, logs: Vec<Value>) -> Value {
        serde_json::json!({
            "transactionHash": tx_hash,
            "status": "0x1",
            "logs": logs
        })
    }

    #[test]
    fn log_filter_address_single() {
        let log = make_log("0xabc123", &["0xtopic1"]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: Some(vec!["0xabc123".to_string()]),
            topics: vec![],
        };
        assert!(log_matches_filter(&log, &filter));

        let filter_miss = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: Some(vec!["0xdef456".to_string()]),
            topics: vec![],
        };
        assert!(!log_matches_filter(&log, &filter_miss));
    }

    #[test]
    fn log_filter_address_list() {
        let log = make_log("0xabc123", &[]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: Some(vec!["0xdef456".to_string(), "0xabc123".to_string()]),
            topics: vec![],
        };
        assert!(log_matches_filter(&log, &filter));
    }

    #[test]
    fn log_filter_address_wildcard() {
        let log = make_log("0xabc123", &[]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![],
        };
        assert!(log_matches_filter(&log, &filter));
    }

    #[test]
    fn log_filter_address_case_insensitive() {
        let log = make_log("0xAbC123", &[]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: Some(vec!["0xabc123".to_string()]),
            topics: vec![],
        };
        assert!(log_matches_filter(&log, &filter));
    }

    #[test]
    fn log_filter_topic_exact() {
        let log = make_log("0xabc", &["0xdead", "0xbeef"]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![TopicFilter::Exact("0xdead".to_string())],
        };
        assert!(log_matches_filter(&log, &filter));

        let filter_miss = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![TopicFilter::Exact("0xbeef".to_string())],
        };
        assert!(!log_matches_filter(&log, &filter_miss));
    }

    #[test]
    fn log_filter_topic_any() {
        let log = make_log("0xabc", &["0xdead", "0xbeef"]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![TopicFilter::Any, TopicFilter::Exact("0xbeef".to_string())],
        };
        assert!(log_matches_filter(&log, &filter));
    }

    #[test]
    fn log_filter_topic_one_of() {
        let log = make_log("0xabc", &["0xdead"]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![TopicFilter::OneOf(vec![
                "0xcafe".to_string(),
                "0xdead".to_string(),
            ])],
        };
        assert!(log_matches_filter(&log, &filter));

        let filter_miss = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![TopicFilter::OneOf(vec![
                "0xcafe".to_string(),
                "0xbabe".to_string(),
            ])],
        };
        assert!(!log_matches_filter(&log, &filter_miss));
    }

    #[test]
    fn log_filter_topic_position_out_of_range() {
        // Log has only 1 topic, filter asks for topic at position 1
        let log = make_log("0xabc", &["0xdead"]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![TopicFilter::Any, TopicFilter::Exact("0xbeef".to_string())],
        };
        assert!(!log_matches_filter(&log, &filter));
    }

    #[test]
    fn log_filter_mixed_address_and_topics() {
        let log = make_log("0xabc", &["0xdead", "0xbeef"]);
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: Some(vec!["0xabc".to_string()]),
            topics: vec![
                TopicFilter::Exact("0xdead".to_string()),
                TopicFilter::Exact("0xbeef".to_string()),
            ],
        };
        assert!(log_matches_filter(&log, &filter));

        // Wrong address
        let filter_wrong_addr = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: Some(vec!["0xdef".to_string()]),
            topics: vec![TopicFilter::Exact("0xdead".to_string())],
        };
        assert!(!log_matches_filter(&log, &filter_wrong_addr));
    }

    #[test]
    fn get_logs_from_cache() {
        let mut cache = BlockCache::new();

        let log1 = make_log("0xaaa", &["0xevent1"]);
        let log2 = make_log("0xbbb", &["0xevent2"]);
        let log3 = make_log("0xaaa", &["0xevent3"]);

        let r1 = make_receipt_with_logs(
            "0xaa00000000000000000000000000000000000000000000000000000000000000",
            vec![log1.clone()],
        );
        let r2 = make_receipt_with_logs(
            "0xbb00000000000000000000000000000000000000000000000000000000000000",
            vec![log2.clone(), log3.clone()],
        );

        let h1 = [0xAA; 32];
        let h2 = [0xBB; 32];
        cache.insert_block(100, [0x01; 32], vec![], vec![(h1, r1)]);
        cache.insert_block(101, [0x02; 32], vec![], vec![(h2, r2)]);

        // All logs (no filter)
        let filter_all = LogFilter {
            from_block: 100,
            to_block: 101,
            addresses: None,
            topics: vec![],
        };
        assert_eq!(cache.get_logs(&filter_all).len(), 3);

        // Filter by address
        let filter_aaa = LogFilter {
            from_block: 100,
            to_block: 101,
            addresses: Some(vec!["0xaaa".to_string()]),
            topics: vec![],
        };
        let results = cache.get_logs(&filter_aaa);
        assert_eq!(results.len(), 2);

        // Filter by single block
        let filter_block = LogFilter {
            from_block: 101,
            to_block: 101,
            addresses: None,
            topics: vec![],
        };
        assert_eq!(cache.get_logs(&filter_block).len(), 2);
    }

    #[test]
    fn has_block_range_checks() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);
        cache.insert_block(101, [0x02; 32], vec![], vec![]);
        cache.insert_block(102, [0x03; 32], vec![], vec![]);

        assert!(cache.has_block_range(100, 102));
        assert!(cache.has_block_range(101, 101));
        assert!(!cache.has_block_range(99, 102));
        assert!(!cache.has_block_range(100, 103));
    }

    #[test]
    fn get_logs_empty_and_no_match() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);

        let filter = LogFilter {
            from_block: 100,
            to_block: 100,
            addresses: Some(vec!["0xnothere".to_string()]),
            topics: vec![],
        };
        assert!(cache.get_logs(&filter).is_empty());
    }

    #[test]
    fn logs_evicted_with_block() {
        let mut cache = BlockCache::new();
        for i in 0..=(MAX_CACHED_BLOCKS as u64) {
            let log = make_log(&format!("0x{:x}", i), &[]);
            let receipt = make_receipt_with_logs(&format!("0x{:0>64x}", i), vec![log]);
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            cache.insert_block(i, [i as u8; 32], vec![], vec![(hash, receipt)]);
        }

        // Block 0 should be evicted
        assert!(!cache.logs.contains_key(&0));
        // Latest block should still have logs
        assert!(cache.logs.contains_key(&(MAX_CACHED_BLOCKS as u64)));
    }

    #[test]
    fn logs_cleaned_on_invalidate() {
        let mut cache = BlockCache::new();
        let log = make_log("0xabc", &[]);
        let receipt = make_receipt_with_logs(
            "0xaa00000000000000000000000000000000000000000000000000000000000000",
            vec![log],
        );
        cache.insert_block(100, [0x01; 32], vec![], vec![([0xAA; 32], receipt)]);

        assert!(cache.logs.contains_key(&100));
        cache.invalidate_from(100);
        assert!(!cache.logs.contains_key(&100));
    }

    // --- Filter tests ---

    #[test]
    fn filter_log_get_changes() {
        let mut cache = BlockCache::new();

        // Insert initial block
        let log1 = make_log("0xaaa", &["0xevent1"]);
        let r1 = make_receipt_with_logs(
            "0xaa00000000000000000000000000000000000000000000000000000000000000",
            vec![log1],
        );
        cache.insert_block(100, [0x01; 32], vec![], vec![([0xAA; 32], r1)]);

        // Create log filter — cursor starts at latest (100)
        let filter = LogFilter {
            from_block: 100,
            to_block: u64::MAX,
            addresses: Some(vec!["0xaaa".to_string()]),
            topics: vec![],
        };
        let id = cache.create_log_filter(filter);

        // No new blocks yet — changes should be empty
        let changes = cache.get_filter_changes(&id).unwrap();
        assert!(changes.is_empty());

        // Insert a new block with a matching log
        let log2 = make_log("0xaaa", &["0xevent2"]);
        let r2 = make_receipt_with_logs(
            "0xbb00000000000000000000000000000000000000000000000000000000000000",
            vec![log2],
        );
        cache.insert_block(101, [0x02; 32], vec![], vec![([0xBB; 32], r2)]);

        let changes = cache.get_filter_changes(&id).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0]["address"].as_str().unwrap(), "0xaaa");

        // Polling again with no new blocks returns empty
        let changes = cache.get_filter_changes(&id).unwrap();
        assert!(changes.is_empty());
    }

    #[test]
    fn filter_block_get_changes() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);

        let id = cache.create_block_filter();

        // No new blocks
        let changes = cache.get_filter_changes(&id).unwrap();
        assert!(changes.is_empty());

        // Add two blocks
        cache.insert_block(101, [0x02; 32], vec![], vec![]);
        cache.insert_block(102, [0x03; 32], vec![], vec![]);

        let changes = cache.get_filter_changes(&id).unwrap();
        assert_eq!(changes.len(), 2);
        // Verify they're hex-encoded block hashes
        assert!(changes[0].as_str().unwrap().starts_with("0x"));
        assert!(changes[1].as_str().unwrap().starts_with("0x"));
    }

    #[test]
    fn filter_pending_tx_returns_empty() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);

        let id = cache.create_pending_tx_filter();
        let changes = cache.get_filter_changes(&id).unwrap();
        assert!(changes.is_empty());

        // Even after new blocks, still empty (no mempool)
        cache.insert_block(101, [0x02; 32], vec![], vec![]);
        let changes = cache.get_filter_changes(&id).unwrap();
        assert!(changes.is_empty());
    }

    #[test]
    fn filter_uninstall() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);

        let id = cache.create_block_filter();
        assert!(cache.uninstall_filter(&id));
        assert!(!cache.uninstall_filter(&id)); // already removed
        assert!(cache.get_filter_changes(&id).is_none()); // not found
    }

    #[test]
    fn filter_expired_cleanup() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);

        let id = cache.create_block_filter();

        // Manually set last_accessed far in the past
        cache.filters.get_mut(&id).unwrap().last_accessed =
            Instant::now() - std::time::Duration::from_secs(FILTER_EXPIRY_SECS + 1);

        // Next filter creation triggers cleanup
        let _id2 = cache.create_block_filter();
        assert!(cache.get_filter_changes(&id).is_none());
    }

    #[test]
    fn filter_get_filter_logs() {
        let mut cache = BlockCache::new();

        let log1 = make_log("0xaaa", &["0xevent1"]);
        let r1 = make_receipt_with_logs(
            "0xaa00000000000000000000000000000000000000000000000000000000000000",
            vec![log1],
        );
        cache.insert_block(100, [0x01; 32], vec![], vec![([0xAA; 32], r1)]);

        let log2 = make_log("0xaaa", &["0xevent2"]);
        let r2 = make_receipt_with_logs(
            "0xbb00000000000000000000000000000000000000000000000000000000000000",
            vec![log2],
        );
        cache.insert_block(101, [0x02; 32], vec![], vec![([0xBB; 32], r2)]);

        // Log filter from block 100
        let filter = LogFilter {
            from_block: 100,
            to_block: u64::MAX,
            addresses: Some(vec!["0xaaa".to_string()]),
            topics: vec![],
        };
        let id = cache.create_log_filter(filter);

        // get_filter_logs returns all matching logs from from_block to latest
        let result = cache.get_filter_logs(&id).unwrap().unwrap();
        assert_eq!(result.len(), 2);

        // Block filter returns Some(None) — not a log filter
        let block_id = cache.create_block_filter();
        let result = cache.get_filter_logs(&block_id);
        assert!(result.unwrap().is_none());

        // Unknown filter returns None
        assert!(cache.get_filter_logs("0xdeadbeef").is_none());
    }

    #[test]
    fn parse_log_filter_object_basic() {
        let obj = serde_json::json!({
            "fromBlock": "0x64",
            "toBlock": "0x65",
            "address": "0xABC",
            "topics": ["0xDEAD", null, ["0xA", "0xB"]]
        });
        let filter = parse_log_filter_object(&obj, 200).unwrap();
        assert_eq!(filter.from_block, 0x64);
        assert_eq!(filter.to_block, 0x65);
        assert_eq!(filter.addresses, Some(vec!["0xabc".to_string()]));
        assert_eq!(filter.topics.len(), 3);
    }

    #[test]
    fn parse_log_filter_object_rejects_reversed_range() {
        let obj = serde_json::json!({
            "fromBlock": "0x100",
            "toBlock": "0x50"
        });
        assert!(parse_log_filter_object(&obj, 200).is_err());
    }

    #[test]
    fn parse_log_filter_object_defaults_to_latest() {
        let obj = serde_json::json!({});
        let filter = parse_log_filter_object(&obj, 500).unwrap();
        assert_eq!(filter.from_block, 500);
        assert_eq!(filter.to_block, 500);
    }

    #[test]
    fn filter_ids_are_unique() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);

        let id1 = cache.create_block_filter();
        let id2 = cache.create_log_filter(LogFilter {
            from_block: 100,
            to_block: 100,
            addresses: None,
            topics: vec![],
        });
        let id3 = cache.create_pending_tx_filter();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn has_block_range_reversed() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);
        cache.insert_block(101, [0x02; 32], vec![], vec![]);
        assert!(!cache.has_block_range(101, 100));
    }

    #[test]
    fn topic_any_requires_position_exists() {
        // Filter [null] should NOT match a log with empty topics
        let log_no_topics = serde_json::json!({"address": "0xabc", "topics": [], "data": "0x"});
        let filter = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![TopicFilter::Any],
        };
        assert!(!log_matches_filter(&log_no_topics, &filter));

        // Filter [null, null] should NOT match a log with only 1 topic
        let log_one_topic = make_log("0xabc", &["0xdead"]);
        let filter2 = LogFilter {
            from_block: 0,
            to_block: 0,
            addresses: None,
            topics: vec![TopicFilter::Any, TopicFilter::Any],
        };
        assert!(!log_matches_filter(&log_one_topic, &filter2));

        // Filter [null] SHOULD match a log with 1 topic
        assert!(log_matches_filter(&log_one_topic, &filter));
    }

    #[test]
    fn filter_changes_respects_to_block() {
        let mut cache = BlockCache::new();
        cache.insert_block(100, [0x01; 32], vec![], vec![]);
        cache.insert_block(101, [0x02; 32], vec![], vec![]);

        // Create a log filter with to_block=100 when latest is already 101
        let filter = LogFilter {
            from_block: 99,
            to_block: 100,
            addresses: None,
            topics: vec![],
        };
        let id = cache.create_log_filter(filter);

        // get_filter_changes should return empty (cursor was set to 101, to_block is 100)
        let changes = cache.get_filter_changes(&id).unwrap();
        assert!(changes.is_empty());

        // Adding more blocks shouldn't change anything — to_block already passed
        cache.insert_block(102, [0x03; 32], vec![], vec![]);
        let changes = cache.get_filter_changes(&id).unwrap();
        assert!(changes.is_empty());
    }

    #[test]
    fn filter_logs_respects_to_block() {
        let mut cache = BlockCache::new();
        let log1 = make_log("0xaaa", &[]);
        let log2 = make_log("0xaaa", &[]);
        let r1 = make_receipt_with_logs(
            "0xaa00000000000000000000000000000000000000000000000000000000000000",
            vec![log1],
        );
        let r2 = make_receipt_with_logs(
            "0xbb00000000000000000000000000000000000000000000000000000000000000",
            vec![log2],
        );
        cache.insert_block(100, [0x01; 32], vec![], vec![([0xAA; 32], r1)]);
        cache.insert_block(101, [0x02; 32], vec![], vec![([0xBB; 32], r2)]);

        let filter = LogFilter {
            from_block: 100,
            to_block: 100,
            addresses: None,
            topics: vec![],
        };
        let id = cache.create_log_filter(filter);

        // get_filter_logs should only return logs up to to_block=100
        let result = cache.get_filter_logs(&id).unwrap().unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn parse_filter_rejects_invalid_address_type() {
        let obj = serde_json::json!({"address": 42});
        assert!(parse_log_filter_object(&obj, 100).is_err());
    }

    #[test]
    fn parse_filter_rejects_invalid_topic_type() {
        let obj = serde_json::json!({"topics": [42]});
        assert!(parse_log_filter_object(&obj, 100).is_err());
    }

    #[test]
    fn parse_filter_rejects_empty_topic_alternatives() {
        let obj = serde_json::json!({"topics": [[]]});
        assert!(parse_log_filter_object(&obj, 100).is_err());
    }

    #[test]
    fn empty_address_array_matches_nothing() {
        let obj = serde_json::json!({"address": []});
        let filter = parse_log_filter_object(&obj, 100).unwrap();
        let log = make_log("0xabc", &[]);
        assert!(!log_matches_filter(&log, &filter));
    }

    #[test]
    fn parse_filter_rejects_safe_finalized() {
        let obj = serde_json::json!({"fromBlock": "safe"});
        assert!(parse_log_filter_object(&obj, 100).is_err());
        let obj = serde_json::json!({"toBlock": "finalized"});
        assert!(parse_log_filter_object(&obj, 100).is_err());
    }
}
