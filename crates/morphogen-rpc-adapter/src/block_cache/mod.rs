use rand::Rng;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// Log filter parsing + matching extracted to `log_filter` (top-level module)
// in TASK-54.18. Re-exported here so `block_cache::LogFilter` /
// `block_cache::parse_log_filter_object` / `block_cache::log_matches_filter`
// keep resolving at all call sites.
#[cfg_attr(not(test), allow(unused_imports))]
pub use crate::log_filter::TopicFilter;
pub use crate::log_filter::{log_matches_filter, parse_log_filter_object, LogFilter};
// Receipt-fetch helpers (rpc_call, fetch_receipts, bounded fan-out)
// extracted to `receipt_fetch` (top-level module) in TASK-54.19.
// (Used only by block_poller.rs.)
// Block-cache poller (start_block_poller + poll_new_blocks) extracted
// to `block_poller` (top-level module) in TASK-54.20.
pub(crate) use crate::block_poller::start_block_poller;

#[cfg(test)]
pub(crate) use crate::receipt_fetch::{
    fetch_receipts_fallback_bounded, ReceiptFetchFuture, RECEIPT_FALLBACK_MAX_IN_FLIGHT,
};

/// Filter expiry timeout (5 minutes).
const FILTER_EXPIRY_SECS: u64 = 300;

/// Maximum number of blocks to keep in cache.
const MAX_CACHED_BLOCKS: usize = 64;

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
    pending_block_zero: bool,
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
        self.get_logs_in_range(filter, filter.from_block, filter.to_block)
    }

    fn get_logs_in_range(&self, filter: &LogFilter, from_block: u64, to_block: u64) -> Vec<Value> {
        if from_block > to_block {
            return Vec::new();
        }

        let mut result = Vec::new();
        for block in &self.cached_blocks {
            if block.number < from_block {
                continue;
            }
            if block.number > to_block {
                continue;
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
        // Use a sentinel for from_block=0 so the first poll can include block 0.
        let cursor = filter.from_block.checked_sub(1).unwrap_or(u64::MAX);
        self.filters.insert(
            hex_id.clone(),
            StoredFilter {
                kind: FilterKind::Log(filter),
                last_polled_block: cursor,
                pending_block_zero: cursor == u64::MAX,
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
                pending_block_zero: false,
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
                pending_block_zero: false,
                last_accessed: Instant::now(),
            },
        );
        hex_id
    }

    /// Get changes since last poll for the given filter.
    /// Returns None if the filter ID doesn't exist.
    pub fn get_filter_changes(&mut self, id: &str) -> Option<Vec<Value>> {
        self.cleanup_expired_filters();
        let has_cached_block_zero = self.cached_blocks.iter().any(|block| block.number == 0);

        enum PollAction {
            Log {
                from_inclusive: u64,
                to_inclusive: u64,
                include_block_zero: bool,
            },
            Block {
                from_exclusive: u64,
                to_inclusive: u64,
            },
            PendingTransaction,
        }

        let action = {
            let filter = self.filters.get_mut(id)?;
            filter.last_accessed = Instant::now();
            let from_exclusive = filter.last_polled_block;

            match &filter.kind {
                FilterKind::Log(log_filter) => {
                    let to_inclusive = self.latest_block.min(log_filter.to_block);
                    let mut from_inclusive = if from_exclusive == u64::MAX {
                        log_filter.from_block
                    } else {
                        from_exclusive.saturating_add(1)
                    };
                    let mut include_block_zero = false;

                    if filter.pending_block_zero {
                        if has_cached_block_zero {
                            // If the cursor has advanced past 0, include block 0 once
                            // separately so late insertion is not lost.
                            include_block_zero = from_inclusive > 0;
                            filter.pending_block_zero = false;
                        } else if from_inclusive == 0 {
                            // Keep block 0 pending, but still allow logs from higher blocks.
                            from_inclusive = 1;
                        }
                    }

                    if to_inclusive >= from_inclusive {
                        filter.last_polled_block = to_inclusive;
                    }
                    PollAction::Log {
                        from_inclusive,
                        to_inclusive,
                        include_block_zero,
                    }
                }
                FilterKind::Block => {
                    let to_inclusive = self.latest_block;
                    filter.last_polled_block = to_inclusive;
                    PollAction::Block {
                        from_exclusive,
                        to_inclusive,
                    }
                }
                FilterKind::PendingTransaction => {
                    filter.last_polled_block = self.latest_block;
                    PollAction::PendingTransaction
                }
            }
        };

        match action {
            PollAction::Log {
                from_inclusive,
                to_inclusive,
                include_block_zero,
            } => {
                let log_filter = match self.filters.get(id)? {
                    StoredFilter {
                        kind: FilterKind::Log(log_filter),
                        ..
                    } => log_filter,
                    // Preserve "filter exists" semantics even if kind unexpectedly diverges.
                    _ => return Some(vec![]),
                };
                let mut logs = if include_block_zero {
                    self.get_logs_in_range(log_filter, 0, 0)
                } else {
                    Vec::new()
                };
                if to_inclusive >= from_inclusive {
                    logs.extend(self.get_logs_in_range(log_filter, from_inclusive, to_inclusive));
                }
                Some(logs)
            }
            PollAction::Block {
                from_exclusive,
                to_inclusive,
            } => Some(self.get_block_hashes_in_range(from_exclusive, to_inclusive)),
            PollAction::PendingTransaction => Some(vec![]),
        }
    }

    /// Get all matching logs for a log filter (used by eth_getFilterLogs).
    /// Returns None if filter ID not found.
    /// Returns Some(None) if filter exists but is not a log filter.
    /// Returns Some(Some(logs)) with matching logs.
    pub fn get_filter_logs(&mut self, id: &str) -> Option<Option<Vec<Value>>> {
        self.cleanup_expired_filters();
        let (from_block, to_block) = {
            let filter = self.filters.get_mut(id)?;
            filter.last_accessed = Instant::now();

            match &filter.kind {
                FilterKind::Log(log_filter) => (
                    log_filter.from_block,
                    self.latest_block.min(log_filter.to_block),
                ),
                _ => return Some(None),
            }
        };

        let log_filter = match self.filters.get(id)? {
            StoredFilter {
                kind: FilterKind::Log(log_filter),
                ..
            } => log_filter,
            // Preserve contract: existing non-log filters map to Some(None), not None.
            _ => return Some(None),
        };

        Some(Some(
            self.get_logs_in_range(log_filter, from_block, to_block),
        ))
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

#[cfg(test)]
mod tests;
