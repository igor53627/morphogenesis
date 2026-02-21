use crate::{
    aggregate_responses, generate_query, AccountData, AggregationError, EpochMetadata,
    ServerResponse, StorageData, QUERIES_PER_REQUEST,
};
use anyhow::{anyhow, Result};
use rand::thread_rng;
use serde::Deserialize;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RETRIES: u32 = 2;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(200);
const DEFAULT_CACHE_CAPACITY: usize = 4096;
const DEFAULT_CACHE_SHARDS: usize = 32;
/// Maximum queries per batch request, matching server's MAX_BATCH_SIZE.
const MAX_BATCH_SIZE: usize = 32;

fn storage_cuckoo_key(address: [u8; 20], slot: [u8; 32]) -> [u8; 8] {
    use alloy_primitives::keccak256;

    let mut storage_key = [0u8; 52];
    storage_key[0..20].copy_from_slice(&address);
    storage_key[20..52].copy_from_slice(&slot);

    let tag = keccak256(storage_key);
    let mut out = [0u8; 8];
    out.copy_from_slice(&tag[0..8]);
    out
}

/// Returns the 8-byte key used for storage query generation.
/// This is the Cuckoo key derived as keccak(address || slot)[0..8].
fn storage_query_key(address: [u8; 20], slot: [u8; 32]) -> [u8; 8] {
    storage_cuckoo_key(address, slot)
}

#[derive(Deserialize)]
struct RawEpochResponse {
    epoch_id: u64,
    num_rows: usize,
    seeds: [u64; 3],
    block_number: u64,
    state_root: String,
}

/// Cache for PIR query results, invalidated on epoch rotation.
struct PirCacheShard {
    epoch_id: u64,
    entries: HashMap<Vec<u8>, [Vec<u8>; QUERIES_PER_REQUEST]>,
}

impl PirCacheShard {
    fn new(epoch_id: u64) -> Self {
        Self {
            epoch_id,
            entries: HashMap::new(),
        }
    }
}

/// Sharded cache to reduce contention under concurrent query load.
struct PirCache {
    epoch_id: AtomicU64,
    shards: Vec<Mutex<PirCacheShard>>,
    shard_mask: usize,
    capacity_per_shard: usize,
}

impl PirCache {
    fn new(capacity: usize) -> Self {
        Self::with_shards(capacity, DEFAULT_CACHE_SHARDS)
    }

    fn with_shards(capacity: usize, shard_count: usize) -> Self {
        assert!(capacity > 0, "cache capacity must be > 0");
        assert!(shard_count > 0, "cache shard count must be > 0");

        let shard_count = shard_count.next_power_of_two();
        let capacity_per_shard = capacity.div_ceil(shard_count).max(1);
        let mut shards = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            shards.push(Mutex::new(PirCacheShard::new(0)));
        }

        Self {
            epoch_id: AtomicU64::new(0),
            shards,
            shard_mask: shard_count - 1,
            capacity_per_shard,
        }
    }

    fn shard_index(&self, key: &[u8]) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) & self.shard_mask
    }

    fn advance_epoch_if_needed(&self, epoch_id: u64) -> u64 {
        let mut current = self.epoch_id.load(Ordering::Acquire);
        while current < epoch_id {
            match self.epoch_id.compare_exchange(
                current,
                epoch_id,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return current,
                Err(observed) => current = observed,
            }
        }
        current
    }

    fn sync_shard_epoch(&self, shard: &mut PirCacheShard, shard_idx: usize, epoch_id: u64) {
        if shard.epoch_id != epoch_id {
            debug!(
                shard = shard_idx,
                old_epoch = shard.epoch_id,
                new_epoch = epoch_id,
                evicted = shard.entries.len(),
                "Cache shard invalidated on epoch rotation"
            );
            shard.entries.clear();
            shard.epoch_id = epoch_id;
        }
    }

    async fn get(&self, epoch_id: u64, key: &[u8]) -> Option<[Vec<u8>; QUERIES_PER_REQUEST]> {
        if self.epoch_id.load(Ordering::Acquire) != epoch_id {
            return None;
        }

        let shard_idx = self.shard_index(key);
        let mut shard = self.shards[shard_idx].lock().await;
        self.sync_shard_epoch(&mut shard, shard_idx, epoch_id);
        shard.entries.get(key).cloned()
    }

    async fn put(&self, epoch_id: u64, key: Vec<u8>, payloads: [Vec<u8>; QUERIES_PER_REQUEST]) {
        let current_epoch = self.advance_epoch_if_needed(epoch_id);
        if current_epoch > epoch_id {
            debug!(
                stale_epoch = epoch_id,
                current_epoch, "Skipping cache insert for stale epoch"
            );
            return;
        }

        let shard_idx = self.shard_index(&key);
        let mut shard = self.shards[shard_idx].lock().await;
        self.sync_shard_epoch(&mut shard, shard_idx, epoch_id);

        if shard.entries.len() >= self.capacity_per_shard {
            debug!(
                shard = shard_idx,
                capacity_per_shard = self.capacity_per_shard,
                "Cache shard full, clearing"
            );
            shard.entries.clear();
        }
        shard.entries.insert(key, payloads);
    }

    #[cfg(test)]
    async fn total_entries(&self) -> usize {
        let mut total = 0;
        for shard in &self.shards {
            total += shard.lock().await.entries.len();
        }
        total
    }
}

/// Internal error type for PIR queries, enabling structured retry decisions.
enum PirQueryError {
    /// Transient network error (timeout, connection refused)
    Transient(String),
    /// Servers returned different epochs from each other
    EpochMismatch { server_a: u64, server_b: u64 },
    /// Server response epoch doesn't match the metadata epoch used for query
    StaleMetadata { expected: u64, got: u64 },
    /// Non-retryable error
    Permanent(String),
}

impl std::fmt::Display for PirQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient(msg) => write!(f, "transient: {}", msg),
            Self::EpochMismatch { server_a, server_b } => {
                write!(
                    f,
                    "epoch mismatch: server_a={}, server_b={}",
                    server_a, server_b
                )
            }
            Self::StaleMetadata { expected, got } => {
                write!(
                    f,
                    "stale metadata: expected epoch {}, servers returned {}",
                    expected, got
                )
            }
            Self::Permanent(msg) => write!(f, "{}", msg),
        }
    }
}

impl PirQueryError {
    fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Transient(_) | Self::EpochMismatch { .. } | Self::StaleMetadata { .. }
        )
    }
}

pub struct PirClient {
    server_a_url: String,
    server_b_url: String,
    http_client: reqwest::Client,
    metadata: RwLock<Option<Arc<EpochMetadata>>>,
    cache: PirCache,
}

/// Determines whether a reqwest error is transient and worth retrying.
fn is_transient(err: &reqwest::Error) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        err.is_timeout()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        err.is_connect() || err.is_timeout()
    }
}

impl PirClient {
    pub fn new(server_a_url: String, server_b_url: String) -> Self {
        let builder = reqwest::Client::builder();
        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
            .timeout(DEFAULT_REQUEST_TIMEOUT);

        let http_client = builder.build().unwrap_or_else(|e| {
            warn!(
                "Failed to build HTTP client with timeouts ({}), using defaults",
                e
            );
            reqwest::Client::new()
        });

        Self {
            server_a_url,
            server_b_url,
            http_client,
            metadata: RwLock::new(None),
            cache: PirCache::new(DEFAULT_CACHE_CAPACITY),
        }
    }

    pub async fn update_metadata(&self) -> Result<Arc<EpochMetadata>> {
        let url = format!("{}/epoch", self.server_a_url);
        let resp = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to fetch epoch metadata from {}: {}", url, e))?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Epoch metadata request failed with status {}",
                resp.status()
            ));
        }

        let resp: RawEpochResponse = resp
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse epoch metadata: {}", e))?;

        let mut state_root = [0u8; 32];
        let hex_root = resp
            .state_root
            .strip_prefix("0x")
            .unwrap_or(&resp.state_root);
        hex::decode_to_slice(hex_root, &mut state_root)?;

        let metadata = Arc::new(EpochMetadata {
            epoch_id: resp.epoch_id,
            num_rows: resp.num_rows,
            seeds: resp.seeds,
            block_number: resp.block_number,
            state_root,
        });

        info!(
            epoch_id = metadata.epoch_id,
            block = metadata.block_number,
            "Updated PIR metadata"
        );

        let mut lock = self.metadata.write().await;
        *lock = Some(metadata.clone());

        Ok(metadata)
    }

    pub async fn get_metadata(&self) -> Result<Arc<EpochMetadata>> {
        let lock = self.metadata.read().await;
        if let Some(metadata) = &*lock {
            return Ok(metadata.clone());
        }
        drop(lock);
        self.update_metadata().await
    }

    /// Send a PIR query to both servers and aggregate the results.
    /// Returns (response_epoch_id, aggregated_payloads).
    async fn execute_pir_query(
        &self,
        key: &[u8],
        metadata: &EpochMetadata,
    ) -> Result<(u64, [Vec<u8>; QUERIES_PER_REQUEST]), PirQueryError> {
        let query_keys = {
            let mut rng = thread_rng();
            generate_query(&mut rng, key, metadata)
        };

        let req_a = self.http_client.post(format!("{}/query", self.server_a_url))
            .json(&serde_json::json!({
                "keys": query_keys.keys_a.iter().map(|k| format!("0x{}", hex::encode(k.to_bytes()))).collect::<Vec<_>>()
            }))
            .send();

        let req_b = self.http_client.post(format!("{}/query", self.server_b_url))
            .json(&serde_json::json!({
                "keys": query_keys.keys_b.iter().map(|k| format!("0x{}", hex::encode(k.to_bytes()))).collect::<Vec<_>>()
            }))
            .send();

        let (resp_a, resp_b) = tokio::try_join!(req_a, req_b).map_err(|e| {
            if is_transient(&e) {
                PirQueryError::Transient(e.to_string())
            } else {
                PirQueryError::Permanent(format!("PIR server request failed: {}", e))
            }
        })?;

        // Check HTTP status before parsing JSON
        if !resp_a.status().is_success() {
            let status = resp_a.status();
            return Err(if status.is_server_error() {
                PirQueryError::Transient(format!("Server A returned {}", status))
            } else {
                PirQueryError::Permanent(format!("Server A returned {}", status))
            });
        }
        if !resp_b.status().is_success() {
            let status = resp_b.status();
            return Err(if status.is_server_error() {
                PirQueryError::Transient(format!("Server B returned {}", status))
            } else {
                PirQueryError::Permanent(format!("Server B returned {}", status))
            });
        }

        #[derive(Deserialize)]
        struct QueryResponse {
            epoch_id: u64,
            payloads: Vec<String>,
        }

        let json_a: QueryResponse = resp_a.json().await.map_err(|e| {
            PirQueryError::Permanent(format!("Failed to parse server A response: {}", e))
        })?;
        let json_b: QueryResponse = resp_b.json().await.map_err(|e| {
            PirQueryError::Permanent(format!("Failed to parse server B response: {}", e))
        })?;

        let parse_payloads =
            |payloads: Vec<String>| -> Result<[Vec<u8>; QUERIES_PER_REQUEST], PirQueryError> {
                if payloads.len() != QUERIES_PER_REQUEST {
                    return Err(PirQueryError::Permanent(format!(
                        "Invalid payload count: got {}, expected {}",
                        payloads.len(),
                        QUERIES_PER_REQUEST
                    )));
                }
                let mut out: [Vec<u8>; QUERIES_PER_REQUEST] = Default::default();
                for (i, p) in payloads.into_iter().enumerate() {
                    let hex_p = p.strip_prefix("0x").unwrap_or(&p);
                    out[i] = hex::decode(hex_p).map_err(|e| {
                        PirQueryError::Permanent(format!("Hex decode error: {}", e))
                    })?;
                }
                Ok(out)
            };

        let server_resp_a = ServerResponse {
            epoch_id: json_a.epoch_id,
            payloads: parse_payloads(json_a.payloads)?,
        };

        let server_resp_b = ServerResponse {
            epoch_id: json_b.epoch_id,
            payloads: parse_payloads(json_b.payloads)?,
        };

        let aggregated =
            aggregate_responses(&server_resp_a, &server_resp_b).map_err(|e| match e {
                AggregationError::EpochMismatch { server_a, server_b } => {
                    PirQueryError::EpochMismatch { server_a, server_b }
                }
                AggregationError::PayloadLengthMismatch {
                    index,
                    len_a,
                    len_b,
                } => PirQueryError::Permanent(format!(
                    "Payload length mismatch at index {}: a={}, b={}",
                    index, len_a, len_b
                )),
            })?;

        let response_epoch = aggregated.epoch_id;

        // Validate that server response epoch matches metadata epoch
        if response_epoch != metadata.epoch_id {
            return Err(PirQueryError::StaleMetadata {
                expected: metadata.epoch_id,
                got: response_epoch,
            });
        }

        Ok((response_epoch, aggregated.payloads))
    }

    /// Execute a PIR query with caching and retry logic.
    /// Checks cache first, falls back to network query with retries.
    /// On epoch mismatch or stale metadata, refreshes metadata and retries.
    async fn execute_pir_query_with_retry(
        &self,
        key: &[u8],
    ) -> Result<[Vec<u8>; QUERIES_PER_REQUEST]> {
        let metadata = self.get_metadata().await?;

        // Check cache first
        if let Some(cached) = self.cache.get(metadata.epoch_id, key).await {
            debug!("PIR cache hit");
            return Ok(cached);
        }

        for attempt in 0..=MAX_RETRIES {
            let current_metadata = if attempt == 0 {
                metadata.clone()
            } else {
                debug!(attempt, "Retrying PIR query, refreshing metadata");
                match self.update_metadata().await {
                    Ok(m) => m,
                    Err(e) => {
                        warn!(attempt, error = %e, "Failed to refresh metadata on retry");
                        metadata.clone()
                    }
                }
            };

            match self.execute_pir_query(key, &current_metadata).await {
                Ok((response_epoch, payloads)) => {
                    // Cache under the response epoch (validated to match metadata)
                    self.cache
                        .put(response_epoch, key.to_vec(), payloads.clone())
                        .await;
                    return Ok(payloads);
                }
                Err(e) => {
                    if attempt < MAX_RETRIES && e.is_retryable() {
                        let delay = RETRY_BASE_DELAY * 2u32.pow(attempt);
                        warn!(
                            attempt,
                            delay_ms = delay.as_millis() as u64,
                            error = %e,
                            "PIR query failed, retrying"
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }

                    return Err(anyhow!("PIR query failed: {}", e));
                }
            }
        }

        Err(anyhow!("PIR query failed after {} retries", MAX_RETRIES))
    }

    /// Send a batch of PIR queries to both servers and aggregate the results.
    /// Returns (response_epoch_id, Vec<aggregated_payloads>).
    async fn execute_batch_pir_query(
        &self,
        keys: &[Vec<u8>],
        metadata: &EpochMetadata,
    ) -> Result<(u64, Vec<[Vec<u8>; QUERIES_PER_REQUEST]>), PirQueryError> {
        let n = keys.len();
        if n == 0 {
            return Ok((metadata.epoch_id, vec![]));
        }

        // Generate N query key sets
        let mut all_query_keys = Vec::with_capacity(n);
        {
            let mut rng = thread_rng();
            for key in keys {
                all_query_keys.push(generate_query(&mut rng, key, metadata));
            }
        }

        // Build batch requests for each server
        let queries_a: Vec<serde_json::Value> = all_query_keys
            .iter()
            .map(|qk| {
                serde_json::json!({
                    "keys": qk.keys_a.iter()
                        .map(|k| format!("0x{}", hex::encode(k.to_bytes())))
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let queries_b: Vec<serde_json::Value> = all_query_keys
            .iter()
            .map(|qk| {
                serde_json::json!({
                    "keys": qk.keys_b.iter()
                        .map(|k| format!("0x{}", hex::encode(k.to_bytes())))
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let req_a = self
            .http_client
            .post(format!("{}/query/batch", self.server_a_url))
            .json(&serde_json::json!({ "queries": queries_a }))
            .send();

        let req_b = self
            .http_client
            .post(format!("{}/query/batch", self.server_b_url))
            .json(&serde_json::json!({ "queries": queries_b }))
            .send();

        let (resp_a, resp_b) = tokio::try_join!(req_a, req_b).map_err(|e| {
            if is_transient(&e) {
                PirQueryError::Transient(e.to_string())
            } else {
                PirQueryError::Permanent(format!("PIR batch request failed: {}", e))
            }
        })?;

        if !resp_a.status().is_success() {
            let status = resp_a.status();
            return Err(if status.is_server_error() {
                PirQueryError::Transient(format!("Server A returned {}", status))
            } else {
                PirQueryError::Permanent(format!("Server A returned {}", status))
            });
        }
        if !resp_b.status().is_success() {
            let status = resp_b.status();
            return Err(if status.is_server_error() {
                PirQueryError::Transient(format!("Server B returned {}", status))
            } else {
                PirQueryError::Permanent(format!("Server B returned {}", status))
            });
        }

        #[derive(Deserialize)]
        struct BatchResult {
            payloads: Vec<String>,
        }

        #[derive(Deserialize)]
        struct BatchResponse {
            epoch_id: u64,
            results: Vec<BatchResult>,
        }

        let json_a: BatchResponse = resp_a.json().await.map_err(|e| {
            PirQueryError::Permanent(format!("Failed to parse server A batch response: {}", e))
        })?;
        let json_b: BatchResponse = resp_b.json().await.map_err(|e| {
            PirQueryError::Permanent(format!("Failed to parse server B batch response: {}", e))
        })?;

        // Validate epoch consistency
        if json_a.epoch_id != json_b.epoch_id {
            return Err(PirQueryError::EpochMismatch {
                server_a: json_a.epoch_id,
                server_b: json_b.epoch_id,
            });
        }
        if json_a.epoch_id != metadata.epoch_id {
            return Err(PirQueryError::StaleMetadata {
                expected: metadata.epoch_id,
                got: json_a.epoch_id,
            });
        }

        if json_a.results.len() != n || json_b.results.len() != n {
            return Err(PirQueryError::Permanent(format!(
                "Batch result count mismatch: expected {}, got a={}, b={}",
                n,
                json_a.results.len(),
                json_b.results.len()
            )));
        }

        let parse_payloads =
            |payloads: Vec<String>| -> Result<[Vec<u8>; QUERIES_PER_REQUEST], PirQueryError> {
                if payloads.len() != QUERIES_PER_REQUEST {
                    return Err(PirQueryError::Permanent(format!(
                        "Invalid payload count: got {}, expected {}",
                        payloads.len(),
                        QUERIES_PER_REQUEST
                    )));
                }
                let mut out: [Vec<u8>; QUERIES_PER_REQUEST] = Default::default();
                for (i, p) in payloads.into_iter().enumerate() {
                    let hex_p = p.strip_prefix("0x").unwrap_or(&p);
                    out[i] = hex::decode(hex_p).map_err(|e| {
                        PirQueryError::Permanent(format!("Hex decode error: {}", e))
                    })?;
                }
                Ok(out)
            };

        // XOR corresponding payloads
        let mut aggregated = Vec::with_capacity(n);
        for (result_a, result_b) in json_a.results.into_iter().zip(json_b.results.into_iter()) {
            let payloads_a = parse_payloads(result_a.payloads)?;
            let payloads_b = parse_payloads(result_b.payloads)?;

            let mut xored: [Vec<u8>; QUERIES_PER_REQUEST] = Default::default();
            for i in 0..QUERIES_PER_REQUEST {
                if payloads_a[i].len() != payloads_b[i].len() {
                    return Err(PirQueryError::Permanent(format!(
                        "Payload length mismatch at index {}: a={}, b={}",
                        i,
                        payloads_a[i].len(),
                        payloads_b[i].len()
                    )));
                }
                xored[i] = payloads_a[i]
                    .iter()
                    .zip(payloads_b[i].iter())
                    .map(|(&a, &b)| a ^ b)
                    .collect();
            }
            aggregated.push(xored);
        }

        Ok((json_a.epoch_id, aggregated))
    }

    /// Execute a batch of PIR queries with caching and retry logic.
    /// Checks cache for each key; only queries misses over the network.
    /// Chunks requests to respect server's MAX_BATCH_SIZE (32).
    async fn execute_batch_pir_query_with_retry(
        &self,
        keys: &[Vec<u8>],
    ) -> Result<Vec<[Vec<u8>; QUERIES_PER_REQUEST]>> {
        let n = keys.len();
        if n == 0 {
            return Ok(vec![]);
        }

        for attempt in 0..=MAX_RETRIES {
            let current_metadata = if attempt == 0 {
                self.get_metadata().await?
            } else {
                debug!(attempt, "Retrying batch PIR query, refreshing metadata");
                match self.update_metadata().await {
                    Ok(m) => m,
                    Err(e) => {
                        warn!(attempt, error = %e, "Failed to refresh metadata on retry");
                        self.get_metadata().await?
                    }
                }
            };

            // Partition into cached hits and uncached misses using current epoch
            let mut results: Vec<Option<[Vec<u8>; QUERIES_PER_REQUEST]>> = vec![None; n];
            let mut miss_indices: Vec<usize> = Vec::new();
            let mut miss_keys: Vec<Vec<u8>> = Vec::new();

            for (i, key) in keys.iter().enumerate() {
                if let Some(cached) = self.cache.get(current_metadata.epoch_id, key).await {
                    debug!("PIR batch cache hit for key {}", i);
                    results[i] = Some(cached);
                } else {
                    miss_indices.push(i);
                    miss_keys.push(key.clone());
                }
            }

            if miss_keys.is_empty() {
                return Ok(results.into_iter().map(|r| r.unwrap()).collect());
            }

            // Chunk misses into batches of MAX_BATCH_SIZE to respect server limits
            let mut all_batch_results: Vec<[Vec<u8>; QUERIES_PER_REQUEST]> =
                Vec::with_capacity(miss_keys.len());
            let mut chunk_error: Option<PirQueryError> = None;

            for chunk in miss_keys.chunks(MAX_BATCH_SIZE) {
                match self.execute_batch_pir_query(chunk, &current_metadata).await {
                    Ok((response_epoch, batch_results)) => {
                        for (j, key) in chunk.iter().enumerate() {
                            self.cache
                                .put(response_epoch, key.clone(), batch_results[j].clone())
                                .await;
                        }
                        all_batch_results.extend(batch_results);
                    }
                    Err(e) => {
                        chunk_error = Some(e);
                        break;
                    }
                }
            }

            if let Some(e) = chunk_error {
                if attempt < MAX_RETRIES && e.is_retryable() {
                    let delay = RETRY_BASE_DELAY * 2u32.pow(attempt);
                    warn!(
                        attempt,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "Batch PIR query failed, retrying"
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                return Err(anyhow!("Batch PIR query failed: {}", e));
            }

            // Reassemble results in original order
            for (j, miss_idx) in miss_indices.iter().enumerate() {
                results[*miss_idx] = Some(all_batch_results[j].clone());
            }
            return Ok(results.into_iter().map(|r| r.unwrap()).collect());
        }

        Err(anyhow!(
            "Batch PIR query failed after {} retries",
            MAX_RETRIES
        ))
    }

    pub async fn query_accounts_batch(&self, addresses: &[[u8; 20]]) -> Result<Vec<AccountData>> {
        let keys: Vec<Vec<u8>> = addresses.iter().map(|a| a.to_vec()).collect();
        let all_payloads = self.execute_batch_pir_query_with_retry(&keys).await?;

        let mut results = Vec::with_capacity(addresses.len());
        for payloads in &all_payloads {
            let mut found = false;
            for payload in payloads.iter() {
                if let Some(data) = AccountData::from_bytes(payload) {
                    if data.balance > 0 || data.nonce > 0 || data.code_id.unwrap_or(0) > 0 {
                        results.push(data);
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                results.push(AccountData {
                    nonce: 0,
                    balance: 0,
                    code_hash: None,
                    code_id: None,
                });
            }
        }

        Ok(results)
    }

    pub async fn query_storages_batch(
        &self,
        queries: &[([u8; 20], [u8; 32])],
    ) -> Result<Vec<StorageData>> {
        use alloy_primitives::keccak256;

        let keys: Vec<Vec<u8>> = queries
            .iter()
            .map(|(address, slot)| storage_query_key(*address, *slot).to_vec())
            .collect();

        let all_payloads = self.execute_batch_pir_query_with_retry(&keys).await?;

        let mut results = Vec::with_capacity(queries.len());
        for (i, payloads) in all_payloads.iter().enumerate() {
            let (address, slot) = &queries[i];

            // Compute expected tag
            let mut storage_key = [0u8; 52];
            storage_key[0..20].copy_from_slice(address);
            storage_key[20..52].copy_from_slice(slot);
            let tag_hash = keccak256(storage_key);
            let expected_tag: [u8; 8] = tag_hash[0..8].try_into().unwrap();

            let mut found = false;

            // Find payload with matching tag (Optimized48)
            for payload in payloads.iter() {
                if let Some(data) = StorageData::from_bytes(payload) {
                    if let Some(tag) = data.tag {
                        if tag == expected_tag {
                            results.push(data);
                            found = true;
                            break;
                        }
                    }
                }
            }

            if !found {
                // Legacy fallback: single non-zero 32-byte payload
                let mut legacy_candidates: Vec<StorageData> = Vec::new();
                for payload in payloads.iter() {
                    if let Some(data) = StorageData::from_bytes(payload) {
                        if data.tag.is_none() && data.value != [0u8; 32] {
                            legacy_candidates.push(data);
                        }
                    }
                }

                if legacy_candidates.len() == 1 {
                    results.push(legacy_candidates.into_iter().next().unwrap());
                } else {
                    results.push(StorageData {
                        value: [0u8; 32],
                        tag: None,
                    });
                }
            }
        }

        Ok(results)
    }

    pub async fn query_account(&self, address: [u8; 20]) -> Result<AccountData> {
        let payloads = self.execute_pir_query_with_retry(&address).await?;

        for payload in payloads.iter() {
            if let Some(data) = AccountData::from_bytes(payload) {
                if data.balance > 0 || data.nonce > 0 || data.code_id.unwrap_or(0) > 0 {
                    return Ok(data);
                }
            }
        }

        Ok(AccountData {
            nonce: 0,
            balance: 0,
            code_hash: None,
            code_id: None,
        })
    }

    pub async fn query_storage(&self, address: [u8; 20], slot: [u8; 32]) -> Result<StorageData> {
        // Construct 52-byte storage key for tag verification: Address (20) + SlotKey (32)
        let mut storage_key = [0u8; 52];
        storage_key[0..20].copy_from_slice(&address);
        storage_key[20..52].copy_from_slice(&slot);

        // Use 8-byte Cuckoo key for query generation
        let cuckoo_key = storage_query_key(address, slot);
        let payloads = self.execute_pir_query_with_retry(&cuckoo_key).await?;

        // Compute expected tag: Keccak256(storage_key)[0..8]
        use alloy_primitives::keccak256;
        let tag_hash = keccak256(storage_key);
        let expected_tag: [u8; 8] = tag_hash[0..8].try_into().unwrap();

        // Find payload with matching tag (Optimized48 scheme)
        for payload in payloads.iter() {
            if let Some(data) = StorageData::from_bytes(payload) {
                if let Some(tag) = data.tag {
                    if tag == expected_tag {
                        return Ok(data);
                    }
                }
            }
        }

        // Fallback: Check for non-zero legacy payloads (32-byte, no tag)
        // Only accept if exactly ONE non-zero legacy payload exists to avoid ambiguity
        let mut legacy_candidates: Vec<StorageData> = Vec::new();
        for payload in payloads.iter() {
            if let Some(data) = StorageData::from_bytes(payload) {
                if data.tag.is_none() && data.value != [0u8; 32] {
                    legacy_candidates.push(data);
                }
            }
        }

        if legacy_candidates.len() == 1 {
            return Ok(legacy_candidates.into_iter().next().unwrap());
        }

        // Return zero value if not found or ambiguous
        Ok(StorageData {
            value: [0u8; 32],
            tag: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn storage_key_hashes_to_8_bytes() {
        let address = [0x11u8; 20];
        let slot = [0x22u8; 32];
        let expected_tag = [0x2b, 0xc0, 0xe4, 0x45, 0x3d, 0x02, 0x2b, 0x3e];

        let actual = storage_cuckoo_key(address, slot);
        assert_eq!(actual, expected_tag);
    }

    #[test]
    fn storage_query_uses_8_byte_cuckoo_key() {
        let address = [0x33u8; 20];
        let slot = [0x44u8; 32];

        let query_key = storage_query_key(address, slot);
        assert_eq!(
            query_key.len(),
            8,
            "storage query must use 8-byte Cuckoo key"
        );

        let expected = storage_cuckoo_key(address, slot);
        assert_eq!(query_key, expected);
    }

    fn make_payloads() -> [Vec<u8>; QUERIES_PER_REQUEST] {
        std::array::from_fn(|i| vec![i as u8 + 1])
    }

    struct SingleLockPirCache {
        epoch_id: u64,
        entries: HashMap<Vec<u8>, [Vec<u8>; QUERIES_PER_REQUEST]>,
        capacity: usize,
    }

    impl SingleLockPirCache {
        fn new(capacity: usize) -> Self {
            Self {
                epoch_id: 0,
                entries: HashMap::new(),
                capacity,
            }
        }

        fn get(&self, epoch_id: u64, key: &[u8]) -> Option<[Vec<u8>; QUERIES_PER_REQUEST]> {
            if self.epoch_id != epoch_id {
                return None;
            }
            self.entries.get(key).cloned()
        }

        fn put(&mut self, epoch_id: u64, key: Vec<u8>, payloads: [Vec<u8>; QUERIES_PER_REQUEST]) {
            if self.epoch_id != epoch_id {
                self.entries.clear();
                self.epoch_id = epoch_id;
            }
            if self.entries.len() >= self.capacity {
                self.entries.clear();
            }
            self.entries.insert(key, payloads);
        }
    }

    #[tokio::test]
    async fn cache_invalidates_on_epoch_change() {
        let cache = PirCache::with_shards(100, 1);
        let key = vec![0x11; 20];
        let payloads = make_payloads();

        cache.put(1, key.clone(), payloads.clone()).await;
        assert!(cache.get(1, &key).await.is_some());

        // Same epoch, different key
        assert!(cache.get(1, &[0x22; 20]).await.is_none());

        // Different epoch: get returns None
        assert!(cache.get(2, &key).await.is_none());

        // put with new epoch clears old entries
        cache.put(2, vec![0xAA], make_payloads()).await;
        assert!(cache.get(2, &[0xAA]).await.is_some());
        assert!(
            cache.get(2, &key).await.is_none(),
            "old key should be cleared after epoch rotation"
        );
    }

    #[tokio::test]
    async fn cache_evicts_when_full() {
        let cache = PirCache::with_shards(2, 1);
        let payloads = make_payloads();

        cache.put(1, vec![0x01], payloads.clone()).await;
        cache.put(1, vec![0x02], payloads.clone()).await;
        assert_eq!(cache.total_entries().await, 2);

        // Third insert triggers clear
        cache.put(1, vec![0x03], payloads.clone()).await;
        assert_eq!(cache.total_entries().await, 1);
        assert!(cache.get(1, &[0x03]).await.is_some());
    }

    #[test]
    fn pir_query_error_retryable() {
        assert!(PirQueryError::Transient("timeout".into()).is_retryable());
        assert!(PirQueryError::EpochMismatch {
            server_a: 1,
            server_b: 2
        }
        .is_retryable());
        assert!(PirQueryError::StaleMetadata {
            expected: 1,
            got: 2
        }
        .is_retryable());
        assert!(!PirQueryError::Permanent("bad data".into()).is_retryable());
    }

    #[tokio::test]
    async fn batch_cache_partitioning() {
        let client = PirClient::new(
            "http://localhost:1".to_string(),
            "http://localhost:2".to_string(),
        );

        // Pre-populate cache with one key
        let epoch_id = 42;
        let key1 = vec![0x11; 20];
        let payloads1 = make_payloads();

        client
            .cache
            .put(epoch_id, key1.clone(), payloads1.clone())
            .await;

        // Set metadata so cache can match
        {
            let mut lock = client.metadata.write().await;
            *lock = Some(Arc::new(EpochMetadata {
                epoch_id,
                num_rows: 1000,
                seeds: [1, 2, 3],
                block_number: 100,
                state_root: [0u8; 32],
            }));
        }

        // Verify cache hit works
        assert!(client.cache.get(epoch_id, &key1).await.is_some());
    }

    #[tokio::test]
    async fn batch_empty_returns_empty() {
        let client = PirClient::new(
            "http://localhost:1".to_string(),
            "http://localhost:2".to_string(),
        );

        // Set metadata
        {
            let mut lock = client.metadata.write().await;
            *lock = Some(Arc::new(EpochMetadata {
                epoch_id: 1,
                num_rows: 1000,
                seeds: [1, 2, 3],
                block_number: 100,
                state_root: [0u8; 32],
            }));
        }

        let keys: Vec<Vec<u8>> = vec![];
        let result = client.execute_batch_pir_query_with_retry(&keys).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn sharded_cache_supports_concurrent_put_get() {
        let cache = Arc::new(PirCache::with_shards(256, 16));
        let epoch_id = 9u64;
        let mut workers = Vec::new();

        for i in 0..128u8 {
            let cache_cloned = cache.clone();
            workers.push(tokio::spawn(async move {
                let key = vec![i; 20];
                let payloads = make_payloads();
                cache_cloned
                    .put(epoch_id, key.clone(), payloads.clone())
                    .await;
                let hit = cache_cloned.get(epoch_id, &key).await;
                assert_eq!(hit, Some(payloads));
            }));
        }

        for worker in workers {
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    async fn single_query_uses_cache_concurrently() {
        let client = Arc::new(PirClient::new(
            "http://localhost:1".to_string(),
            "http://localhost:2".to_string(),
        ));

        let epoch_id = 17;
        let key = vec![0x7B; 20];
        let payloads = make_payloads();

        client
            .cache
            .put(epoch_id, key.clone(), payloads.clone())
            .await;
        {
            let mut lock = client.metadata.write().await;
            *lock = Some(Arc::new(EpochMetadata {
                epoch_id,
                num_rows: 1000,
                seeds: [1, 2, 3],
                block_number: 100,
                state_root: [0u8; 32],
            }));
        }

        let mut workers = Vec::new();
        for _ in 0..32 {
            let client_cloned = client.clone();
            let key_cloned = key.clone();
            let expected = payloads.clone();
            workers.push(tokio::spawn(async move {
                let got = client_cloned
                    .execute_pir_query_with_retry(&key_cloned)
                    .await
                    .unwrap();
                assert_eq!(got, expected);
            }));
        }

        for worker in workers {
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    async fn batch_query_uses_cache_concurrently() {
        let client = Arc::new(PirClient::new(
            "http://localhost:1".to_string(),
            "http://localhost:2".to_string(),
        ));
        let epoch_id = 23;
        let keys: Vec<Vec<u8>> = (0..32u8).map(|i| vec![i; 20]).collect();
        let payloads = make_payloads();

        for key in &keys {
            client
                .cache
                .put(epoch_id, key.clone(), payloads.clone())
                .await;
        }
        {
            let mut lock = client.metadata.write().await;
            *lock = Some(Arc::new(EpochMetadata {
                epoch_id,
                num_rows: 1000,
                seeds: [1, 2, 3],
                block_number: 100,
                state_root: [0u8; 32],
            }));
        }

        let mut workers = Vec::new();
        for _ in 0..16 {
            let client_cloned = client.clone();
            let keys_cloned = keys.clone();
            let expected = payloads.clone();
            workers.push(tokio::spawn(async move {
                let got = client_cloned
                    .execute_batch_pir_query_with_retry(&keys_cloned)
                    .await
                    .unwrap();
                assert_eq!(got.len(), keys_cloned.len());
                assert!(got.into_iter().all(|row| row == expected));
            }));
        }

        for worker in workers {
            worker.await.unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore]
    async fn benchmark_cache_contention_sharded_vs_single_mutex() {
        let key_count = 1024usize;
        let workers = 64usize;
        let iterations_per_worker = 4_000usize;
        let total_ops = workers * iterations_per_worker;
        let epoch_id = 11u64;
        let keys: Arc<Vec<Vec<u8>>> = Arc::new(
            (0..key_count)
                .map(|i| {
                    let mut key = vec![0u8; 20];
                    key[..8].copy_from_slice(&(i as u64).to_le_bytes());
                    key
                })
                .collect(),
        );
        let payloads = make_payloads();

        let single_cache = Arc::new(Mutex::new(SingleLockPirCache::new(4096)));
        {
            let mut cache = single_cache.lock().await;
            for key in keys.iter() {
                cache.put(epoch_id, key.clone(), payloads.clone());
            }
        }

        let sharded_cache = Arc::new(PirCache::with_shards(4096, 32));
        for key in keys.iter() {
            sharded_cache
                .put(epoch_id, key.clone(), payloads.clone())
                .await;
        }

        let start_single_read = Instant::now();
        let mut single_read_workers = Vec::new();
        for w in 0..workers {
            let cache = single_cache.clone();
            let keys = keys.clone();
            single_read_workers.push(tokio::spawn(async move {
                for i in 0..iterations_per_worker {
                    let idx = ((w * iterations_per_worker) + i) % keys.len();
                    let key = &keys[idx];
                    let guard = cache.lock().await;
                    let hit = guard.get(epoch_id, key);
                    drop(guard);
                    assert!(hit.is_some());
                }
            }));
        }
        for worker in single_read_workers {
            worker.await.unwrap();
        }
        let single_read_elapsed = start_single_read.elapsed();

        let start_sharded_read = Instant::now();
        let mut sharded_read_workers = Vec::new();
        for w in 0..workers {
            let cache = sharded_cache.clone();
            let keys = keys.clone();
            sharded_read_workers.push(tokio::spawn(async move {
                for i in 0..iterations_per_worker {
                    let idx = ((w * iterations_per_worker) + i) % keys.len();
                    let key = &keys[idx];
                    let hit = cache.get(epoch_id, key).await;
                    assert!(hit.is_some());
                }
            }));
        }
        for worker in sharded_read_workers {
            worker.await.unwrap();
        }
        let sharded_read_elapsed = start_sharded_read.elapsed();

        let start_single_mixed = Instant::now();
        let mut single_mixed_workers = Vec::new();
        for w in 0..workers {
            let cache = single_cache.clone();
            let keys = keys.clone();
            let payloads = payloads.clone();
            single_mixed_workers.push(tokio::spawn(async move {
                for i in 0..iterations_per_worker {
                    let idx = ((w * iterations_per_worker) + i) % keys.len();
                    let key = keys[idx].clone();
                    if i % 8 == 0 {
                        let mut guard = cache.lock().await;
                        guard.put(epoch_id, key, payloads.clone());
                    } else {
                        let guard = cache.lock().await;
                        let _ = guard.get(epoch_id, &key);
                    }
                }
            }));
        }
        for worker in single_mixed_workers {
            worker.await.unwrap();
        }
        let single_mixed_elapsed = start_single_mixed.elapsed();

        let start_sharded_mixed = Instant::now();
        let mut sharded_mixed_workers = Vec::new();
        for w in 0..workers {
            let cache = sharded_cache.clone();
            let keys = keys.clone();
            let payloads = payloads.clone();
            sharded_mixed_workers.push(tokio::spawn(async move {
                for i in 0..iterations_per_worker {
                    let idx = ((w * iterations_per_worker) + i) % keys.len();
                    let key = keys[idx].clone();
                    if i % 8 == 0 {
                        cache.put(epoch_id, key, payloads.clone()).await;
                    } else {
                        let _ = cache.get(epoch_id, &key).await;
                    }
                }
            }));
        }
        for worker in sharded_mixed_workers {
            worker.await.unwrap();
        }
        let sharded_mixed_elapsed = start_sharded_mixed.elapsed();

        let single_read_ops_per_sec = total_ops as f64 / single_read_elapsed.as_secs_f64();
        let sharded_read_ops_per_sec = total_ops as f64 / sharded_read_elapsed.as_secs_f64();
        let single_mixed_ops_per_sec = total_ops as f64 / single_mixed_elapsed.as_secs_f64();
        let sharded_mixed_ops_per_sec = total_ops as f64 / sharded_mixed_elapsed.as_secs_f64();

        println!(
            "cache_bench_read total_ops={} single_mutex_ms={:.2} sharded_ms={:.2} speedup={:.2}x single_ops_s={:.0} sharded_ops_s={:.0}",
            total_ops,
            single_read_elapsed.as_secs_f64() * 1000.0,
            sharded_read_elapsed.as_secs_f64() * 1000.0,
            sharded_read_ops_per_sec / single_read_ops_per_sec.max(1.0),
            single_read_ops_per_sec,
            sharded_read_ops_per_sec
        );
        println!(
            "cache_bench_mixed total_ops={} single_mutex_ms={:.2} sharded_ms={:.2} speedup={:.2}x single_ops_s={:.0} sharded_ops_s={:.0}",
            total_ops,
            single_mixed_elapsed.as_secs_f64() * 1000.0,
            sharded_mixed_elapsed.as_secs_f64() * 1000.0,
            sharded_mixed_ops_per_sec / single_mixed_ops_per_sec.max(1.0),
            single_mixed_ops_per_sec,
            sharded_mixed_ops_per_sec
        );
    }
}
