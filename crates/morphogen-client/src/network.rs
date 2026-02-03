use crate::{
    aggregate_responses, generate_query, AccountData, EpochMetadata, ServerResponse,
    StorageData, QUERIES_PER_REQUEST,
};
use anyhow::{anyhow, Result};
use rand::thread_rng;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Deserialize)]
struct RawEpochResponse {
    epoch_id: u64,
    num_rows: usize,
    seeds: [u64; 3],
    block_number: u64,
    state_root: String,
}

pub struct PirClient {
    server_a_url: String,
    server_b_url: String,
    http_client: reqwest::Client,
    metadata: RwLock<Option<Arc<EpochMetadata>>>,
}

impl PirClient {
    pub fn new(server_a_url: String, server_b_url: String) -> Self {
        Self {
            server_a_url,
            server_b_url,
            http_client: reqwest::Client::new(),
            metadata: RwLock::new(None),
        }
    }

    pub async fn update_metadata(&self) -> Result<Arc<EpochMetadata>> {
        let url = format!("{}/epoch", self.server_a_url);
        let resp: RawEpochResponse = self.http_client.get(url).send().await?.json().await?;

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

    pub async fn query_account(&self, address: [u8; 20]) -> Result<AccountData> {
        let metadata = self.get_metadata().await?;

        let query_keys = {
            let mut rng = thread_rng();
            generate_query(&mut rng, &address, &metadata)
        };

        // Send queries in parallel

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

        let (resp_a, resp_b) = tokio::try_join!(req_a, req_b)?;

        #[derive(Deserialize)]
        struct QueryResponse {
            #[allow(dead_code)]
            epoch_id: u64,
            payloads: Vec<String>,
        }

        let json_a: QueryResponse = resp_a.json().await?;
        let json_b: QueryResponse = resp_b.json().await?;

        let parse_payloads = |payloads: Vec<String>| -> Result<[Vec<u8>; QUERIES_PER_REQUEST]> {
            if payloads.len() != QUERIES_PER_REQUEST {
                return Err(anyhow!("Invalid payload count"));
            }
            let mut out: [Vec<u8>; QUERIES_PER_REQUEST] = Default::default();
            for (i, p) in payloads.into_iter().enumerate() {
                let hex_p = p.strip_prefix("0x").unwrap_or(&p);
                out[i] = hex::decode(hex_p)?;
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

        let aggregated = aggregate_responses(&server_resp_a, &server_resp_b)
            .map_err(|e| anyhow!("Aggregation error: {:?}", e))?;

        for payload in aggregated.payloads.iter() {
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

    pub async fn query_storage(
        &self,
        address: [u8; 20],
        slot: [u8; 32],
    ) -> Result<StorageData> {
        let metadata = self.get_metadata().await?;

        // Construct 52-byte storage key: Address (20) + SlotKey (32)
        let mut storage_key = [0u8; 52];
        storage_key[0..20].copy_from_slice(&address);
        storage_key[20..52].copy_from_slice(&slot);

        let query_keys = {
            let mut rng = thread_rng();
            generate_query(&mut rng, &storage_key, &metadata)
        };

        // Send queries in parallel
        let req_a = self
            .http_client
            .post(format!("{}/query", self.server_a_url))
            .json(&serde_json::json!({
                "keys": query_keys.keys_a.iter().map(|k| format!("0x{}", hex::encode(k.to_bytes()))).collect::<Vec<_>>()
            }))
            .send();

        let req_b = self
            .http_client
            .post(format!("{}/query", self.server_b_url))
            .json(&serde_json::json!({
                "keys": query_keys.keys_b.iter().map(|k| format!("0x{}", hex::encode(k.to_bytes()))).collect::<Vec<_>>()
            }))
            .send();

        let (resp_a, resp_b) = tokio::try_join!(req_a, req_b)?;

        #[derive(Deserialize)]
        struct QueryResponse {
            #[allow(dead_code)]
            epoch_id: u64,
            payloads: Vec<String>,
        }

        let json_a: QueryResponse = resp_a.json().await?;
        let json_b: QueryResponse = resp_b.json().await?;

        let parse_payloads = |payloads: Vec<String>| -> Result<[Vec<u8>; QUERIES_PER_REQUEST]> {
            if payloads.len() != QUERIES_PER_REQUEST {
                return Err(anyhow!("Invalid payload count"));
            }
            let mut out: [Vec<u8>; QUERIES_PER_REQUEST] = Default::default();
            for (i, p) in payloads.into_iter().enumerate() {
                let hex_p = p.strip_prefix("0x").unwrap_or(&p);
                out[i] = hex::decode(hex_p)?;
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

        let aggregated = aggregate_responses(&server_resp_a, &server_resp_b)
            .map_err(|e| anyhow!("Aggregation error: {:?}", e))?;

        // Compute expected tag: Keccak256(storage_key)[0..8]
        // This matches the server-side tag computation in reth-adapter
        use alloy_primitives::keccak256;
        let tag_hash = keccak256(&storage_key);
        let expected_tag: [u8; 8] = tag_hash[0..8].try_into().unwrap();

        // Find payload with matching tag (Optimized48 scheme)
        for payload in aggregated.payloads.iter() {
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
        for payload in aggregated.payloads.iter() {
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
