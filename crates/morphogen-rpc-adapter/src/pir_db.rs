use crate::code_resolver::CodeResolver;
use morphogen_client::network::PirClient;
use revm::{
    bytecode::Bytecode,
    database::Database,
    database_interface::DBErrorMarker,
    primitives::{Address, B256, KECCAK_EMPTY, U256},
    state::AccountInfo,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use tokio::runtime::Handle;
use tracing::{debug, warn};

/// Error type for PIR-backed database operations.
#[derive(Debug)]
pub struct PirDbError(pub String);

impl std::fmt::Display for PirDbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PirDbError {}
impl DBErrorMarker for PirDbError {}

/// Tracks account/storage touches during EVM execution to build an EIP-2930 access list.
#[derive(Debug, Default)]
pub struct AccessListCollector {
    entries: BTreeMap<[u8; 20], BTreeSet<[u8; 32]>>,
}

impl AccessListCollector {
    /// Record account access (without storage touch).
    pub fn record_account(&mut self, address: [u8; 20]) {
        self.entries.entry(address).or_default();
    }

    /// Record storage access for a given account/slot pair.
    pub fn record_storage(&mut self, address: [u8; 20], slot: [u8; 32]) {
        self.entries.entry(address).or_default().insert(slot);
    }

    /// Remove an account only when no storage keys are recorded for it.
    pub fn remove_account_if_empty(&mut self, address: &[u8; 20]) {
        let should_remove = self
            .entries
            .get(address)
            .is_some_and(std::collections::BTreeSet::is_empty);
        if should_remove {
            self.entries.remove(address);
        }
    }

    /// Serialize collected entries into Ethereum JSON-RPC access list shape.
    pub fn to_rpc_entries(&self) -> Value {
        Value::Array(
            self.entries
                .iter()
                .map(|(address, slots)| {
                    json!({
                        "address": format!("0x{}", hex::encode(address)),
                        "storageKeys": slots
                            .iter()
                            .map(|slot| Value::String(format!("0x{}", hex::encode(slot))))
                            .collect::<Vec<_>>()
                    })
                })
                .collect(),
        )
    }
}

pub type SharedAccessListCollector = Arc<Mutex<AccessListCollector>>;

/// A revm Database implementation backed by PIR queries for private state access.
/// Account data, code, and storage are fetched via PIR (private information retrieval)
/// so the server never learns which accounts/slots are being accessed.
/// Block hashes are proxied to upstream (not privacy-sensitive).
pub struct PirDatabase {
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    handle: Handle,
    access_list_collector: Option<SharedAccessListCollector>,
}

impl PirDatabase {
    pub fn new(
        pir_client: Arc<PirClient>,
        code_resolver: Arc<CodeResolver>,
        upstream_client: reqwest::Client,
        upstream_url: String,
        handle: Handle,
    ) -> Self {
        Self {
            pir_client,
            code_resolver,
            upstream_client,
            upstream_url,
            handle,
            access_list_collector: None,
        }
    }

    pub fn new_with_access_list_collector(
        pir_client: Arc<PirClient>,
        code_resolver: Arc<CodeResolver>,
        upstream_client: reqwest::Client,
        upstream_url: String,
        handle: Handle,
        access_list_collector: SharedAccessListCollector,
    ) -> Self {
        Self {
            pir_client,
            code_resolver,
            upstream_client,
            upstream_url,
            handle,
            access_list_collector: Some(access_list_collector),
        }
    }

    fn record_account_access(&self, address: [u8; 20]) {
        if let Some(collector) = &self.access_list_collector {
            let mut guard = match collector.lock() {
                Ok(g) => g,
                Err(poisoned) => {
                    warn!(
                        "AccessListCollector mutex poisoned in record_account_access; recovering"
                    );
                    poisoned.into_inner()
                }
            };
            guard.record_account(address);
        }
    }

    fn record_storage_access(&self, address: [u8; 20], slot: [u8; 32]) {
        if let Some(collector) = &self.access_list_collector {
            let mut guard = match collector.lock() {
                Ok(g) => g,
                Err(poisoned) => {
                    warn!(
                        "AccessListCollector mutex poisoned in record_storage_access; recovering"
                    );
                    poisoned.into_inner()
                }
            };
            guard.record_storage(address, slot);
        }
    }
}

impl Database for PirDatabase {
    type Error = PirDbError;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        let addr_bytes: [u8; 20] = address.into_array();
        self.record_account_access(addr_bytes);

        let account = self
            .handle
            .block_on(self.pir_client.query_account(addr_bytes))
            .map_err(|e| PirDbError(format!("PIR account query: {}", e)))?;

        // Non-existent account: return None so EXTCODEHASH returns 0x0
        let has_code = account.code_id.is_some_and(|id| id > 0) || account.code_hash.is_some();
        if account.nonce == 0 && account.balance == 0 && !has_code {
            return Ok(None);
        }

        // Resolve code_hash from code_id via dictionary
        let code_hash = if let Some(code_id) = account.code_id {
            if code_id > 0 {
                let hash = self
                    .handle
                    .block_on(self.code_resolver.resolve_code_hash(code_id))
                    .map_err(|e| PirDbError(format!("Code hash resolution: {}", e)))?;
                B256::from(hash)
            } else {
                KECCAK_EMPTY
            }
        } else if let Some(hash) = account.code_hash {
            B256::from(hash)
        } else {
            KECCAK_EMPTY
        };

        debug!(
            %address,
            nonce = account.nonce,
            balance = account.balance,
            "PIR: loaded account"
        );

        Ok(Some(AccountInfo {
            balance: U256::from(account.balance),
            nonce: account.nonce,
            code_hash,
            account_id: None,
            code: None, // Loaded lazily via code_by_hash
        }))
    }

    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        if code_hash == KECCAK_EMPTY || code_hash == B256::ZERO {
            return Ok(Bytecode::default());
        }

        let bytes = self
            .handle
            .block_on(self.code_resolver.fetch_bytecode(code_hash.0))
            .map_err(|e| PirDbError(format!("Bytecode fetch: {}", e)))?;

        debug!(%code_hash, len = bytes.len(), "PIR: loaded bytecode");

        Ok(Bytecode::new_raw(bytes.into()))
    }

    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        let addr_bytes: [u8; 20] = address.into_array();
        let slot_bytes: [u8; 32] = index.to_be_bytes();
        self.record_storage_access(addr_bytes, slot_bytes);

        let storage = self
            .handle
            .block_on(self.pir_client.query_storage(addr_bytes, slot_bytes))
            .map_err(|e| PirDbError(format!("PIR storage query: {}", e)))?;

        let value = U256::from_be_bytes(storage.value);

        debug!(
            %address,
            slot = %index,
            %value,
            "PIR: loaded storage"
        );

        Ok(value)
    }

    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        // Block hashes are not privacy-sensitive; proxy to upstream
        self.handle.block_on(async {
            let request = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBlockByNumber",
                "params": [format!("0x{:x}", number), false]
            });

            let resp = self
                .upstream_client
                .post(&self.upstream_url)
                .json(&request)
                .send()
                .await
                .map_err(|e| PirDbError(format!("Block hash fetch: {}", e)))?;

            if !resp.status().is_success() {
                return Err(PirDbError(format!(
                    "Block hash upstream returned {}",
                    resp.status()
                )));
            }

            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| PirDbError(format!("Block hash parse: {}", e)))?;

            if let Some(err) = json.get("error") {
                let msg = err
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown");
                return Err(PirDbError(format!(
                    "Block hash RPC error for block {}: {}",
                    number, msg
                )));
            }

            // BLOCKHASH for unavailable blocks returns 0x0 per EVM spec
            let result = match json.get("result") {
                None => return Ok(B256::ZERO),
                Some(v) if v.is_null() => return Ok(B256::ZERO),
                Some(v) => v,
            };
            // hash can be null for pending blocks; treat as unavailable
            // missing hash field is an upstream schema error
            let hash_val = match result.get("hash") {
                None => {
                    return Err(PirDbError(format!(
                        "Block {} result missing 'hash' field",
                        number
                    )))
                }
                Some(v) if v.is_null() => return Ok(B256::ZERO),
                Some(v) => v,
            };
            let hash_str = hash_val.as_str().ok_or_else(|| {
                PirDbError(format!("Block {} 'hash' field is not a string", number))
            })?;

            let hash_hex = hash_str.strip_prefix("0x").unwrap_or(hash_str);
            let mut hash = [0u8; 32];
            hex::decode_to_slice(hash_hex, &mut hash)
                .map_err(|e| PirDbError(format!("Block hash decode: {}", e)))?;

            Ok(B256::from(hash))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::AccessListCollector;
    use serde_json::json;

    #[test]
    fn access_list_collector_deduplicates_and_sorts_entries() {
        let mut collector = AccessListCollector::default();
        let addr_a = [0x11_u8; 20];
        let addr_b = [0x22_u8; 20];
        let slot_1 = [0x01_u8; 32];
        let slot_2 = [0x02_u8; 32];

        collector.record_account(addr_b);
        collector.record_storage(addr_a, slot_2);
        collector.record_storage(addr_a, slot_1);
        collector.record_storage(addr_a, slot_1);

        assert_eq!(
            collector.to_rpc_entries(),
            json!([
                {
                    "address": format!("0x{}", hex::encode(addr_a)),
                    "storageKeys": [
                        format!("0x{}", hex::encode(slot_1)),
                        format!("0x{}", hex::encode(slot_2))
                    ]
                },
                {
                    "address": format!("0x{}", hex::encode(addr_b)),
                    "storageKeys": []
                }
            ])
        );
    }

    #[test]
    fn access_list_collector_records_account_without_storage() {
        let mut collector = AccessListCollector::default();
        let addr = [0xab_u8; 20];
        collector.record_account(addr);

        assert_eq!(
            collector.to_rpc_entries(),
            json!([{
                "address": format!("0x{}", hex::encode(addr)),
                "storageKeys": []
            }])
        );
    }
}
