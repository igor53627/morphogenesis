use crate::code_resolver::CodeResolver;
use morphogen_client::network::PirClient;
use revm::{
    bytecode::Bytecode,
    database::Database,
    database_interface::DBErrorMarker,
    primitives::{Address, B256, KECCAK_EMPTY, U256},
    state::AccountInfo,
};
use std::sync::Arc;
use tokio::runtime::Handle;
use tracing::debug;

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
        }
    }
}

impl Database for PirDatabase {
    type Error = PirDbError;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        let addr_bytes: [u8; 20] = address.into_array();

        let account = self
            .handle
            .block_on(self.pir_client.query_account(addr_bytes))
            .map_err(|e| PirDbError(format!("PIR account query: {}", e)))?;

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

            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| PirDbError(format!("Block hash parse: {}", e)))?;

            let hash_str = json
                .get("result")
                .and_then(|r| r.get("hash"))
                .and_then(|h| h.as_str())
                .ok_or_else(|| PirDbError(format!("No block hash for block {}", number)))?;

            let hash_hex = hash_str.strip_prefix("0x").unwrap_or(hash_str);
            let mut hash = [0u8; 32];
            hex::decode_to_slice(hash_hex, &mut hash)
                .map_err(|e| PirDbError(format!("Block hash decode: {}", e)))?;

            Ok(B256::from(hash))
        })
    }
}
