use anyhow::{anyhow, Result};
use reqwest::Client;
use tracing::info;

#[derive(Clone)]
pub struct CodeResolver {
    dict_url: String,
    cas_base_url: String,
    client: Client,
}

impl CodeResolver {
    pub fn new(dict_url: String, cas_base_url: String) -> Self {
        Self {
            dict_url,
            cas_base_url,
            client: Client::new(),
        }
    }

    /// Resolves a 4-byte CodeID to a 32-byte CodeHash using the public dictionary.
    /// Uses HTTP Range Requests to fetch only the needed 32 bytes.
    pub async fn resolve_code_hash(&self, code_id: u32) -> Result<[u8; 32]> {
        let offset = code_id as u64 * 32;
        let range_header = format!("bytes={}-{}", offset, offset + 31);

        info!("Resolving CodeID {} via Range: {}", code_id, range_header);

        let response = self
            .client
            .get(&self.dict_url)
            .header("Range", range_header)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("Dictionary fetch failed: {}", response.status()));
        }

        let bytes = response.bytes().await?;
        if bytes.len() != 32 {
            return Err(anyhow!("Invalid hash length: {}", bytes.len()));
        }

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes);
        Ok(hash)
    }

    /// Fetches the contract bytecode from the Content Addressable Storage (CAS).
    /// Path format: /shard1/shard2/hash.bin
    pub async fn fetch_bytecode(&self, code_hash: [u8; 32]) -> Result<Vec<u8>> {
        if code_hash == [0u8; 32] {
            return Ok(Vec::new()); // EOA has empty code
        }

        let hex_hash = hex::encode(code_hash);
        let shard1 = &hex_hash[0..2];
        let shard2 = &hex_hash[2..4];
        let url = format!(
            "{}/{}/{}/{}.bin",
            self.cas_base_url, shard1, shard2, hex_hash
        );

        info!("Fetching bytecode from CAS: {}", url);

        let response = self.client.get(&url).send().await?;
        if !response.status().is_success() {
            return Err(anyhow!("CAS fetch failed: {}", response.status()));
        }

        let bytes = response.bytes().await?;
        Ok(bytes.to_vec())
    }
}
