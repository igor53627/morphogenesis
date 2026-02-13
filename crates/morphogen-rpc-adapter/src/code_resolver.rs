use anyhow::{anyhow, Result};
use reqwest::Client;
use std::fs::File;
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{info, warn};

const CODE_RESOLVER_TIMEOUT: Duration = Duration::from_secs(10);
const CODE_RESOLVER_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
pub struct CodeResolver {
    dict_url: String,
    cas_base_url: String,
    file_url_root: Option<PathBuf>,
    client: Client,
}

impl CodeResolver {
    fn canonical_file_url_root(&self) -> Result<PathBuf> {
        let root = self
            .file_url_root
            .as_ref()
            .ok_or_else(|| anyhow!("File URL requires --file-url-root"))?;
        root.canonicalize()
            .map_err(|e| anyhow!("File URL root canonicalization failed: {}", e))
    }

    fn ensure_within_file_url_root(&self, canonical_path: &Path) -> Result<()> {
        let canonical_root = self.canonical_file_url_root()?;
        if !canonical_path.starts_with(&canonical_root) {
            return Err(anyhow!(
                "File URL path '{}' is outside allowed root '{}'",
                canonical_path.display(),
                canonical_root.display()
            ));
        }
        Ok(())
    }

    fn file_path_from_url(&self, url: &str) -> Result<Option<PathBuf>> {
        if !url.starts_with("file://") {
            return Ok(None);
        }

        let parsed =
            reqwest::Url::parse(url).map_err(|e| anyhow!("Invalid file URL '{}': {}", url, e))?;
        let path = parsed
            .to_file_path()
            .map_err(|_| anyhow!("Invalid file URL path '{}'", url))?;
        let canonical = path
            .canonicalize()
            .map_err(|e| anyhow!("File URL path canonicalization failed '{}': {}", url, e))?;
        if self.file_url_root.is_none() {
            return Err(anyhow!("File URL '{}' requires --file-url-root", url));
        }
        self.ensure_within_file_url_root(&canonical)?;
        Ok(Some(canonical))
    }

    pub fn new(dict_url: String, cas_base_url: String) -> Self {
        Self::new_with_file_url_root(dict_url, cas_base_url, None)
    }

    pub fn new_with_file_url_root(
        dict_url: String,
        cas_base_url: String,
        file_url_root: Option<PathBuf>,
    ) -> Self {
        let client = Client::builder()
            .timeout(CODE_RESOLVER_TIMEOUT)
            .connect_timeout(CODE_RESOLVER_CONNECT_TIMEOUT)
            .build()
            .unwrap_or_else(|e| {
                warn!(
                    "Failed to build CodeResolver HTTP client ({}), using defaults",
                    e
                );
                Client::new()
            });

        Self {
            dict_url,
            cas_base_url,
            file_url_root,
            client,
        }
    }

    /// Resolves a 4-byte CodeID to a 32-byte CodeHash using the public dictionary.
    /// Uses HTTP Range Requests to fetch only the needed 32 bytes.
    pub async fn resolve_code_hash(&self, code_id: u32) -> Result<[u8; 32]> {
        let offset = code_id as u64 * 32;
        let range_header = format!("bytes={}-{}", offset, offset + 31);

        info!("Resolving CodeID {} via Range: {}", code_id, range_header);

        if let Some(path) = self.file_path_from_url(&self.dict_url)? {
            let hash = tokio::task::spawn_blocking(move || {
                let mut file =
                    File::open(&path).map_err(|e| anyhow!("Dictionary file open failed: {}", e))?;
                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| anyhow!("Dictionary seek failed: {}", e))?;
                let mut hash = [0u8; 32];
                file.read_exact(&mut hash).map_err(|e| {
                    if e.kind() == ErrorKind::UnexpectedEof {
                        anyhow!("Dictionary file too short for CodeID {}", code_id)
                    } else {
                        anyhow!("Dictionary read failed for CodeID {}: {}", code_id, e)
                    }
                })?;
                Ok::<[u8; 32], anyhow::Error>(hash)
            })
            .await
            .map_err(|e| anyhow!("Dictionary file task join failed: {}", e))??;
            return Ok(hash);
        }

        let response = self
            .client
            .get(&self.dict_url)
            .header("Range", range_header)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    anyhow!("Dictionary request timed out for CodeID {}", code_id)
                } else if e.is_connect() {
                    anyhow!("Dictionary server unreachable for CodeID {}", code_id)
                } else {
                    anyhow!("Dictionary fetch failed for CodeID {}: {}", code_id, e)
                }
            })?;

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

        if let Some(base_path) = self.file_path_from_url(&self.cas_base_url)? {
            let file_path = base_path
                .join(shard1)
                .join(shard2)
                .join(format!("{hex_hash}.bin"));
            let canonical_file_path = file_path
                .canonicalize()
                .map_err(|e| anyhow!("CAS file canonicalization failed for {}: {}", hex_hash, e))?;
            self.ensure_within_file_url_root(&canonical_file_path)?;
            let read_hash = hex_hash.clone();
            let join_hash = hex_hash.clone();
            let bytes = tokio::task::spawn_blocking(move || {
                std::fs::read(&canonical_file_path)
                    .map_err(|e| anyhow!("CAS file read failed for {}: {}", read_hash, e))
            })
            .await
            .map_err(|e| anyhow!("CAS file task join failed for {}: {}", join_hash, e))??;
            return Ok(bytes);
        }

        let response = self.client.get(&url).send().await.map_err(|e| {
            if e.is_timeout() {
                anyhow!("CAS request timed out for {}", hex_hash)
            } else if e.is_connect() {
                anyhow!("CAS server unreachable for {}", hex_hash)
            } else {
                anyhow!("CAS fetch failed for {}: {}", hex_hash, e)
            }
        })?;

        if !response.status().is_success() {
            return Err(anyhow!("CAS fetch failed: {}", response.status()));
        }

        let bytes = response.bytes().await?;
        Ok(bytes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use std::path::Path;
    use tempfile::{Builder, TempDir};

    fn file_url(path: &Path) -> String {
        reqwest::Url::from_file_path(path)
            .expect("path should convert to file URL")
            .to_string()
    }

    fn workspace_temp_dir(name: &str) -> TempDir {
        let prefix = format!("morphogen_code_resolver_{name}_");
        let cwd = std::env::current_dir().expect("read current dir");
        Builder::new()
            .prefix(&prefix)
            .tempdir_in(cwd)
            .expect("create temp dir in workspace")
    }

    fn resolver_with_root(dict_path: &Path, cas_path: &Path, root: &Path) -> CodeResolver {
        CodeResolver::new_with_file_url_root(
            file_url(dict_path),
            file_url(cas_path),
            Some(root.to_path_buf()),
        )
    }

    #[tokio::test]
    async fn resolve_code_hash_from_file_url_reads_correct_offset() {
        let dir = workspace_temp_dir("dict_ok");
        let dict_path = dir.path().join("mainnet_compact.dict");

        // code_id=0 => 32x00, code_id=1 => 32xAA
        let mut dict = vec![0u8; 64];
        dict[32..64].fill(0xAA);
        fs::write(&dict_path, dict).expect("write dictionary");

        let resolver = resolver_with_root(&dict_path, dir.path(), dir.path());

        let hash = resolver
            .resolve_code_hash(1)
            .await
            .expect("resolve code hash from file");
        assert_eq!(hash, [0xAA; 32]);
    }

    #[tokio::test]
    async fn fetch_bytecode_from_file_url_reads_expected_blob() {
        let dir = workspace_temp_dir("cas_ok");
        let cas_base = dir.path().join("cas");
        let hash_hex =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        let shard1 = &hash_hex[0..2];
        let shard2 = &hash_hex[2..4];
        let blob_dir = cas_base.join(shard1).join(shard2);
        fs::create_dir_all(&blob_dir).expect("create blob dir");
        let blob_path = blob_dir.join(format!("{hash_hex}.bin"));
        fs::write(&blob_path, [0x60, 0x01, 0x60, 0x01]).expect("write blob");

        let resolver = resolver_with_root(&dir.path().join("dict"), &cas_base, dir.path());

        let bytes = resolver
            .fetch_bytecode([0xAA; 32])
            .await
            .expect("fetch bytecode from file");
        assert_eq!(bytes, vec![0x60, 0x01, 0x60, 0x01]);
    }

    #[tokio::test]
    async fn resolve_code_hash_rejects_file_url_without_explicit_root() {
        let dir = workspace_temp_dir("dict_requires_root");
        let dict_path = dir.path().join("mainnet_compact.dict");
        fs::write(&dict_path, vec![0u8; 32]).expect("write dictionary");
        let cas_path = dir.path().join("cas");
        fs::create_dir_all(&cas_path).expect("create cas dir");

        let resolver = CodeResolver::new(file_url(&dict_path), file_url(&cas_path));
        let err = resolver
            .resolve_code_hash(0)
            .await
            .expect_err("file URL without explicit root must be rejected");
        assert!(
            err.to_string().contains("requires --file-url-root"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn resolve_code_hash_rejects_file_url_outside_root() {
        let allowed = workspace_temp_dir("allowed_root");
        let outside = tempfile::tempdir().expect("create temp dir");
        let dict_path = outside.path().join("mainnet_compact.dict");
        fs::write(&dict_path, vec![0u8; 32]).expect("write dictionary");
        let cas_path = outside.path().join("cas");
        fs::create_dir_all(&cas_path).expect("create cas dir");

        let resolver = resolver_with_root(&dict_path, &cas_path, allowed.path());
        let err = resolver
            .resolve_code_hash(0)
            .await
            .expect_err("outside-root file URL must be rejected");
        assert!(
            err.to_string().contains("outside allowed root"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn fetch_bytecode_rejects_file_url_outside_root() {
        let allowed = workspace_temp_dir("allowed_root_cas");
        let outside = tempfile::tempdir().expect("create temp dir");
        let cas_base = outside.path().join("cas");
        let hash_hex =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        let shard1 = &hash_hex[0..2];
        let shard2 = &hash_hex[2..4];
        let blob_dir = cas_base.join(shard1).join(shard2);
        fs::create_dir_all(&blob_dir).expect("create blob dir");
        let blob_path = blob_dir.join(format!("{hash_hex}.bin"));
        fs::write(&blob_path, [0x60, 0x01, 0x60, 0x01]).expect("write blob");

        let resolver = resolver_with_root(&outside.path().join("dict"), &cas_base, allowed.path());
        let err = resolver
            .fetch_bytecode([0xAA; 32])
            .await
            .expect_err("outside-root CAS file URL must be rejected");
        assert!(
            err.to_string().contains("outside allowed root"),
            "unexpected error: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fetch_bytecode_rejects_symlink_escape_from_allowed_root() {
        let allowed = workspace_temp_dir("allowed_root_symlink");
        let outside = tempfile::tempdir().expect("create temp dir");
        let cas_base = allowed.path().join("cas");
        fs::create_dir_all(&cas_base).expect("create cas dir");

        let hash_hex =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        let shard1 = &hash_hex[0..2];
        let shard2 = &hash_hex[2..4];

        let outside_blob_dir = outside.path().join(shard1).join(shard2);
        fs::create_dir_all(&outside_blob_dir).expect("create outside blob dir");
        let outside_blob_path = outside_blob_dir.join(format!("{hash_hex}.bin"));
        fs::write(&outside_blob_path, [0x60, 0x01, 0x60, 0x01]).expect("write outside blob");

        symlink(outside.path().join(shard1), cas_base.join(shard1)).expect("create symlink");

        let resolver = resolver_with_root(&allowed.path().join("dict"), &cas_base, allowed.path());
        let err = resolver
            .fetch_bytecode([0xAA; 32])
            .await
            .expect_err("symlink-based root escape must be rejected");
        assert!(
            err.to_string().contains("outside allowed root"),
            "unexpected error: {err}"
        );
    }
}
