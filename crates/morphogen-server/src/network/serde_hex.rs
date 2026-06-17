//! Serde helpers for hex-encoded byte payloads in JSON-RPC request/response
//! types.
//!
//! Extracted from `network/api.rs` in TASK-54.8. Private to the `network`
//! module — NOT part of the crate's public API (kept out of `pub use api::*`).
//!
//! Three submodules cover the payload shapes used by the API:
//!
//! - [`hex_bytes`] — fixed-size `[u8; 32]` (e.g. hashes, addresses packed to
//!   32 bytes). Serialize + an optional-deserialize variant for fields that
//!   may be absent.
//! - [`hex_bytes_vec`] — variable-length `Vec<u8>` (e.g. opaque PIR payloads).
//!   Serialize + deserialize.
//! - [`hex_bytes_array`] — fixed `[[u8; 16]; 2]` (e.g. DPF key pairs).
//!   Serialize-only; no deserialize path is used by the API.
//!
//! Serialized form is always `0x`-prefixed hex; the deserialize paths accept
//! either `0x`-prefixed or bare hex.
//!
//! Referenced from `#[serde(with = "...")]` attributes on the API DTOs in
//! `network/api.rs`; the path is brought into scope there via a `use`.

/// Serde helpers for fixed-size `[u8; 32]` payloads.
pub mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("0x{}", hex::encode(bytes)))
    }

    pub fn deserialize_option<'de, D>(deserializer: D) -> Result<Option<[u8; 32]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<String>::deserialize(deserializer)?;
        match opt {
            Some(s) => {
                let hex_str = s.strip_prefix("0x").unwrap_or(&s);
                let bytes = hex::decode(hex_str).map_err(serde::de::Error::custom)?;
                if bytes.len() != 32 {
                    return Err(serde::de::Error::custom(format!(
                        "expected 32 bytes, got {}",
                        bytes.len()
                    )));
                }
                let mut out = [0u8; 32];
                out.copy_from_slice(&bytes);
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }
}

/// Serde helpers for variable-length `Vec<u8>` payloads.
pub mod hex_bytes_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &[Vec<u8>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded: Vec<String> = bytes
            .iter()
            .map(|b| format!("0x{}", hex::encode(b)))
            .collect();
        encoded.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let items: Vec<String> = Vec::deserialize(deserializer)?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let hex_str = item.strip_prefix("0x").unwrap_or(&item);
            let bytes = hex::decode(hex_str).map_err(serde::de::Error::custom)?;
            out.push(bytes);
        }
        Ok(out)
    }
}

/// Serde helpers for fixed `[[u8; 16]; 2]` payloads (DPF key pairs).
/// Serialize-only — the API never deserializes this shape.
pub mod hex_bytes_array {
    use serde::Serializer;

    pub fn serialize<S>(keys: &[[u8; 16]; 2], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(2))?;
        for key in keys {
            seq.serialize_element(&format!("0x{}", hex::encode(key)))?;
        }
        seq.end()
    }
}
