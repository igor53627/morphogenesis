use crate::code_resolver::CodeResolver;
use crate::pir_db::PirDatabase;
use alloy_primitives::{Address, Bytes, U256};
use morphogen_client::network::PirClient;
use revm::{
    context::TxEnv,
    context_interface::result::{ExecutionResult, Output},
    primitives::TxKind,
    Context, ExecuteEvm, MainBuilder, MainContext,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::runtime::Handle;
use tracing::info;

/// Error types for eth_call, distinguishing input validation from internal failures.
#[derive(Debug)]
pub enum EthCallError {
    /// Invalid input parameters (maps to JSON-RPC -32602)
    InvalidParams(String),
    /// Internal execution failure (maps to JSON-RPC -32000)
    Internal(String),
}

/// Block environment fetched from upstream for accurate EVM execution.
struct BlockInfo {
    number: u64,
    timestamp: u64,
    gas_limit: u64,
    basefee: u64,
}

/// Validate block tag: must be a string (named tag or hex quantity) or null.
/// Rejects EIP-1898 object forms since PIR state is epoch-based.
fn validate_block_tag(block_tag: &Value) -> Result<String, String> {
    if block_tag.is_null() {
        return Ok("latest".to_string());
    }
    if let Some(s) = block_tag.as_str() {
        return Ok(s.to_string());
    }
    Err("unsupported block parameter (object form not supported)".to_string())
}

/// Fetch block header fields from upstream for the EVM block env.
async fn fetch_block_info(
    client: &reqwest::Client,
    upstream_url: &str,
    block_tag: &Value,
) -> Result<BlockInfo, String> {
    let tag = validate_block_tag(block_tag)?;

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getBlockByNumber",
        "params": [tag, false]
    });

    let resp = client
        .post(upstream_url)
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Block info fetch: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Block info HTTP {}", resp.status()));
    }

    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("Block info parse: {}", e))?;

    if let Some(err) = json.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown");
        return Err(format!("Block info RPC error: {}", msg));
    }

    let result = json
        .get("result")
        .filter(|v| !v.is_null())
        .ok_or("No block result from upstream")?;

    let parse_hex_u64 = |field: &str| -> Result<u64, String> {
        let s = result
            .get(field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("missing block field '{}'", field))?;
        let hex = s.strip_prefix("0x").unwrap_or(s);
        u64::from_str_radix(hex, 16).map_err(|e| format!("invalid block field '{}': {}", field, e))
    };

    // baseFeePerGas may be absent on pre-London blocks; default to 0 only if missing
    let basefee = match result.get("baseFeePerGas") {
        None => 0,
        Some(v) => {
            let s = v
                .as_str()
                .ok_or("baseFeePerGas: expected hex string")?;
            let hex = s.strip_prefix("0x").unwrap_or(s);
            u64::from_str_radix(hex, 16)
                .map_err(|e| format!("invalid baseFeePerGas: {}", e))?
        }
    };

    Ok(BlockInfo {
        number: parse_hex_u64("number")?,
        timestamp: parse_hex_u64("timestamp")?,
        gas_limit: parse_hex_u64("gasLimit")?,
        basefee,
    })
}

/// Execute an eth_call privately using revm with PIR-backed state.
/// All account, storage, and code lookups go through PIR so the
/// server never learns which addresses/slots are being accessed.
pub async fn execute_eth_call(
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    call_params: &Value,
    block_tag: &Value,
) -> Result<Bytes, EthCallError> {
    // Parse call parameters before moving into blocking task
    let from = parse_address_strict(call_params.get("from"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'from': {}", e)))?
        .unwrap_or_default();
    let to = parse_address_strict(call_params.get("to"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'to': {}", e)))?;
    let data = parse_bytes_strict(call_params.get("data").or(call_params.get("input")))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'data': {}", e)))?;
    let value = parse_u256_strict(call_params.get("value"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'value': {}", e)))?;
    let gas = parse_u64_strict(call_params.get("gas"))
        .map_err(|e| EthCallError::InvalidParams(format!("invalid 'gas': {}", e)))?
        .unwrap_or(30_000_000);

    // Validate block tag early so input errors get the right error code
    validate_block_tag(block_tag)
        .map_err(EthCallError::InvalidParams)?;

    // Fetch block env from upstream so NUMBER/TIMESTAMP/BASEFEE are accurate
    let block_info = fetch_block_info(&upstream_client, &upstream_url, block_tag)
        .await
        .map_err(EthCallError::Internal)?;

    info!(
        %from,
        to = ?to,
        data_len = data.len(),
        block = block_info.number,
        "Executing private eth_call via revm"
    );

    let handle = Handle::current();

    tokio::task::spawn_blocking(move || {
        let pir_db = PirDatabase::new(
            pir_client,
            code_resolver,
            upstream_client,
            upstream_url,
            handle,
        );

        let tx = TxEnv::builder()
            .caller(from)
            .kind(match to {
                Some(addr) => TxKind::Call(addr),
                None => TxKind::Create,
            })
            .data(data)
            .value(value)
            .gas_limit(gas)
            .build()
            .map_err(|e| EthCallError::Internal(format!("Failed to build tx: {:?}", e)))?;

        let mut evm = Context::mainnet()
            .with_db(pir_db)
            .modify_block_chained(|b| {
                b.number = U256::from(block_info.number);
                b.timestamp = U256::from(block_info.timestamp);
                b.gas_limit = block_info.gas_limit;
                b.basefee = block_info.basefee;
            })
            .build_mainnet();

        let result = evm
            .transact(tx)
            .map_err(|e| EthCallError::Internal(format!("EVM execution failed: {:?}", e)))?;

        match result.result {
            ExecutionResult::Success { output, .. } => match output {
                Output::Call(data) => Ok(data),
                Output::Create(data, _) => Ok(data),
            },
            ExecutionResult::Revert { output, .. } => Err(EthCallError::Internal(format!(
                "execution reverted: 0x{}",
                hex::encode(&output)
            ))),
            ExecutionResult::Halt { reason, .. } => {
                Err(EthCallError::Internal(format!("execution halted: {:?}", reason)))
            }
        }
    })
    .await
    .map_err(|e| EthCallError::Internal(format!("spawn_blocking: {}", e)))?
}

/// Parse an address field. Returns Ok(None) if absent, Err if present but invalid.
fn parse_address_strict(val: Option<&Value>) -> Result<Option<Address>, String> {
    let Some(v) = val else { return Ok(None) };
    if v.is_null() {
        return Ok(None);
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(hex).map_err(|e| e.to_string())?;
    if bytes.len() != 20 {
        return Err(format!("expected 20 bytes, got {}", bytes.len()));
    }
    Ok(Some(Address::from_slice(&bytes)))
}

/// Parse a bytes field. Returns Ok(empty) if absent, Err if present but invalid hex.
fn parse_bytes_strict(val: Option<&Value>) -> Result<Bytes, String> {
    let Some(v) = val else {
        return Ok(Bytes::new());
    };
    if v.is_null() {
        return Ok(Bytes::new());
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    if hex.is_empty() {
        return Ok(Bytes::new());
    }
    hex::decode(hex).map(Bytes::from).map_err(|e| e.to_string())
}

/// Parse a U256 field. Returns Ok(ZERO) if absent, Err if present but invalid.
fn parse_u256_strict(val: Option<&Value>) -> Result<U256, String> {
    let Some(v) = val else {
        return Ok(U256::ZERO);
    };
    if v.is_null() {
        return Ok(U256::ZERO);
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    if hex.is_empty() {
        return Ok(U256::ZERO);
    }
    U256::from_str_radix(hex, 16).map_err(|e| e.to_string())
}

/// Parse a u64 field. Returns Ok(None) if absent, Err if present but invalid.
fn parse_u64_strict(val: Option<&Value>) -> Result<Option<u64>, String> {
    let Some(v) = val else { return Ok(None) };
    if v.is_null() {
        return Ok(None);
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s.strip_prefix("0x").unwrap_or(s);
    if hex.is_empty() {
        return Ok(Some(0));
    }
    u64::from_str_radix(hex, 16)
        .map(Some)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_address_with_0x_prefix() {
        let val = json!("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045");
        let addr = parse_address_strict(Some(&val)).unwrap().unwrap();
        let expected: [u8; 20] = hex::decode("d8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(addr.into_array(), expected);
    }

    #[test]
    fn parse_address_without_prefix() {
        let val = json!("d8dA6BF26964aF9D7eEd9e03E53415D37aA96045");
        assert!(parse_address_strict(Some(&val)).unwrap().is_some());
    }

    #[test]
    fn parse_address_none_on_absent() {
        assert_eq!(parse_address_strict(None).unwrap(), None);
        assert_eq!(parse_address_strict(Some(&json!(null))).unwrap(), None);
    }

    #[test]
    fn parse_address_err_on_invalid() {
        assert!(parse_address_strict(Some(&json!("0xinvalid"))).is_err());
        assert!(parse_address_strict(Some(&json!("0xaabb"))).is_err()); // too short
        assert!(parse_address_strict(Some(&json!(42))).is_err()); // not a string
    }

    #[test]
    fn parse_bytes_with_data() {
        let val = json!("0x60806040");
        let bytes = parse_bytes_strict(Some(&val)).unwrap();
        assert_eq!(bytes.as_ref(), &[0x60, 0x80, 0x60, 0x40]);
    }

    #[test]
    fn parse_bytes_empty_on_none() {
        assert!(parse_bytes_strict(None).unwrap().is_empty());
        assert!(parse_bytes_strict(Some(&json!(null))).unwrap().is_empty());
    }

    #[test]
    fn parse_bytes_err_on_invalid_hex() {
        assert!(parse_bytes_strict(Some(&json!("0xZZZZ"))).is_err());
    }

    #[test]
    fn parse_u256_hex_value() {
        let val = json!("0xde0b6b3a7640000"); // 1 ETH in wei
        let result = parse_u256_strict(Some(&val)).unwrap();
        assert_eq!(result, U256::from(1_000_000_000_000_000_000u64));
    }

    #[test]
    fn parse_u256_zero_on_none() {
        assert_eq!(parse_u256_strict(None).unwrap(), U256::ZERO);
    }

    #[test]
    fn parse_u256_err_on_invalid() {
        assert!(parse_u256_strict(Some(&json!("not_hex"))).is_err());
    }

    #[test]
    fn parse_u64_gas_value() {
        let val = json!("0x1c9c380"); // 30_000_000
        assert_eq!(parse_u64_strict(Some(&val)).unwrap(), Some(30_000_000));
    }

    #[test]
    fn parse_u64_none_on_missing() {
        assert_eq!(parse_u64_strict(None).unwrap(), None);
    }

    #[test]
    fn parse_u64_err_on_invalid() {
        assert!(parse_u64_strict(Some(&json!("xyz"))).is_err());
    }

    #[test]
    fn parse_u64_bare_0x_is_zero() {
        assert_eq!(parse_u64_strict(Some(&json!("0x"))).unwrap(), Some(0));
    }

    #[test]
    fn validate_block_tag_string() {
        assert_eq!(validate_block_tag(&json!("latest")).unwrap(), "latest");
        assert_eq!(validate_block_tag(&json!("0x1")).unwrap(), "0x1");
    }

    #[test]
    fn validate_block_tag_null_defaults_latest() {
        assert_eq!(validate_block_tag(&json!(null)).unwrap(), "latest");
    }

    #[test]
    fn validate_block_tag_rejects_object() {
        let obj = json!({"blockNumber": "0x1"});
        assert!(validate_block_tag(&obj).is_err());
    }
}
