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
}

/// Known named block tags per Ethereum JSON-RPC specification.
const NAMED_BLOCK_TAGS: &[&str] = &["latest", "earliest", "pending", "safe", "finalized"];

/// Validate block tag: must be a named tag, hex quantity ("0x..."), or null.
/// Rejects EIP-1898 object forms since PIR state is epoch-based.
fn validate_block_tag(block_tag: &Value) -> Result<String, String> {
    if block_tag.is_null() {
        return Ok("latest".to_string());
    }
    let s = block_tag.as_str().ok_or_else(|| {
        if block_tag.is_object() {
            "EIP-1898 object form not supported".to_string()
        } else {
            "block parameter must be a string or null".to_string()
        }
    })?;
    if NAMED_BLOCK_TAGS.contains(&s) {
        return Ok(s.to_string());
    }
    // Must be a hex quantity: "0x" followed by one or more hex digits
    let hex = s
        .strip_prefix("0x")
        .ok_or_else(|| format!("invalid block tag '{}': expected named tag or hex quantity", s))?;
    if hex.is_empty() || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "invalid block tag '{}': expected named tag or hex quantity",
            s
        ));
    }
    Ok(s.to_string())
}

/// Fetch block header fields from upstream for the EVM block env.
/// `block_tag` must be a pre-validated tag string (from `validate_block_tag`).
async fn fetch_block_info(
    client: &reqwest::Client,
    upstream_url: &str,
    block_tag: &str,
) -> Result<BlockInfo, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getBlockByNumber",
        "params": [block_tag, false]
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

    Ok(BlockInfo {
        number: parse_hex_u64("number")?,
        timestamp: parse_hex_u64("timestamp")?,
        gas_limit: parse_hex_u64("gasLimit")?,
    })
}

/// Parsed call parameters for EVM execution.
struct CallParams {
    from: Address,
    to: Option<Address>,
    data: Bytes,
    value: U256,
    gas: u64,
}

/// Parse and validate call parameters from JSON-RPC input.
fn parse_call_params(call_params: &Value) -> Result<CallParams, EthCallError> {
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
    Ok(CallParams { from, to, data, value, gas })
}

/// Run EVM execution and return the raw ExecutionResult.
/// Sets basefee=0 for simulation so calls with gas_price=0 don't fail
/// (matches Geth's behavior for eth_call/eth_estimateGas).
async fn run_evm(
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    call_params: &Value,
    block_tag: &Value,
) -> Result<(ExecutionResult, CallParams), EthCallError> {
    let parsed = parse_call_params(call_params)?;

    let validated_tag = validate_block_tag(block_tag)
        .map_err(EthCallError::InvalidParams)?;

    let block_info = fetch_block_info(&upstream_client, &upstream_url, &validated_tag)
        .await
        .map_err(EthCallError::Internal)?;

    let from = parsed.from;
    let to = parsed.to;
    let data = parsed.data.clone();
    let value = parsed.value;
    let gas = parsed.gas;

    let handle = Handle::current();

    let result = tokio::task::spawn_blocking(move || {
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
                // Set basefee=0 for simulation so calls with gas_price=0 don't fail
                // with GasPriceLessThanBasefee. Matches Geth's eth_call behavior.
                b.basefee = 0;
            })
            .build_mainnet();

        let result = evm
            .transact(tx)
            .map_err(|e| EthCallError::Internal(format!("EVM execution failed: {:?}", e)))?;

        Ok(result.result)
    })
    .await
    .map_err(|e| EthCallError::Internal(format!("spawn_blocking: {}", e)))??;

    Ok((result, parsed))
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
    info!("Executing private eth_call via revm");

    let (result, _) = run_evm(
        pir_client, code_resolver, upstream_client, upstream_url,
        call_params, block_tag,
    ).await?;

    match result {
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
}

/// Execute an eth_estimateGas privately using revm with PIR-backed state.
/// Returns gas_used with a 20% safety margin (standard practice, matches Geth).
pub async fn execute_eth_estimate_gas(
    pir_client: Arc<PirClient>,
    code_resolver: Arc<CodeResolver>,
    upstream_client: reqwest::Client,
    upstream_url: String,
    call_params: &Value,
    block_tag: &Value,
) -> Result<u64, EthCallError> {
    info!("Executing private eth_estimateGas via revm");

    let (result, _) = run_evm(
        pir_client, code_resolver, upstream_client, upstream_url,
        call_params, block_tag,
    ).await?;

    match result {
        ExecutionResult::Success { gas_used, .. } => {
            // 20% safety margin, standard practice
            Ok(gas_used * 120 / 100)
        }
        ExecutionResult::Revert { output, .. } => Err(EthCallError::Internal(format!(
            "execution reverted: 0x{}",
            hex::encode(&output)
        ))),
        ExecutionResult::Halt { reason, .. } => {
            Err(EthCallError::Internal(format!("execution halted: {:?}", reason)))
        }
    }
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
/// Follows Ethereum JSON-RPC hex quantity rules: requires "0x" prefix, "0x0" for zero.
fn parse_u64_strict(val: Option<&Value>) -> Result<Option<u64>, String> {
    let Some(v) = val else { return Ok(None) };
    if v.is_null() {
        return Ok(None);
    }
    let s = v.as_str().ok_or("expected hex string")?;
    let hex = s
        .strip_prefix("0x")
        .ok_or_else(|| format!("expected 0x-prefixed hex quantity, got '{}'", s))?;
    if hex.is_empty() {
        return Err("invalid hex quantity: no digits after prefix".to_string());
    }
    // Reject leading zeros per JSON-RPC quantity encoding (except "0x0")
    if hex.len() > 1 && hex.starts_with('0') {
        return Err(format!("invalid hex quantity '{}': leading zeros not allowed", s));
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
    fn parse_u64_bare_0x_is_error() {
        assert!(parse_u64_strict(Some(&json!("0x"))).is_err());
    }

    #[test]
    fn parse_u64_0x0_is_zero() {
        assert_eq!(parse_u64_strict(Some(&json!("0x0"))).unwrap(), Some(0));
    }

    #[test]
    fn validate_block_tag_named_tags() {
        for tag in &["latest", "earliest", "pending", "safe", "finalized"] {
            assert_eq!(validate_block_tag(&json!(tag)).unwrap(), *tag);
        }
    }

    #[test]
    fn validate_block_tag_hex_quantity() {
        assert_eq!(validate_block_tag(&json!("0x1")).unwrap(), "0x1");
        assert_eq!(validate_block_tag(&json!("0xff")).unwrap(), "0xff");
    }

    #[test]
    fn validate_block_tag_null_defaults_latest() {
        assert_eq!(validate_block_tag(&json!(null)).unwrap(), "latest");
    }

    #[test]
    fn validate_block_tag_rejects_object() {
        assert!(validate_block_tag(&json!({"blockNumber": "0x1"})).is_err());
    }

    #[test]
    fn validate_block_tag_rejects_invalid_string() {
        assert!(validate_block_tag(&json!("foo")).is_err());
        assert!(validate_block_tag(&json!("0x")).is_err()); // no digits
        assert!(validate_block_tag(&json!("0xZZ")).is_err()); // not hex
    }

    #[test]
    fn validate_block_tag_rejects_non_string_types() {
        assert!(validate_block_tag(&json!(1)).is_err());
        assert!(validate_block_tag(&json!(true)).is_err());
        assert!(validate_block_tag(&json!([1, 2])).is_err());
    }

    #[test]
    fn parse_u64_rejects_no_prefix() {
        assert!(parse_u64_strict(Some(&json!("10"))).is_err());
    }

    #[test]
    fn parse_u64_rejects_leading_zeros() {
        assert!(parse_u64_strict(Some(&json!("0x00"))).is_err());
        assert!(parse_u64_strict(Some(&json!("0x01"))).is_err());
    }
}
